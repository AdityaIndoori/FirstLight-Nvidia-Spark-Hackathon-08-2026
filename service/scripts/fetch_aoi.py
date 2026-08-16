#!/usr/bin/env python3
"""Fetch the configured AOI's county GIS. Run ONCE while a link exists.

WHY this is a table and not three hardcoded URLs: the AOI moved once already,
from West Seattle to Pinellas, and this script did not move with it. It happily
queried King County layers with a Florida bbox, returned zero features, and wrote
an empty FeatureCollection that looked like a successful run. So: sources are
keyed by AOI, the script refuses to run against an AOI it has no sources for, and
it FAILS LOUDLY on an empty result instead of writing an empty file.

Every endpoint and count below was verified live against its own bbox.

Owner-identity columns are dropped here, at ingest, before anything downstream
can see them, because the assessor join is exactly where they would leak in.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.datasets import DROP_COLUMNS  # noqa: E402

PAGE = 1000

# Per-AOI sources. `layer` matters: Pinellas parcels answer on layer 1 and layer
# 0 returns HTTP 400, which is the kind of detail that turns into an empty file
# if it is guessed rather than probed.
SOURCES: dict[str, dict[str, dict]] = {
    "pinellas": {
        "parcels": {
            "url": "https://egis.pinellas.gov/gis/rest/services/WebGIS/Parcels/MapServer/1/query",
            "expect": 39166,
            "note": "SITE_ADDRESS, SITE_CITY, USE_CODE, FIRE_DISTRICT",
        },
        # Pinellas publishes no building-footprint service. That is why the
        # footprint tier for this AOI is Microsoft GlobalMLBuildingFootprints,
        # refreshed by name through the librarian.
        "facilities": {
            "url": "https://egis.pinellas.gov/gis/rest/services/PublicWebGIS/General/MapServer/{layer}/query",
            "layers": {1: "fire_station", 2: "hospital", 0: "police_station", 4: "school", 9: "school"},
            "expect": 36,
            "note": "layers 3 and 18 return HTTP 400, so failures are skipped not fatal",
        },
        "road_closures": {
            "url": "https://egis.pinellas.gov/gis/rest/services/RoadClosures/GraySkyRoadClosures_Public/MapServer/{layer}/query",
            "layers": {i: f"closure_layer_{i}" for i in range(12)},
            "expect": 0,
            "note": (
                "live during-the-event feed. Zero features on a clear day is CORRECT, "
                "not a failure. B4's blocked roads come from the operator on a dry run."
            ),
        },
    },
    "sarasota": {
        "footprints": {
            "url": "https://services1.arcgis.com/l0ykbnLPGRAKyBmN/arcgis/rest/services/BuildingFootprint/FeatureServer/0/query",
            "expect": 34620,
            "note": "the one Florida AOI with a county footprint layer",
        },
    },
    "bay": {
        "parcels": {
            # The Florida GIO statewide layer, not a county service. Bay County's
            # own Parcels FeatureServer answers HTTP 400 for every layer id, and
            # the state layer covers every Florida AOI with one endpoint, which is
            # also why it is the right tier for the next county after this one.
            # PHY_ADDR1 is the physical site address, which is what a dispatch label
            # needs; OWN_ADDR1 and OWN_CITY are owner fields and the scrub drops
            # them.
            "url": (
                "https://services9.arcgis.com/Gh9awoU677aKree0/arcgis/rest/services/"
                "Florida_Statewide_Parcel_Centroid_Version/FeatureServer/0/query"
            ),
            "expect": 24114,
            "note": "Michael 2018, the AOI with a true pre/post aerial pair",
        },
    },
}


def _page(url: str, bbox: str, offset: int) -> dict:
    params = {
        "where": "1=1",
        "geometry": bbox,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE),
        "f": "geojson",
    }
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=90) as r:
        return json.load(r)


def count(url: str, bbox: str) -> int | None:
    """Probe before paging. A count of zero here is the finding, not the pages."""
    params = {
        "where": "1=1",
        "geometry": bbox,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnCountOnly": "true",
        "f": "json",
    }
    try:
        # Generous: a statewide layer counting inside a bbox took 87 s measured,
        # and a probe that times out reads as "endpoint unavailable" and skips a
        # dataset that was in fact there.
        with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=180) as r:
            body = json.load(r)
        if "error" in body:
            return None
        return int(body.get("count", 0))
    except Exception:
        return None


# Exact-match column lists do not survive a county change. DROP_COLUMNS was
# written against King County (OWNER, OWNER_NAME) and Pinellas ships OWNER1,
# OWNER2, MAILTO, OWNADD_1, so an exact list silently passed real owner names
# through. Match on substrings instead: over-dropping a column costs us a field
# we were not going to use, under-dropping one puts a resident's name in an
# export.
OWNER_PATTERNS = (
    "own",      # OWNER, OWNER1, OWNADD_1, OWNERNAME
    "taxpay",
    "mail",     # MAILTO, MAILING_ADDRESS, MAILADD
    "grantee",
    "grantor",
    "deed",
)


def _scrub(props: dict) -> tuple[dict, set[str]]:
    """Drop owner-identity columns and report which ones were actually there.

    The measured set is what the privacy write-up quotes, so a claim about
    dropped fields names real column names from real data.
    """
    exact = {c.lower() for c in DROP_COLUMNS}
    kept, dropped = {}, set()
    for k, v in (props or {}).items():
        low = k.lower()
        if low in exact or any(pat in low for pat in OWNER_PATTERNS):
            dropped.add(k)
            continue
        kept[k] = v
    return kept, dropped


def fetch(url: str, bbox: str, label: str, max_pages: int = 60) -> dict:
    feats: list[dict] = []
    dropped_seen: set[str] = set()
    for i in range(max_pages):
        page = _page(url, bbox, i * PAGE)
        got = page.get("features", []) or []
        for f in got:
            f["properties"], dropped = _scrub(f.get("properties"))
            dropped_seen |= dropped
        feats.extend(got)
        print(f"  {label}: page {i + 1}, {len(got)} features, {len(feats)} total", flush=True)
        if len(got) < PAGE:
            break
        time.sleep(0.3)
    if dropped_seen:
        print(
            f"  dropped {len(dropped_seen)} owner-identity columns at ingest: "
            f"{', '.join(sorted(dropped_seen))}",
            flush=True,
        )
    return {"type": "FeatureCollection", "features": feats, "firstlight_dropped": sorted(dropped_seen)}


def fetch_layered(spec: dict, bbox: str, label: str) -> dict:
    """Enumerate layers and SKIP the ones that 400, because several do."""
    feats: list[dict] = []
    for layer, kind in spec["layers"].items():
        url = spec["url"].format(layer=layer)
        n = count(url, bbox)
        if n is None:
            print(f"  {label} layer {layer}: unavailable, skipped", flush=True)
            continue
        if n == 0:
            print(f"  {label} layer {layer} ({kind}): 0 features", flush=True)
            continue
        fc = fetch(url, bbox, f"{label} layer {layer} ({kind})")
        for f in fc["features"]:
            f.setdefault("properties", {})["firstlight_kind"] = kind
        feats.extend(fc["features"])
    return {"type": "FeatureCollection", "features": feats}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", default=getattr(config, "AOI_NAME", "custom"))
    ap.add_argument("--bbox", default=",".join(str(x) for x in config.AOI))
    ap.add_argument("--out", default=str(config.DATASET_DIR))
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="write a dataset even when it came back empty (default: refuse)",
    )
    args = ap.parse_args(argv)

    aoi_name = args.aoi.strip().lower()
    sources = SOURCES.get(aoi_name)
    if not sources:
        print(f"No sources are registered for AOI {aoi_name!r}.")
        print(f"Registered: {', '.join(sorted(SOURCES))}")
        print("Add the county's endpoints to SOURCES rather than querying another")
        print("county's layers with this bbox, which returns zero and looks like success.")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"AOI {aoi_name} {args.bbox}")

    failures: list[str] = []
    for name, spec in sources.items():
        print(f"\n{name}: {spec.get('note', '')}")
        if "layers" in spec:
            fc = fetch_layered(spec, args.bbox, name)
        else:
            n = count(spec["url"], args.bbox)
            if n is None:
                print("  endpoint did not answer a count probe, skipping")
                failures.append(f"{name}: endpoint unavailable")
                continue
            print(f"  count probe: {n} features (expected about {spec.get('expect', '?')})")
            if n == 0 and spec.get("expect", 0) > 0:
                print("  REFUSING to page: zero features where this AOI should have many.")
                print("  That means the endpoint and the bbox are different geographies.")
                failures.append(f"{name}: zero features against this bbox")
                continue
            fc = fetch(spec["url"], args.bbox, name)

        got = len(fc["features"])
        expected_empty = spec.get("expect", 0) == 0
        if got == 0 and not expected_empty and not args.allow_empty:
            print(f"  NOT WRITING {name}.geojson: empty result, and an empty file")
            print("  reads downstream as 'this county has none' rather than 'the fetch failed'.")
            failures.append(f"{name}: empty, not written")
            continue

        path = out / f"{name}.geojson"
        path.write_text(json.dumps(fc), encoding="utf-8")
        print(f"  wrote {path.name}, {got} features{' (zero is expected here)' if expected_empty and got == 0 else ''}")

    print("\nStill needed, and not ArcGIS queries:")
    print("  facilities from CMS Care Compare, national, refresh by name: cms_facilities")
    print("  svi.geojson from CDC SVI, national, refresh by name: cdc_svi")
    print("  roads.geojson from an OSM extract for this AOI")
    print("  footprints from Microsoft GlobalMLBuildingFootprints where the county")
    print("    publishes none, which is Pinellas, refresh by name: ms_building_footprints")
    print("  web/tiles basemap cache: scripts/fetch_tiles.py")

    if failures:
        print("\nFAILURES, nothing was faked to hide them:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
