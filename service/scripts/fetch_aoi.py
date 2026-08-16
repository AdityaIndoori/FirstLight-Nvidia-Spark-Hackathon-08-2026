#!/usr/bin/env python3
"""Fetch the AOI's authoritative county GIS. Run ONCE while a link exists.

Sources verified reachable by live query: the Seattle 2023 building outlines
carry a parcel PIN on every polygon, so building identity is a key join rather
than a spatial guess, and King County parcels and address points fill in the
street address. Everything lands as local GeoJSON so the box works with no
connectivity afterwards.

Owner names are dropped here, at ingest, before anything downstream can see
them: the assessor join is exactly where they would leak in.
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

SEATTLE_FOOTPRINTS = (
    "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/"
    "Building_Outlines_2023/FeatureServer/0/query"
)
KC_PARCELS = (
    "https://gismaps.kingcounty.gov/arcgis/rest/services/Property/"
    "KingCo_Parcels/MapServer/0/query"
)
PAGE = 1000


def _page(url: str, bbox: str, offset: int, out_fields: str = "*") -> dict:
    params = {
        "where": "1=1",
        "geometry": bbox,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "returnGeometry": "true",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE),
        "f": "geojson",
    }
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=90) as r:
        return json.load(r)


def _scrub(props: dict) -> dict:
    """Drop owner-name columns case-insensitively, then hand back what is left."""
    lowered = {c.lower() for c in DROP_COLUMNS}
    return {k: v for k, v in (props or {}).items() if k.lower() not in lowered}


def fetch(url: str, bbox: str, label: str, max_pages: int = 40) -> dict:
    feats: list[dict] = []
    for i in range(max_pages):
        page = _page(url, bbox, i * PAGE)
        got = page.get("features", []) or []
        for f in got:
            f["properties"] = _scrub(f.get("properties"))
        feats.extend(got)
        print(f"  {label}: page {i + 1}, {len(got)} features, {len(feats)} total", flush=True)
        if len(got) < PAGE:
            break
        time.sleep(0.3)
    return {"type": "FeatureCollection", "features": feats}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", default=",".join(str(x) for x in config.AOI))
    ap.add_argument("--out", default=str(config.DATASET_DIR))
    ap.add_argument("--skip-parcels", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bbox = args.aoi

    print(f"AOI {bbox}")
    print("Seattle Building Outlines 2023 (each polygon carries a parcel PIN)")
    fp = fetch(SEATTLE_FOOTPRINTS, bbox, "footprints")
    (out / "footprints.geojson").write_text(json.dumps(fp), encoding="utf-8")
    print(f"  wrote footprints.geojson, {len(fp['features'])} features")

    if not args.skip_parcels:
        print("King County parcels (owner columns dropped at ingest)")
        pc = fetch(KC_PARCELS, bbox, "parcels")
        (out / "parcels.geojson").write_text(json.dumps(pc), encoding="utf-8")
        print(f"  wrote parcels.geojson, {len(pc['features'])} features")

    print()
    print("Still needed, and they are not ArcGIS queries:")
    print("  facilities.csv  from CMS Care Compare (facility level only)")
    print("  svi.geojson     from CDC SVI, Washington block groups")
    print("  roads.geojson   from OSM extract for the AOI")
    print("  web/tiles/      basemap raster tiles for the AOI, z12 to z18")
    print("Refresh any of the five allowlisted datasets later by NAME:")
    print("  POST /api/datasets/refresh {\"name\": \"cms_facilities\"}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
