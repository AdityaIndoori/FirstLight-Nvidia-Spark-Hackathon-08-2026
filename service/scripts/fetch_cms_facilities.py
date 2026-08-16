#!/usr/bin/env python3
"""Fetch CMS Care Compare care facilities for the configured AOI.

WHY this is separate from fetch_aoi.py: CMS is national and county-agnostic, so
it works for any AOI, while the county endpoints are per-AOI. It is also the only
source of the two facility types the vulnerability join actually cares most about
and that no county publishes: nursing homes and dialysis centres. A county's own
facility layers give fire stations, police stations and schools, none of which
earn the medical-cross marker.

FACILITY LEVEL ONLY. CMS publishes per-facility quality measures and staffing
detail; none of it is read here. The join needs a name, a type and a location.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

METASTORE = (
    "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items"
    "?show-reference-ids=true"
)

# Dataset title in the CMS metastore -> the facility type we record.
WANTED = {
    "Provider Information": "nursing_home",
    "Dialysis Facility - Listing by Facility": "dialysis",
}

LAT_KEYS = ("latitude", "lat", "Latitude", "LATITUDE")
LNG_KEYS = ("longitude", "lng", "lon", "Longitude", "LONGITUDE")
NAME_KEYS = (
    "provider_name",
    "Provider Name",
    "facility_name",
    "Facility Name",
    "provider name",
)
STATE_KEYS = ("state", "State", "provider_state", "Provider State")
CITY_KEYS = ("city", "City", "city/town", "provider_city", "Provider City")
ZIP_KEYS = ("zip_code", "ZIP Code", "zip", "provider_zip_code")
ADDR_KEYS = ("address", "Address", "provider_address", "address_line_1")


def _first(row: dict, keys) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    lowered = {k.lower(): v for k, v in row.items()}
    for k in keys:
        v = lowered.get(k.lower())
        if v and str(v).strip():
            return str(v).strip()
    return ""


def resolve_urls(timeout: float = 60.0) -> dict[str, str]:
    """Ask the metastore rather than hardcoding a resource hash that rotates."""
    with urllib.request.urlopen(METASTORE, timeout=timeout) as r:
        items = json.load(r)
    out: dict[str, str] = {}
    for item in items:
        title = item.get("title", "")
        if title not in WANTED:
            continue
        for dist in item.get("distribution") or []:
            url = (dist.get("data") or {}).get("downloadURL") or dist.get("downloadURL")
            if url:
                out[title] = url
                break
    return out


def fetch_csv(url: str, timeout: float = 180.0) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "FIRST-LIGHT/1.0 (hackathon)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    text = raw.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def in_aoi(lng: float, lat: float, aoi: list[float], pad: float = 0.05) -> bool:
    """Padded, because a facility just outside the flown box still matters: an
    evacuation goes TO a hospital, which is frequently over the boundary."""
    w, s, e, n = aoi
    return (w - pad) <= lng <= (e + pad) and (s - pad) <= lat <= (n + pad)


def zip_centroids(aoi: list[float], pad: float) -> dict[str, list[float]]:
    """Mean parcel centroid per ZIP, from the county parcels already on disk.

    CMS ships the dialysis listing with an address and a ZIP but NO coordinates,
    and a dialysis centre is the single most rescue-relevant facility type in the
    plan, so dropping the whole layer for want of a lat/lng would be the wrong
    call. A ZIP centroid is coarse, and it is labelled coarse in the output, so
    the join can weight it accordingly rather than pretending to street accuracy.
    """
    parcels = config.DATASET_DIR / "parcels.geojson"
    if not parcels.exists():
        return {}
    data = json.loads(parcels.read_text(encoding="utf-8"))
    acc: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for f in data.get("features", []):
        props = f.get("properties") or {}
        z = str(props.get("SITE_ZIP") or props.get("ZIP") or "").strip()[:5]
        if not z or not z.isdigit():
            continue
        geom = f.get("geometry") or {}
        pt = _any_point(geom)
        if pt is None:
            continue
        if z not in acc:
            acc[z], counts[z] = [0.0, 0.0], 0
        acc[z][0] += pt[0]
        acc[z][1] += pt[1]
        counts[z] += 1
    return {
        z: [acc[z][0] / counts[z], acc[z][1] / counts[z]]
        for z in acc
        if counts[z] >= 3 and in_aoi(acc[z][0] / counts[z], acc[z][1] / counts[z], aoi, pad)
    }


def _any_point(geom: dict) -> list[float] | None:
    coords = geom.get("coordinates")
    while isinstance(coords, list) and coords and isinstance(coords[0], list):
        coords = coords[0]
    if isinstance(coords, list) and len(coords) >= 2:
        try:
            return [float(coords[0]), float(coords[1])]
        except (TypeError, ValueError):
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", default=",".join(str(x) for x in config.AOI))
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--out", default=str(config.DATASET_DIR / "cms_facilities.geojson"))
    ap.add_argument("--merge-into", default=str(config.DATASET_DIR / "facilities.geojson"))
    args = ap.parse_args(argv)

    aoi = [float(x) for x in args.aoi.split(",")]
    print(f"AOI {aoi}, pad {args.pad} degrees")

    urls = resolve_urls()
    if not urls:
        print("the CMS metastore returned no download URL for either dataset")
        return 1

    feats: list[dict] = []
    zips = zip_centroids(aoi, args.pad)
    if zips:
        print(f"ZIP centroids available for {len(zips)} AOI ZIPs, from county parcels")
    for title, ftype in WANTED.items():
        url = urls.get(title)
        if not url:
            print(f"{title}: no download URL, skipped")
            continue
        print(f"{title} -> {ftype}")
        rows = fetch_csv(url)
        print(f"  {len(rows)} rows nationally")
        kept = coarse = 0
        for row in rows:
            precision = "point"
            try:
                lat = float(_first(row, LAT_KEYS))
                lng = float(_first(row, LNG_KEYS))
            except (TypeError, ValueError):
                # No coordinates published (the dialysis listing ships none), so
                # fall back to the ZIP centroid and LABEL it coarse.
                z = _first(row, ZIP_KEYS)[:5]
                pt = zips.get(z)
                if pt is None:
                    continue
                lng, lat = pt
                precision = "zip-centroid"
            if not in_aoi(lng, lat, aoi, args.pad):
                continue
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lng, lat]},
                    "properties": {
                        "NAME": _first(row, NAME_KEYS),
                        "firstlight_kind": ftype,
                        "city": _first(row, CITY_KEYS),
                        "state": _first(row, STATE_KEYS),
                        "zip": _first(row, ZIP_KEYS),
                        "address": _first(row, ADDR_KEYS),
                        "location_precision": precision,
                        "source": "CMS Care Compare, facility level only",
                    },
                }
            )
            kept += 1
            coarse += precision == "zip-centroid"
        note = f", {coarse} located by ZIP centroid only" if coarse else ""
        print(f"  {kept} inside the AOI{note}")

    if not feats:
        print("no CMS facilities fell inside this AOI, nothing written")
        return 1

    out = Path(args.out)
    out.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8"
    )
    print(f"\nwrote {out.name}, {len(feats)} facilities")

    # Merge into the file the loader reads, keeping the county's own hospitals.
    merge = Path(args.merge_into)
    if merge.exists():
        existing = json.loads(merge.read_text(encoding="utf-8"))
        county = existing.get("features", [])
        seen = {
            (round(f["geometry"]["coordinates"][0], 5), round(f["geometry"]["coordinates"][1], 5))
            for f in feats
        }
        keep = [
            f
            for f in county
            if (
                round(f["geometry"]["coordinates"][0], 5),
                round(f["geometry"]["coordinates"][1], 5),
            )
            not in seen
        ]
        merged = {"type": "FeatureCollection", "features": keep + feats}
        merge.write_text(json.dumps(merged), encoding="utf-8")
        print(f"merged into {merge.name}: {len(keep)} county + {len(feats)} CMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
