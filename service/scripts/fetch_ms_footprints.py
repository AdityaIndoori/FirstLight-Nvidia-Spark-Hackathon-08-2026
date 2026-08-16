#!/usr/bin/env python3
"""Fetch Microsoft GlobalMLBuildingFootprints for the configured AOI.

WHY this tier exists: most counties publish no building-footprint service, and
Pinellas is one of them. Without footprints the grading path falls through to a
deterministic synthetic grid, which is honest but useless: identical axis-aligned
squares sitting across streets and parking lots. This gives the pipeline real
building polygons for any AOI on earth.

The dataset is sharded by quadkey (a slippy-map tile path at zoom 9), one
newline-delimited GeoJSON file per shard, gzipped. The AOI usually falls inside
one or two shards, so this computes the covering quadkeys rather than downloading
a state.

Licence: ODbL. Attribution is written next to the output.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

INDEX = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
QUADKEY_ZOOM = 9
ATTRIBUTION = (
    "Building footprints: Microsoft GlobalMLBuildingFootprints, ODbL.\n"
    "https://github.com/microsoft/GlobalMLBuildingFootprints\n"
    "Machine-learning derived geometry only: no addresses, no owner identity.\n"
)


def tile_xy(lng: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lng + 180.0) / 360.0 * n)
    lat_r = math.radians(max(-85.05112878, min(85.05112878, lat)))
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def quadkey(x: int, y: int, z: int) -> str:
    """Bing quadkey: the shard naming this dataset uses."""
    out = []
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        out.append(str(digit))
    return "".join(out)


def covering_quadkeys(aoi: list[float], z: int = QUADKEY_ZOOM) -> list[str]:
    w, s, e, n = aoi
    x0, y0 = tile_xy(w, n, z)
    x1, y1 = tile_xy(e, s, z)
    keys = []
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for y in range(min(y0, y1), max(y0, y1) + 1):
            keys.append(quadkey(x, y, z))
    return keys


def index_rows(timeout: float = 90.0) -> list[dict]:
    req = urllib.request.Request(INDEX, headers={"User-Agent": "FIRST-LIGHT/1.0 (hackathon)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def centroid_of(ring: list) -> tuple[float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def inside(lng: float, lat: float, aoi: list[float]) -> bool:
    w, s, e, n = aoi
    return w <= lng <= e and s <= lat <= n


def harvest(url: str, aoi: list[float], *, timeout: float = 600.0, log=print) -> list[dict]:
    """Stream the shard and keep only footprints whose centroid is in the AOI.

    Streamed rather than buffered: a shard is hundreds of megabytes and the AOI
    is a few square kilometres of it, so holding the whole thing would be silly.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "FIRST-LIGHT/1.0 (hackathon)"})
    kept: list[dict] = []
    seen = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = gzip.GzipFile(fileobj=resp) if url.endswith(".gz") else resp
        for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            seen += 1
            if seen % 250_000 == 0:
                log(f"    scanned {seen:,} footprints, kept {len(kept):,}")
            try:
                feat = json.loads(line)
                ring = feat["geometry"]["coordinates"][0]
                lng, lat = centroid_of(ring)
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            if not inside(lng, lat, aoi):
                continue
            props = feat.get("properties") or {}
            kept.append(
                {
                    "type": "Feature",
                    "geometry": feat["geometry"],
                    "properties": {
                        # Geometry only by construction: this dataset carries no
                        # address and no owner identity, so there is nothing to
                        # scrub. The join gets an address from parcels instead.
                        "source": "microsoft-globalml",
                        "height": props.get("height"),
                        "confidence": props.get("confidence"),
                    },
                }
            )
    log(f"    scanned {seen:,} footprints, kept {len(kept):,}")
    return kept


def _mb(size: str | None) -> float:
    """The index writes sizes like '95.1MB' or '1.2GB', no space. Unknown is 0."""
    if not size:
        return 0.0
    text = str(size).strip().upper().replace(" ", "")
    scale = 1.0
    for suffix, factor in (("GB", 1024.0), ("MB", 1.0), ("KB", 1.0 / 1024.0)):
        if text.endswith(suffix):
            text, scale = text[: -len(suffix)], factor
            break
    try:
        return float(text) * scale
    except ValueError:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", default=",".join(str(x) for x in config.AOI))
    ap.add_argument("--location", default="UnitedStates")
    ap.add_argument("--out", default=str(config.DATASET_DIR / "footprints.geojson"))
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    aoi = [float(x) for x in args.aoi.split(",")]
    keys = covering_quadkeys(aoi)
    print(f"AOI {aoi}")
    print(f"covering quadkeys at zoom {QUADKEY_ZOOM}: {', '.join(keys)}")

    rows = index_rows()
    wanted = [r for r in rows if r.get("QuadKey") in keys and r.get("Location") == args.location]
    if not wanted:
        # Some AOIs sit in a shard filed under a neighbouring location name.
        wanted = [r for r in rows if r.get("QuadKey") in keys]
    if not wanted:
        print("no shard in the index covers this AOI")
        print("the dataset is sharded by quadkey; check the AOI or the location name")
        return 1

    total_mb = sum(_mb(r.get("Size")) for r in wanted)
    for r in wanted:
        print(f"  shard {r['QuadKey']} ({r.get('Location')}), {r.get('Size', 'unknown size')}")
    if args.dry_run:
        print(f"dry run: {len(wanted)} shard(s), about {total_mb:.0f} MB to stream")
        return 0

    feats: list[dict] = []
    for r in wanted:
        print(f"streaming shard {r['QuadKey']} ...")
        feats.extend(harvest(r["Url"], aoi, timeout=args.timeout))

    if not feats:
        print("no footprint centroid fell inside the AOI, nothing written")
        return 1

    out = Path(args.out)
    out.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8"
    )
    (out.parent / "footprints.ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    print(f"\nwrote {out.name}: {len(feats):,} real building footprints")
    print(f"attribution: {out.parent / 'footprints.ATTRIBUTION.txt'}")
    print("re-analyze the tiles so grading picks up real outlines instead of the grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
