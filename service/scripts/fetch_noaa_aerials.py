#!/usr/bin/env python3
"""Fetch real post-disaster aerial imagery from NOAA's emergency response tiles.

WHY this matters more than any other dataset: every geographic layer in FIRST
LIGHT is now authentic county data, but the drone imagery has been synthetic, so
the damage grades describe a cartoon. A vision model looking at tan rectangles on
green fields honestly reports "no structure is visible", and the whole ranking
downstream inherits that. Real imagery is the last synthetic thing in the stack.

NOAA flies these within hours of landfall and publishes them as slippy-map tiles
per flight day, which is exactly the shape a drone-tile pipeline wants: each tile
is a georeferenced aerial photograph. Bay County FL after Hurricane Michael
(landfall 2018-10-10, Cat 5) is the AOI with unambiguous catastrophic damage.

The flight days are separate tile sets, so a pre/post pair is available where the
same ground was flown twice, which is what the xView2 cls path needs.

Source: https://storms.ngs.noaa.gov/ (NOAA National Geodetic Survey, public domain)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

CDN = "https://stormscdn.ngs.noaa.gov"
UA = "FIRST-LIGHT/1.0 (NVIDIA DGX Spark hackathon, offline disaster triage)"

# Post-landfall flight days, earliest first. Michael made landfall 2018-10-10, so
# 20181011a is the first response flight. Not every day covers every tile; the
# fetcher probes and reports which one answered.
STORMS = {
    "michael": {
        "flights": ["20181011a", "20181012a", "20181012b", "20181013a", "20181014a"],
        "aoi": [-85.72, 30.13, -85.62, 30.22],
        "note": "Hurricane Michael, Cat 5, Bay County and Panama City FL",
    },
    "milton": {
        "flights": ["20241010a", "20241011a", "20241012a"],
        "aoi": [-82.78, 27.75, -82.70, 27.82],
        "note": "Hurricane Milton, Pinellas County FL. Flight ids need probing.",
    },
}

ATTRIBUTION = (
    "Post-disaster aerial imagery: NOAA National Geodetic Survey emergency\n"
    "response imagery, public domain. https://storms.ngs.noaa.gov/\n"
    "Flown within days of landfall. Each tile is a georeferenced aerial photograph.\n"
)


def tile_xy(lng: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lng + 180.0) / 360.0 * n)
    r = math.radians(max(-85.05112878, min(85.05112878, lat)))
    y = int((1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds(x: int, y: int, z: int) -> list[float]:
    """The [w, s, e, n] this tile covers, which becomes the tile's georeference.

    This is what makes a NOAA tile usable as a drone frame: the pipeline needs
    bounds per image, and a slippy tile carries them implicitly.
    """
    n = 2.0 ** z

    def lat_of(ty: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))

    return [x / n * 360.0 - 180.0, lat_of(y + 1), (x + 1) / n * 360.0 - 180.0, lat_of(y)]


def fetch_tile(flight: str, z: int, x: int, y: int, *, timeout: float = 30.0) -> bytes | None:
    url = f"{CDN}/{flight}-rgb/{z}/{x}/{y}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            body = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    # A redirect lands here as an empty body, and an error page is not a JPEG.
    if len(body) < 1200 or body[:3] != b"\xff\xd8\xff":
        return None
    return body


def probe_flight(flights: list[str], aoi: list[float], z: int) -> str | None:
    """Find a flight day that covers ANY part of the AOI, not just its centre.

    Several days exist per storm and each covers a different swath, so probing
    beats assuming: a missing tile returns a redirect, not a 404, which would
    otherwise be written to disk as a zero-byte "image".

    The probe samples a grid across the AOI rather than testing the centre alone.
    Coverage is a ragged flight path, so the middle of a bbox is frequently
    unflown while most of the box is fine: testing one point rejects an AOI that
    is in fact covered.
    """
    x0, y0 = tile_xy(aoi[0], aoi[3], z)
    x1, y1 = tile_xy(aoi[2], aoi[1], z)
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    samples = []
    for fx in (0.5, 0.25, 0.75, 0.1, 0.9):
        for fy in (0.5, 0.25, 0.75, 0.1, 0.9):
            samples.append(
                (lo_x + int((hi_x - lo_x) * fx), lo_y + int((hi_y - lo_y) * fy))
            )
    for flight in flights:
        for x, y in samples:
            if fetch_tile(flight, z, x, y) is not None:
                return flight
    return None


def stitch(cells: dict[tuple[int, int], bytes], z: int, span: int, out: Path, flight: str) -> list[dict]:
    """Mosaic contiguous tiles into drone-sized frames.

    A single 256 px tile is not a survey frame: a building crop out of it is a few
    dozen pixels, and the VL model is being asked to grade a thumbnail. Stitching
    span x span tiles gives a frame with enough ground per image for per-building
    crops to carry real detail, and the mosaic's bounds are just the union of its
    tiles, so georeferencing stays exact rather than approximated.
    """
    from PIL import Image

    frames: list[dict] = []
    xs = sorted({x for x, _ in cells})
    ys = sorted({y for _, y in cells})
    for bx in range(0, len(xs) - span + 1, span):
        for by in range(0, len(ys) - span + 1, span):
            block = [(xs[bx + i], ys[by + j]) for i in range(span) for j in range(span)]
            if any(key not in cells for key in block):
                continue  # a hole in the mosaic would be a black square in a frame
            side = 256 * span
            canvas = Image.new("RGB", (side, side))
            for i in range(span):
                for j in range(span):
                    tile = Image.open(io.BytesIO(cells[(xs[bx + i], ys[by + j])])).convert("RGB")
                    canvas.paste(tile, (i * 256, j * 256))
            x0, y0 = xs[bx], ys[by]
            x1, y1 = xs[bx + span - 1], ys[by + span - 1]
            w, s0, _, n0 = tile_bounds(x0, y0, z)
            _, s1, e1, _ = tile_bounds(x1, y1, z)
            bounds = [w, s1, e1, n0]
            name = f"noaa_{flight}_{z}_{x0}_{y0}_{span}x{span}.jpg"
            canvas.save(out / name, "JPEG", quality=92)
            (out / f"{name}.bounds.json").write_text(
                json.dumps({"bounds": bounds, "source": f"noaa-{flight}", "zoom": z, "span": span}),
                encoding="utf-8",
            )
            frames.append({"file": name, "bounds": bounds, "tiles": span * span})
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storm", default="michael", choices=sorted(STORMS))
    ap.add_argument("--zoom", type=int, default=17)
    ap.add_argument(
        "--span",
        type=int,
        default=1,
        help="mosaic span x span tiles into one frame, so 4 gives a 1024px frame",
    )
    ap.add_argument("--aoi", default="")
    ap.add_argument("--flight", default="", help="skip probing and use this flight id")
    ap.add_argument("--limit", type=int, default=24, help="tiles to keep as drone frames")
    ap.add_argument("--out", default="")
    ap.add_argument("--rate", type=float, default=8.0, help="requests per second")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    storm = STORMS[args.storm]
    aoi = [float(v) for v in args.aoi.split(",")] if args.aoi else storm["aoi"]
    z = args.zoom
    out = Path(args.out) if args.out else config.DATA / "noaa_tiles"

    print(f"{args.storm}: {storm['note']}")
    print(f"AOI {aoi}, zoom {z}")

    flight = args.flight or probe_flight(storm["flights"], aoi, z)
    if not flight:
        print("no flight day covers this AOI centre at this zoom.")
        print(f"tried: {', '.join(storm['flights'])}")
        print("the AOI may be outside the flown swath, or the flight ids have changed")
        return 1
    print(f"flight {flight} covers the AOI centre")

    x0, y0 = tile_xy(aoi[0], aoi[3], z)
    x1, y1 = tile_xy(aoi[2], aoi[1], z)
    total = (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
    print(f"AOI spans {total} tiles at z{z}, keeping up to {args.limit} as drone frames")
    if args.dry_run:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")

    # Walk from the centre out: the middle of the AOI is the built-up ground, and
    # the edges are frequently water or unflown.
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    order = sorted(
        (
            (x, y)
            for x in range(min(x0, x1), max(x0, x1) + 1)
            for y in range(min(y0, y1), max(y0, y1) + 1)
        ),
        key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
    )

    kept = 0
    missed = 0
    delay = 1.0 / max(0.5, args.rate)
    manifest = []
    span = max(1, int(args.span))
    # With span > 1 the tiles are collected in memory and mosaicked, so the target
    # is tiles-to-fetch rather than frames-to-write.
    target = args.limit * span * span if span > 1 else args.limit
    cells: dict[tuple[int, int], bytes] = {}
    for x, y in order:
        if kept >= target:
            break
        body = fetch_tile(flight, z, x, y)
        if body is None:
            missed += 1
            continue
        kept += 1
        if span > 1:
            cells[(x, y)] = body
        else:
            name = f"noaa_{flight}_{z}_{x}_{y}.jpg"
            (out / name).write_bytes(body)
            bounds = tile_bounds(x, y, z)
            # The sidecar is what makes this a drone frame the pipeline
            # understands: geo.extract reads it third, after GeoTIFF and EXIF.
            (out / f"{name}.bounds.json").write_text(
                json.dumps({"bounds": bounds, "source": f"noaa-{flight}", "zoom": z}),
                encoding="utf-8",
            )
            manifest.append({"file": name, "bounds": bounds, "x": x, "y": y, "z": z})
        if kept % 16 == 0:
            print(f"  {kept} tiles fetched, {missed} not flown")
        time.sleep(delay)

    if span > 1:
        manifest = stitch(cells, z, span, out, flight)
        print(f"  mosaicked {len(cells)} tiles into {len(manifest)} frames of {256 * span}px")

    (out / "MANIFEST.json").write_text(
        json.dumps(
            {
                "storm": args.storm,
                "flight": flight,
                "zoom": z,
                "aoi": aoi,
                "frames": len(manifest),
                "source": f"{CDN}/{flight}-rgb",
                "note": storm["note"],
                "tiles": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{kept} real aerial frames in {out}")
    print(f"{missed} tiles in the AOI were not flown, which is normal at the swath edge")
    print("\nto ingest them as drone imagery:")
    print(f"  cp {out}/*.jpg {out}/*.bounds.json {config.WATCH_DIR}/")
    print("each frame carries its own bounds, so grading joins against real footprints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
