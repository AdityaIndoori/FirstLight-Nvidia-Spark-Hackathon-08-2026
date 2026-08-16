#!/usr/bin/env python3
"""Generate the demo kit: sample tiles, a person fixture, seeded availability.

C9. Runs fully offline. Everything it makes is synthetic except the shapes,
which follow the real contracts, so nothing here can be mistaken for evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from app import config, db, scorer  # noqa: E402

AOI = config.AOI
ROADS = ["SW Alaska St", "35th Ave SW", "California Ave SW", "SW Admiral Way", "42nd Ave SW"]


def _tile_image(path: Path, seed: int, *, people: int = 0) -> None:
    """A nadir-ish scene: roads, roofs, water, and optionally small figures.

    The person fixture is the demo's honesty beat: it must be analyzed and it
    must never be stored, so the kit ships one on purpose.
    """
    rng = random.Random(seed)
    w, h = 1024, 1024
    img = Image.new("RGB", (w, h), (62, 74, 54))
    d = ImageDraw.Draw(img)
    for y in (240, 560, 860):
        d.rectangle([0, y, w, y + 34], fill=(92, 92, 98))
    for x in (180, 520, 840):
        d.rectangle([x, 0, x + 30, h], fill=(92, 92, 98))
    d.ellipse([620, 640, 900, 830], fill=(46, 74, 96))  # standing water
    for _ in range(rng.randint(14, 22)):
        x, y = rng.randint(30, w - 150), rng.randint(30, h - 150)
        bw, bh = rng.randint(70, 130), rng.randint(60, 110)
        roof = rng.choice(
            [(150, 128, 106), (150, 128, 106), (128, 108, 92), (86, 66, 58), (58, 46, 42)]
        )
        d.rectangle([x, y, x + bw, y + bh], fill=roof, outline=(40, 36, 32))
        if roof[0] < 90:  # a dark roof reads as damage to the pixel-stat stub
            for _ in range(rng.randint(3, 7)):
                px, py = rng.randint(x, x + bw), rng.randint(y, y + bh)
                d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(30, 26, 24))
    for i in range(people):
        px, py = 300 + i * 46, 300 + (i % 3) * 40
        d.ellipse([px, py, px + 9, py + 9], fill=(224, 196, 168))  # head
        d.rectangle([px + 2, py + 9, px + 7, py + 26], fill=(60, 80, 150))  # torso
    img.save(path, "JPEG", quality=88)


def _bounds_for(i: int, n: int) -> list[float]:
    """Lay the tiles out over the AOI in a rough grid so the map has spread."""
    w, s, e, nn = AOI
    cols = max(1, int(math.sqrt(n)))
    col, row = i % cols, i // cols
    tw, th = (e - w) / cols, (nn - s) / max(1, math.ceil(n / cols))
    return [w + col * tw, s + row * th, w + (col + 1) * tw, s + (row + 1) * th]


def make_tiles(out: Path, count: int) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for i in range(count):
        p = out / f"tile_{i:02d}.jpg"
        _tile_image(p, seed=1000 + i)
        (out / f"{p.name}.bounds.json").write_text(
            json.dumps({"bounds": _bounds_for(i, count)}), encoding="utf-8"
        )
        made.append(p)
    fixture = out / "fixture_person.jpg"
    _tile_image(fixture, seed=99, people=4)
    (out / f"{fixture.name}.bounds.json").write_text(
        json.dumps({"bounds": _bounds_for(count // 2, count)}), encoding="utf-8"
    )
    made.append(fixture)
    return made


def make_datasets(out: Path) -> None:
    """Labelled fakes with real schemas, so a missing county download never
    dead-ends the demo. Named so nobody mistakes them for county data."""
    out.mkdir(parents=True, exist_ok=True)
    w, s, e, n = AOI
    rng = random.Random(7)

    feats = []
    for i in range(400):
        lng = rng.uniform(w, e)
        lat = rng.uniform(s, n)
        d = 0.00022
        feats.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[lng, lat], [lng + d, lat], [lng + d, lat + d], [lng, lat + d], [lng, lat]]
                    ],
                },
                "properties": {
                    "PIN": f"{rng.randint(100000, 999999)}{rng.randint(1000, 9999)}",
                    "ADDRESS": f"{rng.randint(3000, 6500)} {rng.choice(ROADS)}",
                    "SOURCE": "SAMPLE-NOT-COUNTY-DATA",
                },
            }
        )
    (out / "footprints.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8"
    )

    road_feats = []
    for i, name in enumerate(ROADS):
        if i % 2 == 0:
            y = s + (n - s) * (i + 1) / (len(ROADS) + 1)
            coords = [[w, y], [e, y]]
        else:
            x = w + (e - w) * (i + 1) / (len(ROADS) + 1)
            coords = [[x, s], [x, n]]
        road_feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"name": name, "highway": "residential"},
            }
        )
    (out / "roads.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": road_feats}), encoding="utf-8"
    )

    (out / "facilities.csv").write_text(
        "name,type,longitude,latitude,beds\n"
        f"Providence Mount St. Vincent,nursing_home,{w + (e - w) * 0.35:.6f},{s + (n - s) * 0.55:.6f},112\n"
        f"DaVita West Seattle Dialysis,dialysis,{w + (e - w) * 0.62:.6f},{s + (n - s) * 0.38:.6f},34\n",
        encoding="utf-8",
    )

    svi_feats = []
    cols = 4
    for i in range(cols * cols):
        cx, cy = i % cols, i // cols
        x0 = w + (e - w) * cx / cols
        y0 = s + (n - s) * cy / cols
        x1 = w + (e - w) * (cx + 1) / cols
        y1 = s + (n - s) * (cy + 1) / cols
        svi_feats.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
                },
                "properties": {"RPL_THEMES": round(rng.uniform(0.2, 0.97), 2), "SOURCE": "SAMPLE"},
            }
        )
    (out / "svi.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": svi_feats}), encoding="utf-8"
    )

    # Parcels carry owner columns on purpose: the loader must drop them, and the
    # privacy write-up quotes the measured count of dropped fields.
    (out / "parcels.csv").write_text(
        "PIN,ADDRESS,OWNER,OWNER_NAME,TAXPAYER,MAILING_ADDRESS,USE_CODE\n"
        "1234560001,4200 SW Admiral Way,DOE JOHN,DOE JOHN,DOE JOHN,PO BOX 1,SINGLE FAMILY\n"
        "1234560002,4210 SW Admiral Way,ROE JANE,ROE JANE,ROE JANE,PO BOX 2,SINGLE FAMILY\n",
        encoding="utf-8",
    )


def seed_availability() -> None:
    """Pre-seed so the over-commitment flag fires on cue instead of by luck."""
    db.init()
    for agency, units in (("fire", 4), ("ems", 3), ("police", 6), ("public_works", 2)):
        scorer.set_availability(agency, units, "demo-seed")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", type=int, default=10)
    ap.add_argument("--out", default=str(config.DATA / "sample_tiles"))
    ap.add_argument("--datasets", action="store_true", help="also write labelled fake datasets")
    ap.add_argument("--seed-availability", action="store_true")
    args = ap.parse_args(argv)

    made = make_tiles(Path(args.out), args.tiles)
    print(f"sample tiles: {len(made)} in {args.out}")
    print(f"  person fixture: {made[-1].name} (must be analyzed, must never be stored)")
    if args.datasets:
        make_datasets(config.DATASET_DIR)
        print(f"labelled fake datasets in {config.DATASET_DIR}")
    if args.seed_availability:
        seed_availability()
        print("availability seeded: fire 4, ems 3, police 6, public_works 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
