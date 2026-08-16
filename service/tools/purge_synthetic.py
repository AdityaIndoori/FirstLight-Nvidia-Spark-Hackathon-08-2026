#!/usr/bin/env python3
"""Remove buildings that came from the synthetic grid fallback.

WHY a migration and not just a code fix: the grid used to run whenever the
footprint layer returned nothing, so tiles over open ground wrote fabricated
rectangles into `buildings` - and the address join then labelled them with real
street addresses off the nearest road. Those rows are in the rank list and the
agency assignments right now. Deleting the code path does not delete them.

Identifies them structurally rather than by a flag, because the flag was dropped
before the row was written: a grid cell is one of 12 identically-sized rectangles
in a tile, and real footprints are never all the same size.

Dry run by default. Pass --apply to delete.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402


def size_m(geom: dict) -> tuple[float, float]:
    ring = (geom.get("coordinates") or [[]])[0]
    if not ring:
        return 0.0, 0.0
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    lat = sum(ys) / len(ys)
    w = (max(xs) - min(xs)) * 111_320.0 * math.cos(math.radians(lat))
    h = (max(ys) - min(ys)) * 110_540.0
    return round(w, 1), round(h, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete")
    # The grid was 4x3. A real layer producing 12 pixel-identical footprints in one
    # tile does not happen, so this threshold is safe; it is a floor, not a guess.
    ap.add_argument("--min-cluster", type=int, default=8)
    args = ap.parse_args()

    rows = db.q("SELECT footprint_id, label, geom_json, source_tile FROM buildings")
    by_size: dict[tuple, list] = defaultdict(list)
    for r in rows:
        geom = db.jload(r["geom_json"], None)
        if not geom or geom.get("type") != "Polygon":
            continue
        by_size[(r["source_tile"], size_m(geom))].append(r)

    doomed = []
    for (tile, size), group in sorted(by_size.items()):
        if len(group) >= args.min_cluster:
            doomed.extend(group)
            print(
                f"{tile}: {len(group)} identical {size[0]} x {size[1]} m rectangles "
                f"-> synthetic grid"
            )
            for r in group[:3]:
                print(f"    {r['footprint_id']}  labelled {r['label']!r}")

    print(f"\n{len(doomed)} fabricated buildings of {len(rows)} total")
    if not doomed:
        return 0
    if not args.apply:
        print("dry run: pass --apply to delete")
        return 0

    ids = [r["footprint_id"] for r in doomed]
    for i in range(0, len(ids), 200):
        chunk = ids[i : i + 200]
        marks = ",".join("?" * len(chunk))
        db.run(f"DELETE FROM buildings WHERE footprint_id IN ({marks})", tuple(chunk))
    db.log("maintenance", "purge-synthetic-outlines", {"removed": len(ids)})
    print(f"deleted {len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
