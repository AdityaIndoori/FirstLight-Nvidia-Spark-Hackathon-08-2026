#!/usr/bin/env python3
"""Time outline_and_grade on real tiles, serial vs concurrent.

The per-tile p50 is a headline number on the deck, so it gets measured on the
same code path the pipeline runs, on real imagery, not estimated from a single
call multiplied by a budget.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, grading  # noqa: E402


def tiles(n: int) -> list[Path]:
    found = sorted(config.ANALYZED_DIR.glob("*.jpg"))
    return found[:n]


def bounds_for(path: Path) -> list[float] | None:
    from app import db

    row = db.q1("SELECT bounds_json FROM tiles WHERE filename=?", (path.name,))
    if not row or not row["bounds_json"]:
        return None
    return json.loads(row["bounds_json"])


def run(paths: list[Path], lanes: int) -> dict:
    os.environ["FIRSTLIGHT_VL_CONCURRENCY"] = str(lanes)
    per_tile, calls, model, stub = [], [], [], []
    for p in paths:
        b = bounds_for(p)
        if b is None:
            continue
        t0 = time.time()
        graded = grading.outline_and_grade(p, b)
        per_tile.append((time.time() - t0) * 1000.0)
        last = grading.last_run()
        calls.append(last.get("vl_calls", 0))
        model.append(last.get("model_graded", 0))
        stub.append(last.get("stub_graded", 0))
        print(
            f"  lanes={lanes} {p.name}: {per_tile[-1]/1000:.1f}s "
            f"buildings={len(graded)} vl_calls={calls[-1]} model={model[-1]} stub={stub[-1]}",
            flush=True,
        )
    if not per_tile:
        return {"error": "no tiles with bounds"}
    return {
        "lanes": lanes,
        "tiles": len(per_tile),
        "p50_ms": int(statistics.median(per_tile)),
        "min_ms": int(min(per_tile)),
        "max_ms": int(max(per_tile)),
        "vl_calls_median": int(statistics.median(calls)) if calls else 0,
        "model_graded_median": int(statistics.median(model)) if model else 0,
        "stub_graded_median": int(statistics.median(stub)) if stub else 0,
    }


def main() -> int:
    paths = tiles(int(os.environ.get("N_TILES", "3")))
    if not paths:
        print("no analyzed tiles on disk", file=sys.stderr)
        return 1
    out = {}
    for lanes in (1, 6):
        print(f"--- {lanes} lane(s) ---", flush=True)
        out[f"lanes_{lanes}"] = run(paths, lanes)
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
