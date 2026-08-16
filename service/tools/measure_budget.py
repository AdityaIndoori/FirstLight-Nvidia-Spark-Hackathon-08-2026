#!/usr/bin/env python3
"""Sweep the VL budget at fixed concurrency, so the per-tile latency and the model
coverage it buys are chosen from measurements rather than guessed.

Prints one line per budget plus a JSON block for the deck.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, db, grading  # noqa: E402


def main() -> int:
    lanes = os.environ.get("FIRSTLIGHT_VL_CONCURRENCY", "8")
    os.environ["FIRSTLIGHT_VL_CONCURRENCY"] = lanes
    paths = sorted(config.ANALYZED_DIR.glob("*.jpg"))[:4]
    out = {"lanes": int(lanes), "tiles": len(paths), "sweep": []}
    for budget in (6, 8, 10, 12):
        lat, model, stub = [], [], []
        for p in paths:
            row = db.q1("SELECT bounds_json FROM tiles WHERE filename=?", (p.name,))
            if not row or not row["bounds_json"]:
                continue
            bounds = json.loads(row["bounds_json"])
            t0 = time.time()
            grading.outline_and_grade(p, bounds, vl_budget_override=budget)
            lat.append(time.time() - t0)
            last = grading.last_run()
            model.append(last.get("model_graded", 0))
            stub.append(last.get("stub_graded", 0))
        if not lat:
            continue
        row = {
            "vl_budget": budget,
            "p50_s": round(statistics.median(lat), 1),
            "max_s": round(max(lat), 1),
            "model_graded_median": int(statistics.median(model)),
            "stub_graded_median": int(statistics.median(stub)),
        }
        out["sweep"].append(row)
        print(
            f"budget={budget:>2} lanes={lanes}: p50={row['p50_s']}s max={row['max_s']}s "
            f"model={row['model_graded_median']} stub={row['stub_graded_median']}",
            flush=True,
        )
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
