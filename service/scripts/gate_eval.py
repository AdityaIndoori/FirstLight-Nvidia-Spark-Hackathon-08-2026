#!/usr/bin/env python3
"""A5 gate eval. Measures person recall THROUGH the tiled path, so the number we
publish is the number operators actually get.

The threshold is set FROM this measurement, not chosen and then defended. The two
error classes are not symmetric and the report says so in its own column headers:
a false withhold costs an operator one review click, a false clear costs the whole
privacy claim.

Runs fully offline. Nothing here downloads weights, datasets or fixtures.

  python -m scripts.gate_eval --tiles data/eval/tiles --labels data/eval/labels.json
  python -m scripts.gate_eval --tiles T --labels L --sweep 0.15,0.2,0.25,0.3
  python -m scripts.gate_eval --tiles T --labels L --no-tiling --out single_pass.json

labels.json maps a filename to whether a person is present:

  {"tile_0001.jpg": true, "tile_0002.jpg": false}

A list of {"filename": ..., "has_person": ...} objects is accepted too.

Exit status is 1 when the run cannot support a published claim: any false clear,
or a detector that never loaded. A green exit means the recall figure is real.

The written report is a local development artifact. It names the false-clear
files because someone has to go look at them, and it is never served by the API.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Run as `python scripts/gate_eval.py` from service/ as well as `python -m`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, privacy_gate  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


# ---------------------------------------------------------------- inputs
def load_labels(path: Path) -> dict[str, bool]:
    """Filename to has_person. Keys are compared by basename, because a label
    file written against one directory layout should survive a move."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("images"), (dict, list)):
        raw = raw["images"]
    items: list[tuple[str, Any]]
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [
            (str(r.get("filename") or r.get("file") or r.get("name") or ""),
             r.get("has_person", r.get("person")))
            for r in raw
            if isinstance(r, dict)
        ]
    else:
        raise ValueError("labels.json must be an object or a list of objects")
    out: dict[str, bool] = {}
    for name, value in items:
        if not name:
            continue
        out[Path(str(name)).name] = _truthy(value)
    if not out:
        raise ValueError(f"no usable labels in {path}")
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "person", "people"}
    return bool(value)


def find_tiles(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


# ---------------------------------------------------------------- scoring
def score_at(verdict: privacy_gate.GateVerdict, threshold: float) -> tuple[bool, int]:
    """Re-score one already-computed verdict at a different threshold.

    This is what makes --sweep a single detector pass: run once at the lowest
    threshold in the sweep, then filter the retained detections upward. Running
    the detector once per threshold would give the same answer at four times the
    cost, and on a 100-tile tiled eval that is the difference between a coffee
    and an afternoon. A detector error withholds at every threshold.
    """
    if verdict.detector_error:
        return True, 0
    persons = [
        d
        for d in verdict.all_detections
        if privacy_gate.is_person_class(d) and float(d.get("conf", 0.0)) >= threshold
    ]
    return bool(persons), len(persons)


def confusion(rows: list[dict], threshold: float) -> dict:
    """Positive class is WITHHOLD, so recall is person-recall: of the tiles that
    really contain a person, the share the gate kept out of storage."""
    tp = fn = fp = tn = 0
    false_clears: list[str] = []
    false_withholds: list[str] = []
    for r in rows:
        withheld, _ = score_at(r["verdict"], threshold)
        if r["has_person"]:
            if withheld:
                tp += 1
            else:
                fn += 1
                false_clears.append(r["filename"])
        else:
            if withheld:
                fp += 1
                false_withholds.append(r["filename"])
            else:
                tn += 1
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    return {
        "conf": round(float(threshold), 4),
        "person_tiles": tp + fn,
        "clear_tiles": fp + tn,
        "recall": None if recall is None else round(recall, 4),
        "precision": None if precision is None else round(precision, 4),
        "false_clears": fn,
        "false_withholds": fp,
        "true_withholds": tp,
        "true_clears": tn,
        "false_clear_files": false_clears,
        "false_withhold_files": false_withholds,
    }


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy dependency for six lines of arithmetic,
    and nearest-rank is what a latency budget is checked against anyway."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[k])


def recommend(sweep: list[dict]) -> tuple[Optional[float], str]:
    """Pick the threshold the measurement supports, per A5's asymmetry.

    Highest threshold that clears nothing it should have withheld: highest keeps
    false withholds down, zero false clears keeps the claim. If nothing reaches
    zero, take the best recall and say plainly that the claim is not yet clean.
    """
    if not sweep:
        return None, "no sweep run"
    clean = [s for s in sweep if s["false_clears"] == 0 and s["person_tiles"]]
    if clean:
        best = max(clean, key=lambda s: s["conf"])
        return best["conf"], "highest threshold with zero false clears"
    best = max(sweep, key=lambda s: (s["recall"] or 0.0, -s["conf"]))
    return best["conf"], "best recall available, still leaks: do not publish"


# ---------------------------------------------------------------- run
def evaluate(
    tiles_dir: Path,
    labels: dict[str, bool],
    *,
    conf: float,
    tiled: bool,
    sweep: list[float],
) -> dict:
    paths = find_tiles(tiles_dir)
    # One pass at the lowest threshold in play, then score upward.
    base_conf = min([conf] + sweep)
    rows: list[dict] = []
    unlabeled: list[str] = []
    errors = 0
    t0 = time.perf_counter()
    for p in paths:
        if p.name not in labels:
            unlabeled.append(p.name)
            continue
        v = privacy_gate.check(p, conf=base_conf, tiled=tiled)
        if v.detector_error:
            errors += 1
        rows.append({"filename": p.name, "has_person": labels[p.name], "verdict": v})
        withheld, n = score_at(v, conf)
        print(
            f"  {p.name:<34} label={'person' if labels[p.name] else 'clear ':<6} "
            f"gate={'WITHHELD' if withheld else 'stored  '} "
            f"persons={n:<3} tiles={v.tiles_scanned:<3} {v.took_ms:>6} ms"
            + (f"  [{v.detector_error}]" if v.detector_error else ""),
            flush=True,
        )
    wall_s = time.perf_counter() - t0

    if not rows:
        raise ValueError(
            f"no labeled images found in {tiles_dir} "
            f"({len(paths)} image files, {len(labels)} labels, none matched by name)"
        )

    latencies = [float(r["verdict"].took_ms) for r in rows]
    scanned = [float(r["verdict"].tiles_scanned) for r in rows]
    headline = confusion(rows, conf)
    sweep_rows = [confusion(rows, t) for t in sorted(sweep)] if sweep else []
    rec_conf, rec_why = recommend(sweep_rows or [headline])

    return {
        "gate": privacy_gate.model_version(),
        "weights": Path(config.GATE_WEIGHTS).name,
        "detector_available": errors == 0,
        "detector_errors": errors,
        "tiled": tiled,
        "tile_px": int(config.GATE_TILE) if tiled else None,
        "tile_overlap": float(config.GATE_TILE_OVERLAP) if tiled else None,
        "tile_min_side": int(config.GATE_TILE_MIN_SIDE) if tiled else None,
        "person_classes": sorted(config.GATE_PERSON_CLASSES),
        "images_evaluated": len(rows),
        "images_unlabeled_skipped": len(unlabeled),
        "unlabeled": unlabeled[:20],
        "conf": round(float(conf), 4),
        "base_conf_inferred_at": round(float(base_conf), 4),
        "headline": headline,
        "sweep": sweep_rows,
        "recommended_conf": rec_conf,
        "recommended_because": rec_why,
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "mean": round(sum(latencies) / len(latencies), 1),
            "max": round(max(latencies), 1),
        },
        "tiles_scanned_mean": round(sum(scanned) / len(scanned), 2),
        "wall_s": round(wall_s, 2),
        "measured_at": time.time(),
    }


# ---------------------------------------------------------------- output
def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def print_report(rep: dict) -> None:
    h = rep["headline"]
    path = "tiled" if rep["tiled"] else "single pass"
    if rep["tiled"]:
        path += f" {rep['tile_px']}px / {int(round(rep['tile_overlap'] * 100))}% overlap"
    line = "-" * 66
    print(f"\n{line}")
    print(f"A5 GATE EVAL   {rep['images_evaluated']} labeled tiles   {path}")
    print(f"detector: {rep['gate']}")
    print(line)
    print(f"  {'person tiles':<26} {h['person_tiles']}")
    print(f"  {'clear tiles':<26} {h['clear_tiles']}")
    print(f"  {'person recall':<26} {_pct(h['recall'])}   at conf {h['conf']:.2f}")
    print(f"  {'withhold precision':<26} {_pct(h['precision'])}")
    print(f"  {'FALSE CLEARS':<26} {h['false_clears']}   (costs the privacy claim)")
    print(f"  {'false withholds':<26} {h['false_withholds']}   (costs a review click)")
    print(f"  {'latency p50':<26} {rep['latency_ms']['p50']:.0f} ms")
    print(f"  {'latency p95':<26} {rep['latency_ms']['p95']:.0f} ms")
    print(f"  {'tiles scanned, mean':<26} {rep['tiles_scanned_mean']}")
    if rep["detector_errors"]:
        print(f"  {'DETECTOR ERRORS':<26} {rep['detector_errors']}   (all withheld)")
    if rep["images_unlabeled_skipped"]:
        print(f"  {'unlabeled, skipped':<26} {rep['images_unlabeled_skipped']}")
    if h["false_clear_files"]:
        print("\n  false clears, go look at these:")
        for name in h["false_clear_files"]:
            print(f"    {name}")

    if rep["sweep"]:
        print(f"\n{line}")
        print("  THRESHOLD SWEEP   set the threshold from this table, per A5")
        print(f"  {'conf':>6}  {'recall':>8}  {'precision':>10}  "
              f"{'false clears':>13}  {'false withholds':>16}")
        for s in rep["sweep"]:
            print(
                f"  {s['conf']:>6.2f}  {_pct(s['recall']):>8}  {_pct(s['precision']):>10}  "
                f"{s['false_clears']:>13}  {s['false_withholds']:>16}"
            )
        print(f"\n  recommended conf: {rep['recommended_conf']}  ({rep['recommended_because']})")
        if rep["recommended_conf"] is not None and rep["recommended_conf"] != rep["conf"]:
            print(f"  set FIRSTLIGHT_GATE_CONF={rep['recommended_conf']} to adopt it")

    print(f"\n{line}")
    if h["recall"] is not None and rep["detector_errors"] == 0:
        print(
            f"  README line: person recall {_pct(h['recall'])} on "
            f"{h['person_tiles']} held-out person tiles through the {path} path, "
            f"conf {h['conf']:.2f}, {h['false_clears']} false clears, "
            f"p50 {rep['latency_ms']['p50']:.0f} ms per tile. Measured on this Spark."
        )
    else:
        print("  Not publishable: see the detector errors above.")
    print(f"{line}\n")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gate_eval",
        description="A5: measure privacy-gate person recall through the tiled path.",
    )
    ap.add_argument("--tiles", required=True, type=Path, help="directory of eval tiles")
    ap.add_argument("--labels", required=True, type=Path, help="filename to has_person JSON")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    ap.add_argument("--conf", type=float, default=None, help=f"default {config.GATE_CONF}")
    ap.add_argument(
        "--no-tiling",
        action="store_true",
        help="single downscaled pass, for the comparison that justifies tiling",
    )
    ap.add_argument(
        "--sweep",
        default="",
        help="comma-separated thresholds, e.g. 0.15,0.2,0.25,0.3 (one detector pass)",
    )
    args = ap.parse_args(argv)

    conf = float(config.GATE_CONF) if args.conf is None else float(args.conf)
    try:
        sweep = [float(x) for x in args.sweep.split(",") if x.strip()]
    except ValueError:
        print(f"gate_eval: --sweep must be comma-separated numbers, got {args.sweep!r}")
        return 2
    if any(not 0.0 < t <= 1.0 for t in sweep) or not 0.0 < conf <= 1.0:
        print("gate_eval: thresholds must sit in (0, 1]")
        return 2

    try:
        labels = load_labels(args.labels)
        print(
            f"gate_eval: {len(labels)} labels, "
            f"{sum(labels.values())} marked as containing a person"
        )
        report = evaluate(
            args.tiles, labels, conf=conf, tiled=not args.no_tiling, sweep=sweep
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"gate_eval: {exc}")
        return 2

    print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

    return 1 if (report["headline"]["false_clears"] or report["detector_errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
