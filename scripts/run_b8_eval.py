#!/usr/bin/env python3
"""B8 evaluation runner.

Default is OFFLINE (safe, no network, no real model) -- matches this
project's existing convention of every live_check/eval script defaulting
to the safe path and requiring the caller to opt into live behavior
(see e.g. scripts/agency_plan_live_check.py, scripts/nano_live_check.py).

Modes:
    --offline (default)  every deterministic/fixture metric; makes no
                          network call.
    --live                additionally calls the real Nano (localhost:8000)
                          and Lightning (localhost:8001) servers, and
                          attempts real BGE-small-en-v1.5 ONLY if already
                          cached locally -- never downloads it.

Writes the full machine-readable report to artifacts/b8_eval.json (created
if missing) in addition to the printed summary.

Usage:
    python scripts/run_b8_eval.py
    python scripts/run_b8_eval.py --offline
    python scripts/run_b8_eval.py --live
    FIRSTLIGHT_NANO_BASE_URL=... FIRSTLIGHT_LIGHTNING_BASE_URL=... python scripts/run_b8_eval.py --live
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.eval.report import STATUS_DEFERRED, STATUS_FAIL, STATUS_MEASURED, STATUS_PASS  # noqa: E402
from backend.decision.eval.runner import run_live_eval, run_offline_eval  # noqa: E402

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "b8_eval.json"

_DISPLAY_NAMES = {
    "rationale_faithfulness": "Rationale faithfulness",
    "fema_field_accuracy": "FEMA field accuracy",
    "lightning_self_agreement": "Lightning self-agreement",
    "lightning_self_agreement_live": "Lightning self-agreement",
    "lightning_vs_nano_agreement": "Lightning-vs-Nano",
    "agency_plan_precision_recall": "Agency plan precision/recall",
    "agency_plan_precision_recall_live": "Agency plan precision/recall",
    "agency_unit_accounting_sanity": "Agency unit accounting",
    "tag_precision_recall": "Tag precision/recall",
    "tag_precision_recall_live": "Tag precision/recall",
    "search_precision_recall_at_k": "Search precision@k / recall@k",
    "search_precision_recall_at_k_live": "Search precision@k / recall@k",
    "assistant_citation_faithfulness": "Citation faithfulness",
    "injection_battery": "Injection battery",
    "injection_battery_live": "Injection battery",
    "openshell_policy_tamper": "OpenShell tamper",
}


def _format_value(metric: dict) -> str:
    status = metric["status"]
    value = metric["value"]

    if status == STATUS_DEFERRED:
        return "DEFERRED"
    if status == STATUS_PASS:
        return f"PASS {value}" if value is not None else "PASS"
    if status == STATUS_FAIL:
        return f"FAIL {value}" if value is not None else "FAIL"
    # measured
    if isinstance(value, dict):
        parts = []
        for key, v in value.items():
            if isinstance(v, float):
                parts.append(f"{key}={v:.2f}")
            else:
                parts.append(f"{key}={v}")
        return ", ".join(parts)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_summary(report: dict) -> None:
    print("B8 EVALUATION\n")

    deferred = []
    for metric in report["metrics"]:
        label = _DISPLAY_NAMES.get(metric["name"], metric["name"])
        dots = "." * max(1, 32 - len(label))
        print(f"{label} {dots} {_format_value(metric)}")
        if metric["status"] == STATUS_DEFERRED:
            deferred.append(metric)

    print()
    print(f"OVERALL: {report['overall'].upper()}")

    if deferred:
        print("\n" + "=" * 60)
        print("DEFERRED / NOT MEASURED")
        print("=" * 60)
        for metric in deferred:
            label = _DISPLAY_NAMES.get(metric["name"], metric["name"])
            reason = metric["details"].get("reason", "no reason recorded")
            print(f"- {label}: {reason}")


def _write_artifact(report: dict) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\nFull report written to {ARTIFACT_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="run only deterministic/fixture metrics (default)")
    mode.add_argument("--live", action="store_true", help="also call real Nano/Lightning/BGE where available")
    args = parser.parse_args()

    if args.live:
        print("Running B8 evaluation in LIVE mode (Nano :8000, Lightning :8001, real BGE if cached)...\n")
        report = run_live_eval()
    else:
        print("Running B8 evaluation in OFFLINE mode (no network, deterministic fixtures only)...\n")
        report = run_offline_eval()

    _print_summary(report)
    _write_artifact(report)

    if report["overall"] == STATUS_FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
