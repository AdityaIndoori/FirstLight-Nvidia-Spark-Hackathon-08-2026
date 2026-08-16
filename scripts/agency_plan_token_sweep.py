#!/usr/bin/env python3
"""B6b opt-in MAX_TOKENS sweep for per-building agency-plan drafting.

Not part of the pytest suite, and makes NO production changes. A live run
at the current production cap (max_tokens=80) showed mostly finish_reason
== "length" truncation (1 Nano success, 3 fallbacks out of 4 buildings) --
this script measures whether raising max_tokens fixes that, and if so, the
SMALLEST value that does, WITHOUT touching the prompt, schema, grounding
rules, concurrency, per-request timeout, or fallback policy.

Runs the REAL, unmodified production orchestrator
(agency_plan_drafter.draft_agency_plan_with_diagnostics) against the real
Nano server, using the existing 4-building fixture
(agency_plan_live_check.PROCESSED_FIXTURES) and the production
RealAgencyPlanDraftClient (default 10s per-building timeout,
DEFAULT_MAX_CONCURRENCY=4, /no_think, json_schema) -- nothing here is a
simplified prompt, schema, or a reimplementation of the retry/fallback
logic.

max_tokens override: RealAgencyPlanDraftClient.propose_assignments_for_building
reads agency_plan_client._MAX_TOKENS as a module global at call time (not
captured at import time), so this script temporarily monkey-patches that
module attribute for the duration of each sweep value's run, then restores
the original value in a `finally` block -- a RUNTIME override for
benchmarking only. The committed value of _MAX_TOKENS in
agency_plan_client.py is never edited by this script; no production
behavior changes as a result of running it.

Fallback reason buckets (per building that used the deterministic
fallback), derived from the SAME per-building diagnostics
draft_agency_plan_with_diagnostics() already returns
(agency_plan_diagnostics.categorize_client_error /
categorize_validation_error -- unchanged, not reimplemented here):
  truncated_output    -- finish_reason == "length" (the failure this sweep
                          targets)
  timeout              -- transport-level timeout
  semantic_validation  -- structurally valid JSON but an invalid/ungrounded
                          agency, action, footprint reference, or units
                          value (includes grounding rejections, e.g.
                          "fire_suppression unsupported: ...")
  other                -- connection failure, HTTP failure, malformed JSON,
                          or any other transport-side parsing failure

A building's own LAST attempt determines which category is used: if only
attempt 1 ran (a transport-level exception aborts before a re-prompt),
attempt_1_error_category is used; if attempt 2 ran, attempt_2_error_category
is used (mirrors agency_plan_drafter._draft_one_building_inner's own
control flow exactly -- see DEFAULT_MAX_CONCURRENCY docstring there for the
one-reprompt-then-fallback sequence).

Deterministic-fallback latency for a building is never counted as a Nano
success; model_successes only counts diagnostics["model_building_count"].

Usage:
    python scripts/agency_plan_token_sweep.py
    FIRSTLIGHT_NANO_BASE_URL=http://localhost:8000 python scripts/agency_plan_token_sweep.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.decision import agency_plan_client  # noqa: E402  (module patched at runtime, see _run_once)
from backend.decision.agency_plan import AvailabilityRegistry  # noqa: E402
from backend.decision.agency_plan_client import RealAgencyPlanDraftClient  # noqa: E402
from backend.decision.agency_plan_diagnostics import (  # noqa: E402
    CATEGORY_INVALID_UNITS,
    CATEGORY_MISSING_REQUIRED_FIELD,
    CATEGORY_OTHER_VALIDATION_ERROR,
    CATEGORY_TIMEOUT,
    CATEGORY_TRUNCATED_OUTPUT,
    CATEGORY_UNKNOWN_FOOTPRINT_ID,
    CATEGORY_UNSUPPORTED_AGENCY,
)
from backend.decision.agency_plan_drafter import (  # noqa: E402
    DEFAULT_MAX_CONCURRENCY,
    build_planner_candidate,
    draft_agency_plan_with_diagnostics,
)
from agency_plan_live_check import PROCESSED_FIXTURES  # noqa: E402  (the exact live-check fixtures)

MAX_TOKENS_VALUES = [80, 96, 112, 128, 144]
PRODUCTION_TIMEOUT_S = 10.0

_SEMANTIC_VALIDATION_CATEGORIES = {
    CATEGORY_UNSUPPORTED_AGENCY,
    CATEGORY_UNKNOWN_FOOTPRINT_ID,
    CATEGORY_INVALID_UNITS,
    CATEGORY_MISSING_REQUIRED_FIELD,
    CATEGORY_OTHER_VALIDATION_ERROR,
}


def _fallback_reason_bucket(building: dict) -> str:
    """Map one fallback building's diagnostics to one of the four buckets
    this sweep reports. Uses whichever attempt was LAST (see module
    docstring) -- the same attempt whose category triggered the fallback.
    """
    category = (
        building["attempt_2_error_category"]
        if building["attempt_count"] == 2
        else building["attempt_1_error_category"]
    )
    if category == CATEGORY_TRUNCATED_OUTPUT:
        return "truncated_output"
    if category == CATEGORY_TIMEOUT:
        return "timeout"
    if category in _SEMANTIC_VALIDATION_CATEGORIES:
        return "semantic_validation"
    return "other"


def _make_candidates():
    return [build_planner_candidate(fixture, confirmed=False) for fixture in PROCESSED_FIXTURES]


def _make_availability() -> AvailabilityRegistry:
    availability = AvailabilityRegistry()
    for agency, units in (("fire", 2), ("ems", 2), ("police", 1), ("public_works", 1)):
        availability.set_availability(agency, units, "operator:demo")
    return availability


def _run_once(max_tokens: int) -> dict:
    """Runs the complete 4-building plan exactly once, through the real,
    unmodified production path, with agency_plan_client._MAX_TOKENS
    temporarily patched to `max_tokens`. Restores the original value
    afterward regardless of outcome -- this is a benchmarking-only runtime
    override, never a change to the committed production constant.
    """
    original_max_tokens = agency_plan_client._MAX_TOKENS
    agency_plan_client._MAX_TOKENS = max_tokens
    try:
        client = RealAgencyPlanDraftClient()
        candidates = _make_candidates()
        availability = _make_availability()

        start = time.perf_counter()
        result = draft_agency_plan_with_diagnostics(
            candidates, availability, client=client, max_concurrency=DEFAULT_MAX_CONCURRENCY
        )
        elapsed_s = time.perf_counter() - start
    finally:
        agency_plan_client._MAX_TOKENS = original_max_tokens

    diagnostics = result["diagnostics"]
    total_completion_tokens = sum(u.get("completion_tokens", 0) for u in client.usage_log if u)

    reason_counts = {"truncated_output": 0, "semantic_validation": 0, "timeout": 0, "other": 0}
    for building in diagnostics["buildings"]:
        if building["recovery"] == "stub":
            reason_counts[_fallback_reason_bucket(building)] += 1

    return {
        "max_tokens": max_tokens,
        "total_elapsed_s": elapsed_s,
        "model_successes": diagnostics["model_building_count"],
        "fallbacks": diagnostics["fallback_building_count"],
        "total_completion_tokens": total_completion_tokens,
        "reason_counts": reason_counts,
    }


def main():
    print(f"Target: {agency_plan_client._resolve_base_url()}/v1/chat/completions (model=nano, /no_think, json_schema)")
    print(f"production_timeout_s (per building): {PRODUCTION_TIMEOUT_S}")
    print(f"max_concurrency: {DEFAULT_MAX_CONCURRENCY}")
    print(f"buildings: {len(PROCESSED_FIXTURES)}")
    print(f"sweep values: {MAX_TOKENS_VALUES}\n")

    print("Warm-up run (max_tokens=80, discarded, not part of the comparison)...")
    _run_once(80)
    print("warm-up done.\n")

    rows = []
    for max_tokens in MAX_TOKENS_VALUES:
        print(f"Running max_tokens={max_tokens} ...")
        row = _run_once(max_tokens)
        rows.append(row)
        rc = row["reason_counts"]
        print(
            f"  elapsed_s={row['total_elapsed_s']:.2f}  nano={row['model_successes']}/4  "
            f"fallback={row['fallbacks']}  completion_tokens={row['total_completion_tokens']}  "
            f"truncated={rc['truncated_output']}  semantic={rc['semantic_validation']}  "
            f"timeout={rc['timeout']}  other={rc['other']}"
        )
    print()

    header = f"{'tokens':<8}{'elapsed':<10}{'nano':<7}{'fallback':<10}{'truncated':<11}{'semantic':<10}{'timeout':<9}{'other':<7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        rc = row["reason_counts"]
        print(
            f"{row['max_tokens']:<8}{row['total_elapsed_s']:<10.2f}{row['model_successes']:<7}"
            f"{row['fallbacks']:<10}{rc['truncated_output']:<11}{rc['semantic_validation']:<10}"
            f"{rc['timeout']:<9}{rc['other']:<7}"
        )
    print()

    # Goal (1) + (3): zero truncated_output AND whole plan under budget.
    # (2) 4/4 Nano successes and (4) grounding validation are preferred but
    # not required to call a value "viable" -- grounding is always enforced
    # regardless of max_tokens (unchanged this sweep), and semantic-
    # validation fallbacks are a different failure mode than truncation.
    viable = [
        row
        for row in rows
        if row["reason_counts"]["truncated_output"] == 0 and row["total_elapsed_s"] <= PRODUCTION_TIMEOUT_S
    ]

    if not viable:
        print(
            "RECOMMENDATION: no tested max_tokens value eliminated truncation within the "
            f"{PRODUCTION_TIMEOUT_S}s budget -- sweep a higher range before changing production."
        )
    else:
        smallest_viable = min(viable, key=lambda r: r["max_tokens"])
        print(
            f"RECOMMENDATION: smallest measured max_tokens with zero truncation and "
            f"total_elapsed_s <= {PRODUCTION_TIMEOUT_S}s is {smallest_viable['max_tokens']} "
            f"(elapsed_s={smallest_viable['total_elapsed_s']:.2f}, "
            f"nano={smallest_viable['model_successes']}/4, fallback={smallest_viable['fallbacks']})."
        )
        fully_successful = [r for r in viable if r["model_successes"] == 4]
        if fully_successful:
            smallest_full = min(fully_successful, key=lambda r: r["max_tokens"])
            if smallest_full["max_tokens"] != smallest_viable["max_tokens"]:
                print(
                    f"  Smallest value that ALSO reached 4/4 Nano successes: "
                    f"max_tokens={smallest_full['max_tokens']} "
                    f"(elapsed_s={smallest_full['total_elapsed_s']:.2f})."
                )
        else:
            print(
                "  None of the truncation-free values reached 4/4 Nano successes in this run "
                "-- remaining fallbacks were semantic/timeout/other, not truncation."
            )

    print(
        f"\nNo production code was modified. agency_plan_client._MAX_TOKENS remains "
        f"{agency_plan_client._MAX_TOKENS} on disk; this script only patches it in-process, "
        "per run, and restores it afterward."
    )


if __name__ == "__main__":
    main()
