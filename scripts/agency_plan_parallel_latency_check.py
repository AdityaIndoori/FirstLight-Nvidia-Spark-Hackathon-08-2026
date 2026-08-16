#!/usr/bin/env python3
"""B6b opt-in LATENCY benchmark for the per-building parallel agency-plan
drafting strategy.

Not part of the pytest suite, and makes NO production changes. Runs the
REAL, unmodified production orchestrator
(agency_plan_drafter.draft_agency_plan_with_diagnostics) against the real
Nano server, using the existing 4-building fixture
(agency_plan_live_check.PROCESSED_FIXTURES) and the production
RealAgencyPlanDraftClient (default 10s per-building timeout,
DEFAULT_MAX_CONCURRENCY=4) -- nothing here is a simplified prompt, schema,
or a reimplementation of the retry/fallback logic.

Goal: does the ENTIRE 4-building plan complete within the existing 10s
production timeout budget now that drafting is per-building and parallel,
replacing the earlier whole-plan request that measured 220 completion
tokens / finish_reason="length" / ~9.25s and was still truncated?

Measures:
  - total wall-clock latency for the whole 4-building plan
  - latency per building (from per-building diagnostics)
  - prompt tokens total / completion tokens total (client.usage_log)
  - model successes / fallbacks (diagnostics.model_building_count /
    fallback_building_count)
  - assignment count

Does NOT assert or require a particular stochastic assignment -- the real
model's sampling is not pinned.

Usage:
    python scripts/agency_plan_parallel_latency_check.py
    FIRSTLIGHT_NANO_BASE_URL=http://localhost:8000 python scripts/agency_plan_parallel_latency_check.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.decision.agency_plan import AvailabilityRegistry  # noqa: E402
from backend.decision.agency_plan_client import RealAgencyPlanDraftClient  # noqa: E402
from backend.decision.agency_plan_drafter import (  # noqa: E402
    DEFAULT_MAX_CONCURRENCY,
    build_planner_candidate,
    draft_agency_plan_with_diagnostics,
)
from agency_plan_live_check import PROCESSED_FIXTURES  # noqa: E402  (the exact live-check fixtures)


def main():
    client = RealAgencyPlanDraftClient()
    candidates = [build_planner_candidate(fixture, confirmed=False) for fixture in PROCESSED_FIXTURES]

    availability = AvailabilityRegistry()
    for agency in ("fire", "ems", "police", "public_works"):
        availability.set_availability(agency, 2, "operator:demo")

    print(f"Target: {client.base_url}/v1/chat/completions (model=nano)")
    print(f"production_timeout_s (per building): {client.timeout_s}")
    print(f"max_concurrency: {DEFAULT_MAX_CONCURRENCY}")
    print(f"buildings: {len(candidates)}\n")

    start = time.perf_counter()
    result = draft_agency_plan_with_diagnostics(candidates, availability, client=client)
    total_elapsed_s = time.perf_counter() - start

    diagnostics = result["diagnostics"]
    total_assignments = sum(len(g["steps"]) for g in result["plan"]["agencies"])
    total_prompt_tokens = sum(u.get("prompt_tokens", 0) for u in client.usage_log if u)
    total_completion_tokens = sum(u.get("completion_tokens", 0) for u in client.usage_log if u)

    print(f"total_elapsed_s: {total_elapsed_s:.2f}")
    print(f"model successes: {diagnostics['model_building_count']}")
    print(f"fallbacks: {diagnostics['fallback_building_count']}")
    print(f"assignment_count: {total_assignments}")
    print(f"prompt_tokens_total: {total_prompt_tokens}")
    print(f"completion_tokens_total: {total_completion_tokens}\n")

    print("Per-building latency:")
    for building in diagnostics["buildings"]:
        print(
            f"  {building['footprint_id']}: elapsed_s={building['elapsed_s']:.2f}, "
            f"recovery={building['recovery']}, attempt_count={building['attempt_count']}"
        )
        if building["fallback_reason"]:
            print(f"    fallback_reason: {building['fallback_reason']}")
    print()

    if total_elapsed_s <= client.timeout_s:
        print(f"RESULT: entire 4-building plan completed in {total_elapsed_s:.2f}s, "
              f"within the {client.timeout_s}s production budget.")
    else:
        print(f"RESULT: entire 4-building plan took {total_elapsed_s:.2f}s, "
              f"OVER the {client.timeout_s}s production budget.")

    print(f"drafted_by: {result['plan']['drafted_by']}")


if __name__ == "__main__":
    main()
