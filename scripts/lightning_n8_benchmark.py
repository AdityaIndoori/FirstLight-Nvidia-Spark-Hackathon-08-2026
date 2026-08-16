#!/usr/bin/env python3
"""Opt-in benchmark: 8 separate parallel n=1 requests vs. 1 request with n=8,
against the REAL Lightning vLLM server.

Not part of the pytest suite -- run by hand once the DGX Spark's Lightning
vLLM server is reachable (directly, or via the Mac's SSH tunnel to
localhost:8001).

Mode A (parallel_8_requests): the EXISTING production-shaped path --
lightning_ballot.request_lightning_ballot_parallel, eight separate n=1
requests through a bounded thread pool.

Mode B (single_request_n8): the EXPERIMENTAL path --
lightning_n8_experiment.request_n8_ballot, one request with n=8.

Both modes use the exact same BuildingEvidence-derived fixture, model,
temperature, chat_template_kwargs.enable_thinking, and structured_outputs.choice.
Neither mode is assumed faster -- this only measures.

If the server rejects n=8 together with structured decoding, that error is
reported as-is (as benchmark failures) rather than worked around.

Usage:
    python scripts/lightning_n8_benchmark.py
    FIRSTLIGHT_LIGHTNING_BASE_URL=http://localhost:8001 python scripts/lightning_n8_benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.lightning_ballot import K_VOTES, request_lightning_ballot_parallel
from backend.decision.lightning_client import LightningClientError, RealLightningSeverityClient
from backend.decision.lightning_n8_experiment import request_n8_ballot
from backend.decision.lightning_perf import summarize_latencies_ms

# The exact same BuildingEvidence-derived Lightning ballot input for every
# trial, in both modes.
BUILDING_CONTEXT = {
    "grader_class": 2,
    "grader_confidence": 0.85,
    "vl_caption": "Two-storey structure with partial roof collapse and debris along the eastern wall.",
    "footprint_area_m2": 120.0,
    "facility_context": "150 m from Riverside Clinic",
    "neighbor_damage_classes": [1, 2, 2],
}

TRIAL_COUNT = 10


def _run_trials(name: str, ballot_fn, client) -> None:
    print(f"Warming up {name}...")
    try:
        warmup_result = ballot_fn(BUILDING_CONTEXT, client=client)
        print(f"  warm-up ok, votes={warmup_result['votes']}")
    except LightningClientError as exc:
        print(f"  warm-up FAILED: {exc}")
        print(f"  ({name}: stopping -- reporting the actual error rather than working around it)\n")
        return

    latencies_ms = []
    failures = 0
    example_votes = None
    for trial in range(TRIAL_COUNT):
        start = time.perf_counter()
        try:
            result = ballot_fn(BUILDING_CONTEXT, client=client)
        except LightningClientError as exc:
            failures += 1
            print(f"  trial {trial}: FAILED ({exc})")
            continue
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if example_votes is None:
            example_votes = result["votes"]

    summary = summarize_latencies_ms(latencies_ms)
    trial_count = len(latencies_ms)
    print(f"\nmode: {name}")
    print(f"trial_count: {trial_count}")
    print(f"failures: {failures}")
    print(f"mean_latency_ms: {summary['mean_ms']:.1f}")
    print(f"p50_latency_ms: {summary['p50_ms']:.1f}")
    print(f"p95_latency_ms: {summary['p95_ms']:.1f}")
    print(f"total_sampled_generations: {trial_count * K_VOTES}")
    print(f"example vote array: {example_votes}\n")


def main():
    client = RealLightningSeverityClient()
    print(f"Benchmarking parallel_8_requests vs. single_request_n8 against {client.base_url}\n")

    def mode_a(building_context, client):
        return request_lightning_ballot_parallel(building_context, client=client)

    def mode_b(building_context, client):
        return request_n8_ballot(building_context, base_url=client.base_url, timeout_s=client.timeout_s)

    _run_trials("parallel_8_requests", mode_a, client)
    _run_trials("single_request_n8", mode_b, client)

    print("Do not assume either mode is faster from a single run -- compare the numbers above.")


if __name__ == "__main__":
    main()
