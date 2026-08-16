#!/usr/bin/env python3
"""Opt-in benchmark: sequential k=8 ballot vs. parallel k=8 ballot, one
building, against the REAL Lightning vLLM server.

Not part of the pytest suite -- run by hand once the DGX Spark's Lightning
vLLM server is reachable (directly, or via the Mac's SSH tunnel to
localhost:8001). Runs the EXISTING unmodified sequential ballot
(lightning_ballot.request_lightning_ballot) and the new bounded-concurrency
ballot (lightning_ballot.request_lightning_ballot_parallel) against the same
fixed BuildingEvidence-derived context, model, temperature, and structured
decoding -- only execution shape differs.

This is a THROUGHPUT benchmark, not a correctness test: it does not compare
the actual vote labels between modes (real-model sampling is stochastic). It
only validates that each ballot still has exactly 8 labels in 0-3 with a
valid agreement/doubt.

Usage:
    python scripts/lightning_parallel_benchmark.py
    FIRSTLIGHT_LIGHTNING_BASE_URL=http://localhost:8001 python scripts/lightning_parallel_benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.lightning_ballot import (
    K_VOTES,
    request_lightning_ballot,
    request_lightning_ballot_parallel,
)
from backend.decision.lightning_client import LightningClientError, RealLightningSeverityClient
from backend.decision.lightning_perf import summarize_latencies_ms

# One fixed BuildingEvidence-derived Lightning ballot input, identical for
# every trial and every mode.
BUILDING_CONTEXT = {
    "grader_class": 2,
    "grader_confidence": 0.85,
    "vl_caption": "Two-storey structure with partial roof collapse and debris along the eastern wall.",
    "footprint_area_m2": 120.0,
    "facility_context": "150 m from Riverside Clinic",
    "neighbor_damage_classes": [1, 2, 2],
}

TRIAL_COUNT = 5
_VALID_LABELS = (0, 1, 2, 3)


def _validate_ballot(result: dict) -> None:
    votes = result["votes"]
    if len(votes) != K_VOTES or any(v not in _VALID_LABELS for v in votes):
        raise AssertionError(f"malformed ballot: {votes!r}")
    if not (0.0 <= result["vote_agreement"] <= 1.0):
        raise AssertionError(f"vote_agreement out of range: {result['vote_agreement']!r}")
    if not (0.05 <= result["doubt"] <= 1.0):
        raise AssertionError(f"doubt out of range: {result['doubt']!r}")


def _run_mode(name: str, ballot_fn, client) -> dict:
    print(f"Warming up {name}...")
    try:
        ballot_fn(BUILDING_CONTEXT, client=client)
    except LightningClientError as exc:
        print(f"  warm-up FAILED: {exc}")

    latencies_ms = []
    failures = 0
    for trial in range(TRIAL_COUNT):
        start = time.perf_counter()
        try:
            result = ballot_fn(BUILDING_CONTEXT, client=client)
            _validate_ballot(result)
        except (LightningClientError, AssertionError) as exc:
            failures += 1
            print(f"  trial {trial}: FAILED ({exc})")
            continue
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    summary = summarize_latencies_ms(latencies_ms)
    print(f"\nmode: {name}")
    print(f"trial count: {len(latencies_ms)} (failures: {failures})")
    print(f"k: {K_VOTES}")
    print(f"mean ballot latency_ms: {summary['mean_ms']:.1f}")
    print(f"p50 ballot latency_ms: {summary['p50_ms']:.1f}")
    print(f"p95 ballot latency_ms: {summary['p95_ms']:.1f}")
    print(f"total generations: {len(latencies_ms) * K_VOTES}\n")
    return summary


def main():
    client = RealLightningSeverityClient()
    print(f"Benchmarking sequential vs. parallel k=8 against {client.base_url}\n")

    _run_mode("sequential", request_lightning_ballot, client)
    _run_mode("parallel", request_lightning_ballot_parallel, client)


if __name__ == "__main__":
    main()
