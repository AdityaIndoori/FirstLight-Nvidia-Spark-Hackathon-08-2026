#!/usr/bin/env python3
"""Opt-in 50-building x k=8 concurrency sweep against the REAL Lightning
vLLM server.

Not part of the pytest suite -- run by hand once the DGX Spark's Lightning
vLLM server is reachable (directly, or via the Mac's SSH tunnel to
localhost:8001). 50 synthetic, decision-layer-only fixtures (no VL server
call, no images -- see lightning_perf.generate_synthetic_fixtures) x k=8 =
400 total Lightning generations per concurrency level.

FAIRNESS RULE: the exact same 50 fixtures are reused, unmutated, at every
concurrency level. temperature, k, prompt, structured decoding, and model
never change between levels -- only max concurrently in-flight generations
does (bounded_concurrency, via lightning_perf.run_batch_sweep's shared
thread pool). Concurrency levels stay <= the server's --max-num-seqs=16.

This benchmark measures the REAL Lightning model only: a failed generation
during the sweep is counted in "failures" and that building is dropped from
the results -- never silently patched with a stub value or a fabricated
timing (see lightning_perf.run_batch_sweep's docstring).

Token-usage rates are reported with a precise metric name
(completion_tokens_per_second, decode throughput only) -- never a combined
(prompt + completion) / wall_time figure mislabeled as generation speed.

Usage:
    python scripts/lightning_batch_sweep.py
    FIRSTLIGHT_LIGHTNING_BASE_URL=http://localhost:8001 python scripts/lightning_batch_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.lightning_client import RealLightningSeverityClient
from backend.decision.lightning_perf import generate_synthetic_fixtures, run_batch_sweep

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
WARMUP_BUILDING_COUNT = 2


def _usage_totals(usage_slice: list) -> tuple:
    total_prompt = sum(u.get("prompt_tokens", 0) for u in usage_slice if u)
    total_completion = sum(u.get("completion_tokens", 0) for u in usage_slice if u)
    return total_prompt, total_completion


def main():
    client = RealLightningSeverityClient()
    buildings = generate_synthetic_fixtures(50)
    print(f"Sweeping Lightning batch concurrency against {client.base_url}")
    print(f"{len(buildings)} buildings x 8 votes = {len(buildings) * 8} generations per concurrency level\n")

    print("Warming up...")
    run_batch_sweep(buildings[:WARMUP_BUILDING_COUNT], concurrency=2, client=client)
    usage_marker = len(getattr(client, "usage_log", []))
    print("Warm-up done.\n")

    for concurrency in CONCURRENCY_LEVELS:
        result = run_batch_sweep(buildings, concurrency=concurrency, client=client)

        print(f"concurrency: {concurrency}")
        print(f"buildings processed: {result['buildings_processed']}")
        print(f"total generations: {result['total_generations']}")
        print(f"failures: {result['failures']}")
        print(f"elapsed_s: {result['elapsed_s']:.3f}")
        print(f"buildings_per_second: {result['buildings_per_second']:.3f}")
        print(f"generations_per_second: {result['generations_per_second']:.3f}")
        print(f"ballot latency p50_ms: {result['ballot_latency_ms']['p50_ms']:.1f}")
        print(f"ballot latency p95_ms: {result['ballot_latency_ms']['p95_ms']:.1f}")
        print(f"mean_vote_agreement: {result['mean_vote_agreement']:.3f}")

        usage_log = getattr(client, "usage_log", [])
        level_usage = usage_log[usage_marker:]
        usage_marker = len(usage_log)
        if level_usage:
            total_prompt, total_completion = _usage_totals(level_usage)
            completion_tokens_per_second = total_completion / result["elapsed_s"] if result["elapsed_s"] > 0 else 0.0
            print(f"total_prompt_tokens: {total_prompt}")
            print(f"total_completion_tokens: {total_completion}")
            print(f"completion_tokens_per_second: {completion_tokens_per_second:.2f}  (decode throughput only)")
        print()


if __name__ == "__main__":
    main()
