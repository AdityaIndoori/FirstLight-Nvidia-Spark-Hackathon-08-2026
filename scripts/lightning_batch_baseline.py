#!/usr/bin/env python3
"""Opt-in real Lightning multi-building baseline/benchmark.

Not part of the pytest suite -- run by hand once the DGX Spark's Lightning
vLLM server is reachable (directly, or via the Mac's SSH tunnel to
localhost:8001). Runs the EXISTING, unchanged k=8 ballot
(lightning_ballot.request_lightning_ballot, via
lightning_baseline.run_lightning_baseline) over 5 fixture buildings --
5 x 8 = 40 real model generations total -- and reports timing/agreement
statistics, plus token usage if the server's responses actually include it.

Usage:
    python scripts/lightning_batch_baseline.py
    FIRSTLIGHT_LIGHTNING_BASE_URL=http://localhost:8001 python scripts/lightning_batch_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.lightning_baseline import BuildingBallotError, run_lightning_baseline
from backend.decision.lightning_client import RealLightningSeverityClient


def main():
    client = RealLightningSeverityClient()
    print(f"Running Lightning batch baseline against {client.base_url} (model=lightning)...")
    print("5 buildings x 8 votes = 40 real model generations.\n")

    try:
        summary = run_lightning_baseline(client=client)
    except BuildingBallotError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    for result in summary["results"]:
        print(f"Building: {result['name']}")
        print(f"votes: {result['votes']}")
        print(f"voted_class: {result['voted_class']}")
        print(f"vote_agreement: {result['vote_agreement']}")
        print(f"doubt: {result['doubt']}")
        print()

    print("Summary:")
    print(f"buildings: {summary['buildings_processed']}")
    print(f"generations: {summary['total_generations']}")
    print(f"elapsed_s: {summary['elapsed_s']:.3f}")
    print(f"avg_s_per_generation: {summary['avg_s_per_generation']:.3f}")
    print(f"mean_vote_agreement: {summary['mean_vote_agreement']:.3f}")
    print(f"min_vote_agreement: {summary['min_vote_agreement']:.3f}")
    print(f"max_vote_agreement: {summary['max_vote_agreement']:.3f}")

    if "total_prompt_tokens" in summary:
        print(f"total_prompt_tokens: {summary['total_prompt_tokens']}")
        print(f"total_completion_tokens: {summary['total_completion_tokens']}")
        print(f"tokens_per_sec: {summary['tokens_per_sec']:.2f}")


if __name__ == "__main__":
    main()
