"""Lightning throughput measurement helpers.

Opt-in benchmark scripts (scripts/lightning_parallel_benchmark.py,
scripts/lightning_batch_sweep.py) are thin CLI wrappers around this module.
This module itself makes no network calls -- it only calls
LightningSeverityClient.sample_severity() through whatever client it is
given, so it is fully testable offline with a stub or a mocked real client.

percentile()/summarize_latencies_ms() are the ONLY place p50/p95 are
computed; run_batch_sweep() reuses lightning_ballot._aggregate_votes() for
vote aggregation instead of reimplementing it.

generate_synthetic_fixtures() produces synthetic Lightning ballot fixtures
(building_context dicts, see lightning_ballot.py) for throughput
benchmarking ONLY -- these are not real BuildingEvidence, and building them
never calls the VL model or loads an image.
"""

import concurrent.futures
import math
import time

from backend.decision.lightning_ballot import (
    K_VOTES,
    _DEFAULT_TEMPERATURE,
    _aggregate_votes,
)
from backend.decision.lightning_client import LightningSeverityClient, StubLightningSeverityClient

_SYNTHETIC_CAPTIONS = {
    0: "Building appears fully intact with no visible structural damage.",
    1: "Minor visible damage; walls and roofline appear largely intact.",
    2: "Significant structural damage including partial roof collapse.",
    3: "Structure has collapsed; only rubble and foundation remnants are visible.",
}


def percentile(values: list, pct: float) -> float:
    """Nearest-rank percentile (no interpolation) over values; pct in [0, 100].
    Empty input returns 0.0.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def summarize_latencies_ms(latencies_ms: list) -> dict:
    """{"mean_ms", "p50_ms", "p95_ms"} over a list of per-trial or
    per-building latencies in milliseconds. Empty input -> all zeros.
    """
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": sum(latencies_ms) / len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
    }


def generate_synthetic_fixtures(count: int = 50) -> list:
    """count deterministic synthetic Lightning ballot fixtures:
    [{"name": str, "context": building_context}, ...]. Cycles grader_class
    0-3 so the set is varied but perfectly reproducible run to run -- no
    randomness, no VL model, no images.
    """
    fixtures = []
    for i in range(count):
        grader_class = i % 4
        fixtures.append(
            {
                "name": f"synthetic_{i:03d}",
                "context": {
                    "grader_class": grader_class,
                    "grader_confidence": round(0.5 + 0.1 * (i % 5), 2),
                    "vl_caption": _SYNTHETIC_CAPTIONS[grader_class],
                    "footprint_area_m2": 80.0 + 5.0 * (i % 20),
                    "facility_context": "150 m from Riverside Clinic" if i % 7 == 0 else None,
                    "neighbor_damage_classes": [(grader_class + j) % 4 for j in range(3)],
                },
            }
        )
    return fixtures


def run_batch_sweep(
    buildings: list,
    concurrency: int,
    client: LightningSeverityClient = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> dict:
    """Run k=8 Lightning ballots for every building in `buildings` (each item
    {"name": str, "context": building_context}), issuing ALL
    len(buildings) * K_VOTES individual sample_severity() calls through ONE
    shared, bounded thread pool -- `concurrency` is the maximum number of
    Lightning generations in flight at once, across the whole sweep, not
    per building. Calling this with the same `buildings` list at different
    `concurrency` values is exactly the fairness rule: identical fixtures,
    k, prompt, structured decoding, and model at every concurrency level.

    Never mutates `buildings` or any building's context.

    Each generation is tagged with (building_index, vote_index) so votes are
    regrouped back to the correct building regardless of completion order,
    even under concurrent, out-of-order completion.

    A failed generation (the client raises) is counted in "failures" and
    that building is EXCLUDED from "results" -- never silently patched with
    a stub-produced vote or a fabricated timing. Compare
    buildings_processed to len(buildings) to see what was dropped.

    Returns {
        "concurrency": int,
        "buildings_processed": int,
        "total_generations": int,
        "failures": int,
        "elapsed_s": float,
        "buildings_per_second": float,
        "generations_per_second": float,
        "results": [{"name", "votes", "voted_class", "vote_agreement",
                      "doubt", "latency_ms"}, ...],
        "ballot_latency_ms": {"mean_ms", "p50_ms", "p95_ms"},
        "mean_vote_agreement": float,
    }
    """
    active_client = client if client is not None else StubLightningSeverityClient()
    bounded_workers = max(1, concurrency)

    votes_by_building = [[None] * K_VOTES for _ in buildings]
    building_start = [None] * len(buildings)
    building_end = [None] * len(buildings)
    failures = 0

    def _one_generation(building_index):
        context = buildings[building_index]["context"]
        return active_client.sample_severity(context, temperature=temperature)

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=bounded_workers) as executor:
        future_to_indices = {}
        for building_index in range(len(buildings)):
            building_start[building_index] = time.monotonic()
            for vote_index in range(K_VOTES):
                future = executor.submit(_one_generation, building_index)
                future_to_indices[future] = (building_index, vote_index)

        for future in concurrent.futures.as_completed(future_to_indices):
            building_index, vote_index = future_to_indices[future]
            try:
                votes_by_building[building_index][vote_index] = future.result()
            except Exception:
                failures += 1
            building_end[building_index] = time.monotonic()
    elapsed_s = time.monotonic() - start

    results = []
    ballot_latencies_ms = []
    for building_index, fixture in enumerate(buildings):
        votes = votes_by_building[building_index]
        if any(vote is None for vote in votes):
            continue  # this building had at least one failed generation
        aggregated = _aggregate_votes(votes)
        latency_ms = (building_end[building_index] - building_start[building_index]) * 1000.0
        results.append({"name": fixture["name"], "latency_ms": latency_ms, **aggregated})
        ballot_latencies_ms.append(latency_ms)

    buildings_processed = len(results)
    total_generations = len(buildings) * K_VOTES
    agreements = [r["vote_agreement"] for r in results]

    return {
        "concurrency": concurrency,
        "buildings_processed": buildings_processed,
        "total_generations": total_generations,
        "failures": failures,
        "elapsed_s": elapsed_s,
        "buildings_per_second": buildings_processed / elapsed_s if elapsed_s > 0 else 0.0,
        "generations_per_second": total_generations / elapsed_s if elapsed_s > 0 else 0.0,
        "results": results,
        "ballot_latency_ms": summarize_latencies_ms(ballot_latencies_ms),
        "mean_vote_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
    }
