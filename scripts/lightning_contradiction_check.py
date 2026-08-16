#!/usr/bin/env python3
"""Opt-in diagnostic: does Lightning's vote_agreement/doubt actually move when
the VL caption contradicts the primary grader class?

Not part of the pytest suite -- run by hand once the DGX Spark's Lightning
vLLM server is reachable (directly, or via the Mac's SSH tunnel to
localhost:8001). Runs the EXISTING, unchanged k=8 ballot
(lightning_ballot.request_lightning_ballot) via the EXISTING real client
(lightning_client.RealLightningSeverityClient) over two fixtures that are
identical in every field except vl_caption -- one where the caption agrees
with grader_class, one where it directly contradicts it.

This is an EMPIRICAL diagnostic, not a unit test: it prints both ballots'
results side by side and does not assert a PASS/FAIL verdict, because the
real model's sampling is stochastic and no specific outcome is guaranteed.

Usage:
    python scripts/lightning_contradiction_check.py
    FIRSTLIGHT_LIGHTNING_BASE_URL=http://localhost:8001 python scripts/lightning_contradiction_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.lightning_ballot import request_lightning_ballot
from backend.decision.lightning_client import LightningClientError, RealLightningSeverityClient

_SHARED_FIELDS = {
    "grader_class": 2,
    "grader_confidence": 0.85,
    "footprint_area_m2": 120.0,
    "facility_context": "150 m from Riverside Clinic",
    "neighbor_damage_classes": [1, 2, 2],
}

FIXTURES = [
    {
        "name": "consistent_evidence",
        "context": {
            **_SHARED_FIELDS,
            "vl_caption": (
                "Two-storey structure with partial roof collapse and debris "
                "along the eastern wall."
            ),
        },
    },
    {
        "name": "contradictory_evidence",
        "context": {
            **_SHARED_FIELDS,
            "vl_caption": (
                "Two-storey structure appears intact, with no visible roof "
                "collapse, wall failure, or major structural damage."
            ),
        },
    },
]


def main():
    client = RealLightningSeverityClient()
    print(f"Requesting Lightning k=8 ballots from {client.base_url} (model=lightning, enable_thinking=false)...\n")

    results = {}
    for fixture in FIXTURES:
        print(f"Fixture: {fixture['name']}")
        for key, value in fixture["context"].items():
            print(f"  {key} = {value}")
        print()

        try:
            result = request_lightning_ballot(fixture["context"], client=client)
        except LightningClientError as exc:
            print(f"FAILED: {exc}")
            sys.exit(1)

        results[fixture["name"]] = result

        print(f"votes: {result['votes']}")
        print(f"voted_class: {result['voted_class']}")
        print(f"vote_agreement: {result['vote_agreement']}")
        print(f"doubt: {result['doubt']}\n")

    consistent = results["consistent_evidence"]
    contradictory = results["contradictory_evidence"]

    print("Comparison:")
    print(f"consistent agreement: {consistent['vote_agreement']}")
    print(f"consistent doubt: {consistent['doubt']}")
    print()
    print(f"contradictory agreement: {contradictory['vote_agreement']}")
    print(f"contradictory doubt: {contradictory['doubt']}")


if __name__ == "__main__":
    main()
