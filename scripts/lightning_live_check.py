#!/usr/bin/env python3
"""Manual, opt-in check against the REAL Lightning vLLM server.

Not part of the pytest suite -- run by hand once the DGX Spark's vLLM server
is reachable (directly, or via the Mac's SSH tunnel to localhost:8001).
Executes one real k=8 ballot -- eight independently sampled requests, each
temperature=0.7, structured-decoded to a single "0"/"1"/"2"/"3" -- through
the existing, unchanged ballot logic (lightning_ballot.request_lightning_ballot).

Usage:
    python scripts/lightning_live_check.py
    FIRSTLIGHT_LIGHTNING_BASE_URL=http://localhost:8001 python scripts/lightning_live_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.lightning_ballot import request_lightning_ballot
from backend.decision.lightning_client import LightningClientError, RealLightningSeverityClient

BUILDING_CONTEXT = {
    "grader_class": 2,
    "grader_confidence": 0.85,
    "vl_caption": "Two-storey structure with partial roof collapse and debris along the eastern wall.",
    "footprint_area_m2": 120.0,
    "facility_context": "150 m from Riverside Clinic",
    "neighbor_damage_classes": [1, 2, 2],
}


def main():
    client = RealLightningSeverityClient()
    print(f"Requesting Lightning k=8 ballot from {client.base_url} (model=lightning, enable_thinking=false)...")
    print("Building fixture (VL model's two independent observations + GIS context):")
    for key, value in BUILDING_CONTEXT.items():
        if key == "vl_caption":
            continue
        print(f"  {key} = {value}")
    print(f'  vl_caption = "{BUILDING_CONTEXT["vl_caption"]}"\n')

    try:
        result = request_lightning_ballot(BUILDING_CONTEXT, client=client)
    except LightningClientError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    votes = result["votes"]
    if len(votes) != 8 or any(v not in (0, 1, 2, 3) for v in votes):
        print(f"FAILED: unexpected votes {votes!r}")
        sys.exit(1)

    print(f"Lightning votes:\n{votes}\n")
    print(f"voted_class: {result['voted_class']}")
    print(f"vote_agreement: {result['vote_agreement']}")
    print(f"doubt: {result['doubt']}")


if __name__ == "__main__":
    main()
