#!/usr/bin/env python3
"""Manual, opt-in check against the REAL Nano vLLM server.

Not part of the pytest suite -- run by hand once the DGX Spark's vLLM server
is reachable (directly, or via the Mac's SSH tunnel to localhost:8000).

The fixture below deliberately gives every faithfulness-sensitive field a
distinct, non-null, non-trivial value so a live response exercises all of
them at once:
    staleness_h        6.5   (hours -- must read as increasing priority)
    vulnerable_density 2.3   (dimensionless -- must never become a headcount)
    doubt               0.12  (dimensionless -- must never become a bare %)
    road_cutoff         1.8   (dimensionless multiplier -- must never become a distance)
    facility_near.dist_m 180 (the ONE field allowed to carry meters)

If the live model reintroduces the reported bug (e.g. describing road_cutoff
as "1.8m"), RealNanoRationaleClient raises NanoClientError from its
faithfulness check instead of returning the bad text -- this script reports
that as a FAILED run so a regression is never silently missed.

Usage:
    python scripts/nano_live_check.py
    FIRSTLIGHT_NANO_BASE_URL=http://localhost:8000 python scripts/nano_live_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.nano_client import NanoClientError, RealNanoRationaleClient

SAMPLE_RANK_ITEM = {
    "footprint_id": "fp-live-check",
    "label": "412 Elm St",
    "centroid": [-122.4194, 37.7749],
    "damage_class": 3,
    "confidence": 0.91,
    "confirmed": True,
    "graded_by": "operator:jsmith",
    "facility_near": {"name": "Riverside Clinic", "type": "clinic", "dist_m": 180},
    "inputs": {
        "staleness_h": 6.5,
        "vulnerable_density": 2.3,
        "doubt": 0.12,
        "road_cutoff": 1.8,
    },
    "priority": 24.19320,
    "rationale": "",
    "rationale_by": "nano",
}


def main():
    client = RealNanoRationaleClient()
    print(f"Requesting hero rationale from {client.base_url} (model=nano, /no_think)...")
    inputs = SAMPLE_RANK_ITEM["inputs"]
    print("Fixture under test:")
    print(f"  staleness_h        = {inputs['staleness_h']} (hours; must increase priority)")
    print(f"  vulnerable_density = {inputs['vulnerable_density']} (dimensionless; no headcount)")
    print(f"  doubt              = {inputs['doubt']} (dimensionless; no bare %)")
    print(f"  road_cutoff        = {inputs['road_cutoff']} (dimensionless multiplier; no distance unit)")
    print(f"  facility_near.dist_m = {SAMPLE_RANK_ITEM['facility_near']['dist_m']} (the only field in meters)")
    print()

    try:
        rationale = client.generate_rationale(SAMPLE_RANK_ITEM)
    except NanoClientError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    print("OK. Rationale (passed the faithfulness check):")
    print(rationale)


if __name__ == "__main__":
    main()
