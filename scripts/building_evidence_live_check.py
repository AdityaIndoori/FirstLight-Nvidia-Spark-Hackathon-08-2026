#!/usr/bin/env python3
"""Manual, opt-in check of the full B-side BuildingEvidence processing path
against the REAL Lightning vLLM server.

Not part of the pytest suite -- run by hand once the DGX Spark's Lightning
vLLM server is reachable (directly, or via the Mac's SSH tunnel to
localhost:8001). Runs the EXISTING, unchanged pipeline
(building_evidence.process_building_evidence -> lightning_ballot.request_lightning_ballot
-> scoring.calculate_priority) over one deterministic BuildingEvidence
fixture, with a fixed scored_at so staleness_h is reproducible.

Usage:
    python scripts/building_evidence_live_check.py
    FIRSTLIGHT_LIGHTNING_BASE_URL=http://localhost:8001 python scripts/building_evidence_live_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.building_evidence import BuildingEvidenceError, process_building_evidence
from backend.decision.lightning_client import RealLightningSeverityClient

CAPTURED_AT = 1_700_000_000.0  # fixed timestamp
SCORED_AT = CAPTURED_AT + 6.5 * 3600.0  # exactly 6.5 hours later

BUILDING_EVIDENCE = {
    "footprint_id": "bldg-0042",
    "image_id": "img-0187",
    "label": "412 Elm St",
    "centroid": [-122.3764, 47.5581],
    "captured_at": CAPTURED_AT,
    "damage_class": 2,
    "confidence": 0.85,
    "graded_by": "nemotron-vl",
    "vl_caption": "Two-storey structure with partial roof collapse and debris along the eastern wall.",
    "footprint_area_m2": 120.0,
    "facility_near": {"name": "Riverside Clinic", "type": "dialysis", "dist_m": 150},
    "neighbor_damage_classes": [1, 2, 2],
    "vulnerable_density": 2.31,
}

ROAD_CUTOFF = 1.8


def main():
    client = RealLightningSeverityClient()
    print(f"Requesting Lightning k=8 ballot from {client.base_url} (model=lightning, enable_thinking=false)...\n")

    print(f"footprint_id: {BUILDING_EVIDENCE['footprint_id']}")
    print(f"damage_class (original grader value): {BUILDING_EVIDENCE['damage_class']}")
    print(f'vl_caption: "{BUILDING_EVIDENCE["vl_caption"]}"\n')

    try:
        result = process_building_evidence(
            BUILDING_EVIDENCE, scored_at=SCORED_AT, road_cutoff=ROAD_CUTOFF, real_client=client
        )
    except BuildingEvidenceError as exc:
        print(f"FAILED (invalid BuildingEvidence): {exc}")
        sys.exit(1)

    print(f"Lightning votes: {result['votes']}")
    print(f"voted_class: {result['voted_class']}  <- Lightning's own, separate result")
    print(f"vote_agreement: {result['vote_agreement']}")
    print(f"doubt: {result['doubt']}\n")

    print(f"staleness_h: {result['staleness_h']}")
    print(f"vulnerable_density: {BUILDING_EVIDENCE['vulnerable_density']}")
    print(f"road_cutoff: {result['road_cutoff']}")
    print(f"priority: {result['priority']}\n")

    print(f"recovery: {result['lightning_recovery']}\n")

    print(f"damage_class remains the original grader value: {result['evidence']['damage_class']}")
    print(f"voted_class is Lightning's separate result: {result['voted_class']}")


if __name__ == "__main__":
    main()
