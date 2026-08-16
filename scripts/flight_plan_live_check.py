#!/usr/bin/env python3
"""B2 live check -- runs the REAL Nano next-flight-tasking path
(RealNanoFlightClient, reasoning ENABLED) against localhost:8000.

Not part of the pytest suite. Uses a small realistic ranked fixture (4
candidate buildings, spanning damage classes and one facility-adjacent
case) and the production request_flight_plan() orchestration
(flight_planner.py, completely unchanged by this script) -- nothing here
reimplements validation, retry, or fallback logic.

Usage:
    python scripts/flight_plan_live_check.py
    python scripts/flight_plan_live_check.py --force-invalid-first
    FIRSTLIGHT_NANO_BASE_URL=http://localhost:8000 python scripts/flight_plan_live_check.py

--force-invalid-first proves the retry path deterministically (first
attempt rejected without even calling the model, second attempt calls the
real model normally) -- see flight_client.RealNanoFlightClient's
force_invalid_first, default False, never affects production behavior.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.flight_client import RealNanoFlightClient  # noqa: E402
from backend.decision.flight_planner import request_flight_plan, validate_flight_plan  # noqa: E402

CANDIDATES = [
    {
        "footprint_id": "fp-001",
        "label": "412 Elm St",
        "centroid": [-122.4001, 47.6003],
        "damage_class": 1,
        "confirmed": False,
        "facility_near": None,
        "priority": 3.20000,
        "rationale": "Minor cracking observed; low operational urgency.",
    },
    {
        "footprint_id": "fp-002",
        "label": "Riverside Dialysis Center",
        "centroid": [-122.3850, 47.6050],
        "damage_class": 3,
        "confirmed": True,
        "facility_near": {"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0},
        "priority": 24.19320,
        "rationale": "Confirmed destroyed, adjacent to a dialysis facility -- highest operational priority.",
    },
    {
        "footprint_id": "fp-003",
        "label": "88 Oak Ave",
        "centroid": [-122.3951, 47.5990],
        "damage_class": 2,
        "confirmed": False,
        "facility_near": None,
        "priority": 9.50000,
        "rationale": "Major damage, unconfirmed grade -- moderate priority pending verification.",
    },
    {
        "footprint_id": "fp-004",
        "label": "35th Ave SW & Elm St",
        "centroid": [-122.3902, 47.5981],
        "damage_class": 3,
        "confirmed": True,
        "facility_near": None,
        "priority": 15.75000,
        "rationale": "Confirmed destroyed structure, no nearby vulnerable facility.",
    },
]

PLANNING_INPUT = {
    "footprint_id": CANDIDATES[0]["footprint_id"],  # deterministic fallback target if Nano/retry both fail
    "centroid": CANDIDATES[0]["centroid"],
    "area_radius_m": 150.0,
    "altitude_m_agl": 60.0,
    "line_spacing_m": 40.0,
    "candidates": CANDIDATES,
}


def _print_geojson_summary(flight_plan: dict) -> None:
    for feature in flight_plan["features"]:
        role = feature["properties"]["role"]
        geom = feature["geometry"]
        if role == "survey-area":
            ring = geom["coordinates"][0]
            print(f"  survey-area: Polygon, {len(ring)} ring points, first={ring[0]}")
        else:
            coords = geom["coordinates"]
            print(
                f"  survey-path: LineString, {len(coords)} points, "
                f"altitude_m_agl={feature['properties']['altitude_m_agl']}, "
                f"line_spacing_m={feature['properties']['line_spacing_m']}, "
                f"transects={feature['properties']['transects']}, "
                f"est_flight_min={feature['properties']['est_flight_min']}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-invalid-first",
        action="store_true",
        help="deterministically reject the first attempt (no model call) to prove retry recovery",
    )
    args = parser.parse_args()

    client = RealNanoFlightClient(force_invalid_first=args.force_invalid_first)

    print(f"Nano endpoint: {client.base_url}/v1/chat/completions")
    print("model: nano")
    print("reasoning enabled: true (/think)")
    print(f"candidate count: {len(CANDIDATES)}")
    print(f"force_invalid_first: {args.force_invalid_first}\n")

    start = time.perf_counter()
    result = request_flight_plan(PLANNING_INPUT, client=client)
    elapsed_s = time.perf_counter() - start

    # finish_reason_log only grows on successful HTTP responses (a
    # transport failure never reaches it), so its length is exactly how
    # many real model calls actually received a response this run.
    real_calls_with_response = len(client.finish_reason_log)

    print(f"elapsed_s: {elapsed_s:.2f}")
    print(f"real model calls that received a response: {real_calls_with_response}")
    print(f"recovery: {result['recovery']!r}  (None=nano first try, 'model'=nano after retry, 'stub'=deterministic fallback)")
    print(f"fallback: {result['recovery'] == 'stub'}")
    if client.finish_reason_log:
        print(f"finish_reason(s): {client.finish_reason_log}")
    if client.usage_log:
        total_prompt = sum(u.get("prompt_tokens", 0) for u in client.usage_log if u)
        total_completion = sum(u.get("completion_tokens", 0) for u in client.usage_log if u)
        print(f"prompt_tokens_total: {total_prompt}")
        print(f"completion_tokens_total: {total_completion}")
    if client.last_grounding_error:
        print(f"last_grounding_error: {client.last_grounding_error}")
    print(f"final selected footprint_id: {client.last_selected_footprint_id!r}")
    print()

    print("Final GeoJSON summary:")
    _print_geojson_summary(result["flight_plan"])
    print()

    validation_errors = validate_flight_plan(result["flight_plan"])
    print(f"grounding validation: {'PASSED' if not validation_errors else 'FAILED'}")
    if validation_errors:
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
