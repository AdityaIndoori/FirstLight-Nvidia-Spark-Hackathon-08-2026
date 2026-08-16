#!/usr/bin/env python3
"""B6b live check -- runs the REAL Nano agency-plan drafter against
localhost:8000, PER BUILDING with bounded parallelism.

Not part of the pytest suite. Uses 4 deterministic processed-building
fixtures (constructed directly in the ProcessedBuildingEvidence shape --
not re-run through the full B3 Lightning pipeline, since this script tests
B6b specifically) and a pre-seeded OPERATOR availability state.

Also performs the faithfulness check this task requires: scans every
resulting step's task text for fabricated casualties, trapped people,
occupancy counts, resource availability, property value, or owner names --
none of which were placed in any fixture below. A hit is reported as an
explicit FAITHFULNESS FAILURE, never silently accepted.

GROUNDING CHECK: since B (not Nano) now writes task text from a fixed
action->text mapping (_ACTION_TASK_TEXT), this script reverse-maps each
final step's task text back to its action code and re-runs
agency_plan_client.is_action_supported() against the ORIGINAL fixture
evidence for that footprint_id. Any assignment whose action the original
evidence does not support is an explicit GROUNDING FAILURE -- this is
exactly the class of live bug this task fixes (Nano assigning
fire_suppression to a dialysis building with no fire evidence). Grounding
failures make the faithfulness check fail loudly, the same as any other
fabrication.

Always prints "model buildings" / "fallback buildings" (recovery is now
PER BUILDING, so a mix is a normal, expected outcome -- not just an
all-or-nothing "fallback occurred" flag). When at least one building fell
back, also prints that building's MODEL RECOVERY DIAGNOSTICS (attempt_count,
error/category). This is DEVELOPMENT DIAGNOSTIC OUTPUT ONLY: it comes from
draft_agency_plan_with_diagnostics(), an internal wrapper never used by the
production draft_agency_plan() / public AgencyPlan contract.

Usage:
    python scripts/agency_plan_live_check.py
    FIRSTLIGHT_NANO_BASE_URL=http://localhost:8000 python scripts/agency_plan_live_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.agency_plan import AvailabilityRegistry, is_overcommitted, units_shortfall
from backend.decision.agency_plan_client import (
    _ACTION_TASK_TEXT,
    RealAgencyPlanDraftClient,
    is_action_supported,
)
from backend.decision.agency_plan_drafter import build_planner_candidate, draft_agency_plan_with_diagnostics

# Reverse of _ACTION_TASK_TEXT: display task text -> action code, so this
# script can check grounding against B's own deterministic output without
# a second classifier.
_TASK_TEXT_TO_ACTION = {text: action for action, text in _ACTION_TASK_TEXT.items()}

_FORBIDDEN_TERMS = (
    "casualt", "trapped", "occupant", "resident", "injured", "fatal", "death",
    "people inside", "property value", "owner", "$",
    "ambulance available", "units available", "resource availab",
)

# 4 deterministic processed-building fixtures, constructed directly rather
# than run through the full B3 Lightning ballot -- this script tests B6b
# (Nano agency-plan drafting) in isolation.
PROCESSED_FIXTURES = [
    {
        "evidence": {
            "footprint_id": "fp-001",
            "image_id": "img-001",
            "label": "412 Elm St",
            "centroid": [-122.400, 47.600],
            "captured_at": 1_700_000_000.0,
            "damage_class": 3,
            "confidence": 0.91,
            "graded_by": "nemotron-vl",
            "vl_caption": "Two-storey structure with visible flames and heavy roof damage.",
            "footprint_area_m2": 140.0,
            "facility_near": None,
            "neighbor_damage_classes": [2, 3, 1],
            "vulnerable_density": 1.8,
        },
        "votes": [3] * 8,
        "voted_class": 3,
        "vote_agreement": 1.0,
        "doubt": 0.05,
        "staleness_h": 3.0,
        "road_cutoff": None,
        "priority": 5.4,
        "lightning_recovery": "model",
    },
    {
        "evidence": {
            "footprint_id": "fp-002",
            "image_id": "img-002",
            "label": "Riverside Dialysis Center",
            "centroid": [-122.385, 47.605],
            "captured_at": 1_700_000_000.0,
            "damage_class": 2,
            "confidence": 0.85,
            "graded_by": "nemotron-vl",
            "vl_caption": "Building has significant exterior damage and obstructed entrance.",
            "footprint_area_m2": 200.0,
            "facility_near": {"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0},
            "neighbor_damage_classes": [1, 2, 2],
            "vulnerable_density": 2.4,
        },
        "votes": [2, 2, 2, 2, 2, 2, 3, 3],
        "voted_class": 2,
        "vote_agreement": 0.75,
        "doubt": 0.25,
        "staleness_h": 4.5,
        "road_cutoff": None,
        "priority": 4.9,
        "lightning_recovery": "model",
    },
    {
        "evidence": {
            "footprint_id": "fp-003",
            "image_id": "img-003",
            "label": "88 Oak Ave",
            "centroid": [-122.395, 47.598],
            "captured_at": 1_700_000_000.0,
            "damage_class": 1,
            "confidence": 0.7,
            "graded_by": "nemotron-vl",
            "vl_caption": "Large debris field obstructs roadway access beside the structure.",
            "footprint_area_m2": 95.0,
            "facility_near": None,
            "neighbor_damage_classes": [1, 1, 0],
            "vulnerable_density": 0.9,
        },
        "votes": [1] * 8,
        "voted_class": 1,
        "vote_agreement": 1.0,
        "doubt": 0.05,
        "staleness_h": 5.0,
        "road_cutoff": None,
        "priority": 1.2,
        "lightning_recovery": "model",
    },
    {
        "evidence": {
            "footprint_id": "fp-004",
            "image_id": "img-004",
            "label": "35th Ave SW & Elm St",
            "centroid": [-122.390, 47.598],
            "captured_at": 1_700_000_000.0,
            "damage_class": 1,
            "confidence": 0.66,
            "graded_by": "nemotron-vl",
            "vl_caption": "Damaged commercial structure adjacent to an active road closure.",
            "footprint_area_m2": 180.0,
            "facility_near": None,
            "neighbor_damage_classes": [0, 1, 1],
            "vulnerable_density": 1.1,
        },
        "votes": [1] * 8,
        "voted_class": 1,
        "vote_agreement": 1.0,
        "doubt": 0.05,
        "staleness_h": 5.5,
        "road_cutoff": None,
        "priority": 1.0,
        "lightning_recovery": "model",
    },
]

_LABELS = {"fire": "FIRE", "ems": "EMS", "police": "POLICE", "public_works": "PUBLIC WORKS"}


def _faithfulness_check(plan: dict, candidates_by_footprint_id: dict) -> list:
    """Scan every step's task text for fabricated content, AND verify
    grounding: reverse-map the step's task text back to its action code and
    confirm the ORIGINAL candidate evidence for that footprint_id actually
    supports it (agency_plan_client.is_action_supported). Returns a list of
    violation strings; empty means the plan is faithful and grounded.
    """
    violations = []
    for group in plan["agencies"]:
        for step in group["steps"]:
            lowered = step["task"].lower()
            for term in _FORBIDDEN_TERMS:
                if term in lowered:
                    violations.append(
                        f"{group['agency']} step {step['n']} ({step['footprint_id']}) task "
                        f"contains forbidden term {term!r}: {step['task']!r}"
                    )

            action = _TASK_TEXT_TO_ACTION.get(step["task"])
            if action is None:
                violations.append(
                    f"{group['agency']} step {step['n']} ({step['footprint_id']}) task "
                    f"{step['task']!r} does not match any known _ACTION_TASK_TEXT value"
                )
                continue
            candidate = candidates_by_footprint_id.get(step["footprint_id"])
            if candidate is None:
                violations.append(
                    f"{group['agency']} step {step['n']} references unknown footprint_id "
                    f"{step['footprint_id']!r}"
                )
                continue
            if not is_action_supported(action, candidate):
                violations.append(
                    f"GROUNDING FAILURE: {group['agency']} step {step['n']} "
                    f"({step['footprint_id']}) assigns {action!r} ({step['task']!r}) but the "
                    f"original evidence does not support it: {candidate['vl_caption']!r}, "
                    f"facility_near={candidate['facility_near']!r}"
                )
            else:
                print(f"  grounding: SUPPORTED  {step['footprint_id']} -> {action} ({step['task']!r})")
    return violations


def _print_building_recovery_diagnostics(building: dict) -> None:
    print(f"  footprint_id: {building['footprint_id']}")
    print(f"  recovery: {building['recovery']}")
    print(f"  attempt_count: {building['attempt_count']}")
    print(f"  elapsed_s: {building['elapsed_s']:.2f}")
    if building["attempt_1_error"]:
        print(f"  attempt 1 error: {building['attempt_1_error']}  [{building['attempt_1_error_category']}]")
    if building["attempt_2_error"]:
        print(f"  attempt 2 error: {building['attempt_2_error']}  [{building['attempt_2_error_category']}]")
    if building["fallback_reason"]:
        print(f"  fallback_reason: {building['fallback_reason']}")
    print()


def main():
    client = RealAgencyPlanDraftClient()
    print(f"Requesting agency-plan draft from {client.base_url} (model=nano, /no_think, json_schema)...\n")

    availability = AvailabilityRegistry()
    print("OPERATOR-ENTERED availability (fixture values, not observed/invented):")
    for agency, units in (("fire", 2), ("ems", 2), ("police", 1), ("public_works", 1)):
        availability.set_availability(agency, units, "operator:demo")
        print(f"  {agency} = {units}")
    print()

    candidates = [build_planner_candidate(fixture, confirmed=False) for fixture in PROCESSED_FIXTURES]
    candidates_by_footprint_id = {c["footprint_id"]: c for c in candidates}

    result = draft_agency_plan_with_diagnostics(candidates, availability, client=client)
    plan = result["plan"]
    diagnostics = result["diagnostics"]

    print(f"model buildings: {diagnostics['model_building_count']}")
    print(f"fallback buildings: {diagnostics['fallback_building_count']}\n")
    print(f"drafted_by: {plan['drafted_by']}\n")

    if diagnostics["fallback_building_count"] > 0:
        print("=" * 60)
        print("MODEL RECOVERY DIAGNOSTICS (buildings that used the fallback)")
        print("=" * 60)
        for building in diagnostics["buildings"]:
            if building["recovery"] == "stub":
                _print_building_recovery_diagnostics(building)

    for group in plan["agencies"]:
        print(_LABELS[group["agency"]])
        print(f"required: {group['units_required']}")
        print(f"available: {group['units_available']}")
        print(f"shortfall: {units_shortfall(group)}")
        print(f"overcommitted: {is_overcommitted(group)}")
        if not group["steps"]:
            print("  (no steps)")
        for step in group["steps"]:
            print(f"{step['n']}. [{step['footprint_id']}] {step['label']} -- {step['task']} ({step['units']} units)")
        print()

    print("GROUNDING CHECK (each assignment's action re-verified against its original evidence):")
    violations = _faithfulness_check(plan, candidates_by_footprint_id)
    print()
    if violations:
        print("FAITHFULNESS/GROUNDING FAILURE -- the live output contains unsupported claims:")
        for violation in violations:
            print(f"  - {violation}")
        sys.exit(1)

    print(
        "Faithfulness check: PASSED (no invented casualties/occupancy/resources/property "
        "value/owner names, and every assignment's action is grounded in its own evidence)."
    )


if __name__ == "__main__":
    main()
