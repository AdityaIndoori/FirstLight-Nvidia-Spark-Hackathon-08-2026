#!/usr/bin/env python3
"""B6a deterministic demo/live check -- no model, no network, no NemoClaw.

Builds a sample AgencyPlan (all four agencies, Fire deliberately
overcommitted), prints it, then performs the demoed edit -- reassigning one
step from Fire to EMS -- and prints the plan again so the numbering,
units_required, and overcommitment recompute are all visibly different,
while units_available stays put.

Usage:
    python scripts/agency_plan_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.agency_plan import (
    AvailabilityRegistry,
    apply_plan_edit,
    build_agency_plan,
    is_overcommitted,
    units_shortfall,
)

ASSIGNMENTS = [
    {
        "agency": "fire",
        "footprint_id": "fp-100",
        "label": "412 Elm St",
        "centroid": [-122.400, 47.600],
        "task": "Structure fire - extinguish",
        "units": 2,
    },
    {
        "agency": "fire",
        "footprint_id": "fp-101",
        "label": "88 Oak Ave",
        "centroid": [-122.395, 47.605],
        "task": "Collapsed structure - entrapment rescue",
        "units": 1,
    },
    {
        "agency": "ems",
        "footprint_id": "fp-200",
        "label": "Riverside Dialysis Center",
        "centroid": [-122.385, 47.605],
        "task": "Dialysis facility - patient evacuation support",
        "units": 2,
    },
    {
        "agency": "police",
        "footprint_id": "fp-300",
        "label": "35th Ave SW & Elm St",
        "centroid": [-122.390, 47.598],
        "task": "Road perimeter control",
        "units": 1,
    },
    {
        "agency": "public_works",
        "footprint_id": "fp-400",
        "label": "Harbor Ave SW",
        "centroid": [-122.393, 47.601],
        "task": "Debris clearance",
        "units": 1,
    },
]

_LABELS = {"fire": "FIRE", "ems": "EMS", "police": "POLICE", "public_works": "PUBLIC WORKS"}


def print_plan(plan: dict) -> None:
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


def main():
    availability = AvailabilityRegistry()
    availability.set_availability("fire", 2, "operator:jsmith")  # deliberately under required=3
    availability.set_availability("ems", 2, "operator:jsmith")
    availability.set_availability("police", 1, "operator:jsmith")
    availability.set_availability("public_works", 1, "operator:jsmith")

    plan = build_agency_plan(ASSIGNMENTS, drafted_by="stub", availability=availability)

    print("=" * 60)
    print("BEFORE")
    print("=" * 60)
    print_plan(plan)

    print("Demoed edit: reassign fp-100 (Fire step 1) -> EMS\n")
    reassigned_plan = apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "reassign",
            "step_n": 1,
            "payload": {"to_agency": "ems"},
            "operator": "operator:jsmith",
        },
    )

    print("=" * 60)
    print("AFTER")
    print("=" * 60)
    print_plan(reassigned_plan)

    fire_before = next(g for g in plan["agencies"] if g["agency"] == "fire")
    fire_after = next(g for g in reassigned_plan["agencies"] if g["agency"] == "fire")
    ems_before = next(g for g in plan["agencies"] if g["agency"] == "ems")
    ems_after = next(g for g in reassigned_plan["agencies"] if g["agency"] == "ems")

    print("=" * 60)
    print("DIFF SUMMARY")
    print("=" * 60)
    print(f"Fire numbering:        {[s['n'] for s in fire_before['steps']]} -> {[s['n'] for s in fire_after['steps']]}")
    print(f"EMS numbering:         {[s['n'] for s in ems_before['steps']]} -> {[s['n'] for s in ems_after['steps']]}")
    print(f"Fire units_required:   {fire_before['units_required']} -> {fire_after['units_required']}")
    print(f"EMS units_required:    {ems_before['units_required']} -> {ems_after['units_required']}")
    print(f"Fire units_available:  {fire_before['units_available']} -> {fire_after['units_available']} (unchanged)")
    print(f"EMS units_available:   {ems_before['units_available']} -> {ems_after['units_available']} (unchanged)")
    print(f"Fire overcommitted:    {is_overcommitted(fire_before)} -> {is_overcommitted(fire_after)}")
    print(f"Fire shortfall:        {units_shortfall(fire_before)} -> {units_shortfall(fire_after)}")
    print(f"EMS overcommitted:     {is_overcommitted(ems_before)} -> {is_overcommitted(ems_after)}")
    print(f"EMS shortfall:         {units_shortfall(ems_before)} -> {units_shortfall(ems_after)}")


if __name__ == "__main__":
    main()
