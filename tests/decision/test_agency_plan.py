import copy

import pytest

from backend.decision.agency_plan import (
    SUPPORTED_AGENCIES,
    AgencyPlanError,
    AvailabilityRegistry,
    apply_plan_edit,
    build_agency_plan,
    is_overcommitted,
    units_shortfall,
)

# Fixture: Fire is overcommitted (required=3, available=2), the rest are not.
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


def make_availability() -> AvailabilityRegistry:
    availability = AvailabilityRegistry()
    availability.set_availability("fire", 2, "operator:jsmith")  # overcommitted (3 required)
    availability.set_availability("ems", 2, "operator:jsmith")
    availability.set_availability("police", 1, "operator:jsmith")
    availability.set_availability("public_works", 1, "operator:jsmith")
    return availability


def make_plan() -> dict:
    return build_agency_plan(ASSIGNMENTS, drafted_by="stub", availability=make_availability())


def group(plan: dict, agency: str) -> dict:
    for g in plan["agencies"]:
        if g["agency"] == agency:
            return g
    raise AssertionError(f"agency {agency!r} not in plan")


# 1: all four agency groups exist
def test_all_four_agency_groups_exist():
    plan = build_agency_plan(
        [ASSIGNMENTS[0]], drafted_by="stub", availability=AvailabilityRegistry()
    )  # only a fire assignment
    agencies_present = {g["agency"] for g in plan["agencies"]}

    assert agencies_present == set(SUPPORTED_AGENCIES)
    assert group(plan, "ems")["steps"] == []
    assert group(plan, "police")["steps"] == []
    assert group(plan, "public_works")["steps"] == []


# 2: unsupported agency rejected
def test_unsupported_agency_rejected_in_build():
    bad_assignment = dict(ASSIGNMENTS[0], agency="hazmat")
    with pytest.raises(AgencyPlanError):
        build_agency_plan([bad_assignment], drafted_by="stub", availability=AvailabilityRegistry())


def test_unsupported_agency_rejected_in_availability():
    with pytest.raises(AgencyPlanError):
        AvailabilityRegistry().set_availability("hazmat", 2, "operator:jsmith")


def test_unsupported_agency_rejected_in_plan_edit():
    plan = make_plan()
    with pytest.raises(AgencyPlanError):
        apply_plan_edit(
            plan,
            {
                "agency": "hazmat",
                "op": "delete",
                "step_n": 1,
                "payload": {},
                "operator": "operator:jsmith",
            },
        )


# 3: units_available defaults according to chosen project convention (0, undocumented default elsewhere)
def test_units_available_defaults_to_zero():
    availability = AvailabilityRegistry()
    for agency in SUPPORTED_AGENCIES:
        assert availability.get_availability(agency) == 0


# 4: availability can only be operator supplied
def test_availability_only_settable_via_set_availability():
    availability = AvailabilityRegistry()
    plan = build_agency_plan(ASSIGNMENTS, drafted_by="stub", availability=availability)
    # Nothing supplied availability yet -> all zero, regardless of steps/units_required.
    for g in plan["agencies"]:
        assert g["units_available"] == 0

    availability.set_availability("fire", 5, "operator:jsmith")
    plan2 = build_agency_plan(ASSIGNMENTS, drafted_by="stub", availability=availability)
    assert group(plan2, "fire")["units_available"] == 5
    assert group(plan2, "ems")["units_available"] == 0  # untouched agencies stay at their set value (0)


# 5: negative availability rejected
def test_negative_availability_rejected():
    with pytest.raises(AgencyPlanError):
        AvailabilityRegistry().set_availability("fire", -1, "operator:jsmith")


def test_empty_operator_rejected_in_set_availability():
    with pytest.raises(AgencyPlanError):
        AvailabilityRegistry().set_availability("fire", 2, "")


# 6: units_required equals sum of step units
def test_units_required_equals_sum_of_step_units():
    plan = make_plan()
    assert group(plan, "fire")["units_required"] == 3  # 2 + 1
    assert group(plan, "ems")["units_required"] == 2
    assert group(plan, "police")["units_required"] == 1
    assert group(plan, "public_works")["units_required"] == 1


def test_units_required_is_never_trusted_from_caller():
    # build_agency_plan's assignment shape has no units_required field at
    # all -- it cannot be supplied, only derived.
    assert "units_required" not in ASSIGNMENTS[0]


# 7: add renumbers correctly
def test_add_renumbers_correctly():
    plan = make_plan()
    edited = apply_plan_edit(
        plan,
        {
            "agency": "police",
            "op": "add",
            "step_n": None,
            "payload": {
                "footprint_id": "fp-301",
                "label": "36th Ave SW",
                "centroid": [-122.391, 47.599],
                "task": "Traffic control",
                "units": 1,
            },
            "operator": "operator:jsmith",
        },
    )
    police = group(edited, "police")
    assert [s["n"] for s in police["steps"]] == [1, 2]
    assert police["steps"][1]["footprint_id"] == "fp-301"
    assert police["units_required"] == 2


# 8: move renumbers correctly
def test_move_renumbers_correctly():
    plan = make_plan()
    edited = apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "move",
            "step_n": 1,
            "payload": {"to_n": 2},
            "operator": "operator:jsmith",
        },
    )
    fire = group(edited, "fire")
    assert [s["n"] for s in fire["steps"]] == [1, 2]
    assert fire["steps"][0]["footprint_id"] == "fp-101"  # what was step 2 is now step 1
    assert fire["steps"][1]["footprint_id"] == "fp-100"


def test_move_out_of_bounds_rejected():
    plan = make_plan()
    with pytest.raises(AgencyPlanError):
        apply_plan_edit(
            plan,
            {"agency": "fire", "op": "move", "step_n": 1, "payload": {"to_n": 99}, "operator": "operator:jsmith"},
        )


# 9: edit updates allowed fields
def test_edit_updates_allowed_fields():
    plan = make_plan()
    edited = apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "edit",
            "step_n": 1,
            "payload": {"units": 4, "task": "Structure fire - defensive"},
            "operator": "operator:jsmith",
        },
    )
    fire = group(edited, "fire")
    assert fire["steps"][0]["units"] == 4
    assert fire["steps"][0]["task"] == "Structure fire - defensive"
    assert fire["units_required"] == 4 + 1  # recomputed


def test_edit_cannot_change_footprint_id():
    plan = make_plan()
    with pytest.raises(AgencyPlanError):
        apply_plan_edit(
            plan,
            {
                "agency": "fire",
                "op": "edit",
                "step_n": 1,
                "payload": {"footprint_id": "fp-999"},
                "operator": "operator:jsmith",
            },
        )


# 10: delete renumbers correctly
def test_delete_renumbers_correctly():
    plan = make_plan()
    edited = apply_plan_edit(
        plan,
        {"agency": "fire", "op": "delete", "step_n": 1, "payload": {}, "operator": "operator:jsmith"},
    )
    fire = group(edited, "fire")
    assert [s["n"] for s in fire["steps"]] == [1]
    assert fire["steps"][0]["footprint_id"] == "fp-101"
    assert fire["units_required"] == 1


# 11, 12, 13, 14, 23: reassign
def test_reassign_removes_from_source_and_adds_exactly_once_to_destination():
    plan = make_plan()
    edited = apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "reassign",
            "step_n": 1,  # fp-100
            "payload": {"to_agency": "ems"},
            "operator": "operator:jsmith",
        },
    )

    fire = group(edited, "fire")
    ems = group(edited, "ems")

    # 11
    assert "fp-100" not in [s["footprint_id"] for s in fire["steps"]]
    # 12
    matches = [s for s in ems["steps"] if s["footprint_id"] == "fp-100"]
    assert len(matches) == 1
    # 13
    assert [s["n"] for s in fire["steps"]] == list(range(1, len(fire["steps"]) + 1))
    assert [s["n"] for s in ems["steps"]] == list(range(1, len(ems["steps"]) + 1))
    # 14
    assert fire["units_required"] == 1  # only fp-101 (1 unit) left
    assert ems["units_required"] == 4  # original fp-200 (2) + reassigned fp-100 (2)
    # 23
    assert matches[0]["footprint_id"] == "fp-100"
    assert matches[0]["label"] == "412 Elm St"
    assert matches[0]["centroid"] == [-122.400, 47.600]


def test_reassign_preserves_task_and_units_unless_overridden():
    plan = make_plan()
    edited = apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "reassign",
            "step_n": 1,
            "payload": {"to_agency": "ems"},
            "operator": "operator:jsmith",
        },
    )
    reassigned = [s for s in group(edited, "ems")["steps"] if s["footprint_id"] == "fp-100"][0]
    assert reassigned["task"] == "Structure fire - extinguish"  # preserved, no override supplied
    assert reassigned["units"] == 2  # preserved


def test_reassign_can_override_task_and_units():
    plan = make_plan()
    edited = apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "reassign",
            "step_n": 1,
            "payload": {"to_agency": "ems", "task": "Patient transport", "units": 3},
            "operator": "operator:jsmith",
        },
    )
    reassigned = [s for s in group(edited, "ems")["steps"] if s["footprint_id"] == "fp-100"][0]
    assert reassigned["task"] == "Patient transport"
    assert reassigned["units"] == 3


def test_reassign_never_duplicates_the_step():
    plan = make_plan()
    edited = apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "reassign",
            "step_n": 1,
            "payload": {"to_agency": "police"},
            "operator": "operator:jsmith",
        },
    )
    all_footprint_ids = [s["footprint_id"] for g in edited["agencies"] for s in g["steps"]]
    assert all_footprint_ids.count("fp-100") == 1


def test_reassign_to_same_agency_rejected():
    plan = make_plan()
    with pytest.raises(AgencyPlanError):
        apply_plan_edit(
            plan,
            {
                "agency": "fire",
                "op": "reassign",
                "step_n": 1,
                "payload": {"to_agency": "fire"},
                "operator": "operator:jsmith",
            },
        )


# 15: availability survives plan edits
def test_availability_survives_plan_edits():
    plan = make_plan()
    before = {g["agency"]: g["units_available"] for g in plan["agencies"]}

    edited = apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "reassign",
            "step_n": 1,
            "payload": {"to_agency": "ems"},
            "operator": "operator:jsmith",
        },
    )
    edited = apply_plan_edit(
        edited,
        {"agency": "police", "op": "delete", "step_n": 1, "payload": {}, "operator": "operator:jsmith"},
    )

    after = {g["agency"]: g["units_available"] for g in edited["agencies"]}
    assert after == before


# 16, 17, 18: overcommitment / shortfall
def test_overcommitment_detected_correctly():
    plan = make_plan()
    assert is_overcommitted(group(plan, "fire")) is True


def test_shortfall_calculated_correctly():
    plan = make_plan()
    assert units_shortfall(group(plan, "fire")) == 1  # 3 required - 2 available


def test_non_overcommitted_agency_returns_false():
    plan = make_plan()
    assert is_overcommitted(group(plan, "ems")) is False
    assert units_shortfall(group(plan, "ems")) == 0


# 19: invalid step units rejected
def test_invalid_step_units_rejected():
    plan = make_plan()
    for bad_units in (0, -1, "two", 1.5):
        with pytest.raises(AgencyPlanError):
            apply_plan_edit(
                plan,
                {
                    "agency": "police",
                    "op": "add",
                    "step_n": None,
                    "payload": {
                        "footprint_id": "fp-999",
                        "label": "Somewhere",
                        "centroid": [0.0, 0.0],
                        "task": "Task",
                        "units": bad_units,
                    },
                    "operator": "operator:jsmith",
                },
            )


# 20: invalid centroid rejected
def test_invalid_centroid_rejected():
    plan = make_plan()
    for bad_centroid in ([1.0], [1.0, 2.0, 3.0], "1,2", None):
        with pytest.raises(AgencyPlanError):
            apply_plan_edit(
                plan,
                {
                    "agency": "police",
                    "op": "add",
                    "step_n": None,
                    "payload": {
                        "footprint_id": "fp-999",
                        "label": "Somewhere",
                        "centroid": bad_centroid,
                        "task": "Task",
                        "units": 1,
                    },
                    "operator": "operator:jsmith",
                },
            )


# 21: caller's original plan is not unexpectedly mutated
def test_original_plan_not_mutated_by_edits():
    plan = make_plan()
    snapshot = copy.deepcopy(plan)

    apply_plan_edit(
        plan,
        {
            "agency": "fire",
            "op": "reassign",
            "step_n": 1,
            "payload": {"to_agency": "ems"},
            "operator": "operator:jsmith",
        },
    )

    assert plan == snapshot


def test_build_agency_plan_does_not_mutate_assignments():
    snapshot = copy.deepcopy(ASSIGNMENTS)
    build_agency_plan(ASSIGNMENTS, drafted_by="stub", availability=make_availability())
    assert ASSIGNMENTS == snapshot


# 22: drafted_by preserved
def test_drafted_by_preserved():
    plan = build_agency_plan(ASSIGNMENTS, drafted_by="operator:jsmith", availability=make_availability())
    assert plan["drafted_by"] == "operator:jsmith"

    edited = apply_plan_edit(
        plan,
        {"agency": "police", "op": "delete", "step_n": 1, "payload": {}, "operator": "operator:jsmith"},
    )
    assert edited["drafted_by"] == "operator:jsmith"


def test_empty_operator_rejected_in_plan_edit():
    plan = make_plan()
    with pytest.raises(AgencyPlanError):
        apply_plan_edit(
            plan, {"agency": "fire", "op": "delete", "step_n": 1, "payload": {}, "operator": ""}
        )


def test_invalid_op_rejected():
    plan = make_plan()
    with pytest.raises(AgencyPlanError):
        apply_plan_edit(
            plan, {"agency": "fire", "op": "explode", "step_n": 1, "payload": {}, "operator": "operator:jsmith"}
        )
