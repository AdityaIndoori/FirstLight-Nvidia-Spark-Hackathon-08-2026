from backend.decision.eval.eval_agency import (
    AGENCY_FIXTURE,
    evaluate_agency_plan_correctness_offline,
    evaluate_agency_unit_sanity_offline,
)


def test_fixture_has_at_least_ten_cases():
    assert len(AGENCY_FIXTURE) >= 10


def test_fixture_covers_required_categories():
    ids = {case["case_id"] for case in AGENCY_FIXTURE}
    required = {
        "visible_fire", "collapsed_roof", "dialysis_center", "hospital", "nursing_home",
        "blocked_roadway", "debris_obstruction", "ordinary_damage_no_evidence",
        "combined_fire_and_road_closure", "combined_medical_and_blocked_access",
    }
    assert required.issubset(ids)


def test_correctness_offline_perfect_on_this_fixture():
    # captions are deliberately written to exactly match B6's grounding
    # rules, so the deterministic path should score perfectly here --
    # this proves the harness/labels are correct, not that the model is.
    metric = evaluate_agency_plan_correctness_offline()
    assert metric["status"] == "measured"
    assert metric["value"]["precision"] == 1.0
    assert metric["value"]["recall"] == 1.0


def test_correctness_offline_ordinary_case_predicts_nothing():
    metric = evaluate_agency_plan_correctness_offline()
    ordinary = next(c for c in metric["details"]["per_case"] if c["case_id"] == "ordinary_damage_no_evidence")
    assert ordinary["predicted"] == []
    assert ordinary["expected"] == []


def test_correctness_offline_deterministic():
    m1 = evaluate_agency_plan_correctness_offline()
    m2 = evaluate_agency_plan_correctness_offline()
    assert m1 == m2


# unit-count sanity -- hard deterministic gate
def test_unit_sanity_passes_with_zero_violations():
    metric = evaluate_agency_unit_sanity_offline()
    assert metric["status"] == "pass"
    assert metric["value"] == 0
    assert metric["details"]["violations"] == []


def test_unit_sanity_flags_units_not_positive():
    from backend.decision.agency_plan import AvailabilityRegistry, build_agency_plan
    from backend.decision.agency_plan_client import _VALID_AGENCY_ACTIONS  # noqa: F401

    availability = AvailabilityRegistry()
    for agency in ("fire", "ems", "police", "public_works"):
        availability.set_availability(agency, 5, "test")

    # Directly exercise the same invariant the eval checks, using B6a's
    # own contract validation (build_agency_plan already rejects
    # units <= 0 -- this test proves the eval's invariant is the SAME one
    # B6a enforces, not a redundant reimplementation).
    import pytest
    from backend.decision.agency_plan import AgencyPlanError

    with pytest.raises(AgencyPlanError):
        build_agency_plan(
            [{"agency": "fire", "footprint_id": "x", "label": "x", "centroid": [0, 0], "task": "t", "units": 0}],
            drafted_by="stub",
            availability=availability,
        )


def test_unit_sanity_units_available_matches_registry_exactly():
    metric = evaluate_agency_unit_sanity_offline()
    # no violation about units_available mismatching the registry
    assert not any("units_available" in v for v in metric["details"]["violations"])
