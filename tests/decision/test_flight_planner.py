import copy

from backend.decision.flight_client import (
    NanoFlightClient,
    StubNanoFlightClient,
    build_deterministic_flight_plan,
)
from backend.decision.flight_planner import request_flight_plan, validate_flight_plan

PLANNING_INPUT = {
    "footprint_id": "fp-900",
    "centroid": [-122.4194, 37.7749],  # [lng, lat]
    "area_radius_m": 150.0,
    "altitude_m_agl": 60.0,
    "line_spacing_m": 40.0,
}


class ValidFirstClient(NanoFlightClient):
    """Always returns a valid candidate on the first call; records call count."""

    def __init__(self):
        self.calls = []

    def request_flight_plan(self, planning_input, validation_error=None):
        self.calls.append(validation_error)
        return build_deterministic_flight_plan(planning_input)


class InvalidThenValidClient(NanoFlightClient):
    """Invalid on the first attempt (validation_error is None), valid on retry."""

    def __init__(self):
        self.calls = []

    def request_flight_plan(self, planning_input, validation_error=None):
        self.calls.append(validation_error)
        if validation_error is None:
            plan = build_deterministic_flight_plan(planning_input)
            for feature in plan["features"]:
                if feature["properties"]["role"] == "survey-path":
                    del feature["properties"]["line_spacing_m"]
            return plan
        return build_deterministic_flight_plan(planning_input)


class AlwaysInvalidClient(NanoFlightClient):
    """Never produces a schema-valid candidate; records call count."""

    def __init__(self):
        self.calls = []

    def request_flight_plan(self, planning_input, validation_error=None):
        self.calls.append(validation_error)
        return {"type": "FeatureCollection", "features": []}


def _survey_path_props(flight_plan):
    for feature in flight_plan["features"]:
        if feature["properties"]["role"] == "survey-path":
            return feature["properties"]
    raise AssertionError("no survey-path feature")


# 1 & 2: valid first result returned without retry, recovery is None
def test_valid_first_result_returned_without_retry():
    client = ValidFirstClient()
    result = request_flight_plan(PLANNING_INPUT, client=client)

    assert len(client.calls) == 1
    assert validate_flight_plan(result["flight_plan"]) == []
    assert result["recovery"] is None


# 3: invalid first result causes exactly ONE retry
def test_invalid_first_result_causes_exactly_one_retry():
    client = InvalidThenValidClient()
    request_flight_plan(PLANNING_INPUT, client=client)

    assert len(client.calls) == 2


# 4: the validation error is supplied to the retry request
def test_validation_error_supplied_to_retry():
    client = InvalidThenValidClient()
    request_flight_plan(PLANNING_INPUT, client=client)

    first_call_error, retry_call_error = client.calls
    assert first_call_error is None
    assert isinstance(retry_call_error, str) and retry_call_error
    assert "line_spacing_m" in retry_call_error


# 5: invalid first -> valid second returns recovery = "model"
def test_invalid_first_then_valid_second_recovers_as_model():
    client = InvalidThenValidClient()
    result = request_flight_plan(PLANNING_INPUT, client=client)

    assert result["recovery"] == "model"
    assert validate_flight_plan(result["flight_plan"]) == []


# 6 & 7: invalid first -> invalid second uses deterministic fallback, recovery = "stub"
def test_invalid_first_and_second_uses_deterministic_fallback():
    client = AlwaysInvalidClient()
    result = request_flight_plan(PLANNING_INPUT, client=client)

    assert len(client.calls) == 2  # never more than one retry
    assert result["recovery"] == "stub"
    assert validate_flight_plan(result["flight_plan"]) == []
    assert result["flight_plan"] == build_deterministic_flight_plan(PLANNING_INPUT)


# 8: missing survey-area is rejected
def test_missing_survey_area_is_rejected():
    plan = build_deterministic_flight_plan(PLANNING_INPUT)
    plan["features"] = [f for f in plan["features"] if f["properties"]["role"] != "survey-area"]

    errors = validate_flight_plan(plan)
    assert any("survey-area" in e for e in errors)


# 9: missing survey-path is rejected
def test_missing_survey_path_is_rejected():
    plan = build_deterministic_flight_plan(PLANNING_INPUT)
    plan["features"] = [f for f in plan["features"] if f["properties"]["role"] != "survey-path"]

    errors = validate_flight_plan(plan)
    assert any("survey-path" in e for e in errors)


# 10: wrong geometry types are rejected
def test_wrong_geometry_types_are_rejected():
    plan = build_deterministic_flight_plan(PLANNING_INPUT)
    for feature in plan["features"]:
        if feature["properties"]["role"] == "survey-area":
            feature["geometry"]["type"] = "LineString"
        if feature["properties"]["role"] == "survey-path":
            feature["geometry"]["type"] = "Polygon"

    errors = validate_flight_plan(plan)
    assert any("Polygon" in e for e in errors)
    assert any("LineString" in e for e in errors)


# 11: missing required survey-path properties are rejected
def test_missing_survey_path_properties_are_rejected():
    for missing_field in ("altitude_m_agl", "line_spacing_m", "transects", "est_flight_min"):
        plan = build_deterministic_flight_plan(PLANNING_INPUT)
        for feature in plan["features"]:
            if feature["properties"]["role"] == "survey-path":
                del feature["properties"][missing_field]

        errors = validate_flight_plan(plan)
        assert any(missing_field in e for e in errors), f"expected error mentioning {missing_field}"


def test_non_feature_collection_is_rejected():
    assert validate_flight_plan({"type": "Feature"}) != []
    assert validate_flight_plan("not even a dict") != []


def test_malformed_coordinates_are_rejected():
    plan = build_deterministic_flight_plan(PLANNING_INPUT)
    for feature in plan["features"]:
        if feature["properties"]["role"] == "survey-path":
            feature["geometry"]["coordinates"] = [[999.0, 999.0], [0, 0]]  # out of [lng,lat] range

    errors = validate_flight_plan(plan)
    assert any("coordinate" in e for e in errors)


# 12: demo-force-invalid-first behavior deterministically exercises recovery
def test_demo_force_invalid_first_exercises_recovery():
    demo_client = StubNanoFlightClient(force_invalid_first=True)

    first_attempt = demo_client.request_flight_plan(PLANNING_INPUT)
    assert validate_flight_plan(first_attempt) != []  # deterministically invalid

    result = request_flight_plan(PLANNING_INPUT, client=demo_client)
    assert result["recovery"] == "model"
    assert validate_flight_plan(result["flight_plan"]) == []

    # Deterministic, not randomized: repeating the same request behaves identically.
    result_again = request_flight_plan(PLANNING_INPUT, client=StubNanoFlightClient(force_invalid_first=True))
    assert result_again["recovery"] == "model"
    assert result_again["flight_plan"] == result["flight_plan"]


# 13: identical fallback inputs produce identical fallback outputs
def test_identical_fallback_inputs_produce_identical_outputs():
    first = build_deterministic_flight_plan(copy.deepcopy(PLANNING_INPUT))
    second = build_deterministic_flight_plan(copy.deepcopy(PLANNING_INPUT))
    assert first == second

    result_a = request_flight_plan(PLANNING_INPUT, client=AlwaysInvalidClient())
    result_b = request_flight_plan(PLANNING_INPUT, client=AlwaysInvalidClient())
    assert result_a["flight_plan"] == result_b["flight_plan"]


# 14: input planning data is not mutated
def test_planning_input_not_mutated():
    snapshot = copy.deepcopy(PLANNING_INPUT)

    for client in (ValidFirstClient(), InvalidThenValidClient(), AlwaysInvalidClient()):
        working_copy = copy.deepcopy(PLANNING_INPUT)
        request_flight_plan(working_copy, client=client)
        assert working_copy == snapshot

    build_deterministic_flight_plan(copy.deepcopy(PLANNING_INPUT))
    assert PLANNING_INPUT == snapshot


# Sanity: the shared valid-plan properties actually carry the expected values.
def test_deterministic_flight_plan_has_expected_survey_path_properties():
    plan = build_deterministic_flight_plan(PLANNING_INPUT)
    props = _survey_path_props(plan)

    assert props["altitude_m_agl"] == PLANNING_INPUT["altitude_m_agl"]
    assert props["line_spacing_m"] == PLANNING_INPUT["line_spacing_m"]
    assert props["transects"] >= 1
    assert props["est_flight_min"] > 0
