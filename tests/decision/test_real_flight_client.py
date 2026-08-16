import json

import pytest

from backend.decision.flight_client import RealNanoFlightClient, build_deterministic_flight_plan
from backend.decision.flight_planner import request_flight_plan, validate_flight_plan
from backend.decision.nano_client import NanoClientError

PLANNING_INPUT = {
    "footprint_id": "fp-900",
    "centroid": [-122.4194, 37.7749],
    "area_radius_m": 150.0,
    "altitude_m_agl": 60.0,
    "line_spacing_m": 40.0,
    "candidates": [
        {
            "footprint_id": "fp-001",
            "label": "Elm St Home",
            "centroid": [-122.400, 37.770],
            "damage_class": 1,
            "confirmed": False,
            "facility_near": None,
            "priority": 5.0,
            "rationale": "Minor damage, low priority.",
        },
        {
            "footprint_id": "fp-002",
            "label": "Riverside Dialysis Center",
            "centroid": [-122.410, 37.780],
            "damage_class": 3,
            "confirmed": True,
            "facility_near": {"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0},
            "priority": 24.19320,
            "rationale": "Destroyed, confirmed, adjacent to a dialysis facility.",
        },
        {
            "footprint_id": "fp-003",
            "label": "Oak Ave Building",
            "centroid": [-122.395, 37.775],
            "damage_class": 2,
            "confirmed": False,
            "facility_near": None,
            "priority": 10.0,
            "rationale": None,
        },
    ],
}


class FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _install_fake_urlopen(monkeypatch, content_by_call):
    """content_by_call: list of raw content strings, or Exception instances
    to raise, one per HTTP call in order."""
    import urllib.request

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        item = content_by_call[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        body = {"choices": [{"message": {"content": item}, "finish_reason": "stop"}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _sent_payload(request) -> dict:
    return json.loads(request.data)


def _valid_content(footprint_id: str) -> str:
    return json.dumps({"target_footprint_id": footprint_id})


# 1: real-client request uses model=nano
def test_real_client_request_uses_model_nano(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, [_valid_content("fp-002")])
    RealNanoFlightClient(base_url="http://localhost:8000").request_flight_plan(PLANNING_INPUT)

    payload = _sent_payload(calls[0])
    assert payload["model"] == "nano"


# 2: flight tasking enables reasoning (/think, not /no_think)
def test_flight_tasking_enables_reasoning(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, [_valid_content("fp-002")])
    RealNanoFlightClient(base_url="http://localhost:8000").request_flight_plan(PLANNING_INPUT)

    payload = _sent_payload(calls[0])
    assert payload["messages"][0] == {"role": "system", "content": "/think"}


# 3: structured output schema is supplied, enum-constrained to candidates
def test_structured_output_schema_supplied_with_candidate_enum(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, [_valid_content("fp-002")])
    RealNanoFlightClient(base_url="http://localhost:8000").request_flight_plan(PLANNING_INPUT)

    payload = _sent_payload(calls[0])
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["target_footprint_id"]
    assert set(schema["properties"]["target_footprint_id"]["enum"]) == {"fp-001", "fp-002", "fp-003"}
    assert schema["additionalProperties"] is False


def test_max_tokens_is_bounded(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, [_valid_content("fp-002")])
    RealNanoFlightClient(base_url="http://localhost:8000").request_flight_plan(PLANNING_INPUT)

    payload = _sent_payload(calls[0])
    assert isinstance(payload["max_tokens"], int) and payload["max_tokens"] > 0


# 4: candidate IDs are grounded -- selecting a real candidate produces a
# valid plan built from ITS authoritative centroid, never Nano's own text
def test_candidate_selection_is_grounded_to_authoritative_centroid(monkeypatch):
    _install_fake_urlopen(monkeypatch, [_valid_content("fp-002")])
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    plan = client.request_flight_plan(PLANNING_INPUT)

    assert validate_flight_plan(plan) == []
    assert client.last_selected_footprint_id == "fp-002"

    expected = build_deterministic_flight_plan(
        {
            "centroid": [-122.410, 37.780],  # fp-002's own centroid, not planning_input's default
            "area_radius_m": PLANNING_INPUT["area_radius_m"],
            "altitude_m_agl": PLANNING_INPUT["altitude_m_agl"],
            "line_spacing_m": PLANNING_INPUT["line_spacing_m"],
        }
    )
    assert plan == expected


# 5: invented footprint rejected
def test_invented_footprint_id_is_rejected(monkeypatch):
    _install_fake_urlopen(monkeypatch, [_valid_content("fp-does-not-exist")])
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    plan = client.request_flight_plan(PLANNING_INPUT)

    assert validate_flight_plan(plan) != []
    assert client.last_selected_footprint_id is None
    assert client.last_grounding_error is not None
    assert "fp-does-not-exist" in client.last_grounding_error


# 6: invented coordinates rejected -- Nano's schema has no coordinate
# field at all; a raw-response attempt to smuggle one in is ignored
def test_invented_coordinates_are_structurally_impossible_to_apply(monkeypatch):
    content = json.dumps({"target_footprint_id": "fp-002", "centroid": [999.0, 999.0]})
    _install_fake_urlopen(monkeypatch, [content])
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    plan = client.request_flight_plan(PLANNING_INPUT)

    assert validate_flight_plan(plan) == []
    # the geometry must be centered on fp-002's real centroid, not the injected one
    survey_area = next(f for f in plan["features"] if f["properties"]["role"] == "survey-area")
    lngs = [pt[0] for pt in survey_area["geometry"]["coordinates"][0]]
    assert max(lngs) < 0  # nowhere near the injected 999.0


# 7: malformed GeoJSON / malformed model response rejected
def test_malformed_json_response_raises_nano_client_error(monkeypatch):
    _install_fake_urlopen(monkeypatch, ["not valid json"])
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    with pytest.raises(NanoClientError):
        client.request_flight_plan(PLANNING_INPUT)


def test_missing_target_field_raises_nano_client_error(monkeypatch):
    _install_fake_urlopen(monkeypatch, [json.dumps({"something_else": "x"})])
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    with pytest.raises(NanoClientError):
        client.request_flight_plan(PLANNING_INPUT)


# 8: wrong feature count rejected -- exercised through the retry/fallback
# integration below (invalid candidate always has an incomplete
# survey-path), and directly via the existing validate_flight_plan tests
def test_grounding_invalid_candidate_fails_existing_structural_validator(monkeypatch):
    _install_fake_urlopen(monkeypatch, [_valid_content("fp-invented")])
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    plan = client.request_flight_plan(PLANNING_INPUT)
    errors = validate_flight_plan(plan)
    assert any("line_spacing_m" in e for e in errors)


# 9 & 10: retry occurs exactly once for invalid model output, valid retry accepted
def test_retry_occurs_exactly_once_and_valid_retry_is_accepted(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, [_valid_content("fp-invented"), _valid_content("fp-002")])
    client = RealNanoFlightClient(base_url="http://localhost:8000")

    result = request_flight_plan(PLANNING_INPUT, client=client)

    assert len(calls) == 2
    assert result["recovery"] == "model"
    assert validate_flight_plan(result["flight_plan"]) == []


def test_retry_includes_validation_error_text(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, [_valid_content("fp-invented"), _valid_content("fp-002")])
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    request_flight_plan(PLANNING_INPUT, client=client)

    retry_prompt = _sent_payload(calls[1])["messages"][-1]["content"]
    assert "invalid" in retry_prompt.lower()


# 11: second invalid response triggers deterministic fallback
def test_second_invalid_response_triggers_deterministic_fallback(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, [_valid_content("fp-a"), _valid_content("fp-b")])
    client = RealNanoFlightClient(base_url="http://localhost:8000")

    result = request_flight_plan(PLANNING_INPUT, client=client)

    assert len(calls) == 2
    assert result["recovery"] == "stub"
    assert validate_flight_plan(result["flight_plan"]) == []
    assert result["flight_plan"] == build_deterministic_flight_plan(PLANNING_INPUT)


# 12: transport failure triggers recovery
def test_transport_failure_on_first_attempt_triggers_stub_recovery(monkeypatch):
    import urllib.error

    _install_fake_urlopen(monkeypatch, [urllib.error.URLError("timed out")])
    client = RealNanoFlightClient(base_url="http://localhost:8000")

    result = request_flight_plan(PLANNING_INPUT, client=client)
    assert result["recovery"] == "stub"
    assert validate_flight_plan(result["flight_plan"]) == []


def test_transport_failure_on_retry_triggers_stub_recovery(monkeypatch):
    import urllib.error

    _install_fake_urlopen(monkeypatch, [_valid_content("fp-invented"), urllib.error.URLError("timed out")])
    client = RealNanoFlightClient(base_url="http://localhost:8000")

    result = request_flight_plan(PLANNING_INPUT, client=client)
    assert result["recovery"] == "stub"
    assert validate_flight_plan(result["flight_plan"]) == []


def test_missing_candidates_raises_nano_client_error():
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    with pytest.raises(NanoClientError):
        client.request_flight_plan({**PLANNING_INPUT, "candidates": []})


def test_missing_candidates_via_orchestration_falls_back():
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    result = request_flight_plan({**PLANNING_INPUT, "candidates": []}, client=client)
    assert result["recovery"] == "stub"


# 13: deterministic fallback remains valid
def test_deterministic_fallback_remains_valid():
    assert validate_flight_plan(build_deterministic_flight_plan(PLANNING_INPUT)) == []


# force_invalid_first diagnostic flag: first attempt rejected without a
# real model call, retry calls the real model normally
def test_force_invalid_first_rejects_without_calling_model_then_retries(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, [_valid_content("fp-002")])
    client = RealNanoFlightClient(base_url="http://localhost:8000", force_invalid_first=True)

    result = request_flight_plan(PLANNING_INPUT, client=client)

    assert len(calls) == 1  # first attempt never touched the network
    assert result["recovery"] == "model"
    assert validate_flight_plan(result["flight_plan"]) == []


def test_force_invalid_first_default_is_false():
    client = RealNanoFlightClient(base_url="http://localhost:8000")
    assert client.force_invalid_first is False


# usage_log captures token usage when present
def test_usage_log_captures_usage(monkeypatch):
    import urllib.request

    def fake_urlopen(request, timeout=None):
        body = {
            "choices": [{"message": {"content": _valid_content("fp-002")}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 150, "total_tokens": 450},
        }
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = RealNanoFlightClient(base_url="http://localhost:8000")
    client.request_flight_plan(PLANNING_INPUT)
    assert client.usage_log == [{"prompt_tokens": 300, "completion_tokens": 150, "total_tokens": 450}]


# planning_input is never mutated
def test_planning_input_not_mutated(monkeypatch):
    import copy

    snapshot = copy.deepcopy(PLANNING_INPUT)
    _install_fake_urlopen(monkeypatch, [_valid_content("fp-002")])
    working_copy = copy.deepcopy(PLANNING_INPUT)
    RealNanoFlightClient(base_url="http://localhost:8000").request_flight_plan(working_copy)
    assert working_copy == snapshot


# no real network access
def test_no_real_network_access(monkeypatch):
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(socket, "create_connection", _forbidden)

    # constructing the client performs no I/O
    RealNanoFlightClient(base_url="http://localhost:8000")
