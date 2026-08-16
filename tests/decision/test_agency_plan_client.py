import json

import pytest

from backend.decision.agency_plan_client import (
    RealAgencyPlanDraftClient,
    StubAgencyPlanDraftClient,
    is_action_supported,
)
from backend.decision.nano_client import NanoClientError

CANDIDATE_FIRE = {
    "footprint_id": "fp-100",
    "label": "412 Elm St",
    "centroid": [-122.400, 47.600],
    "damage_class": 3,
    "confidence": 0.91,
    "confirmed": True,
    "priority": 24.19320,
    "vl_caption": "Two-storey structure with visible flames and heavy roof damage.",
    "facility_near": None,
}

CANDIDATE_EMS = {
    "footprint_id": "fp-200",
    "label": "Riverside Dialysis Center",
    "centroid": [-122.385, 47.605],
    "damage_class": 2,
    "confidence": 0.78,
    "confirmed": False,
    "priority": 12.5,
    "vl_caption": "Building has significant exterior damage and obstructed entrance.",
    "facility_near": {"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0},
}

# Fixture A (section 8): dialysis facility, damage + obstructed entrance.
FIXTURE_A = {
    "footprint_id": "fp-a",
    "label": "Fixture A",
    "centroid": [0.0, 0.0],
    "damage_class": 2,
    "confidence": 0.8,
    "confirmed": False,
    "priority": 1.0,
    "vl_caption": "Building has significant exterior damage and obstructed entrance.",
    "facility_near": {"name": "Some Dialysis Center", "type": "dialysis", "dist_m": 0},
}

# Fixture B: damaged commercial structure adjacent to an active road closure.
FIXTURE_B = {
    "footprint_id": "fp-b",
    "label": "Fixture B",
    "centroid": [0.0, 0.0],
    "damage_class": 1,
    "confidence": 0.6,
    "confirmed": False,
    "priority": 1.0,
    "vl_caption": "Damaged commercial structure adjacent to an active road closure.",
    "facility_near": None,
}

# Fixture C: visible flames and heavy roof damage.
FIXTURE_C = {
    "footprint_id": "fp-c",
    "label": "Fixture C",
    "centroid": [0.0, 0.0],
    "damage_class": 3,
    "confidence": 0.9,
    "confirmed": False,
    "priority": 1.0,
    "vl_caption": "Two-storey structure with visible flames and heavy roof damage.",
    "facility_near": None,
}

# Fixture D: roof collapsed with structural debris around the entrance.
FIXTURE_D = {
    "footprint_id": "fp-d",
    "label": "Fixture D",
    "centroid": [0.0, 0.0],
    "damage_class": 3,
    "confidence": 0.85,
    "confirmed": False,
    "priority": 1.0,
    "vl_caption": "Roof collapsed with structural debris around the entrance.",
    "facility_near": None,
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


def _install_fake_urlopen(monkeypatch, content_obj):
    import urllib.request

    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        body = {"choices": [{"message": {"content": json.dumps(content_obj)}}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _sent_payload(captured_request) -> dict:
    return json.loads(captured_request.data)


# --------------------------------------------------------------------------
# Section 8 fixtures -- exact assertions
# --------------------------------------------------------------------------


def test_fixture_a_dialysis_damage_obstructed_entrance():
    assert is_action_supported("fire_suppression", FIXTURE_A) is False
    assert is_action_supported("medical_support", FIXTURE_A) is True
    assert is_action_supported("debris_clearance", FIXTURE_A) is True


def test_fixture_b_damaged_commercial_road_closure():
    assert is_action_supported("fire_suppression", FIXTURE_B) is False
    assert is_action_supported("collapse_response", FIXTURE_B) is False
    assert is_action_supported("road_closure", FIXTURE_B) is True


def test_fixture_c_visible_flames():
    assert is_action_supported("fire_suppression", FIXTURE_C) is True


def test_fixture_d_roof_collapsed_structural_debris():
    assert is_action_supported("collapse_response", FIXTURE_D) is True
    assert is_action_supported("debris_clearance", FIXTURE_D) is True


# 6, 7: dialysis / hospital support medical_support
def test_dialysis_supports_medical_support():
    candidate = dict(FIXTURE_A, vl_caption="No specific damage described.")
    assert is_action_supported("medical_support", candidate) is True


def test_hospital_facility_type_supports_medical_support():
    candidate = {**FIXTURE_A, "facility_near": {"name": "General Hospital", "type": "hospital", "dist_m": 5}}
    assert is_action_supported("medical_support", candidate) is True


def test_nursing_home_facility_type_supports_medical_support():
    candidate = {**FIXTURE_A, "facility_near": {"name": "Sunset Nursing Home", "type": "nursing_home", "dist_m": 5}}
    assert is_action_supported("medical_support", candidate) is True


def test_no_facility_and_no_medical_caption_does_not_support_medical_support():
    candidate = dict(FIXTURE_C, facility_near=None)  # flames caption, no facility
    assert is_action_supported("medical_support", candidate) is False


# damage_class alone is never evidence for any action
def test_damage_class_alone_does_not_support_any_action():
    bland = {
        "footprint_id": "fp-bland",
        "label": "Bland Building",
        "centroid": [0.0, 0.0],
        "damage_class": 3,
        "confidence": 0.9,
        "confirmed": False,
        "priority": 1.0,
        "vl_caption": "Structure has extensive damage.",
        "facility_near": None,
    }
    for action in ("fire_suppression", "collapse_response", "road_closure", "perimeter_control", "debris_clearance"):
        assert is_action_supported(action, bland) is False


# --------------------------------------------------------------------------
# Real client: request/schema/prompt content
# --------------------------------------------------------------------------


def test_request_contains_only_one_buildings_evidence(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch, {"assignments": []})
    RealAgencyPlanDraftClient(base_url="http://localhost:8000").propose_assignments_for_building(CANDIDATE_FIRE)

    prompt = _sent_payload(captured[0])["messages"][-1]["content"]
    assert "fp-100" in prompt
    assert "412 Elm St" in prompt
    assert CANDIDATE_FIRE["vl_caption"] in prompt
    assert "fp-200" not in prompt
    assert "Riverside Dialysis Center" not in prompt


def test_property_value_owner_and_units_available_not_included(monkeypatch):
    for candidate in (CANDIDATE_FIRE, CANDIDATE_EMS):
        assert "owner" not in candidate
        assert "property_value" not in candidate
        assert "units_available" not in candidate

    captured = _install_fake_urlopen(monkeypatch, {"assignments": []})
    RealAgencyPlanDraftClient(base_url="http://localhost:8000").propose_assignments_for_building(CANDIDATE_FIRE)

    prompt = _sent_payload(captured[0])["messages"][-1]["content"]
    assert "$" not in prompt
    assert "units_available" not in prompt.lower()


# 1, 2: Nano no longer returns free-form task; only allowed action enum accepted
def test_output_schema_requires_only_agency_action_units(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch, {"assignments": []})
    RealAgencyPlanDraftClient(base_url="http://localhost:8000").propose_assignments_for_building(CANDIDATE_FIRE)

    payload = _sent_payload(captured[0])
    assert payload["model"] == "nano"
    assert payload["messages"][0] == {"role": "system", "content": "/no_think"}
    schema = payload["response_format"]["json_schema"]["schema"]
    item_schema = schema["properties"]["assignments"]["items"]
    assert set(item_schema["required"]) == {"agency", "action", "units"}
    assert set(item_schema["properties"].keys()) == {"agency", "action", "units"}
    assert "task" not in item_schema["properties"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["agency"]["enum"] == ["fire", "ems", "police", "public_works"]
    assert set(item_schema["properties"]["action"]["enum"]) == {
        "fire_suppression",
        "collapse_response",
        "medical_support",
        "perimeter_control",
        "road_closure",
        "debris_clearance",
    }


# 16: max_tokens=80 is sent
def test_max_tokens_80_is_sent(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch, {"assignments": []})
    RealAgencyPlanDraftClient(base_url="http://localhost:8000").propose_assignments_for_building(CANDIDATE_FIRE)

    payload = _sent_payload(captured[0])
    assert payload["max_tokens"] == 80


def test_finish_reason_length_is_treated_as_failure(monkeypatch):
    import urllib.request

    def fake_urlopen(request, timeout=None):
        body = {"choices": [{"message": {"content": '{"assignments": [{"agency": "fire"'}, "finish_reason": "length"}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(NanoClientError):
        RealAgencyPlanDraftClient(base_url="http://localhost:8000").propose_assignments_for_building(CANDIDATE_FIRE)


def test_valid_response_parses_correctly(monkeypatch):
    _install_fake_urlopen(
        monkeypatch,
        {"assignments": [{"agency": "fire", "action": "fire_suppression", "units": 2}]},
    )
    result = RealAgencyPlanDraftClient(base_url="http://localhost:8000").propose_assignments_for_building(
        CANDIDATE_FIRE
    )
    assert result["assignments"][0] == {"agency": "fire", "action": "fire_suppression", "units": 2}


def test_usage_log_captures_token_usage_when_present(monkeypatch):
    import urllib.request

    def fake_urlopen(request, timeout=None):
        body = {
            "choices": [{"message": {"content": '{"assignments": []}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 7, "total_tokens": 130},
        }
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = RealAgencyPlanDraftClient(base_url="http://localhost:8000")
    client.propose_assignments_for_building(CANDIDATE_FIRE)

    assert client.usage_log == [{"prompt_tokens": 123, "completion_tokens": 7, "total_tokens": 130}]


def test_no_real_network_access_for_stub(monkeypatch):
    import socket
    import urllib.request

    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    result = StubAgencyPlanDraftClient().propose_assignments_for_building(CANDIDATE_FIRE)
    assert isinstance(result["assignments"], list)


# --------------------------------------------------------------------------
# Deterministic fallback: same evidence-support rules, action-based output
# --------------------------------------------------------------------------


def test_deterministic_fallback_evidence_based_and_no_invented_content():
    fire_result = StubAgencyPlanDraftClient().propose_assignments_for_building(FIXTURE_C)
    ems_result = StubAgencyPlanDraftClient().propose_assignments_for_building(FIXTURE_A)

    assert any(a["agency"] == "fire" and a["action"] == "fire_suppression" for a in fire_result["assignments"])
    assert any(a["agency"] == "ems" and a["action"] == "medical_support" for a in ems_result["assignments"])

    all_text = json.dumps(fire_result["assignments"] + ems_result["assignments"]).lower()
    for forbidden in ("casualt", "trapped", "occupant", "resident", "injured", "property value"):
        assert forbidden not in all_text


def test_deterministic_fallback_assigns_nothing_when_no_evidence_supports_it():
    bland_candidate = {
        "footprint_id": "fp-999",
        "label": "Some Building",
        "centroid": [0.0, 0.0],
        "damage_class": 0,
        "confidence": 0.5,
        "confirmed": False,
        "priority": 0.1,
        "vl_caption": "Building appears intact with no visible damage.",
        "facility_near": None,
    }
    result = StubAgencyPlanDraftClient().propose_assignments_for_building(bland_candidate)
    assert result["assignments"] == []


def test_deterministic_fallback_debris_and_road_closure_rules():
    debris_result = StubAgencyPlanDraftClient().propose_assignments_for_building(FIXTURE_A)
    closure_result = StubAgencyPlanDraftClient().propose_assignments_for_building(FIXTURE_B)

    assert any(a["agency"] == "public_works" and a["action"] == "debris_clearance" for a in debris_result["assignments"])
    assert any(a["agency"] == "police" and a["action"] == "road_closure" for a in closure_result["assignments"])


def test_deterministic_fallback_collapse_response_rule():
    result = StubAgencyPlanDraftClient().propose_assignments_for_building(FIXTURE_D)
    assert any(a["agency"] == "fire" and a["action"] == "collapse_response" for a in result["assignments"])
    assert any(a["agency"] == "public_works" and a["action"] == "debris_clearance" for a in result["assignments"])


def test_deterministic_fallback_never_pairs_invalid_agency_action():
    from backend.decision.agency_plan_client import _VALID_AGENCY_ACTIONS

    for fixture in (FIXTURE_A, FIXTURE_B, FIXTURE_C, FIXTURE_D):
        result = StubAgencyPlanDraftClient().propose_assignments_for_building(fixture)
        for assignment in result["assignments"]:
            assert assignment["action"] in _VALID_AGENCY_ACTIONS[assignment["agency"]]
