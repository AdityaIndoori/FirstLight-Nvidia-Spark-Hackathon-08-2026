import copy
import json

import pytest

from backend.decision import lightning_n8_experiment
from backend.decision.lightning_client import LightningClientError, RealLightningSeverityClient
from backend.decision.lightning_n8_experiment import _post_n8_votes, request_n8_ballot

BUILDING_CONTEXT = {
    "grader_class": 2,
    "grader_confidence": 0.85,
    "vl_caption": "Two-storey structure with partial roof collapse and debris along the eastern wall.",
    "footprint_area_m2": 120.0,
    "facility_context": "150 m from Riverside Clinic",
    "neighbor_damage_classes": [1, 2, 2],
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


def _choices(labels):
    return [{"message": {"content": label}} for label in labels]


def _install_fake_urlopen(monkeypatch, raw_body: dict = None, raw_body_bytes: bytes = None):
    import urllib.request

    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        if raw_body_bytes is not None:
            return FakeHTTPResponse(raw_body_bytes)
        return FakeHTTPResponse(json.dumps(raw_body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _default_body(labels=("2", "2", "2", "2", "2", "2", "3", "3")):
    return {"choices": _choices(labels)}


# 1: experimental request contains n=8
def test_request_contains_n_equals_eight(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch, raw_body=_default_body())
    request_n8_ballot(BUILDING_CONTEXT, base_url="http://localhost:8001")

    payload = json.loads(captured[0].data)
    assert payload["n"] == 8
    assert payload["model"] == "lightning"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["structured_outputs"] == {"choice": ["0", "1", "2", "3"]}
    assert payload["temperature"] == 0.7
    # one request, one message list -- not eight separate requests
    assert isinstance(payload["messages"], list) and len(payload["messages"]) == 1


# 2: exactly eight response choices are parsed
def test_exactly_eight_choices_are_parsed(monkeypatch):
    _install_fake_urlopen(monkeypatch, raw_body=_default_body(("0", "1", "2", "3", "0", "1", "2", "3")))
    votes, _usage = _post_n8_votes(BUILDING_CONTEXT, temperature=0.7, base_url="http://localhost:8001", timeout_s=5)

    assert len(votes) == 8
    assert votes == [0, 1, 2, 3, 0, 1, 2, 3]


# 3: every choice must be 0..3
def test_out_of_range_choice_is_rejected(monkeypatch):
    _install_fake_urlopen(monkeypatch, raw_body=_default_body(("2", "2", "2", "2", "2", "2", "2", "9")))
    with pytest.raises(LightningClientError):
        request_n8_ballot(BUILDING_CONTEXT, base_url="http://localhost:8001")


# 4: missing choices are rejected
def test_missing_choices_are_rejected(monkeypatch):
    _install_fake_urlopen(monkeypatch, raw_body=_default_body(("2", "2", "2")))  # only 3, not 8
    with pytest.raises(LightningClientError):
        request_n8_ballot(BUILDING_CONTEXT, base_url="http://localhost:8001")


# 5: extra/malformed choices are rejected appropriately
def test_extra_choices_are_rejected(monkeypatch):
    _install_fake_urlopen(monkeypatch, raw_body=_default_body(("2",) * 9))  # 9, not 8
    with pytest.raises(LightningClientError):
        request_n8_ballot(BUILDING_CONTEXT, base_url="http://localhost:8001")


def test_malformed_choice_structure_is_rejected(monkeypatch):
    _install_fake_urlopen(monkeypatch, raw_body={"choices": [{"unexpected": "shape"}] * 8})
    with pytest.raises(LightningClientError):
        request_n8_ballot(BUILDING_CONTEXT, base_url="http://localhost:8001")


def test_non_json_response_is_rejected(monkeypatch):
    _install_fake_urlopen(monkeypatch, raw_body_bytes=b"not json at all")
    with pytest.raises(LightningClientError):
        request_n8_ballot(BUILDING_CONTEXT, base_url="http://localhost:8001")


# 6: aggregation reuses existing ballot logic, not a reimplementation
def test_aggregation_reuses_existing_ballot_logic(monkeypatch):
    calls = []
    original = lightning_n8_experiment._aggregate_votes

    def spy(votes):
        calls.append(list(votes))
        return original(votes)

    monkeypatch.setattr(lightning_n8_experiment, "_aggregate_votes", spy)
    _install_fake_urlopen(monkeypatch, raw_body=_default_body(("2", "2", "2", "2", "2", "2", "3", "3")))

    result = request_n8_ballot(BUILDING_CONTEXT, base_url="http://localhost:8001")

    assert len(calls) == 1
    assert calls[0] == [2, 2, 2, 2, 2, 2, 3, 3]
    assert result["voted_class"] == 2
    assert result["vote_agreement"] == 0.75
    assert result["doubt"] == 0.25


# 7: building_context (BuildingEvidence-derived ballot input) is not mutated
def test_building_context_not_mutated(monkeypatch):
    _install_fake_urlopen(monkeypatch, raw_body=_default_body())
    snapshot = copy.deepcopy(BUILDING_CONTEXT)

    request_n8_ballot(BUILDING_CONTEXT, base_url="http://localhost:8001")

    assert BUILDING_CONTEXT == snapshot


# 8: existing production ballot remains unchanged -- the production
# single-vote request never gains an "n" parameter and behaves exactly as
# before, independent of this experimental module existing.
def test_production_single_vote_request_has_no_n_parameter(monkeypatch):
    import urllib.request

    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        body = {"choices": [{"message": {"content": "2"}}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    payload = json.loads(captured[0].data)
    assert "n" not in payload
    assert result == 2
