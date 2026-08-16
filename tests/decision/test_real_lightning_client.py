import copy
import json
import socket
import urllib.error
import urllib.request

import pytest

from backend.decision.lightning_ballot import K_VOTES, request_lightning_ballot
from backend.decision.lightning_client import (
    LightningClientError,
    LightningSeverityClient,
    RealLightningSeverityClient,
    StubLightningSeverityClient,
)

BUILDING_CONTEXT = {
    "grader_class": 2,
    "grader_confidence": 0.85,
    "vl_caption": "Two-storey structure with partial roof collapse and debris along the eastern wall.",
    "footprint_area_m2": 120.0,
    "facility_context": "150m from Riverside Clinic",
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


def _install_fake_urlopen(monkeypatch, contents=None, raise_exc=None, raw_body_bytes=None):
    """contents: fixed str, or list[str] cycled per call (default "2").
    raise_exc: exception instance raised on every call instead of responding.
    raw_body_bytes: bypasses the {"choices": [...]} wrapping entirely, to
    simulate a genuinely malformed HTTP body.
    """
    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append({"request": request, "timeout": timeout})
        if raise_exc is not None:
            raise raise_exc
        if raw_body_bytes is not None:
            return FakeHTTPResponse(raw_body_bytes)
        if isinstance(contents, list):
            content = contents[(len(captured) - 1) % len(contents)]
        else:
            content = contents if contents is not None else "2"
        body = {"choices": [{"message": {"content": content}}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


# 1: request targets /v1/chat/completions
def test_real_client_targets_chat_completions_endpoint(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)

    client = RealLightningSeverityClient(base_url="http://localhost:8001")
    client.sample_severity(BUILDING_CONTEXT)

    assert captured[0]["request"].full_url == "http://localhost:8001/v1/chat/completions"
    assert captured[0]["request"].get_method() == "POST"


# 2: model is "lightning"
def test_request_uses_served_model_name_lightning(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)

    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    payload = json.loads(captured[0]["request"].data)
    assert payload["model"] == "lightning"


# 3: enable_thinking=false is sent
def test_enable_thinking_false_is_sent(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)

    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    payload = json.loads(captured[0]["request"].data)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


# 4: structured_outputs.choice contains exactly ["0","1","2","3"]
def test_structured_outputs_choice_is_exactly_the_four_labels(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)

    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    payload = json.loads(captured[0]["request"].data)
    assert payload["structured_outputs"] == {"choice": ["0", "1", "2", "3"]}


# 5: temperature=0.7 is sent
def test_temperature_default_is_sent_as_zero_point_seven(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)

    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    payload = json.loads(captured[0]["request"].data)
    assert payload["temperature"] == 0.7


# 6: valid "2" response parses to integer 2
def test_valid_response_parses_to_integer(monkeypatch):
    _install_fake_urlopen(monkeypatch, contents="2")

    result = RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    assert result == 2
    assert isinstance(result, int)


# 7: label outside 0-3 is rejected
def test_label_outside_range_is_rejected(monkeypatch):
    _install_fake_urlopen(monkeypatch, contents="4")

    with pytest.raises(LightningClientError):
        RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)


# 8: malformed response is rejected
def test_malformed_response_is_rejected(monkeypatch):
    _install_fake_urlopen(monkeypatch, raw_body_bytes=b"not json at all")
    with pytest.raises(LightningClientError):
        RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    _install_fake_urlopen(monkeypatch, raw_body_bytes=json.dumps({"unexpected": "shape"}).encode())
    with pytest.raises(LightningClientError):
        RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)


# 9: timeout is handled cleanly
def test_timeout_is_handled_cleanly(monkeypatch):
    _install_fake_urlopen(monkeypatch, raise_exc=socket.timeout("timed out"))

    with pytest.raises(LightningClientError):
        RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)


# 10: connection failure is handled cleanly
def test_connection_failure_is_handled_cleanly(monkeypatch):
    _install_fake_urlopen(monkeypatch, raise_exc=urllib.error.URLError(OSError("connection refused")))

    with pytest.raises(LightningClientError):
        RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)


# 11: real client fits the existing Lightning abstraction -- request_lightning_ballot
# (unchanged) works identically regardless of stub vs real client
def test_real_client_fits_existing_ballot_abstraction(monkeypatch):
    _install_fake_urlopen(monkeypatch, contents="2")

    assert isinstance(RealLightningSeverityClient(), LightningSeverityClient)

    real_result = request_lightning_ballot(
        BUILDING_CONTEXT, client=RealLightningSeverityClient(base_url="http://localhost:8001")
    )
    stub_result = request_lightning_ballot(
        BUILDING_CONTEXT, client=StubLightningSeverityClient(votes=[2] * 8)
    )

    assert real_result["voted_class"] == stub_result["voted_class"] == 2
    assert real_result["vote_agreement"] == stub_result["vote_agreement"] == 1.0
    assert real_result["doubt"] == stub_result["doubt"] == 0.05


# 12: existing stub tests still pass -- proven by running the full suite; a
# direct sanity check here that the stub is untouched:
def test_stub_still_works_and_is_still_networkless(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)

    result = request_lightning_ballot(BUILDING_CONTEXT, client=StubLightningSeverityClient(votes=[1] * 8))
    assert result["voted_class"] == 1


# 13: the ballot calls the (real) client exactly 8 times
def test_ballot_calls_real_client_exactly_eight_times(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch, contents="2")

    request_lightning_ballot(BUILDING_CONTEXT, client=RealLightningSeverityClient(base_url="http://localhost:8001"))

    assert len(captured) == K_VOTES == 8


# 14: original building context remains unchanged
def test_building_context_not_mutated_by_real_client(monkeypatch):
    _install_fake_urlopen(monkeypatch, contents="2")
    snapshot = copy.deepcopy(BUILDING_CONTEXT)

    request_lightning_ballot(BUILDING_CONTEXT, client=RealLightningSeverityClient(base_url="http://localhost:8001"))

    assert BUILDING_CONTEXT == snapshot


# 15: original grader_class (VL model's primary grade) remains unchanged
# (and is never the ballot's output field)
def test_grader_class_untouched_by_real_ballot(monkeypatch):
    _install_fake_urlopen(monkeypatch, contents="3")
    context = copy.deepcopy(BUILDING_CONTEXT)
    context["grader_class"] = 2

    result = request_lightning_ballot(context, client=RealLightningSeverityClient(base_url="http://localhost:8001"))

    assert context["grader_class"] == 2  # untouched VL model grade
    assert result["voted_class"] == 3  # Lightning's own, separate output
    assert "grader_class" not in result
    assert "damage_class" not in result
    assert "confidence" not in result
    assert "graded_by" not in result


# base_url resolution: explicit argument wins; env var used when no argument given
def test_base_url_can_be_configured_via_environment(monkeypatch):
    monkeypatch.setenv("FIRSTLIGHT_LIGHTNING_BASE_URL", "http://localhost:9998")
    captured = _install_fake_urlopen(monkeypatch)

    RealLightningSeverityClient().sample_severity(BUILDING_CONTEXT)

    assert captured[0]["request"].full_url == "http://localhost:9998/v1/chat/completions"
