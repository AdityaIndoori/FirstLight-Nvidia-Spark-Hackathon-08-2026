import json
import socket
import urllib.error
import urllib.request

import pytest

from backend.decision.nano_client import (
    NanoClientError,
    RealNanoRationaleClient,
    StubNanoRationaleClient,
)
from backend.decision.rationale import generate_rationale, generate_rationale_with_recovery

RANK_ITEM = {
    "footprint_id": "fp-001",
    "label": "412 Elm St",
    "centroid": [-122.4194, 37.7749],
    "damage_class": 3,
    "confidence": 0.91,
    "confirmed": True,
    "graded_by": "operator:jsmith",
    "facility_near": {"name": "Riverside Clinic", "type": "clinic", "dist_m": 180},
    "inputs": {
        "staleness_h": 6.5,
        "vulnerable_density": 2.3,
        "doubt": 0.12,
        "road_cutoff": 1.8,
    },
    "priority": 24.19320,
    "rationale": "",
    "rationale_by": "nano",
}


class FakeHTTPResponse:
    """Minimal stand-in for the object urllib.request.urlopen returns."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _install_fake_urlopen(monkeypatch, response_body: dict = None, raise_exc: Exception = None):
    """Patch urllib.request.urlopen and capture the Request it was called with."""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        if raise_exc is not None:
            raise raise_exc
        return FakeHTTPResponse(json.dumps(response_body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _success_body(content="A concise operator rationale."):
    return {"choices": [{"message": {"content": content}}]}


# 1: real client targets /v1/chat/completions
def test_real_client_targets_chat_completions_endpoint(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch, response_body=_success_body())

    client = RealNanoRationaleClient(base_url="http://localhost:8000")
    client.generate_rationale(RANK_ITEM)

    assert captured["request"].full_url == "http://localhost:8000/v1/chat/completions"
    assert captured["request"].get_method() == "POST"


# 2: model name is "nano"
def test_request_uses_served_model_name_nano(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch, response_body=_success_body())

    RealNanoRationaleClient(base_url="http://localhost:8000").generate_rationale(RANK_ITEM)

    payload = json.loads(captured["request"].data)
    assert payload["model"] == "nano"


# 3: /no_think is included for the hero-rationale request
def test_no_think_directive_sent_for_hero_rationale(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch, response_body=_success_body())

    RealNanoRationaleClient(base_url="http://localhost:8000").generate_rationale(RANK_ITEM)

    payload = json.loads(captured["request"].data)
    assert payload["messages"][0] == {"role": "system", "content": "/no_think"}


# 4: response content is parsed correctly
def test_response_content_is_parsed_correctly(monkeypatch):
    _install_fake_urlopen(monkeypatch, response_body=_success_body("Building destroyed, high priority."))

    result = RealNanoRationaleClient(base_url="http://localhost:8000").generate_rationale(RANK_ITEM)

    assert result == "Building destroyed, high priority."


# 5: timeout triggers the existing deterministic fallback
def test_timeout_triggers_deterministic_fallback(monkeypatch):
    _install_fake_urlopen(monkeypatch, raise_exc=socket.timeout("timed out"))

    with pytest.raises(NanoClientError):
        RealNanoRationaleClient(base_url="http://localhost:8000").generate_rationale(RANK_ITEM)

    result = generate_rationale_with_recovery(
        RANK_ITEM, real_client=RealNanoRationaleClient(base_url="http://localhost:8000")
    )

    assert result["recovery"] == "stub"
    assert result["rationale"] == StubNanoRationaleClient().generate_rationale(RANK_ITEM)


# 6: connection failure triggers fallback
def test_connection_failure_triggers_fallback(monkeypatch):
    _install_fake_urlopen(monkeypatch, raise_exc=urllib.error.URLError(OSError("connection refused")))

    result = generate_rationale_with_recovery(
        RANK_ITEM, real_client=RealNanoRationaleClient(base_url="http://localhost:8000")
    )

    assert result["recovery"] == "stub"
    assert result["rationale"] == StubNanoRationaleClient().generate_rationale(RANK_ITEM)


# 7: malformed model response triggers clean failure/fallback
def test_malformed_response_triggers_clean_failure_and_fallback(monkeypatch):
    _install_fake_urlopen(monkeypatch, response_body={"unexpected": "shape"})

    with pytest.raises(NanoClientError):
        RealNanoRationaleClient(base_url="http://localhost:8000").generate_rationale(RANK_ITEM)

    result = generate_rationale_with_recovery(
        RANK_ITEM, real_client=RealNanoRationaleClient(base_url="http://localhost:8000")
    )
    assert result["recovery"] == "stub"


# 8: application code still uses the same Nano abstraction (generate_rationale
# works unchanged when handed the real client, exactly as it does with the stub)
def test_application_entry_point_is_agnostic_to_real_vs_stub_client(monkeypatch):
    _install_fake_urlopen(monkeypatch, response_body=_success_body("Real model rationale."))

    real_result = generate_rationale(RANK_ITEM, client=RealNanoRationaleClient(base_url="http://localhost:8000"))
    stub_result = generate_rationale(RANK_ITEM, client=StubNanoRationaleClient())

    assert real_result == "Real model rationale."
    assert stub_result != real_result  # different backends, both reached through the same call shape


# base_url resolution: explicit argument wins over environment/default
def test_base_url_can_be_configured_via_environment(monkeypatch):
    monkeypatch.setenv("FIRSTLIGHT_NANO_BASE_URL", "http://localhost:9999")
    captured = _install_fake_urlopen(monkeypatch, response_body=_success_body())

    RealNanoRationaleClient().generate_rationale(RANK_ITEM)

    assert captured["request"].full_url == "http://localhost:9999/v1/chat/completions"
