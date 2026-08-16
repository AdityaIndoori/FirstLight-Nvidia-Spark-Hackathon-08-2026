import inspect
import json

import pytest

from backend.decision.archive_tag_extractor import (
    DeterministicStubTagExtractor,
    LightningTagExtractor,
    _MAX_BATCH_SIZE,
    _normalize_and_validate_tags,
)
from backend.decision.lightning_client import LightningClientError


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
    """content_by_call: list of raw assistant-content strings, one per HTTP
    call in order (so tests can simulate chunked batches)."""
    import urllib.request

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        content = content_by_call[len(calls) - 1]
        body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _sent_payload(request) -> dict:
    return json.loads(request.data)


# 31, 32, 33: shape, lowercase, dedup
def test_stub_extractor_shape_matches_input_length():
    captions = [
        "Two-storey wood structure with roof collapsed and standing water in street.",
        "Undamaged single-family home.",
    ]
    result = DeterministicStubTagExtractor().extract_tags_batch(captions)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(tags, list) for tags in result)


def test_stub_extractor_tags_are_lowercase():
    result = DeterministicStubTagExtractor().extract_tags_batch(["Large FIRE with heavy smoke."])
    for tag in result[0]:
        assert tag == tag.lower()


def test_normalize_dedupes_case_and_whitespace_variants():
    tags = _normalize_and_validate_tags(["Fire", "fire", "  fire  ", "FIRE"], "a large fire visible")
    assert tags == ["fire"]


# 34: prohibited human-related tags rejected
def test_prohibited_person_tags_are_dropped():
    tags = _normalize_and_validate_tags(
        ["person visible", "roof collapse", "victim present"], "roof collapse near the building"
    )
    assert "person visible" not in tags
    assert "victim present" not in tags
    assert "roof collapse" in tags


def test_prohibited_clothing_and_body_tags_are_dropped():
    tags = _normalize_and_validate_tags(["blue shirt", "broken hand", "fire"], "a large fire visible")
    assert tags == ["fire"]


# tag must be supported by caption -- ungrounded tags dropped, not invented
def test_ungrounded_tag_is_dropped():
    tags = _normalize_and_validate_tags(["helicopter present"], "roof has collapsed")
    assert tags == []


def test_grounded_tag_with_tense_variation_is_kept():
    tags = _normalize_and_validate_tags(["roof collapse"], "the roof has collapsed onto the structure")
    assert "roof collapse" in tags


def test_stub_extractor_never_produces_prohibited_tags():
    caption = "A person is visible near a fire with heavy smoke."
    tags = DeterministicStubTagExtractor().extract_tags_batch([caption])[0]
    for tag in tags:
        assert "person" not in tag


# 35: tag extraction never changes damage class -- interface has no such
# parameter at all, and stub output is a pure function of caption text.
def test_extract_tags_batch_signature_has_no_damage_class_parameter():
    signature = inspect.signature(DeterministicStubTagExtractor().extract_tags_batch)
    assert list(signature.parameters.keys()) == ["captions"]


def test_stub_extractor_output_independent_of_call_order_or_external_state():
    caption = "Roof has fully collapsed onto the structure below."
    first = DeterministicStubTagExtractor().extract_tags_batch([caption])
    second = DeterministicStubTagExtractor().extract_tags_batch(["irrelevant caption", caption])
    assert first[0] == second[1]


# 36: normal tests make zero network calls (stub path)
def test_stub_extractor_never_touches_network(monkeypatch):
    import socket
    import urllib.request

    def _forbidden(*args, **kwargs):
        raise AssertionError("stub tag extractor must never touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    result = DeterministicStubTagExtractor().extract_tags_batch(["A large fire with heavy smoke."])
    assert isinstance(result, list)


# --------------------------------------------------------------------------
# LightningTagExtractor -- real HTTP path, always faked via monkeypatch
# --------------------------------------------------------------------------


def test_lightning_extractor_sends_no_think_json_schema_and_zero_temperature(monkeypatch):
    captions = ["A large fire with heavy smoke.", "Roof has fully collapsed."]
    calls = _install_fake_urlopen(
        monkeypatch, [json.dumps({"results": [["fire"], ["roof collapse"]]})]
    )

    result = LightningTagExtractor(base_url="http://localhost:8001").extract_tags_batch(captions)

    assert len(calls) == 1
    payload = _sent_payload(calls[0])
    assert payload["model"] == "lightning"
    assert payload["messages"][0] == {"role": "system", "content": "/no_think"}
    assert payload["temperature"] == 0.0
    assert payload["response_format"]["type"] == "json_schema"
    assert result == [["fire"], ["roof collapse"]]


def test_lightning_extractor_length_mismatch_raises(monkeypatch):
    _install_fake_urlopen(monkeypatch, [json.dumps({"results": [["fire"]]})])  # only 1, for 2 captions

    with pytest.raises(LightningClientError):
        LightningTagExtractor(base_url="http://localhost:8001").extract_tags_batch(
            ["caption one with fire", "caption two with collapse"]
        )


def test_lightning_extractor_malformed_json_raises(monkeypatch):
    _install_fake_urlopen(monkeypatch, ["not valid json"])

    with pytest.raises(LightningClientError):
        LightningTagExtractor(base_url="http://localhost:8001").extract_tags_batch(["a caption"])


def test_lightning_extractor_truncated_output_raises(monkeypatch):
    import urllib.request

    def fake_urlopen(request, timeout=None):
        body = {"choices": [{"message": {"content": '{"results": [['}, "finish_reason": "length"}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(LightningClientError):
        LightningTagExtractor(base_url="http://localhost:8001").extract_tags_batch(["a caption"])


def test_lightning_extractor_drops_prohibited_and_ungrounded_tags(monkeypatch):
    caption = "A large fire with heavy smoke near the structure."
    _install_fake_urlopen(
        monkeypatch,
        [json.dumps({"results": [["fire", "person visible", "helicopter present"]]})],
    )

    result = LightningTagExtractor(base_url="http://localhost:8001").extract_tags_batch([caption])
    assert result == [["fire"]]


def test_lightning_extractor_chunks_batches_over_max_batch_size(monkeypatch):
    captions = [f"caption number {i} with fire" for i in range(_MAX_BATCH_SIZE + 5)]
    responses = [
        json.dumps({"results": [["fire"]] * _MAX_BATCH_SIZE}),
        json.dumps({"results": [["fire"]] * 5}),
    ]
    calls = _install_fake_urlopen(monkeypatch, responses)

    result = LightningTagExtractor(base_url="http://localhost:8001").extract_tags_batch(captions)

    assert len(calls) == 2
    assert len(result) == len(captions)


def test_lightning_extractor_usage_log_captures_usage(monkeypatch):
    import urllib.request

    def fake_urlopen(request, timeout=None):
        body = {
            "choices": [{"message": {"content": json.dumps({"results": [["fire"]]})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
        }
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = LightningTagExtractor(base_url="http://localhost:8001")
    client.extract_tags_batch(["a fire caption"])
    assert client.usage_log == [{"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}]
