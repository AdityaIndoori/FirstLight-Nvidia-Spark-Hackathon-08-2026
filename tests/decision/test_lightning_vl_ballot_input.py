import copy
import json
import urllib.request

from backend.decision.lightning_ballot import K_VOTES, request_lightning_ballot
from backend.decision.lightning_client import RealLightningSeverityClient, StubLightningSeverityClient

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


def _install_fake_urlopen(monkeypatch, content="2"):
    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        body = {"choices": [{"message": {"content": content}}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _sent_prompt(captured_request) -> str:
    payload = json.loads(captured_request.data)
    return payload["messages"][0]["content"]


# 1: vl_caption is present in the real Lightning request prompt
def test_vl_caption_present_in_prompt(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)
    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    prompt = _sent_prompt(captured[0])
    assert BUILDING_CONTEXT["vl_caption"] in prompt


# 2: grader class is present
def test_grader_class_present_in_prompt(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)
    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    prompt = _sent_prompt(captured[0])
    assert f"class: {BUILDING_CONTEXT['grader_class']}" in prompt


# 3: grader confidence is present
def test_grader_confidence_present_in_prompt(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)
    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    prompt = _sent_prompt(captured[0])
    assert f"{BUILDING_CONTEXT['grader_confidence']:.2f}" in prompt


# 4: footprint/facility/neighbour context remains present
def test_gis_context_remains_present_in_prompt(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)
    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    prompt = _sent_prompt(captured[0])
    assert f"{BUILDING_CONTEXT['footprint_area_m2']:.1f}" in prompt
    assert BUILDING_CONTEXT["facility_context"] in prompt
    assert str(BUILDING_CONTEXT["neighbor_damage_classes"]) in prompt


# 5: caption text is not mutated
def test_caption_text_not_mutated(monkeypatch):
    _install_fake_urlopen(monkeypatch)
    snapshot = copy.deepcopy(BUILDING_CONTEXT)

    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    assert BUILDING_CONTEXT == snapshot
    assert BUILDING_CONTEXT["vl_caption"] == snapshot["vl_caption"]


# 6: existing k=8 behavior remains exactly eight independent requests
def test_k_equals_eight_independent_requests(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)

    request_lightning_ballot(BUILDING_CONTEXT, client=RealLightningSeverityClient(base_url="http://localhost:8001"))

    assert len(captured) == K_VOTES == 8
    for request in captured:
        payload = json.loads(request.data)
        # one user message per request -- eight separate generations, never
        # a single prompt asking for eight labels at once
        assert isinstance(payload["messages"], list) and len(payload["messages"]) == 1


# 7: existing structured decoding remains intact
def test_structured_decoding_remains_intact(monkeypatch):
    captured = _install_fake_urlopen(monkeypatch)
    RealLightningSeverityClient(base_url="http://localhost:8001").sample_severity(BUILDING_CONTEXT)

    payload = json.loads(captured[0].data)
    assert payload["model"] == "lightning"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["structured_outputs"] == {"choice": ["0", "1", "2", "3"]}
    assert payload["temperature"] == 0.7


# 8: normal pytest performs no real network calls
def test_no_real_network_calls_in_normal_pytest(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)

    result = request_lightning_ballot(BUILDING_CONTEXT, client=StubLightningSeverityClient(votes=[2] * 8))
    assert result["voted_class"] == 2


# 9: the original damage class (grader_class) is never overwritten by voted_class
def test_grader_class_never_overwritten_by_voted_class(monkeypatch):
    _install_fake_urlopen(monkeypatch, content="0")  # Lightning disagrees with the grader
    context = copy.deepcopy(BUILDING_CONTEXT)  # grader_class == 2

    result = request_lightning_ballot(context, client=RealLightningSeverityClient(base_url="http://localhost:8001"))

    assert context["grader_class"] == 2  # untouched, even though Lightning voted differently
    assert result["voted_class"] == 0
    assert "grader_class" not in result
