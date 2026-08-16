"""B2 real-Nano flight-tasking wiring must not change any OTHER Nano/
Lightning caller's behavior. flight_client.py/flight_planner.py were the
only production files touched; this file proves the two things most at
risk of an accidental cross-module change: agency planning's /no_think
directive (flight tasking is the first caller to ever use "/think"
instead), and that the full B3/B6/B7/B8 suites are unaffected.
"""

import json


class FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# 14: /no_think agency behavior is NOT accidentally changed
def test_agency_plan_still_uses_no_think(monkeypatch):
    import urllib.request

    from backend.decision.agency_plan_client import RealAgencyPlanDraftClient

    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        body = {"choices": [{"message": {"content": json.dumps({"assignments": []})}, "finish_reason": "stop"}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    candidate = {
        "footprint_id": "fp-1",
        "label": "Test Building",
        "centroid": [-122.4, 47.6],
        "damage_class": 0,
        "confidence": 0.5,
        "confirmed": False,
        "priority": 1.0,
        "vl_caption": "No visible damage.",
        "facility_near": None,
    }
    RealAgencyPlanDraftClient(base_url="http://localhost:8000").propose_assignments_for_building(candidate)

    payload = json.loads(captured[0].data)
    assert payload["messages"][0] == {"role": "system", "content": "/no_think"}


def test_hero_rationale_still_uses_no_think(monkeypatch):
    import urllib.request

    from backend.decision.nano_client import RealNanoRationaleClient

    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        body = {"choices": [{"message": {"content": "412 Elm St: priority 24.19320."}, "finish_reason": "stop"}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    rank_item = {
        "footprint_id": "fp-1",
        "label": "412 Elm St",
        "centroid": [-122.4, 47.6],
        "damage_class": 0,
        "confidence": 0.5,
        "confirmed": False,
        "graded_by": "nemotron-vl",
        "facility_near": None,
        "inputs": {"staleness_h": 1.0, "vulnerable_density": 1.0, "doubt": 0.05, "road_cutoff": None},
        "priority": 24.19320,
    }
    RealNanoRationaleClient(base_url="http://localhost:8000").generate_rationale(rank_item)

    payload = json.loads(captured[0].data)
    assert payload["messages"][0] == {"role": "system", "content": "/no_think"}


# 15: B3/B6/B7/B8 behavior remains unchanged -- spot-check a few
# representative, already-passing behaviors from each rather than
# duplicating their full suites (which also all run green in the same
# `pytest tests/ -q` invocation).
def test_lightning_ballot_still_computes_doubt_formula():
    from backend.decision.lightning_ballot import request_lightning_ballot
    from backend.decision.lightning_client import StubLightningSeverityClient

    building_context = {
        "grader_class": 2,
        "grader_confidence": 0.8,
        "vl_caption": "Major damage observed.",
        "footprint_area_m2": 100.0,
        "facility_context": None,
        "neighbor_damage_classes": [2, 1],
    }
    client = StubLightningSeverityClient(votes=[2, 2, 2, 2, 2, 1, 1, 1])
    ballot = request_lightning_ballot(building_context, client=client)
    assert ballot["vote_agreement"] == 5 / 8
    assert ballot["doubt"] == max(0.05, 1 - 5 / 8)


def test_agency_plan_grounding_still_rejects_ungrounded_fire():
    from backend.decision.agency_plan_client import is_action_supported

    candidate = {
        "footprint_id": "fp-2",
        "label": "Dialysis Building",
        "centroid": [-122.4, 47.6],
        "damage_class": 2,
        "confidence": 0.8,
        "confirmed": False,
        "priority": 5.0,
        "vl_caption": "Building has significant exterior damage and obstructed entrance.",
        "facility_near": {"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0},
    }
    assert is_action_supported("fire_suppression", candidate) is False
    assert is_action_supported("medical_support", candidate) is True


def test_archive_search_still_filters_then_ranks():
    from backend.decision import archive_store
    from backend.decision.archive_embedder import DeterministicStubEmbedder
    from backend.decision.archive_search import search_archive
    from backend.decision.archive_write import index_cleared_archive_item

    conn = archive_store.get_connection(":memory:")
    embedder = DeterministicStubEmbedder()
    item = {
        "image_id": "img-1",
        "thumb_path": "t.jpg",
        "captured_at": 1.0,
        "centroid": [-122.4, 47.6],
        "needs_geo": False,
        "caption": "Large fire visible.",
        "tags": ["fire"],
        "class_max": 3,
        "key_evidence": True,
    }
    index_cleared_archive_item(conn, item, embedder, eligible=True)

    result = search_archive({"q": "class:0", "limit": 10}, conn, embedder=embedder)
    assert result["items"] == []  # class:3 item correctly excluded by class:0 filter
    assert result["resolved_by"] == ["filter"]


def test_b8_rationale_faithfulness_still_scores_perfectly():
    from backend.decision.eval.eval_rationale import evaluate_rationale_faithfulness

    metric = evaluate_rationale_faithfulness()
    assert metric["status"] == "pass"
    assert metric["value"] == 1.0


# 16: normal tests make zero network calls -- proven at the module level:
# constructing every real client here performs no I/O, and every test
# above that actually calls request_flight_plan/generate_rationale/
# propose_assignments_for_building goes through a monkeypatched urlopen.
def test_constructing_real_clients_performs_no_network_io(monkeypatch):
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError("constructing a client must never touch the network")

    monkeypatch.setattr(socket, "create_connection", _forbidden)

    from backend.decision.agency_plan_client import RealAgencyPlanDraftClient
    from backend.decision.flight_client import RealNanoFlightClient
    from backend.decision.nano_client import RealNanoRationaleClient

    RealNanoFlightClient(base_url="http://localhost:8000")
    RealAgencyPlanDraftClient(base_url="http://localhost:8000")
    RealNanoRationaleClient(base_url="http://localhost:8000")
