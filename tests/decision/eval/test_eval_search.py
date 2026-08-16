import socket
import urllib.request

from backend.decision.eval.eval_search import (
    SEARCH_QUERY_FIXTURE,
    _precision_recall_at_k,
    evaluate_search_recall_precision_live,
    evaluate_search_recall_precision_offline,
)


def test_exactly_twenty_queries():
    assert len(SEARCH_QUERY_FIXTURE) == 20


def test_every_query_has_known_relevant_ids():
    for case in SEARCH_QUERY_FIXTURE:
        assert isinstance(case["relevant_image_ids"], set)
        assert case["relevant_image_ids"]  # every query has at least one known-relevant id
        assert case["k"] >= 1


def test_resolver_mix_covers_required_composition():
    mixes = {case["resolver_mix"] for case in SEARCH_QUERY_FIXTURE}
    required = {"semantic", "filter", "location", "filter+semantic", "location+semantic", "location+filter+semantic"}
    assert required.issubset(mixes)


def test_offline_evaluation_returns_measured():
    metric = evaluate_search_recall_precision_offline()
    assert metric["status"] == "measured"
    assert metric["sample_count"] == 20
    assert 0.0 <= metric["value"]["mean_precision_at_k"] <= 1.0
    assert 0.0 <= metric["value"]["mean_recall_at_k"] <= 1.0


def test_offline_mode_labeled_as_stub_quality():
    metric = evaluate_search_recall_precision_offline()
    assert "stub" in metric["details"]["mode"].lower()
    assert "not real bge" in metric["details"]["mode"].lower() or "NOT real BGE" in metric["details"]["mode"]


def test_offline_resolved_by_matches_intended_composition():
    metric = evaluate_search_recall_precision_offline()
    for q in metric["details"]["per_query"]:
        resolved = set(q["resolved_by"])
        intended = set(q["resolver_mix_intended"].split("+"))
        assert resolved == intended, f"{q['q']!r}: resolved_by={resolved} != intended={intended}"


def test_offline_structured_and_location_only_queries_are_perfect():
    # these never touch the embedder, so they're genuine even offline
    metric = evaluate_search_recall_precision_offline()
    for q in metric["details"]["per_query"]:
        if q["resolver_mix_intended"] in ("filter", "location"):
            assert q["precision_at_k"] == 1.0
            assert q["recall_at_k"] == 1.0


def test_offline_deterministic_repeated_runs():
    m1 = evaluate_search_recall_precision_offline()
    m2 = evaluate_search_recall_precision_offline()
    assert m1["value"] == m2["value"]
    assert m1["details"]["per_query"] == m2["details"]["per_query"]


def test_live_defers_when_bge_unavailable():
    metric = evaluate_search_recall_precision_live()
    # on this machine BGE is not cached -- assert the honest outcome
    assert metric["status"] in ("deferred", "measured")
    if metric["status"] == "deferred":
        assert "bge" in metric["details"]["reason"].lower()


def test_offline_search_eval_never_touches_network(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("offline search eval must never touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    metric = evaluate_search_recall_precision_offline()
    assert metric["status"] == "measured"


# precision@k / recall@k arithmetic edge cases
def test_precision_recall_at_k_zero_result_query():
    p, r = _precision_recall_at_k([], {"img-001"}, k=5)
    assert p == 0.0
    assert r == 0.0


def test_precision_recall_at_k_fewer_candidates_than_k_uses_actual_count():
    # only 2 retrieved even though k=10 -- precision is over 2, not 10
    p, r = _precision_recall_at_k(["img-001", "img-002"], {"img-001", "img-002"}, k=10)
    assert p == 1.0
    assert r == 1.0


def test_precision_recall_at_k_partial():
    p, r = _precision_recall_at_k(["img-001", "img-999"], {"img-001", "img-002"}, k=2)
    assert p == 0.5
    assert r == 0.5
