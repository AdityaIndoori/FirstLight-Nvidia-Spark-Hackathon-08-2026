import copy
import socket
import urllib.request

import pytest

from backend.decision.lightning_client import LightningSeverityClient, StubLightningSeverityClient
from backend.decision.lightning_perf import (
    generate_synthetic_fixtures,
    percentile,
    run_batch_sweep,
    summarize_latencies_ms,
)


# 7 & 8: p50 / p95 are computed correctly
def test_percentile_p50_and_p95():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 100


def test_percentile_unsorted_input_and_single_value():
    assert percentile([30, 10, 20], 50) == 20
    assert percentile([42], 95) == 42
    assert percentile([], 50) == 0.0


def test_summarize_latencies_matches_percentile():
    values = [5.0, 1.0, 9.0, 3.0, 7.0]
    summary = summarize_latencies_ms(values)
    assert summary["mean_ms"] == pytest.approx(sum(values) / len(values))
    assert summary["p50_ms"] == percentile(values, 50)
    assert summary["p95_ms"] == percentile(values, 95)


def test_summarize_latencies_empty_input():
    assert summarize_latencies_ms([]) == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}


# 5: 50 buildings x 8 votes = 400 generations
def test_fifty_buildings_times_eight_is_four_hundred_generations():
    buildings = generate_synthetic_fixtures(50)
    assert len(buildings) == 50

    result = run_batch_sweep(buildings, concurrency=4, client=StubLightningSeverityClient(votes=[2] * 8))
    assert result["total_generations"] == 400
    assert result["buildings_processed"] == 50


# 4: votes remain associated with the correct building
def test_votes_remain_associated_with_correct_building():
    buildings = generate_synthetic_fixtures(12)  # grader_class cycles 0,1,2,3,...
    # default stub (no configured votes) unanimously echoes building_context["grader_class"]
    result = run_batch_sweep(buildings, concurrency=4, client=StubLightningSeverityClient())

    by_name = {r["name"]: r for r in result["results"]}
    for fixture in buildings:
        expected_grader_class = fixture["context"]["grader_class"]
        assert by_name[fixture["name"]]["voted_class"] == expected_grader_class


# 6: concurrency sweep uses the same fixture set
def test_sweep_reuses_the_same_fixture_set_across_concurrency_levels():
    buildings = generate_synthetic_fixtures(10)
    snapshot = copy.deepcopy(buildings)

    for concurrency in (1, 2, 4):
        run_batch_sweep(buildings, concurrency=concurrency, client=StubLightningSeverityClient(votes=[2] * 8))

    assert buildings == snapshot  # never mutated -- identical fixtures reused at every level


# 9: failures are counted and not silently omitted
def test_failures_are_counted_and_building_is_excluded():
    class FlakyClient(LightningSeverityClient):
        def __init__(self):
            self.calls = 0

        def sample_severity(self, building_context, temperature=0.7):
            self.calls += 1
            if self.calls % 5 == 0:
                raise RuntimeError("simulated Lightning failure")
            return 2

    buildings = generate_synthetic_fixtures(5)
    result = run_batch_sweep(buildings, concurrency=4, client=FlakyClient())

    assert result["failures"] > 0
    assert result["buildings_processed"] < len(buildings)
    assert result["failures"] <= result["total_generations"]


# 10: fixture building contexts are not mutated
def test_fixture_contexts_not_mutated():
    buildings = generate_synthetic_fixtures(5)
    snapshot = copy.deepcopy(buildings)

    run_batch_sweep(buildings, concurrency=2, client=StubLightningSeverityClient(votes=[1] * 8))

    assert buildings == snapshot


def test_no_real_network_access(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    buildings = generate_synthetic_fixtures(20)
    result = run_batch_sweep(buildings, concurrency=8, client=StubLightningSeverityClient(votes=[2] * 8))
    assert result["buildings_processed"] == 20


def test_ballot_latency_summary_present_and_non_negative():
    buildings = generate_synthetic_fixtures(6)
    result = run_batch_sweep(buildings, concurrency=3, client=StubLightningSeverityClient(votes=[2] * 8))

    for key in ("mean_ms", "p50_ms", "p95_ms"):
        assert result["ballot_latency_ms"][key] >= 0.0


def test_mean_vote_agreement_is_within_valid_range():
    buildings = generate_synthetic_fixtures(6)
    result = run_batch_sweep(buildings, concurrency=2, client=StubLightningSeverityClient(votes=[2] * 8))
    assert 0.0 <= result["mean_vote_agreement"] <= 1.0
