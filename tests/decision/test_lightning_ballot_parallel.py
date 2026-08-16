import copy
import threading
import time

import pytest

from backend.decision.lightning_ballot import (
    K_VOTES,
    request_lightning_ballot,
    request_lightning_ballot_parallel,
)
from backend.decision.lightning_client import LightningSeverityClient, StubLightningSeverityClient


def make_building_context(grader_class=1, **overrides):
    context = {
        "grader_class": grader_class,
        "grader_confidence": 0.8,
        "vl_caption": "Building shows visible damage consistent with the assigned grade.",
        "footprint_area_m2": 120.0,
        "facility_context": None,
        "neighbor_damage_classes": [1, 1, 2],
    }
    context.update(overrides)
    return context


class SlowCountingClient(LightningSeverityClient):
    """Simulates a slow real HTTP call so concurrency is observable; tracks
    the peak number of calls in flight simultaneously.
    """

    def __init__(self, delay_s=0.02, fixed_vote=2):
        self.delay_s = delay_s
        self.fixed_vote = fixed_vote
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0
        self.calls = 0

    def sample_severity(self, building_context, temperature=0.7):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            self.calls += 1
        time.sleep(self.delay_s)
        with self._lock:
            self._in_flight -= 1
        return self.fixed_vote


class InvalidVoteClient(LightningSeverityClient):
    def sample_severity(self, building_context, temperature=0.7):
        return 9


# 1: parallel ballot still returns exactly 8 votes
def test_parallel_ballot_returns_exactly_eight_votes():
    client = StubLightningSeverityClient(votes=[0, 1, 2, 3, 0, 1, 2, 3])
    result = request_lightning_ballot_parallel(make_building_context(), client=client)

    assert len(result["votes"]) == K_VOTES == 8
    assert set(result.keys()) == {"votes", "voted_class", "vote_agreement", "doubt"}


# 2: each vote is independently obtained through the Lightning client
def test_each_vote_independently_obtained_through_client():
    client = SlowCountingClient(delay_s=0.0)
    request_lightning_ballot_parallel(make_building_context(), client=client, max_concurrency=8)

    assert client.calls == 8  # eight separate sample_severity() invocations


# 3: maximum concurrency is bounded
def test_max_concurrency_is_bounded():
    client = SlowCountingClient(delay_s=0.05)
    request_lightning_ballot_parallel(make_building_context(), client=client, max_concurrency=3)

    assert client.peak_in_flight <= 3
    assert client.peak_in_flight > 1  # actually ran concurrently, not serially


def test_max_concurrency_clamped_to_k_votes():
    client = SlowCountingClient(delay_s=0.03)
    request_lightning_ballot_parallel(make_building_context(), client=client, max_concurrency=100)

    assert client.peak_in_flight <= K_VOTES  # never more workers than votes needed


def test_parallel_result_shape_matches_sequential():
    votes = [2, 2, 2, 2, 2, 2, 3, 3]
    sequential = request_lightning_ballot(make_building_context(), client=StubLightningSeverityClient(votes=votes))
    parallel = request_lightning_ballot_parallel(
        make_building_context(), client=StubLightningSeverityClient(votes=votes)
    )

    assert set(sequential.keys()) == set(parallel.keys())
    assert sequential["voted_class"] == parallel["voted_class"]
    assert sequential["vote_agreement"] == parallel["vote_agreement"]
    assert sequential["doubt"] == parallel["doubt"]


def test_parallel_ballot_does_not_mutate_building_context():
    context = make_building_context(grader_class=2)
    snapshot = copy.deepcopy(context)

    request_lightning_ballot_parallel(context, client=StubLightningSeverityClient(votes=[2] * 8))

    assert context == snapshot


def test_parallel_ballot_rejects_invalid_labels():
    with pytest.raises(ValueError):
        request_lightning_ballot_parallel(make_building_context(), client=InvalidVoteClient())


def test_sequential_path_is_unmodified_and_still_available():
    # request_lightning_ballot (sequential) still works exactly as before,
    # independent of the new parallel function existing alongside it.
    result = request_lightning_ballot(make_building_context(), client=StubLightningSeverityClient(votes=[1] * 8))
    assert result["voted_class"] == 1
