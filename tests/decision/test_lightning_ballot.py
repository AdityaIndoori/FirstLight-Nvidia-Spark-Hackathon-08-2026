import copy
import socket
import urllib.request

import pytest

from backend.decision.lightning_ballot import (
    K_VOTES,
    _break_tie,
    _voted_class,
    request_lightning_ballot,
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


class CountingClient(LightningSeverityClient):
    """Records every call; always votes a fixed class. Not the stub under test."""

    def __init__(self, fixed_vote=2):
        self.fixed_vote = fixed_vote
        self.calls = []

    def sample_severity(self, building_context, temperature=0.7):
        self.calls.append((building_context, temperature))
        return self.fixed_vote


class InvalidVoteClient(LightningSeverityClient):
    """Always returns an out-of-range severity label."""

    def sample_severity(self, building_context, temperature=0.7):
        return 5


# 1: exactly 8 votes are requested
def test_exactly_eight_votes_are_requested():
    client = CountingClient()
    request_lightning_ballot(make_building_context(), client=client)
    assert len(client.calls) == K_VOTES == 8


# 2 & 3: unanimous votes -> agreement 1.0, doubt 0.05 (floor)
def test_unanimous_votes_produce_full_agreement_and_floored_doubt():
    client = StubLightningSeverityClient(votes=[2, 2, 2, 2, 2, 2, 2, 2])
    result = request_lightning_ballot(make_building_context(), client=client)

    assert result["voted_class"] == 2
    assert result["vote_agreement"] == 1.0
    assert result["doubt"] == 0.05


# 4 & 5: 6/8 modal agreement -> vote_agreement=0.75, doubt=0.25
def test_six_of_eight_modal_agreement():
    client = StubLightningSeverityClient(votes=[2, 2, 2, 2, 2, 2, 3, 3])
    result = request_lightning_ballot(make_building_context(), client=client)

    assert result["vote_agreement"] == 0.75
    assert result["doubt"] == 0.25


# 6: voted_class is the modal class
def test_voted_class_is_the_modal_label():
    client = StubLightningSeverityClient(votes=[1, 1, 1, 2, 2, 2, 2, 3])
    result = request_lightning_ballot(make_building_context(), client=client)

    assert result["voted_class"] == 2
    assert result["vote_agreement"] == 0.5


# 7: severity labels outside 0-3 are rejected
def test_severity_labels_outside_range_are_rejected():
    with pytest.raises(ValueError):
        request_lightning_ballot(make_building_context(), client=InvalidVoteClient())


# 8: original grader class (VL model's primary grade) is not modified / not owned by Lightning
def test_original_grader_class_not_modified_and_not_in_result():
    context = make_building_context(grader_class=3)
    result = request_lightning_ballot(context, client=StubLightningSeverityClient(votes=[1] * 8))

    assert context["grader_class"] == 3  # VL model's grade untouched
    assert set(result.keys()) == {"votes", "voted_class", "vote_agreement", "doubt"}
    assert "grader_class" not in result
    assert "damage_class" not in result
    assert "confidence" not in result
    assert "graded_by" not in result


# 9: original building input is not mutated
def test_building_context_is_not_mutated():
    context = make_building_context(grader_class=2, footprint_area_m2=88.5)
    snapshot = copy.deepcopy(context)

    request_lightning_ballot(context, client=StubLightningSeverityClient(votes=[2, 3, 2, 2, 1, 2, 2, 0]))

    assert context == snapshot


# 10: repeated deterministic stub input produces identical results
def test_repeated_deterministic_stub_calls_are_identical():
    context = make_building_context(grader_class=1)
    votes = [1, 1, 2, 1, 1, 3, 1, 1]

    first = request_lightning_ballot(context, client=StubLightningSeverityClient(votes=votes))
    second = request_lightning_ballot(context, client=StubLightningSeverityClient(votes=votes))

    assert first == second


# 11: client abstraction hides whether the implementation is stub or a future real model
def test_ballot_is_agnostic_to_client_implementation():
    class FakeAlternateClient(LightningSeverityClient):
        """Stands in for a future real Lightning client to prove callers only
        need the interface, not the concrete stub."""

        def sample_severity(self, building_context, temperature=0.7):
            return 3

    context = make_building_context(grader_class=0)
    stub_result = request_lightning_ballot(context, client=StubLightningSeverityClient(votes=[0] * 8))
    fake_result = request_lightning_ballot(context, client=FakeAlternateClient())

    assert stub_result["voted_class"] == 0
    assert fake_result["voted_class"] == 3
    assert isinstance(StubLightningSeverityClient(), LightningSeverityClient)
    assert isinstance(FakeAlternateClient(), LightningSeverityClient)


# 12: temperature=0.7 is passed through the sampling interface
def test_temperature_is_passed_through_to_every_call():
    client = CountingClient()
    request_lightning_ballot(make_building_context(), client=client)

    assert len(client.calls) == 8
    assert all(temperature == 0.7 for _, temperature in client.calls)

    client_override = CountingClient()
    request_lightning_ballot(make_building_context(), client=client_override, temperature=0.2)
    assert all(temperature == 0.2 for _, temperature in client_override.calls)


# 13: no real network/model access
def test_ballot_performs_no_real_network_access(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    result = request_lightning_ballot(make_building_context())

    assert result["voted_class"] in (0, 1, 2, 3)


# 14: tie behavior is deterministic and explicitly documented, not accidental
def test_tie_behavior_is_deterministic_and_documented():
    assert _break_tie.__doc__ and "UNRESOLVED" in _break_tie.__doc__

    tied_votes = [0, 0, 0, 0, 1, 1, 1, 1]  # 4-4 tie between class 0 and class 1
    assert _voted_class(tied_votes) == 1  # documented rule: highest tied class wins
    assert _break_tie([0, 1, 3]) == 3

    client = StubLightningSeverityClient(votes=tied_votes)
    first = request_lightning_ballot(make_building_context(), client=client)
    second = request_lightning_ballot(make_building_context(), client=StubLightningSeverityClient(votes=tied_votes))

    assert first["voted_class"] == second["voted_class"] == 1
