import copy

import pytest

from backend.decision.lightning_baseline import (
    FIXTURE_BUILDINGS,
    BuildingBallotError,
    run_lightning_baseline,
)
from backend.decision.lightning_ballot import K_VOTES
from backend.decision.lightning_client import LightningSeverityClient, StubLightningSeverityClient

# Deterministic 40-vote sequence (5 buildings x 8 votes), chosen so the
# aggregate stats are hand-computable:
#   building 1: [0]*8                        -> agreement 1.0
#   building 2: [1,1,1,1,1,1,1,0]             -> agreement 0.875
#   building 3: [2,2,2,2,2,3,3,3]             -> agreement 0.625
#   building 4: [3]*8                         -> agreement 1.0
#   building 5: [0,1,2,3,0,1,2,3] (4-way tie) -> agreement 0.25
KNOWN_VOTES = (
    [0] * 8
    + [1, 1, 1, 1, 1, 1, 1, 0]
    + [2, 2, 2, 2, 2, 3, 3, 3]
    + [3] * 8
    + [0, 1, 2, 3, 0, 1, 2, 3]
)
EXPECTED_AGREEMENTS = [1.0, 0.875, 0.625, 1.0, 0.25]


class FlakyClient(LightningSeverityClient):
    """Raises on a specific call number; otherwise votes a fixed class."""

    def __init__(self, fail_at_call: int):
        self.fail_at_call = fail_at_call
        self.calls = 0

    def sample_severity(self, building_context, temperature=0.7):
        self.calls += 1
        if self.calls == self.fail_at_call:
            raise RuntimeError("simulated Lightning failure")
        return 2


# 1: exactly 5 building contexts are processed by the baseline helper
def test_exactly_five_fixture_buildings():
    assert len(FIXTURE_BUILDINGS) == 5
    names = {b["name"] for b in FIXTURE_BUILDINGS}
    assert len(names) == 5  # all distinct

    result = run_lightning_baseline(client=StubLightningSeverityClient())
    assert result["buildings_processed"] == 5
    assert len(result["results"]) == 5


# 2: each building receives exactly 8 votes
def test_each_building_receives_exactly_eight_votes():
    result = run_lightning_baseline(client=StubLightningSeverityClient(votes=KNOWN_VOTES))
    for building_result in result["results"]:
        assert len(building_result["votes"]) == K_VOTES == 8


# 3: total expected generation count is 40
def test_total_generation_count_is_forty():
    result = run_lightning_baseline(client=StubLightningSeverityClient())
    assert result["total_generations"] == 40


# 4: aggregate mean agreement is computed correctly from mocked ballots
def test_mean_agreement_computed_correctly():
    result = run_lightning_baseline(client=StubLightningSeverityClient(votes=KNOWN_VOTES))

    actual_agreements = [r["vote_agreement"] for r in result["results"]]
    assert actual_agreements == EXPECTED_AGREEMENTS
    assert result["mean_vote_agreement"] == pytest.approx(sum(EXPECTED_AGREEMENTS) / 5)


# 5: min/max agreement are correct
def test_min_and_max_agreement_correct():
    result = run_lightning_baseline(client=StubLightningSeverityClient(votes=KNOWN_VOTES))

    assert result["min_vote_agreement"] == pytest.approx(0.25)
    assert result["max_vote_agreement"] == pytest.approx(1.0)


# 6: building inputs are not mutated
def test_fixture_building_contexts_not_mutated():
    snapshot = copy.deepcopy(FIXTURE_BUILDINGS)

    run_lightning_baseline(client=StubLightningSeverityClient(votes=KNOWN_VOTES))

    assert FIXTURE_BUILDINGS == snapshot


# 7: one building failure is surfaced clearly rather than silently omitted
def test_building_failure_is_surfaced_not_silently_dropped():
    # fail_at_call=9 is the first call of the 2nd fixture (likely_minor_damage)
    client = FlakyClient(fail_at_call=9)

    with pytest.raises(BuildingBallotError) as exc_info:
        run_lightning_baseline(client=client)

    assert exc_info.value.building_name == FIXTURE_BUILDINGS[1]["name"]
    assert isinstance(exc_info.value.original_exc, RuntimeError)
    assert exc_info.value.__cause__ is exc_info.value.original_exc


# 8: no real HTTP request occurs in normal pytest
def test_baseline_performs_no_real_network_access(monkeypatch):
    import socket
    import urllib.request

    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    result = run_lightning_baseline(client=StubLightningSeverityClient())
    assert result["buildings_processed"] == 5


# 9: usage stats are only reported when the client actually recorded them
def test_usage_stats_absent_when_client_reports_none():
    result = run_lightning_baseline(client=StubLightningSeverityClient())
    assert "total_prompt_tokens" not in result
    assert "total_completion_tokens" not in result
    assert "tokens_per_sec" not in result
