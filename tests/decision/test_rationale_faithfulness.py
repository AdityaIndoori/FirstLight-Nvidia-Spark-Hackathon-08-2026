import copy
import json

import pytest

from backend.decision.nano_client import (
    NanoClientError,
    RealNanoRationaleClient,
    StubNanoRationaleClient,
    _faithfulness_violations,
)
from backend.decision.rationale import generate_rationale_with_recovery

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

# The exact bug report: road_cutoff=1.8 rendered as a distance.
BUGGY_LIVE_OUTPUT = (
    "The operator should prioritize 412 Elm St due to its confirmed destruction "
    "(damage class 3) and high vulnerable density (2.30) near the Riverside Clinic, "
    "located 180m away, despite a 6.5-hour staleness and partial road access up to "
    "1.8m cutoff. The low doubt (0.12) and high confidence (0.91) from operator "
    "jsmith support action to mitigate immediate risks."
)


class FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _install_fake_urlopen(monkeypatch, content: str):
    import urllib.request

    def fake_urlopen(request, timeout=None):
        body = {"choices": [{"message": {"content": content}}]}
        return FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


# 1: road_cutoff is never rendered with a physical distance unit
def test_road_cutoff_never_rendered_with_distance_unit():
    assert _faithfulness_violations(BUGGY_LIVE_OUTPUT, RANK_ITEM)  # the reported bug

    bad = "Access is constrained: 1.8 miles blocked on the approach road."
    assert _faithfulness_violations(bad, RANK_ITEM)

    good = "Road-access disruption contributes a 1.8x priority multiplier."
    assert _faithfulness_violations(good, RANK_ITEM) == []


def test_road_cutoff_bad_output_triggers_deterministic_fallback(monkeypatch):
    _install_fake_urlopen(monkeypatch, BUGGY_LIVE_OUTPUT)

    with pytest.raises(NanoClientError):
        RealNanoRationaleClient(base_url="http://localhost:8000").generate_rationale(RANK_ITEM)

    result = generate_rationale_with_recovery(
        RANK_ITEM, real_client=RealNanoRationaleClient(base_url="http://localhost:8000")
    )
    assert result["recovery"] == "stub"
    assert result["rationale"] == StubNanoRationaleClient().generate_rationale(RANK_ITEM)


# 2: facility_near.dist_m may correctly use meters
def test_dist_m_may_correctly_use_meters():
    good = (
        "412 Elm St is 180 m from Riverside Clinic; a 1.8x road-cutoff multiplier "
        "and 6.5 hours of staleness both raise its priority."
    )
    assert _faithfulness_violations(good, RANK_ITEM) == []


# 3: vulnerable_density is not turned into a population count
def test_vulnerable_density_not_turned_into_population_count():
    bad = "High vulnerable density of 2.30 people nearby drives urgency."
    assert _faithfulness_violations(bad, RANK_ITEM)

    bad_residents = "An estimated 2.30 residents live near this site."
    assert _faithfulness_violations(bad_residents, RANK_ITEM)

    good = "The vulnerable-density factor of 2.30 drives urgency."
    assert _faithfulness_violations(good, RANK_ITEM) == []


# 4: doubt is not turned into a percentage unless explicitly a dimensionless score
def test_doubt_percentage_requires_dimensionless_label():
    bad = "There is a 12% chance of further damage given current doubt."
    assert _faithfulness_violations(bad, RANK_ITEM)

    good_no_percent = "Doubt of 0.12 reflects grading uncertainty."
    assert _faithfulness_violations(good_no_percent, RANK_ITEM) == []

    good_labeled = "Doubt equates to a 12% dimensionless uncertainty score."
    assert _faithfulness_violations(good_labeled, RANK_ITEM) == []


# 5: staleness contributes to/increases priority, never described as reducing it
def test_staleness_direction_is_checked():
    bad = "The 6.5-hour staleness reduces priority for this building."
    assert _faithfulness_violations(bad, RANK_ITEM)

    bad_lower = "Because the report is older, staleness lowers the urgency here."
    assert _faithfulness_violations(bad_lower, RANK_ITEM)

    good = "The observation is 6.5 hours old, increasing staleness and priority."
    assert _faithfulness_violations(good, RANK_ITEM) == []


# 6: confidence is not described as casualty/occupancy probability
def test_confidence_not_described_as_casualty_probability():
    bad = "Confidence of 0.91 reflects a high probability of casualties."
    assert _faithfulness_violations(bad, RANK_ITEM)

    good = "Confidence of 0.91 in the assigned damage grade supports action."
    assert _faithfulness_violations(good, RANK_ITEM) == []


# 7: property value, casualties, occupancy, resource availability stay absent
def test_forbidden_unsupported_quantities_remain_absent():
    for bad in (
        "Estimated property value loss is significant.",
        "Two ambulances are available for dispatch.",
        "Three occupants are believed trapped inside.",
    ):
        assert _faithfulness_violations(bad, RANK_ITEM), f"expected a violation for: {bad!r}"

    good = "412 Elm St: destroyed, operator-confirmed, priority 24.19320."
    assert _faithfulness_violations(good, RANK_ITEM) == []


# 8: priority input values are not mutated by the faithfulness check or the real client
def test_rank_item_not_mutated(monkeypatch):
    snapshot = copy.deepcopy(RANK_ITEM)
    _faithfulness_violations(BUGGY_LIVE_OUTPUT, RANK_ITEM)
    assert RANK_ITEM == snapshot

    _install_fake_urlopen(monkeypatch, "412 Elm St: destroyed, priority 24.19320.")
    working_copy = copy.deepcopy(RANK_ITEM)
    RealNanoRationaleClient(base_url="http://localhost:8000").generate_rationale(working_copy)
    assert working_copy == snapshot
