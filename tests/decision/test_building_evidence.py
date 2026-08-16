import copy

import pytest

from backend.decision import building_evidence
from backend.decision.building_evidence import (
    BuildingEvidenceError,
    compute_staleness_h,
    process_building_evidence,
    validate_building_evidence,
)
from backend.decision.lightning_client import LightningClientError, LightningSeverityClient, StubLightningSeverityClient
from backend.decision.scoring import calculate_priority

CAPTURED_AT = 1_700_000_000.0


def make_evidence(**overrides):
    evidence = {
        "footprint_id": "bldg-0042",
        "image_id": "img-0187",
        "label": "412 Elm St",
        "centroid": [-122.3764, 47.5581],
        "captured_at": CAPTURED_AT,
        "damage_class": 2,
        "confidence": 0.85,
        "graded_by": "nemotron-vl",
        "vl_caption": "Two-storey structure with partial roof collapse and debris along the eastern wall.",
        "footprint_area_m2": 120.0,
        "facility_near": {"name": "Riverside Clinic", "type": "dialysis", "dist_m": 150},
        "neighbor_damage_classes": [1, 2, 2],
        "vulnerable_density": 2.31,
    }
    evidence.update(overrides)
    return evidence


class SpyClient(LightningSeverityClient):
    """Records every building_context it receives; always votes a fixed class."""

    def __init__(self, fixed_vote=2):
        self.fixed_vote = fixed_vote
        self.calls = []

    def sample_severity(self, building_context, temperature=0.7):
        self.calls.append(building_context)
        return self.fixed_vote


class AlwaysFailingClient(LightningSeverityClient):
    def sample_severity(self, building_context, temperature=0.7):
        raise LightningClientError("simulated real Lightning failure")


# 1: valid BuildingEvidence is accepted
def test_valid_building_evidence_is_accepted():
    assert validate_building_evidence(make_evidence()) == []


# 2: invalid damage_class rejected
def test_invalid_damage_class_rejected():
    errors = validate_building_evidence(make_evidence(damage_class=4))
    assert any("damage_class" in e for e in errors)


# 3: invalid confidence rejected
def test_invalid_confidence_rejected():
    errors_high = validate_building_evidence(make_evidence(confidence=1.5))
    errors_low = validate_building_evidence(make_evidence(confidence=-0.1))
    assert any("confidence" in e for e in errors_high)
    assert any("confidence" in e for e in errors_low)


# 4: invalid neighbor class rejected
def test_invalid_neighbor_class_rejected():
    errors = validate_building_evidence(make_evidence(neighbor_damage_classes=[1, 2, 5]))
    assert any("neighbor_damage_classes" in e for e in errors)


# 5: invalid facility type rejected
def test_invalid_facility_type_rejected():
    errors = validate_building_evidence(
        make_evidence(facility_near={"name": "Some School", "type": "school", "dist_m": 100})
    )
    assert any("facility_near.type" in e for e in errors)


# 6: negative vulnerable_density rejected
def test_negative_vulnerable_density_rejected():
    errors = validate_building_evidence(make_evidence(vulnerable_density=-1.0))
    assert any("vulnerable_density" in e for e in errors)


# additional boundary checks matching the frozen contract
def test_empty_footprint_id_rejected():
    assert any("footprint_id" in e for e in validate_building_evidence(make_evidence(footprint_id="")))


def test_empty_image_id_rejected():
    assert any("image_id" in e for e in validate_building_evidence(make_evidence(image_id="")))


def test_malformed_centroid_rejected():
    assert any("centroid" in e for e in validate_building_evidence(make_evidence(centroid=[1.0])))


def test_empty_graded_by_rejected():
    assert any("graded_by" in e for e in validate_building_evidence(make_evidence(graded_by="")))


def test_empty_vl_caption_rejected():
    assert any("vl_caption" in e for e in validate_building_evidence(make_evidence(vl_caption="")))


def test_negative_footprint_area_rejected():
    assert any("footprint_area_m2" in e for e in validate_building_evidence(make_evidence(footprint_area_m2=-5.0)))


def test_negative_facility_dist_m_rejected():
    errors = validate_building_evidence(
        make_evidence(facility_near={"name": "Riverside Clinic", "type": "dialysis", "dist_m": -10})
    )
    assert any("dist_m" in e for e in errors)


def test_null_facility_near_is_valid():
    assert validate_building_evidence(make_evidence(facility_near=None)) == []


# 7: staleness is calculated from scored_at - captured_at
def test_staleness_calculated_from_scored_at_minus_captured_at():
    result = compute_staleness_h(captured_at=1000.0, scored_at=1000.0 + 6.5 * 3600.0)
    assert result == pytest.approx(6.5)


# 8: future captured_at clamps staleness to 0
def test_future_captured_at_clamps_staleness_to_zero():
    result = compute_staleness_h(captured_at=2000.0, scored_at=1000.0)
    assert result == 0.0


# 9: Lightning receives vl_caption
def test_lightning_receives_vl_caption():
    evidence = make_evidence()
    spy = SpyClient()

    process_building_evidence(evidence, scored_at=CAPTURED_AT, real_client=spy)

    assert len(spy.calls) == 8
    assert all(ctx["vl_caption"] == evidence["vl_caption"] for ctx in spy.calls)


# 10: Lightning receives original damage_class
def test_lightning_receives_original_damage_class():
    evidence = make_evidence(damage_class=3)
    spy = SpyClient()

    process_building_evidence(evidence, scored_at=CAPTURED_AT, real_client=spy)

    assert all(ctx["grader_class"] == 3 for ctx in spy.calls)


# 11: original BuildingEvidence is not mutated
def test_original_evidence_not_mutated():
    evidence = make_evidence()
    snapshot = copy.deepcopy(evidence)

    process_building_evidence(evidence, scored_at=CAPTURED_AT, real_client=SpyClient(), road_cutoff=1.5)

    assert evidence == snapshot


# 12: Lightning voted_class does NOT overwrite damage_class
def test_voted_class_does_not_overwrite_damage_class():
    evidence = make_evidence(damage_class=2)
    # Lightning disagrees and votes 0 every time
    disagreeing_client = SpyClient(fixed_vote=0)

    result = process_building_evidence(evidence, scored_at=CAPTURED_AT, real_client=disagreeing_client)

    assert result["evidence"]["damage_class"] == 2  # untouched original grade
    assert result["voted_class"] == 0  # Lightning's own, separate output


# 13: k=8 behavior remains unchanged
def test_k_equals_eight_calls_at_this_layer_too():
    spy = SpyClient()
    process_building_evidence(make_evidence(), scored_at=CAPTURED_AT, real_client=spy)
    assert len(spy.calls) == 8


# 14: doubt from Lightning feeds calculate_priority
def test_doubt_from_lightning_feeds_calculate_priority():
    evidence = make_evidence(vulnerable_density=2.31)
    votes = [2, 2, 2, 2, 2, 2, 3, 3]  # 6/8 agreement -> doubt = 0.25
    client = StubLightningSeverityClient(votes=votes)
    scored_at = CAPTURED_AT + 6.5 * 3600.0

    result = process_building_evidence(evidence, scored_at=scored_at, real_client=client, road_cutoff=1.8)

    expected_priority = calculate_priority(
        staleness_h=6.5, vulnerable_density=2.31, doubt=0.25, road_cutoff=1.8
    )
    assert result["doubt"] == 0.25
    assert result["priority"] == expected_priority


# 15: road_cutoff=None behaves as multiplier 1
def test_road_cutoff_none_behaves_as_multiplier_one():
    evidence = make_evidence(vulnerable_density=1.0)
    client = StubLightningSeverityClient(votes=[2] * 8)  # unanimous -> doubt 0.05
    scored_at = CAPTURED_AT + 3600.0  # staleness_h = 1.0

    result = process_building_evidence(evidence, scored_at=scored_at, real_client=client, road_cutoff=None)

    expected = calculate_priority(staleness_h=1.0, vulnerable_density=1.0, doubt=0.05, road_cutoff=None)
    assert result["road_cutoff"] is None
    assert result["priority"] == expected


# 16: road_cutoff<1 rejected
def test_road_cutoff_below_one_rejected():
    spy = SpyClient()
    with pytest.raises(ValueError):
        process_building_evidence(make_evidence(), scored_at=CAPTURED_AT, real_client=spy, road_cutoff=0.5)

    assert len(spy.calls) == 0  # Lightning must never be invoked for a rejected road_cutoff


# 17: existing calculate_priority is reused, not reimplemented
def test_calculate_priority_is_reused_not_reimplemented(monkeypatch):
    calls = []

    def spy_calculate_priority(staleness_h, vulnerable_density, doubt, road_cutoff=None):
        calls.append((staleness_h, vulnerable_density, doubt, road_cutoff))
        return 999.99999

    monkeypatch.setattr(building_evidence, "calculate_priority", spy_calculate_priority)

    evidence = make_evidence(vulnerable_density=2.31)
    scored_at = CAPTURED_AT + 6.5 * 3600.0
    result = process_building_evidence(
        evidence, scored_at=scored_at, real_client=StubLightningSeverityClient(votes=[2] * 8), road_cutoff=1.8
    )

    assert len(calls) == 1
    staleness_h, vulnerable_density, doubt, road_cutoff = calls[0]
    assert staleness_h == pytest.approx(6.5)
    assert vulnerable_density == 2.31
    assert doubt == 0.05
    assert road_cutoff == 1.8
    assert result["priority"] == 999.99999


# 18: Lightning failure returns recovery="stub"
def test_lightning_failure_returns_recovery_stub():
    result = process_building_evidence(
        make_evidence(),
        scored_at=CAPTURED_AT,
        real_client=AlwaysFailingClient(),
        fallback_client=StubLightningSeverityClient(votes=[1] * 8),
    )

    assert result["lightning_recovery"] == "stub"
    assert result["voted_class"] == 1


def test_real_client_success_returns_recovery_model():
    result = process_building_evidence(
        make_evidence(), scored_at=CAPTURED_AT, real_client=StubLightningSeverityClient(votes=[2] * 8)
    )
    assert result["lightning_recovery"] == "model"


# 19: invalid BuildingEvidence does not invoke Lightning
def test_invalid_evidence_does_not_invoke_lightning():
    spy = SpyClient()
    with pytest.raises(BuildingEvidenceError):
        process_building_evidence(make_evidence(damage_class=9), scored_at=CAPTURED_AT, real_client=spy)

    assert len(spy.calls) == 0


def test_invalid_evidence_error_lists_the_violations():
    with pytest.raises(BuildingEvidenceError) as exc_info:
        process_building_evidence(make_evidence(damage_class=9, vulnerable_density=-1), scored_at=CAPTURED_AT)
    assert "damage_class" in str(exc_info.value)
    assert "vulnerable_density" in str(exc_info.value)


# 11: production fallback behavior is unchanged by the new parallel/perf
# additions -- process_building_evidence still goes through the unmodified
# sequential request_lightning_ballot, never the new parallel path.
def test_production_path_does_not_use_the_new_parallel_ballot(monkeypatch):
    from backend.decision import lightning_ballot

    calls = []
    original = lightning_ballot.request_lightning_ballot_parallel

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(lightning_ballot, "request_lightning_ballot_parallel", spy)

    process_building_evidence(
        make_evidence(), scored_at=CAPTURED_AT, real_client=StubLightningSeverityClient(votes=[2] * 8)
    )

    assert calls == []


# no real network access: SpyClient/StubLightningSeverityClient never touch a socket
def test_no_real_network_access(monkeypatch):
    import socket
    import urllib.request

    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    result = process_building_evidence(
        make_evidence(), scored_at=CAPTURED_AT, real_client=StubLightningSeverityClient(votes=[2] * 8)
    )
    assert result["voted_class"] == 2
