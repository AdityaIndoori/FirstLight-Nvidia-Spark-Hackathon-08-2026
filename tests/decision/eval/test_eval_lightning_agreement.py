from backend.decision.eval.eval_lightning_agreement import (
    AGREEMENT_FIXTURE,
    evaluate_lightning_vs_nano_offline,
    evaluate_self_agreement_offline,
    exact_class_agreement_rate,
    mean_absolute_class_difference,
)


def test_fixture_has_at_least_ten_examples():
    assert len(AGREEMENT_FIXTURE) >= 10


def test_fixture_covers_required_spread():
    ids = {case["case_id"] for case in AGREEMENT_FIXTURE}
    assert "undamaged" in ids
    assert "minor" in ids
    assert "major" in ids
    assert "destroyed" in ids
    assert any("contradictory" in i for i in ids)
    assert "facility_context" in ids
    assert any("neighbor_variation" in i for i in ids)


# C1 offline: agreement math
def test_self_agreement_offline_returns_measured():
    metric = evaluate_self_agreement_offline()
    assert metric["status"] == "measured"
    assert metric["sample_count"] == len(AGREEMENT_FIXTURE)
    assert 0.0 <= metric["value"] <= 1.0


def test_self_agreement_offline_reports_mean_median_min():
    metric = evaluate_self_agreement_offline()
    details = metric["details"]
    assert "mean" in details and "median" in details and "min" in details and "max" in details
    assert details["min"] <= details["mean"] <= details["max"]


def test_self_agreement_offline_labeled_as_stub_not_real():
    metric = evaluate_self_agreement_offline()
    assert "stub" in metric["details"]["mode"].lower()


def test_self_agreement_offline_verifies_doubt_formula():
    metric = evaluate_self_agreement_offline()
    assert metric["details"]["doubt_formula_verified"] is True
    for case in metric["details"]["per_case"]:
        expected_doubt = max(0.05, 1 - case["vote_agreement"])
        assert abs(case["doubt"] - expected_doubt) < 1e-9


def test_self_agreement_offline_deterministic():
    m1 = evaluate_self_agreement_offline()
    m2 = evaluate_self_agreement_offline()
    assert m1 == m2


# C2 offline: no meaningful offline analog -- deferred, not faked
def test_lightning_vs_nano_offline_is_deferred():
    metric = evaluate_lightning_vs_nano_offline()
    assert metric["status"] == "deferred"
    assert "circular" in metric["details"]["reason"]


def test_lightning_vs_nano_offline_never_claims_pass():
    metric = evaluate_lightning_vs_nano_offline()
    assert metric["status"] != "pass"
    assert metric["value"] is None


# agreement math -- pure arithmetic, edge cases
def test_exact_class_agreement_rate_all_agree():
    assert exact_class_agreement_rate([(1, 1), (2, 2), (3, 3)]) == 1.0


def test_exact_class_agreement_rate_none_agree():
    assert exact_class_agreement_rate([(0, 3), (1, 2)]) == 0.0


def test_exact_class_agreement_rate_partial():
    assert exact_class_agreement_rate([(1, 1), (2, 3), (0, 0), (3, 1)]) == 0.5


def test_exact_class_agreement_rate_empty_is_zero():
    assert exact_class_agreement_rate([]) == 0.0


def test_mean_absolute_class_difference():
    assert mean_absolute_class_difference([(0, 3), (1, 1), (2, 0)]) == (3 + 0 + 2) / 3


def test_mean_absolute_class_difference_empty_is_zero():
    assert mean_absolute_class_difference([]) == 0.0
