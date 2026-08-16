import pytest

from backend.decision.eval.report import (
    OVERALL_FAIL,
    OVERALL_PARTIAL,
    OVERALL_PASS,
    build_report,
    deferred_metric,
    make_metric,
)


def test_make_metric_shape():
    metric = make_metric("x", "pass", value=1, threshold=0, sample_count=5, details={"a": 1})
    assert set(metric.keys()) == {"name", "status", "value", "threshold", "sample_count", "details"}


def test_make_metric_rejects_unknown_status():
    with pytest.raises(ValueError):
        make_metric("x", "sort-of-pass")


def test_deferred_metric_shorthand():
    metric = deferred_metric("x", "not available")
    assert metric["status"] == "deferred"
    assert metric["details"]["reason"] == "not available"


def test_build_report_all_pass_is_overall_pass():
    metrics = [make_metric("a", "pass"), make_metric("b", "measured", value=0.9)]
    report = build_report(metrics, now_fn=lambda: 1000.0)
    assert report["overall"] == OVERALL_PASS
    assert report["generated_at"] == 1000.0
    assert report["metrics"] == metrics


def test_build_report_any_fail_is_overall_fail():
    metrics = [make_metric("a", "pass"), make_metric("b", "fail")]
    report = build_report(metrics)
    assert report["overall"] == OVERALL_FAIL


# deferred metrics represented honestly -- never counted as a pass
def test_build_report_any_deferred_without_fail_is_partial_not_pass():
    metrics = [make_metric("a", "pass"), deferred_metric("b", "unavailable")]
    report = build_report(metrics)
    assert report["overall"] == OVERALL_PARTIAL
    assert report["overall"] != OVERALL_PASS


def test_build_report_fail_takes_priority_over_deferred():
    metrics = [make_metric("a", "fail"), deferred_metric("b", "unavailable")]
    report = build_report(metrics)
    assert report["overall"] == OVERALL_FAIL


def test_build_report_empty_metrics_is_pass():
    report = build_report([])
    assert report["overall"] == OVERALL_PASS
    assert report["metrics"] == []
