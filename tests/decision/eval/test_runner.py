import socket
import urllib.request

from backend.decision.eval.report import STATUS_DEFERRED
from backend.decision.eval.runner import run_offline_eval

_EXPECTED_METRIC_NAMES = {
    "rationale_faithfulness",
    "fema_field_accuracy",
    "lightning_self_agreement",
    "lightning_vs_nano_agreement",
    "agency_plan_precision_recall",
    "agency_unit_accounting_sanity",
    "tag_precision_recall",
    "search_precision_recall_at_k",
    "assistant_citation_faithfulness",
    "injection_battery",
    "openshell_policy_tamper",
}


def test_offline_eval_produces_all_expected_metrics():
    report = run_offline_eval()
    names = {m["name"] for m in report["metrics"]}
    assert names == _EXPECTED_METRIC_NAMES


def test_offline_eval_report_has_required_top_level_shape():
    report = run_offline_eval()
    assert set(report.keys()) == {"metrics", "generated_at", "overall"}
    assert report["overall"] in ("pass", "fail", "partial")


def test_offline_eval_deferred_metrics_have_reasons():
    report = run_offline_eval()
    for metric in report["metrics"]:
        if metric["status"] == STATUS_DEFERRED:
            assert metric["details"].get("reason"), f"{metric['name']} deferred without a reason"


def test_offline_eval_overall_reflects_deferred_and_fail_honestly():
    report = run_offline_eval()
    statuses = {m["status"] for m in report["metrics"]}
    if "fail" in statuses:
        assert report["overall"] == "fail"
    elif "deferred" in statuses:
        assert report["overall"] == "partial"


def test_offline_eval_deterministic_metric_names_and_statuses_across_runs():
    r1 = run_offline_eval()
    r2 = run_offline_eval()
    names_statuses_1 = [(m["name"], m["status"]) for m in r1["metrics"]]
    names_statuses_2 = [(m["name"], m["status"]) for m in r2["metrics"]]
    assert names_statuses_1 == names_statuses_2


def test_offline_eval_makes_no_network_calls(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("offline B8 eval must never touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    report = run_offline_eval()
    assert len(report["metrics"]) == len(_EXPECTED_METRIC_NAMES)
