from backend.decision.eval.eval_assistant import evaluate_citation_faithfulness
from backend.decision.eval.eval_fema import evaluate_fema_field_accuracy


def test_fema_metric_is_deferred_with_reason():
    metric = evaluate_fema_field_accuracy()
    assert metric["status"] == "deferred"
    assert "reason" in metric["details"]
    assert metric["details"]["reason"]  # non-empty


def test_fema_metric_never_claims_pass():
    metric = evaluate_fema_field_accuracy()
    assert metric["status"] != "pass"
    assert metric["value"] is None


def test_citation_faithfulness_is_deferred_with_reason():
    metric = evaluate_citation_faithfulness()
    assert metric["status"] == "deferred"
    assert metric["details"]["reason"]


def test_citation_faithfulness_never_claims_pass():
    metric = evaluate_citation_faithfulness()
    assert metric["status"] != "pass"
