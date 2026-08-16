from backend.decision.eval.eval_assistant import evaluate_citation_faithfulness
from backend.decision.eval.eval_fema import evaluate_fema_field_accuracy


def test_fema_metric_measures_the_worksheet_when_a_builder_exists():
    """These two tests used to assert `deferred`, and the reason recorded then was
    accurate: no FEMA row builder existed in backend/. One exists in the service
    tree now, so the assertion that matters is field correctness rather than
    absence. Deferred is still accepted when the builder is not importable,
    because that is a missing component and not a failed gate.
    """
    metric = evaluate_fema_field_accuracy()

    if metric["status"] == "deferred":
        assert metric["details"]["reason"]
        return

    assert metric["status"] == "pass", metric["details"]["failed_checks"]
    assert metric["value"] == 1.0
    assert metric["sample_count"] >= 1


def test_fema_worksheet_excludes_undamaged_and_owner_columns():
    """The two failures that are gates rather than quality scores: an undamaged
    structure on a damage worksheet is wrong, and an owner-identity column on a
    federal export is a privacy incident."""
    metric = evaluate_fema_field_accuracy()
    if metric["status"] == "deferred":
        return
    failed = {c["check"] for c in metric["details"]["failed_checks"]}
    assert "undamaged structures excluded" not in failed
    assert "no owner-identity column" not in failed


def test_citation_faithfulness_is_deferred_with_reason():
    metric = evaluate_citation_faithfulness()
    assert metric["status"] == "deferred"
    assert metric["details"]["reason"]


def test_citation_faithfulness_never_claims_pass():
    metric = evaluate_citation_faithfulness()
    assert metric["status"] != "pass"
