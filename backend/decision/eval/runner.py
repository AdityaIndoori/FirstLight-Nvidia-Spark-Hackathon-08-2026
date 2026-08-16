"""B8: top-level offline/live evaluation runners.

run_offline_eval() runs every metric that can be measured without any
network call or real model -- safe to call from anywhere, including
pytest. run_live_eval() additionally calls the real Nano (:8000) and
Lightning (:8001) servers, and attempts real BGE (never downloading it).

Unit-count sanity always runs via the deterministic path in BOTH modes:
it verifies B6a's accounting ARITHMETIC (units_required == sum(step
units), overcommitment/shortfall formulas), which is identical code
regardless of whether Nano or the stub drafted the assignments being
accounted -- there is no separate "live" version of pure arithmetic.
"""

from backend.decision.eval.eval_agency import (
    evaluate_agency_plan_correctness_live,
    evaluate_agency_plan_correctness_offline,
    evaluate_agency_unit_sanity_offline,
)
from backend.decision.eval.eval_assistant import evaluate_citation_faithfulness
from backend.decision.eval.eval_fema import evaluate_fema_field_accuracy
from backend.decision.eval.eval_injection import (
    evaluate_injection_battery_live,
    evaluate_injection_battery_offline,
    evaluate_openshell_tamper,
)
from backend.decision.eval.eval_lightning_agreement import (
    evaluate_lightning_vs_nano_live,
    evaluate_lightning_vs_nano_offline,
    evaluate_self_agreement_live,
    evaluate_self_agreement_offline,
)
from backend.decision.eval.eval_rationale import evaluate_rationale_faithfulness
from backend.decision.eval.eval_search import evaluate_search_recall_precision_live, evaluate_search_recall_precision_offline
from backend.decision.eval.eval_tags import evaluate_tag_precision_recall_live, evaluate_tag_precision_recall_offline
from backend.decision.eval.report import build_report


def run_offline_eval() -> dict:
    """Every deterministic/fixture-driven B8 metric. No network call, no
    real model, safe from pytest.
    """
    metrics = [
        evaluate_rationale_faithfulness(),
        evaluate_fema_field_accuracy(),
        evaluate_self_agreement_offline(),
        evaluate_lightning_vs_nano_offline(),
        evaluate_agency_plan_correctness_offline(),
        evaluate_agency_unit_sanity_offline(),
        evaluate_tag_precision_recall_offline(),
        evaluate_search_recall_precision_offline(),
        evaluate_citation_faithfulness(),
        evaluate_injection_battery_offline(),
        evaluate_openshell_tamper(),
    ]
    return build_report(metrics)


def run_live_eval(nano_base_url: str = None, lightning_base_url: str = None, timeout_s: float = 10.0) -> dict:
    """Every B8 metric, with model-dependent ones hitting the real Nano
    (:8000) and Lightning (:8001) servers and attempting real BGE (never
    downloading it -- see eval_search.evaluate_search_recall_precision_live).
    A per-sample model failure inside any one metric is recorded and that
    sample skipped; it never aborts the rest of this function.
    """
    metrics = [
        evaluate_rationale_faithfulness(),  # deterministic string-check; identical in every mode
        evaluate_fema_field_accuracy(),
        evaluate_self_agreement_live(base_url=lightning_base_url, timeout_s=timeout_s),
        evaluate_lightning_vs_nano_live(
            lightning_base_url=lightning_base_url, nano_base_url=nano_base_url, timeout_s=timeout_s
        ),
        evaluate_agency_plan_correctness_live(base_url=nano_base_url, timeout_s=timeout_s),
        evaluate_agency_unit_sanity_offline(),  # pure accounting arithmetic -- see module docstring
        evaluate_tag_precision_recall_live(base_url=lightning_base_url, timeout_s=timeout_s),
        evaluate_search_recall_precision_live(),
        evaluate_citation_faithfulness(),
        evaluate_injection_battery_live(
            agency_base_url=nano_base_url, lightning_base_url=lightning_base_url, timeout_s=timeout_s
        ),
        evaluate_openshell_tamper(),
    ]
    return build_report(metrics)
