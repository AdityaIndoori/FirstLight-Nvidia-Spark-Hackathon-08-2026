"""B8 Part A: rationale faithfulness evaluation.

Reuses the existing checker verbatim (nano_client._faithfulness_violations)
against a labeled fixture of RankItem/rationale-text pairs -- this module
adds no new faithfulness logic of its own, it only measures the real one.
Fully deterministic and offline: no model call, just string checks against
fixed text.

--------------------------------------------------------------------------
FORMERLY A GAP, NOW CLOSED
--------------------------------------------------------------------------
An earlier run of this eval found that two violation categories --
"invented facility" and "wrong damage class" -- were not caught by
_faithfulness_violations: it checked unit/percentage misuse on the
dimensionless fields, staleness direction, and a forbidden-content-term
list, but never cross-checked a stated facility name against
rank_item["facility_near"] or a stated severity word against
rank_item["damage_class"]. That measured gap (Part 12's "do not fake a
passing score" applied to the eval suite itself) motivated adding
_damage_class_contradicted and _facility_contradicted to
nano_client._faithfulness_violations. All RATIONALE_FIXTURE cases,
including "invented_facility" and "wrong_damage_class", are now correctly
classified -- see test_eval_rationale.py::test_all_cases_are_correctly_classified.
"""

from backend.decision.eval.report import STATUS_FAIL, STATUS_PASS, make_metric
from backend.decision.nano_client import _faithfulness_violations

_BASE_RANK_ITEM = {
    "footprint_id": "fp-001",
    "label": "412 Elm St",
    "centroid": [-122.4194, 37.7749],
    "damage_class": 3,
    "confidence": 0.91,
    "confirmed": True,
    "graded_by": "operator:jsmith",
    "facility_near": None,
    "inputs": {"staleness_h": 6.5, "vulnerable_density": 2.3, "doubt": 0.12, "road_cutoff": 1.8},
    "priority": 24.19320,
    "rationale": "",
    "rationale_by": "nano",
}

_FACILITY_RANK_ITEM = dict(
    _BASE_RANK_ITEM,
    footprint_id="fp-002",
    facility_near={"name": "Riverside Clinic", "type": "clinic", "dist_m": 180},
)

RATIONALE_FIXTURE = [
    {
        "case_id": "correct_no_facility",
        "rank_item": _BASE_RANK_ITEM,
        "rationale_text": (
            "412 Elm St: destroyed, operator-confirmed, confidence 0.91. Staleness of "
            "6.5 hours and a vulnerable-density factor of 2.30 raise priority; doubt of "
            "0.12 reflects grading uncertainty. A 1.8x road-cutoff multiplier further "
            "raises priority. Priority: 24.19320."
        ),
        "expect_violation": False,
        "category": "correct",
    },
    {
        "case_id": "correct_with_facility",
        "rank_item": _FACILITY_RANK_ITEM,
        "rationale_text": (
            "412 Elm St is 180 m from Riverside Clinic. A 1.8x road-cutoff multiplier and "
            "6.5 hours of staleness both raise its priority; doubt of 0.12 reflects "
            "grading uncertainty. Priority: 24.19320."
        ),
        "expect_violation": False,
        "category": "correct",
    },
    {
        "case_id": "wrong_priority",
        "rank_item": _BASE_RANK_ITEM,
        "rationale_text": "412 Elm St: destroyed. Priority is 24.1932 hours of urgency.",
        "expect_violation": True,
        "category": "wrong_priority",
    },
    {
        "case_id": "wrong_staleness",
        "rank_item": _BASE_RANK_ITEM,
        "rationale_text": "The 6.5-hour staleness reduces priority for this building.",
        "expect_violation": True,
        "category": "wrong_staleness",
    },
    {
        "case_id": "wrong_vulnerable_density",
        "rank_item": _BASE_RANK_ITEM,
        "rationale_text": "High vulnerable density of 2.30 people nearby drives urgency.",
        "expect_violation": True,
        "category": "wrong_vulnerable_density",
    },
    {
        "case_id": "wrong_doubt",
        "rank_item": _BASE_RANK_ITEM,
        "rationale_text": "There is a 12% chance of further damage given current doubt.",
        "expect_violation": True,
        "category": "wrong_doubt",
    },
    {
        "case_id": "wrong_road_cutoff_semantics",
        "rank_item": _BASE_RANK_ITEM,
        "rationale_text": "Access is constrained: 1.8 miles blocked on the approach road.",
        "expect_violation": True,
        "category": "wrong_road_cutoff",
    },
    {
        "case_id": "invented_facility",
        "rank_item": _BASE_RANK_ITEM,  # facility_near is None
        "rationale_text": "412 Elm St, 300 m from Seattle General Hospital, is destroyed.",
        "expect_violation": True,
        "category": "invented_facility",  # KNOWN GAP -- see module docstring
    },
    {
        "case_id": "wrong_damage_class",
        "rank_item": _BASE_RANK_ITEM,  # damage_class is 3 (destroyed)
        "rationale_text": "412 Elm St shows only minor damage and can wait.",
        "expect_violation": True,
        "category": "wrong_damage_class",  # KNOWN GAP -- see module docstring
    },
    {
        "case_id": "forbidden_content_casualties",
        "rank_item": _BASE_RANK_ITEM,
        "rationale_text": "An estimated 3 casualties are believed present at 412 Elm St.",
        "expect_violation": True,
        "category": "forbidden_content",
    },
]


def evaluate_rationale_faithfulness() -> dict:
    """Run every RATIONALE_FIXTURE case through the real
    nano_client._faithfulness_violations and compare against the labeled
    expectation. status is "pass" only if EVERY case matched its label
    (100% of violations detected AND every valid case accepted, Part 3's
    exact pass criterion); "fail" otherwise -- including the two documented
    known-gap categories, reported honestly rather than excluded.
    """
    cases = []
    all_correct = True
    for case in RATIONALE_FIXTURE:
        violations = _faithfulness_violations(case["rationale_text"], case["rank_item"])
        detected = bool(violations)
        correct = detected == case["expect_violation"]
        if not correct:
            all_correct = False
        cases.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "expected_violation": case["expect_violation"],
                "detected_violation": detected,
                "correct": correct,
                "violations": violations,
            }
        )

    accuracy = sum(1 for c in cases if c["correct"]) / len(cases)

    return make_metric(
        name="rationale_faithfulness",
        status=STATUS_PASS if all_correct else STATUS_FAIL,
        value=accuracy,
        threshold="100% of fixture violations detected and all valid cases accepted",
        sample_count=len(cases),
        details={"cases": cases},
    )
