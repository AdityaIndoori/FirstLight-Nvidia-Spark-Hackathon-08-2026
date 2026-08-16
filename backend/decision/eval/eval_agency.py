"""B8 Part D: agency-plan correctness (precision/recall against a labeled
set) and unit-count sanity.

Reuses B6 verbatim: agency_plan_drafter.draft_agency_plan (with
StubAgencyPlanDraftClient offline -- the SAME deterministic,
evidence-grounded fallback production already falls back to, driven by
agency_plan_client.is_action_supported) and agency_plan's
is_overcommitted/units_shortfall. No grounding rule or accounting formula
is reimplemented here.

Per instructions, this does NOT require model_successes==100% -- a
correct, grounded deterministic-fallback plan is a valid, fully scoreable
result; a separate live metric exists for the real Nano planner.
"""

import time

from backend.decision.agency_plan import AvailabilityRegistry, is_overcommitted, units_shortfall
from backend.decision.agency_plan_client import RealAgencyPlanDraftClient, StubAgencyPlanDraftClient
from backend.decision.agency_plan_drafter import build_planner_candidate, draft_agency_plan_with_diagnostics
from backend.decision.eval.report import STATUS_FAIL, STATUS_MEASURED, STATUS_PASS, make_metric

# --------------------------------------------------------------------------
# Fixture: 10 labeled cases, each an internal planner-candidate (see
# agency_plan_drafter.build_planner_candidate) with its expected supported
# (agency, action) pairs -- captions are written specifically to exercise
# B6's deterministic keyword-grounding rules (agency_plan_client.py).
# --------------------------------------------------------------------------

AGENCY_FIXTURE = [
    {
        "case_id": "visible_fire",
        "candidate": {
            "footprint_id": "fp-eval-1",
            "label": "Eval Building 1",
            "centroid": [-122.390, 47.598],
            "damage_class": 3,
            "confidence": 0.9,
            "confirmed": False,
            "priority": 10.0,
            "vl_caption": "Two-storey structure with visible flames and heavy smoke.",
            "facility_near": None,
        },
        "expected": {("fire", "fire_suppression")},
    },
    {
        "case_id": "collapsed_roof",
        "candidate": {
            "footprint_id": "fp-eval-2",
            "label": "Eval Building 2",
            "centroid": [-122.391, 47.599],
            "damage_class": 3,
            "confidence": 0.85,
            "confirmed": False,
            "priority": 9.0,
            "vl_caption": "Roof has fully collapsed with structural debris around the entrance.",
            "facility_near": None,
        },
        "expected": {("fire", "collapse_response"), ("public_works", "debris_clearance")},
    },
    {
        "case_id": "dialysis_center",
        "candidate": {
            "footprint_id": "fp-eval-3",
            "label": "Eval Building 3",
            "centroid": [-122.385, 47.605],
            "damage_class": 2,
            "confidence": 0.8,
            "confirmed": False,
            "priority": 8.0,
            "vl_caption": "Building has significant exterior damage.",
            "facility_near": {"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0},
        },
        "expected": {("ems", "medical_support")},
    },
    {
        "case_id": "hospital",
        "candidate": {
            "footprint_id": "fp-eval-4",
            "label": "Eval Building 4",
            "centroid": [-122.386, 47.606],
            "damage_class": 2,
            "confidence": 0.8,
            "confirmed": False,
            "priority": 8.0,
            "vl_caption": "Hospital entrance is damaged.",
            "facility_near": {"name": "General Hospital", "type": "hospital", "dist_m": 5},
        },
        "expected": {("ems", "medical_support")},
    },
    {
        "case_id": "nursing_home",
        "candidate": {
            "footprint_id": "fp-eval-5",
            "label": "Eval Building 5",
            "centroid": [-122.386, 47.567],
            "damage_class": 2,
            "confidence": 0.75,
            "confirmed": False,
            "priority": 7.0,
            "vl_caption": "Building shows exterior damage near the nursing home.",
            "facility_near": {"name": "Providence Mount St. Vincent", "type": "nursing_home", "dist_m": 10},
        },
        "expected": {("ems", "medical_support")},
    },
    {
        "case_id": "blocked_roadway",
        "candidate": {
            "footprint_id": "fp-eval-6",
            "label": "Eval Building 6",
            "centroid": [-122.390, 47.598],
            "damage_class": 1,
            "confidence": 0.66,
            "confirmed": False,
            "priority": 3.0,
            "vl_caption": "Damaged commercial structure adjacent to an active road closure.",
            "facility_near": None,
        },
        "expected": {("police", "road_closure")},
    },
    {
        "case_id": "debris_obstruction",
        "candidate": {
            "footprint_id": "fp-eval-7",
            "label": "Eval Building 7",
            "centroid": [-122.395, 47.599],
            "damage_class": 1,
            "confidence": 0.7,
            "confirmed": False,
            "priority": 2.0,
            "vl_caption": "Large debris field obstructs roadway access beside the structure.",
            "facility_near": None,
        },
        "expected": {("public_works", "debris_clearance")},
    },
    {
        "case_id": "ordinary_damage_no_evidence",
        "candidate": {
            "footprint_id": "fp-eval-8",
            "label": "Eval Building 8",
            "centroid": [-122.400, 47.600],
            "damage_class": 1,
            "confidence": 0.6,
            "confirmed": False,
            "priority": 1.0,
            "vl_caption": "Building has moderate exterior damage.",
            "facility_near": None,
        },
        "expected": set(),
    },
    {
        "case_id": "combined_fire_and_road_closure",
        "candidate": {
            "footprint_id": "fp-eval-9",
            "label": "Eval Building 9",
            "centroid": [-122.390, 47.598],
            "damage_class": 3,
            "confidence": 0.88,
            "confirmed": False,
            "priority": 11.0,
            "vl_caption": "Vehicle fire on the roadway; an active road closure is in effect.",
            "facility_near": None,
        },
        "expected": {("fire", "fire_suppression"), ("police", "road_closure")},
    },
    {
        "case_id": "combined_medical_and_blocked_access",
        "candidate": {
            "footprint_id": "fp-eval-10",
            "label": "Eval Building 10",
            "centroid": [-122.385, 47.605],
            "damage_class": 2,
            "confidence": 0.8,
            "confirmed": False,
            "priority": 9.0,
            "vl_caption": "Dialysis center entrance is blocked by debris; access obstructed.",
            "facility_near": {"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0},
        },
        "expected": {("ems", "medical_support"), ("public_works", "debris_clearance")},
    },
]


def _predicted_pairs(plan: dict, footprint_id: str) -> set:
    pairs = set()
    for group in plan["agencies"]:
        for step in group["steps"]:
            if step["footprint_id"] == footprint_id:
                pairs.add((group["agency"], step["task"]))
    return pairs


def _precision_recall(predicted: set, expected: set) -> tuple:
    correct = len(predicted & expected)
    precision = correct / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = correct / len(expected) if expected else (1.0 if not predicted else 0.0)
    return precision, recall


def _make_availability() -> AvailabilityRegistry:
    availability = AvailabilityRegistry()
    for agency in ("fire", "ems", "police", "public_works"):
        availability.set_availability(agency, 5, "eval:b8")
    return availability


def evaluate_agency_plan_correctness_offline() -> dict:
    """Runs every AGENCY_FIXTURE candidate through the real
    draft_agency_plan_with_diagnostics using StubAgencyPlanDraftClient for
    BOTH client and fallback -- exactly the deterministic, grounded
    fallback path production already uses, driven by
    agency_plan_client.is_action_supported. Task text is compared via
    (agency, task) since the drafter maps action codes to fixed task text
    (agency_plan_client._ACTION_TASK_TEXT), not via a private action code.
    """
    from backend.decision.agency_plan_client import _ACTION_TASK_TEXT

    candidates = [build_planner_candidate_from_fixture(case["candidate"]) for case in AGENCY_FIXTURE]
    stub = StubAgencyPlanDraftClient()
    result = draft_agency_plan_with_diagnostics(candidates, _make_availability(), client=stub, fallback_client=stub)
    plan = result["plan"]

    per_case = []
    precisions = []
    recalls = []
    for case in AGENCY_FIXTURE:
        footprint_id = case["candidate"]["footprint_id"]
        expected_tasks = {(agency, _ACTION_TASK_TEXT[action]) for agency, action in case["expected"]}
        predicted = _predicted_pairs(plan, footprint_id)
        precision, recall = _precision_recall(predicted, expected_tasks)
        precisions.append(precision)
        recalls.append(recall)
        per_case.append(
            {
                "case_id": case["case_id"],
                "expected": sorted(expected_tasks),
                "predicted": sorted(predicted),
                "precision": precision,
                "recall": recall,
            }
        )

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)

    return make_metric(
        name="agency_plan_precision_recall",
        status=STATUS_MEASURED,
        value={"precision": mean_precision, "recall": mean_recall},
        threshold=None,
        sample_count=len(AGENCY_FIXTURE),
        details={"mode": "offline: deterministic StubAgencyPlanDraftClient", "per_case": per_case},
    )


def build_planner_candidate_from_fixture(candidate: dict) -> dict:
    """AGENCY_FIXTURE candidates are already in the planner-candidate shape
    (build_planner_candidate's own output shape) -- this is an identity
    pass-through kept as a named function so the intent (these ARE
    planner candidates, not raw ProcessedBuildingEvidence) is explicit at
    the call site.
    """
    return dict(candidate)


def evaluate_agency_unit_sanity_offline() -> dict:
    """Hard deterministic gate (Part 12: "unit accounting mismatch: 0"),
    not a quality metric -- runs the same fixture through the real B6a
    accounting (agency_plan.build_agency_plan via draft_agency_plan,
    is_overcommitted, units_shortfall) and checks every invariant:
    every step's units > 0, units_required == sum(step units),
    units_available always exactly what AvailabilityRegistry supplied
    (never model-derived), and overcommitment/shortfall arithmetic correct.
    """
    candidates = [build_planner_candidate_from_fixture(case["candidate"]) for case in AGENCY_FIXTURE]
    availability = _make_availability()
    stub = StubAgencyPlanDraftClient()
    plan = draft_agency_plan_with_diagnostics(candidates, availability, client=stub, fallback_client=stub)["plan"]

    violations = []
    for group in plan["agencies"]:
        for step in group["steps"]:
            if not isinstance(step["units"], int) or step["units"] <= 0:
                violations.append(f"{group['agency']} step {step['n']}: units={step['units']!r} is not > 0")

        expected_required = sum(step["units"] for step in group["steps"])
        if group["units_required"] != expected_required:
            violations.append(
                f"{group['agency']}: units_required={group['units_required']} != "
                f"sum(step units)={expected_required}"
            )

        if group["units_available"] != availability.get_availability(group["agency"]):
            violations.append(
                f"{group['agency']}: units_available={group['units_available']} does not match "
                f"AvailabilityRegistry ({availability.get_availability(group['agency'])})"
            )

        expected_overcommitted = group["units_required"] > group["units_available"]
        if is_overcommitted(group) != expected_overcommitted:
            violations.append(f"{group['agency']}: is_overcommitted disagrees with the raw comparison")

        expected_shortfall = max(0, group["units_required"] - group["units_available"])
        if units_shortfall(group) != expected_shortfall:
            violations.append(f"{group['agency']}: units_shortfall={units_shortfall(group)} != {expected_shortfall}")

    return make_metric(
        name="agency_unit_accounting_sanity",
        status=STATUS_PASS if not violations else STATUS_FAIL,
        value=len(violations),
        threshold="0 accounting mismatches",
        sample_count=len(AGENCY_FIXTURE),
        details={"violations": violations},
    )


def evaluate_agency_plan_correctness_live(base_url: str = None, timeout_s: float = 10.0) -> dict:
    """Same fixture/labels, real RealAgencyPlanDraftClient with the SAME
    stub fallback (production behavior: per-building recovery, some
    buildings may legitimately fall back) -- model_successes==100% is
    explicitly NOT required; diagnostics report the real split honestly.
    """
    candidates = [build_planner_candidate_from_fixture(case["candidate"]) for case in AGENCY_FIXTURE]
    real_client = RealAgencyPlanDraftClient(base_url=base_url, timeout_s=timeout_s) if base_url else RealAgencyPlanDraftClient(timeout_s=timeout_s)
    fallback = StubAgencyPlanDraftClient()

    from backend.decision.agency_plan_client import _ACTION_TASK_TEXT

    started_at = time.perf_counter()
    result = draft_agency_plan_with_diagnostics(
        candidates, _make_availability(), client=real_client, fallback_client=fallback
    )
    elapsed_s = time.perf_counter() - started_at
    plan = result["plan"]
    diagnostics = result["diagnostics"]

    per_case = []
    precisions = []
    recalls = []
    for case in AGENCY_FIXTURE:
        footprint_id = case["candidate"]["footprint_id"]
        expected_tasks = {(agency, _ACTION_TASK_TEXT[action]) for agency, action in case["expected"]}
        predicted = _predicted_pairs(plan, footprint_id)
        precision, recall = _precision_recall(predicted, expected_tasks)
        precisions.append(precision)
        recalls.append(recall)
        per_case.append({"case_id": case["case_id"], "expected": sorted(expected_tasks), "predicted": sorted(predicted)})

    return make_metric(
        name="agency_plan_precision_recall_live",
        status=STATUS_MEASURED,
        value={"precision": sum(precisions) / len(precisions), "recall": sum(recalls) / len(recalls)},
        threshold=None,
        sample_count=len(AGENCY_FIXTURE),
        details={
            "mode": "live: real Nano agency planner, per-building recovery (fallback is valid, not a failure)",
            "model_building_count": diagnostics["model_building_count"],
            "fallback_building_count": diagnostics["fallback_building_count"],
            "elapsed_s": elapsed_s,
            "per_case": per_case,
        },
    )
