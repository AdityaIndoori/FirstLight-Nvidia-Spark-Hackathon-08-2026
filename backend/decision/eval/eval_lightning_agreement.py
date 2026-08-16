"""B8 Parts C1 (Lightning self-agreement) and C2 (Lightning-vs-Nano
agreement).

Reuses the real B3 k=8 ballot (lightning_ballot.request_lightning_ballot)
and lightning_client's real/stub clients -- no ballot logic is
reimplemented here.

--------------------------------------------------------------------------
C1 -- offline vs live
--------------------------------------------------------------------------
Offline mode runs the SAME real request_lightning_ballot() aggregation
code, but through StubLightningSeverityClient with FIXED, varied
per-fixture vote lists (not real model stochasticity) -- this is
meaningful to measure offline because it exercises the actual agreement/
doubt ARITHMETIC across a distribution of vote patterns (unanimous,
6-of-8, ties, 3-way splits), not because it says anything about a real
model's self-consistency. Labeled "measured" with an explicit "offline
(stub vote patterns)" note in details so it is never mistaken for real
Lightning behavior. Live mode calls the real k=8 ballot against
localhost:8001 for the same fixtures and reports genuine self-agreement.

--------------------------------------------------------------------------
C2 -- offline vs live (asymmetric on purpose)
--------------------------------------------------------------------------
C2 has NO meaningful offline analog: comparing a Lightning stub to a Nano
stub would be circular, since both stubs in this repository derive their
answer from building_context["grader_class"] -- guaranteeing fabricated
100% "agreement" that says nothing about the real models. Per Part 12
("do not fake a passing score"), offline mode reports C2 as DEFERRED with
that reasoning made explicit, never as a fake measured number. Live mode
independently asks the real Lightning and real Nano
(model_class_adapters.nano_assess_damage_class, this package's EVAL-ONLY
adapter) for a class in {0,1,2,3} from the literal same prompt text
(lightning_client._ballot_prompt), and reports the genuine agreement rate
plus mean absolute class difference.

INTERPRETATION USED for "Lightning-versus-Nano agreement" (the README's
own wording is ambiguous about what exactly is compared): this module
asks each model INDEPENDENTLY for its own damage-class opinion of the same
textual evidence Lightning's k=8 ballot already grades (grader
class/confidence, VL caption, footprint/facility/neighbor context) --
NOT Lightning's aggregated ballot vote vs. Nano's rationale text, and NOT
either model's opinion of a DIFFERENT input. Neither call ever touches or
overrides the authoritative damage_class/RankItem/BuildingEvidence
contract; this is evaluation-only.
"""

import statistics
import time

from backend.decision.eval.model_class_adapters import nano_assess_damage_class
from backend.decision.eval.report import STATUS_MEASURED, deferred_metric, make_metric
from backend.decision.lightning_ballot import request_lightning_ballot
from backend.decision.lightning_client import (
    LightningClientError,
    RealLightningSeverityClient,
    StubLightningSeverityClient,
)
from backend.decision.nano_client import NanoClientError

# --------------------------------------------------------------------------
# Fixture: 10 BuildingEvidence-shaped building_context examples spanning
# undamaged/minor/major/destroyed, contradictory caption/grade, facility
# context, and neighboring-damage variation (Part 5's exact spread).
# --------------------------------------------------------------------------

AGREEMENT_FIXTURE = [
    {
        "case_id": "undamaged",
        "building_context": {
            "grader_class": 0,
            "grader_confidence": 0.95,
            "vl_caption": "Single-family home shows no visible damage.",
            "footprint_area_m2": 140.0,
            "facility_context": None,
            "neighbor_damage_classes": [0, 0, 1],
        },
        "stub_votes": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    {
        "case_id": "minor",
        "building_context": {
            "grader_class": 1,
            "grader_confidence": 0.8,
            "vl_caption": "Minor cracking visible on the exterior wall.",
            "footprint_area_m2": 110.0,
            "facility_context": None,
            "neighbor_damage_classes": [1, 0, 1],
        },
        "stub_votes": [1, 1, 1, 1, 1, 1, 1, 1],
    },
    {
        "case_id": "major",
        "building_context": {
            "grader_class": 2,
            "grader_confidence": 0.85,
            "vl_caption": "Significant structural damage with a partially collapsed wall.",
            "footprint_area_m2": 200.0,
            "facility_context": None,
            "neighbor_damage_classes": [2, 2, 1],
        },
        "stub_votes": [2, 2, 2, 2, 2, 2, 1, 1],
    },
    {
        "case_id": "destroyed",
        "building_context": {
            "grader_class": 3,
            "grader_confidence": 0.93,
            "vl_caption": "Structure fully collapsed and engulfed in flames.",
            "footprint_area_m2": 160.0,
            "facility_context": None,
            "neighbor_damage_classes": [3, 2, 3],
        },
        "stub_votes": [3, 3, 3, 3, 3, 3, 3, 3],
    },
    {
        "case_id": "contradictory_high_grade_low_caption",
        "building_context": {
            "grader_class": 3,
            "grader_confidence": 0.6,
            "vl_caption": "Building appears intact with no visible damage.",
            "footprint_area_m2": 130.0,
            "facility_context": None,
            "neighbor_damage_classes": [0, 1, 0],
        },
        "stub_votes": [3, 3, 3, 2, 2, 1, 1, 0],
    },
    {
        "case_id": "contradictory_low_grade_high_caption",
        "building_context": {
            "grader_class": 0,
            "grader_confidence": 0.55,
            "vl_caption": "Roof has fully collapsed onto the structure below.",
            "footprint_area_m2": 150.0,
            "facility_context": None,
            "neighbor_damage_classes": [3, 3, 2],
        },
        "stub_votes": [0, 0, 0, 1, 1, 2, 2, 3],
    },
    {
        "case_id": "facility_context",
        "building_context": {
            "grader_class": 2,
            "grader_confidence": 0.82,
            "vl_caption": "Dialysis center exterior shows significant structural damage.",
            "footprint_area_m2": 220.0,
            "facility_context": "0 m from Riverside Dialysis Center (dialysis)",
            "neighbor_damage_classes": [1, 2, 2],
        },
        "stub_votes": [2, 2, 2, 2, 2, 1, 1, 1],
    },
    {
        "case_id": "neighbor_variation_isolated_severe",
        "building_context": {
            "grader_class": 3,
            "grader_confidence": 0.88,
            "vl_caption": "Structure destroyed while surrounding buildings appear intact.",
            "footprint_area_m2": 170.0,
            "facility_context": None,
            "neighbor_damage_classes": [0, 0, 0],
        },
        "stub_votes": [3, 3, 3, 3, 2, 2, 2, 2],
    },
    {
        "case_id": "neighbor_variation_surrounded_by_damage",
        "building_context": {
            "grader_class": 1,
            "grader_confidence": 0.7,
            "vl_caption": "Minor visible damage, though neighboring structures are heavily damaged.",
            "footprint_area_m2": 100.0,
            "facility_context": None,
            "neighbor_damage_classes": [3, 3, 3],
        },
        "stub_votes": [1, 1, 1, 1, 1, 0, 0, 0],
    },
    {
        "case_id": "borderline_low_confidence",
        "building_context": {
            "grader_class": 1,
            "grader_confidence": 0.55,
            "vl_caption": "Ambiguous damage; hard to determine severity from this angle.",
            "footprint_area_m2": 120.0,
            "facility_context": None,
            "neighbor_damage_classes": [1, 2, 0],
        },
        "stub_votes": [1, 1, 1, 1, 1, 0, 0, 0],
    },
]


# --------------------------------------------------------------------------
# C1 -- Lightning self-agreement
# --------------------------------------------------------------------------


def evaluate_self_agreement_offline() -> dict:
    per_case = []
    agreements = []
    for case in AGREEMENT_FIXTURE:
        client = StubLightningSeverityClient(votes=case["stub_votes"])
        ballot = request_lightning_ballot(case["building_context"], client=client)
        expected_doubt = max(0.05, 1 - ballot["vote_agreement"])
        doubt_correct = abs(ballot["doubt"] - expected_doubt) < 1e-9
        agreements.append(ballot["vote_agreement"])
        per_case.append(
            {
                "case_id": case["case_id"],
                "votes": ballot["votes"],
                "voted_class": ballot["voted_class"],
                "vote_agreement": ballot["vote_agreement"],
                "doubt": ballot["doubt"],
                "doubt_formula_correct": doubt_correct,
            }
        )

    return make_metric(
        name="lightning_self_agreement",
        status=STATUS_MEASURED,
        value=statistics.mean(agreements),
        threshold=None,
        sample_count=len(agreements),
        details={
            "mode": (
                "offline: deterministic StubLightningSeverityClient vote patterns -- "
                "exercises the real aggregation/doubt arithmetic across a varied vote "
                "distribution, NOT real Lightning self-consistency. See --live."
            ),
            "mean": statistics.mean(agreements),
            "median": statistics.median(agreements),
            "min": min(agreements),
            "max": max(agreements),
            "doubt_formula_verified": all(c["doubt_formula_correct"] for c in per_case),
            "per_case": per_case,
        },
    )


def evaluate_self_agreement_live(base_url: str = None, timeout_s: float = 10.0) -> dict:
    """Real k=8 Lightning ballot per fixture against the live server. A
    per-sample LightningClientError is recorded and that sample is
    skipped -- one failing building never aborts the rest of the eval.
    """
    client = RealLightningSeverityClient(base_url=base_url, timeout_s=timeout_s)
    per_case = []
    agreements = []
    failures = 0

    started_at = time.perf_counter()
    for case in AGREEMENT_FIXTURE:
        try:
            ballot = request_lightning_ballot(case["building_context"], client=client)
            expected_doubt = max(0.05, 1 - ballot["vote_agreement"])
            agreements.append(ballot["vote_agreement"])
            per_case.append(
                {
                    "case_id": case["case_id"],
                    "votes": ballot["votes"],
                    "voted_class": ballot["voted_class"],
                    "vote_agreement": ballot["vote_agreement"],
                    "doubt": ballot["doubt"],
                    "doubt_formula_correct": abs(ballot["doubt"] - expected_doubt) < 1e-9,
                }
            )
        except LightningClientError as exc:
            failures += 1
            per_case.append({"case_id": case["case_id"], "error": str(exc)})
    elapsed_s = time.perf_counter() - started_at

    if not agreements:
        return deferred_metric(
            "lightning_self_agreement_live",
            f"all {len(AGREEMENT_FIXTURE)} live Lightning ballot calls failed -- see per_case errors",
        )

    return make_metric(
        name="lightning_self_agreement_live",
        status=STATUS_MEASURED,
        value=statistics.mean(agreements),
        threshold=None,
        sample_count=len(agreements),
        details={
            "mode": "live: real k=8 Lightning ballot against the server",
            "mean": statistics.mean(agreements),
            "median": statistics.median(agreements),
            "min": min(agreements),
            "max": max(agreements),
            "failures": failures,
            "elapsed_s": elapsed_s,
            "per_case": per_case,
        },
    )


# --------------------------------------------------------------------------
# C2 -- Lightning-vs-Nano independent class agreement
# --------------------------------------------------------------------------


def exact_class_agreement_rate(pairs: list) -> float:
    """|{(l, n) in pairs : l == n}| / |pairs|; 0.0 for an empty list."""
    if not pairs:
        return 0.0
    return sum(1 for lightning_class, nano_class in pairs if lightning_class == nano_class) / len(pairs)


def mean_absolute_class_difference(pairs: list) -> float:
    """mean(|l - n| for (l, n) in pairs); 0.0 for an empty list."""
    if not pairs:
        return 0.0
    return sum(abs(lightning_class - nano_class) for lightning_class, nano_class in pairs) / len(pairs)


def evaluate_lightning_vs_nano_offline() -> dict:
    """No offline analog exists -- see module docstring's C2 section for
    why a stub-vs-stub comparison here would be circular/fabricated.
    """
    return deferred_metric(
        "lightning_vs_nano_agreement",
        "requires independent live Nano (:8000) and Lightning (:8001) class assessments; "
        "a stub-vs-stub comparison would be circular (both stubs in this repo derive their "
        "answer from the same grader_class input, guaranteeing a fabricated 100% agreement) "
        "-- not run in offline mode, see --live",
    )


def evaluate_lightning_vs_nano_live(
    lightning_base_url: str = None, nano_base_url: str = None, timeout_s: float = 10.0
) -> dict:
    lightning_client = RealLightningSeverityClient(base_url=lightning_base_url, timeout_s=timeout_s)
    pairs = []
    per_case = []
    failures = 0

    started_at = time.perf_counter()
    for case in AGREEMENT_FIXTURE:
        try:
            lightning_class = lightning_client.sample_severity(case["building_context"], temperature=0.0)
            nano_class = nano_assess_damage_class(case["building_context"], base_url=nano_base_url, timeout_s=timeout_s)
            pairs.append((lightning_class, nano_class))
            per_case.append(
                {
                    "case_id": case["case_id"],
                    "lightning_class": lightning_class,
                    "nano_class": nano_class,
                    "agree": lightning_class == nano_class,
                }
            )
        except (LightningClientError, NanoClientError) as exc:
            failures += 1
            per_case.append({"case_id": case["case_id"], "error": str(exc)})
    elapsed_s = time.perf_counter() - started_at

    if not pairs:
        return deferred_metric(
            "lightning_vs_nano_agreement",
            f"all {len(AGREEMENT_FIXTURE)} live class-assessment calls failed -- see per_case errors",
        )

    return make_metric(
        name="lightning_vs_nano_agreement",
        status=STATUS_MEASURED,
        value=exact_class_agreement_rate(pairs),
        threshold=None,
        sample_count=len(pairs),
        details={
            "interpretation": (
                "each model independently classifies the SAME textual evidence "
                "(lightning_client._ballot_prompt) into {0,1,2,3}; this is NOT Lightning's "
                "aggregated k=8 ballot vote vs. Nano's rationale text, and neither call "
                "touches the authoritative damage_class"
            ),
            "mean_absolute_class_difference": mean_absolute_class_difference(pairs),
            "failures": failures,
            "elapsed_s": elapsed_s,
            "per_case": per_case,
        },
    )
