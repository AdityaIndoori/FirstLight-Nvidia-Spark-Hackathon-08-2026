"""Small multi-building Lightning baseline/benchmark.

Runs the EXISTING, unchanged k=8 ballot (lightning_ballot.request_lightning_ballot)
over FIXTURE_BUILDINGS and collects timing + agreement statistics. Vote
counting, voted_class, vote_agreement, and doubt are never recomputed here --
every ballot result comes straight out of request_lightning_ballot().

FIXTURE_BUILDINGS: five deterministic INTERNAL building contexts, using only
the existing ballot input fields (grader_class, grader_confidence,
vl_caption, footprint_area_m2, facility_context, neighbor_damage_classes)
defined in lightning_ballot.py -- no new public contract fields, not the
frozen A -> B RankItem contract. grader_class/grader_confidence/vl_caption
stand in for the VL model's (Nemotron Nano 12B v2, :8002) two independent
observations; this baseline does not call the VL model itself.
"""

import time

from backend.decision.lightning_ballot import K_VOTES, request_lightning_ballot
from backend.decision.lightning_client import LightningSeverityClient, StubLightningSeverityClient

FIXTURE_BUILDINGS = [
    {
        "name": "obvious_no_damage",
        "context": {
            "grader_class": 0,
            "grader_confidence": 0.97,
            "vl_caption": "Building appears fully intact with no visible structural damage.",
            "footprint_area_m2": 140.0,
            "facility_context": None,
            "neighbor_damage_classes": [0, 0, 0],
        },
    },
    {
        "name": "likely_minor_damage",
        "context": {
            "grader_class": 1,
            "grader_confidence": 0.82,
            "vl_caption": "Minor roof shingle damage visible; walls and structure appear intact.",
            "footprint_area_m2": 95.0,
            "facility_context": "200m from Lakeside School",
            "neighbor_damage_classes": [0, 1, 1],
        },
    },
    {
        "name": "likely_major_damage",
        "context": {
            "grader_class": 2,
            "grader_confidence": 0.78,
            "vl_caption": "Partial roof collapse and visible structural cracking along the north wall.",
            "footprint_area_m2": 210.0,
            "facility_context": "60m from Riverside Clinic",
            "neighbor_damage_classes": [2, 2, 3],
        },
    },
    {
        "name": "likely_destroyed",
        "context": {
            "grader_class": 3,
            "grader_confidence": 0.93,
            "vl_caption": "Structure has collapsed; only rubble and foundation remnants are visible.",
            "footprint_area_m2": 180.0,
            "facility_context": "30m from Fire Station 4",
            "neighbor_damage_classes": [3, 3, 2],
        },
    },
    {
        # Deliberately conflicting evidence (low grader confidence, a caption
        # that describes worse damage than the grade, wildly split neighbor
        # classes) -- plausibly lower agreement, but the real model is
        # stochastic, so nothing here asserts it must disagree.
        "name": "ambiguous_conflicting",
        "context": {
            "grader_class": 1,
            "grader_confidence": 0.41,
            "vl_caption": (
                "Roofline appears displaced and a large debris field is visible, "
                "though the primary grader marked only minor damage."
            ),
            "footprint_area_m2": 150.0,
            "facility_context": "adjacent to a collapsed overpass",
            "neighbor_damage_classes": [0, 3, 1, 2],
        },
    },
]


class BuildingBallotError(Exception):
    """Raised by run_lightning_baseline when one building's ballot fails, so
    the failure is surfaced clearly instead of that building being silently
    dropped from the results. building_name identifies which fixture failed;
    the original exception is chained via __cause__.
    """

    def __init__(self, building_name: str, original_exc: Exception):
        self.building_name = building_name
        self.original_exc = original_exc
        super().__init__(f"ballot failed for building {building_name!r}: {original_exc}")


def run_lightning_baseline(client: LightningSeverityClient = None, buildings: list = None) -> dict:
    """Run the existing k=8 ballot over each fixture building (default:
    FIXTURE_BUILDINGS) and collect per-building results plus aggregate
    timing/agreement statistics.

    Raises BuildingBallotError immediately if any one building's ballot
    fails -- never silently omits a failed building.

    Returns {
        "results": [{"name": str, "votes": [...], "voted_class": int,
                      "vote_agreement": float, "doubt": float}, ...],
        "buildings_processed": int,
        "total_generations": int,
        "elapsed_s": float,
        "avg_s_per_generation": float,
        "mean_vote_agreement": float,
        "min_vote_agreement": float,
        "max_vote_agreement": float,
        # present only if the client actually recorded token usage:
        "total_prompt_tokens": int,
        "total_completion_tokens": int,
        "tokens_per_sec": float,
    }
    """
    active_client = client if client is not None else StubLightningSeverityClient()
    fixtures = buildings if buildings is not None else FIXTURE_BUILDINGS

    results = []
    start = time.monotonic()
    for fixture in fixtures:
        try:
            ballot = request_lightning_ballot(fixture["context"], client=active_client)
        except Exception as exc:
            raise BuildingBallotError(fixture["name"], exc) from exc
        results.append({"name": fixture["name"], **ballot})
    elapsed_s = time.monotonic() - start

    buildings_processed = len(results)
    total_generations = buildings_processed * K_VOTES
    agreements = [r["vote_agreement"] for r in results]

    summary = {
        "results": results,
        "buildings_processed": buildings_processed,
        "total_generations": total_generations,
        "elapsed_s": elapsed_s,
        "avg_s_per_generation": elapsed_s / total_generations if total_generations else 0.0,
        "mean_vote_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
        "min_vote_agreement": min(agreements) if agreements else None,
        "max_vote_agreement": max(agreements) if agreements else None,
    }

    usage_log = getattr(active_client, "usage_log", None)
    if usage_log:
        total_prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage_log if u)
        total_completion_tokens = sum(u.get("completion_tokens", 0) for u in usage_log if u)
        summary["total_prompt_tokens"] = total_prompt_tokens
        summary["total_completion_tokens"] = total_completion_tokens
        summary["tokens_per_sec"] = (
            (total_prompt_tokens + total_completion_tokens) / elapsed_s if elapsed_s > 0 else 0.0
        )

    return summary
