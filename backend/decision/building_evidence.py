"""B-side BuildingEvidence processing path.

    BuildingEvidence (frozen A -> B contract)
        -> validate at the B boundary
        -> existing Lightning k=8 ballot (real, falling back to the
           deterministic stub on failure)
        -> vote_agreement + doubt
        -> existing B1 calculate_priority (unchanged, reused)

Member A owns production of BuildingEvidence -- Member B never calls the VL
model, never reads image pixels, never redoes GIS joins, and never invents
missing evidence. This module only validates, maps, and scores.

The frozen BuildingEvidence contract (A -> B):
    {
        footprint_id: str,
        image_id: str,
        label: str,
        centroid: [lng, lat],
        captured_at: float,
        damage_class: 0-3,
        confidence: 0.0-1.0,
        graded_by: str,
        vl_caption: str,
        footprint_area_m2: float,
        facility_near: {name: str, type: "nursing_home"|"dialysis"|"hospital",
                         dist_m: int} or null,
        neighbor_damage_classes: [0-3],
        vulnerable_density: float,
    }

ProcessedBuildingEvidence is an INTERNAL result shape only -- it does not
change the public RankItem contract, and voted_class/vote_agreement are not
added to RankItem here:
    {
        evidence: BuildingEvidence,
        votes: [8 ints],
        voted_class: int,
        vote_agreement: float,
        doubt: float,
        staleness_h: float,
        road_cutoff: float or None,
        priority: float,
        lightning_recovery: "model" | "stub",
    }
"""

from backend.decision.lightning_ballot import request_lightning_ballot
from backend.decision.lightning_client import (
    LightningClientError,
    LightningSeverityClient,
    RealLightningSeverityClient,
    StubLightningSeverityClient,
)
from backend.decision.scoring import calculate_priority

_VALID_SEVERITY_LABELS = (0, 1, 2, 3)
_VALID_FACILITY_TYPES = ("nursing_home", "dialysis", "hospital")


class BuildingEvidenceError(ValueError):
    """Raised when BuildingEvidence fails the frozen A -> B contract
    validation. A contract failure, not a model failure -- process_building_evidence
    raises this BEFORE touching Lightning, so a bad evidence record never
    triggers (or hides behind) the Lightning fallback path.
    """


def validate_building_evidence(evidence: dict) -> list:
    """Validate evidence against the frozen A -> B BuildingEvidence contract.

    Returns a list of human-readable error strings; empty means valid. Never
    corrects, coerces, or defaults a bad value -- only reports it.
    """
    errors = []

    if not isinstance(evidence.get("footprint_id"), str) or not evidence.get("footprint_id"):
        errors.append("footprint_id must be a non-empty string")

    if not isinstance(evidence.get("image_id"), str) or not evidence.get("image_id"):
        errors.append("image_id must be a non-empty string")

    centroid = evidence.get("centroid")
    if (
        not isinstance(centroid, (list, tuple))
        or len(centroid) != 2
        or not all(isinstance(c, (int, float)) for c in centroid)
    ):
        errors.append("centroid must be exactly [lng, lat]")

    if not isinstance(evidence.get("captured_at"), (int, float)):
        errors.append("captured_at must be a number (unix timestamp)")

    damage_class = evidence.get("damage_class")
    if damage_class not in _VALID_SEVERITY_LABELS:
        errors.append(f"damage_class must be one of {_VALID_SEVERITY_LABELS}, got {damage_class!r}")

    confidence = evidence.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        errors.append(f"confidence must be in [0.0, 1.0], got {confidence!r}")

    if not isinstance(evidence.get("graded_by"), str) or not evidence.get("graded_by"):
        errors.append("graded_by must be a non-empty string")

    if not isinstance(evidence.get("vl_caption"), str) or not evidence.get("vl_caption"):
        errors.append("vl_caption must be a non-empty string")

    footprint_area_m2 = evidence.get("footprint_area_m2")
    if not isinstance(footprint_area_m2, (int, float)) or footprint_area_m2 < 0:
        errors.append(f"footprint_area_m2 must be >= 0, got {footprint_area_m2!r}")

    neighbor_damage_classes = evidence.get("neighbor_damage_classes")
    if not isinstance(neighbor_damage_classes, list) or not all(
        c in _VALID_SEVERITY_LABELS for c in neighbor_damage_classes
    ):
        errors.append(f"neighbor_damage_classes must be a list of values in {_VALID_SEVERITY_LABELS}")

    vulnerable_density = evidence.get("vulnerable_density")
    if not isinstance(vulnerable_density, (int, float)) or vulnerable_density < 0:
        errors.append(f"vulnerable_density must be >= 0, got {vulnerable_density!r}")

    facility_near = evidence.get("facility_near")
    if facility_near is not None:
        if not isinstance(facility_near, dict):
            errors.append("facility_near must be null or an object")
        else:
            if not isinstance(facility_near.get("name"), str) or not facility_near.get("name"):
                errors.append("facility_near.name must be a non-empty string")
            if facility_near.get("type") not in _VALID_FACILITY_TYPES:
                errors.append(
                    f"facility_near.type must be one of {_VALID_FACILITY_TYPES}, "
                    f"got {facility_near.get('type')!r}"
                )
            dist_m = facility_near.get("dist_m")
            if not isinstance(dist_m, int) or dist_m < 0:
                errors.append(f"facility_near.dist_m must be an int >= 0, got {dist_m!r}")

    return errors


def compute_staleness_h(captured_at: float, scored_at: float) -> float:
    """staleness_h = max(0.0, (scored_at - captured_at) / 3600.0).

    scored_at is an explicit caller-supplied parameter -- this never calls
    time.time() itself, so callers (and tests) are fully deterministic.
    """
    return max(0.0, (scored_at - captured_at) / 3600.0)


def _to_lightning_building_context(evidence: dict) -> dict:
    """Map validated BuildingEvidence onto the existing Lightning ballot's
    internal building_context shape (lightning_ballot.py). A pure, read-only
    mapping -- never mutates evidence, never calls the VL model, never redoes
    the GIS join (facility_context is derived from the already-joined
    evidence.facility_near, not recomputed).
    """
    facility_near = evidence.get("facility_near")
    facility_context = (
        f"{facility_near['dist_m']} m from {facility_near['name']} ({facility_near['type']})"
        if facility_near is not None
        else None
    )
    return {
        "grader_class": evidence["damage_class"],
        "grader_confidence": evidence["confidence"],
        "vl_caption": evidence["vl_caption"],
        "footprint_area_m2": evidence["footprint_area_m2"],
        "facility_context": facility_context,
        "neighbor_damage_classes": list(evidence["neighbor_damage_classes"]),
    }


def process_building_evidence(
    evidence: dict,
    scored_at: float,
    road_cutoff: float = None,
    real_client: LightningSeverityClient = None,
    fallback_client: LightningSeverityClient = None,
) -> dict:
    """Run the full B-side BuildingEvidence processing path for one building.

    1. Validate evidence against the frozen contract; raise
       BuildingEvidenceError on failure. This is a contract check, not a
       model call -- Lightning is never invoked for invalid evidence.
    2. Validate road_cutoff (None, or >= 1); raise ValueError on failure,
       again before any Lightning call.
    3. Map evidence onto the existing Lightning ballot input and run the
       existing, unchanged k=8 ballot: try the real client first: on any
       LightningClientError, fall back to the deterministic stub through the
       same LightningSeverityClient interface, and record which path
       produced the result as lightning_recovery ("model" | "stub") --
       never silently claiming the real model ran.
    4. Compute staleness_h from evidence["captured_at"] and scored_at.
    5. Feed staleness_h, evidence["vulnerable_density"], Lightning's doubt,
       and road_cutoff into the EXISTING, unmodified
       scoring.calculate_priority -- the arithmetic is never duplicated here.

    Returns the INTERNAL ProcessedBuildingEvidence dict described in this
    module's docstring. Does not mutate evidence and does not touch the
    public RankItem contract.
    """
    errors = validate_building_evidence(evidence)
    if errors:
        raise BuildingEvidenceError("; ".join(errors))

    if road_cutoff is not None and road_cutoff < 1:
        raise ValueError(f"road_cutoff must be None or >= 1, got {road_cutoff!r}")

    building_context = _to_lightning_building_context(evidence)

    active_real_client = real_client if real_client is not None else RealLightningSeverityClient()
    active_fallback_client = fallback_client if fallback_client is not None else StubLightningSeverityClient()

    try:
        ballot = request_lightning_ballot(building_context, client=active_real_client)
        lightning_recovery = "model"
    except LightningClientError:
        ballot = request_lightning_ballot(building_context, client=active_fallback_client)
        lightning_recovery = "stub"

    staleness_h = compute_staleness_h(evidence["captured_at"], scored_at)

    priority = calculate_priority(
        staleness_h=staleness_h,
        vulnerable_density=evidence["vulnerable_density"],
        doubt=ballot["doubt"],
        road_cutoff=road_cutoff,
    )

    return {
        "evidence": evidence,
        "votes": ballot["votes"],
        "voted_class": ballot["voted_class"],
        "vote_agreement": ballot["vote_agreement"],
        "doubt": ballot["doubt"],
        "staleness_h": staleness_h,
        "road_cutoff": road_cutoff,
        "priority": priority,
        "lightning_recovery": lightning_recovery,
    }
