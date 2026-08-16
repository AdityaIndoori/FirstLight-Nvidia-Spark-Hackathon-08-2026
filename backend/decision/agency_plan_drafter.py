"""B6b: real Nemotron Nano agency-plan drafting, PER BUILDING, with bounded
parallelism, orchestrated on top of the existing B6a domain layer
(agency_plan.py).

    candidates (built from ProcessedBuildingEvidence, see
    build_planner_candidate)
        -> for EACH candidate, concurrently (bounded by max_concurrency):
             client.propose_assignments_for_building(candidate)   (attempt 1)
             -> validate
                 valid   -> normalize -> provenance "nano"
                 invalid -> client.propose_assignments_for_building(candidate,
                            validation_error=...)                 (attempt 2)
                     valid   -> normalize -> provenance "nano"
                     invalid/NanoClientError/timeout
                             -> fallback_client.propose_assignments_for_building(candidate)
                             -> normalize -> provenance "stub"
        -> merge all buildings' assignments, in original candidate order
        -> build_agency_plan(all_assignments, drafted_by=..., availability)

This is a straight replacement of an earlier whole-plan-in-one-request
strategy (see agency_plan_client.py's module docstring for the measured
reason: the whole-plan request's completion length was too variable to
reliably fit the 10s production timeout). Recovery is now PER BUILDING: one
building's model failure never discards another building's successful
result, and the mix is tracked in draft_agency_plan_with_diagnostics()'s
diagnostics.

B6a (agency_plan.py) remains solely responsible for: all four agency
groups, step numbering, units_required, units_available, contract
validation, and overcommitment -- this module never recomputes any of that,
and never lets the model supply units_available.

Identity/location protection: the compact per-building schema
(agency_plan_client._ASSIGNMENTS_JSON_SCHEMA) never asks the model for
footprint_id, label, centroid, or free-form task text at all -- only
agency/action/units, for the ONE building the request was already scoped
to. After parsing, _normalize_assignments attaches that SAME candidate's
authoritative footprint_id/label/centroid and maps the chosen action to
fixed task text (agency_plan_client._ACTION_TASK_TEXT); the model has no
way to supply or override any of that.

Evidence grounding: _validate_assignments rejects any action the
candidate's own evidence doesn't support (agency_plan_client.is_action_supported)
-- e.g. Nano choosing fire_suppression for a building whose only evidence
is dialysis-facility damage is a validation failure, triggering the same
one-reprompt-then-fallback recovery as any other invalid response, not a
silently accepted assignment.

CONTRACT GAP (not resolved here): the frozen AgencyPlan drafted_by field is
typed as a plain str, and the established values in use are "nano", "stub",
and "operator:<name>" -- there is no agreed value for "some buildings came
from Nano, some from the deterministic fallback," which is now a normal,
expected outcome of per-building recovery. Per instructions, we do NOT
invent a new public value for this. draft_agency_plan_with_diagnostics()
sets the overall drafted_by to "nano" only if EVERY building succeeded via
the real model, and "stub" otherwise -- the conservative choice, since it
never overclaims that every assignment came from the real model. The true
per-building split is always available via
diagnostics["model_building_count"] / diagnostics["fallback_building_count"]
/ diagnostics["buildings"], for whichever caller (the live-check script
today) wants to report it honestly. This should be revisited when B/C
contracts are next frozen.
"""

import concurrent.futures
import time

from backend.decision.agency_plan import SUPPORTED_AGENCIES, AvailabilityRegistry, build_agency_plan
from backend.decision.agency_plan_client import (
    _ACTION_TASK_TEXT,
    _ACTION_EVIDENCE_DESCRIPTION,
    _VALID_ACTIONS,
    _VALID_AGENCY_ACTIONS,
    AgencyPlanDraftClient,
    RealAgencyPlanDraftClient,
    StubAgencyPlanDraftClient,
    is_action_supported,
)
from backend.decision.agency_plan_diagnostics import categorize_client_error, categorize_validation_error
from backend.decision.nano_client import NanoClientError

DEFAULT_MAX_CONCURRENCY = 4
"""Bounded concurrency for per-building drafting requests. The demo fixture
has only a handful of candidates; this is never meant to spawn uncontrolled
parallel requests -- see draft_agency_plan_with_diagnostics(), which clamps
it to at most len(candidates) workers regardless of the value passed in.
"""


def build_planner_candidate(processed_evidence: dict, confirmed: bool = False) -> dict:
    """INTERNAL ONLY. Extract exactly the fields Nano needs for agency
    assignment from one ProcessedBuildingEvidence record (building_evidence.py):
    footprint_id, label, centroid, damage_class, confidence, priority,
    vl_caption, facility_near -- never property value, owner names, or
    invented facts.

    `confirmed` is operator-confirmation state, which ProcessedBuildingEvidence
    itself does not carry (that concept lives on the public RankItem
    contract, built elsewhere); it defaults to False, mirroring RankItem's
    own default before an operator confirms a grade. Does not mutate
    processed_evidence.
    """
    evidence = processed_evidence["evidence"]
    return {
        "footprint_id": evidence["footprint_id"],
        "label": evidence["label"],
        "centroid": list(evidence["centroid"]),
        "damage_class": evidence["damage_class"],
        "confidence": evidence["confidence"],
        "confirmed": confirmed,
        "priority": processed_evidence["priority"],
        "vl_caption": evidence["vl_caption"],
        "facility_near": processed_evidence["evidence"].get("facility_near"),
    }


def _validate_assignments(response, candidate: dict) -> list:
    """Structural AND evidence-grounding validation of ONE building's
    proposed-assignments response against agency, action, units -- the
    compact per-building schema's only fields. No footprint_id/label/
    centroid/task check -- they are not part of this schema at all (the
    request was already scoped to one building, and B generates task text
    itself, see _normalize_assignments).

    Grounding: an action that IS one of the valid agency/action pairs but
    that `candidate`'s own evidence does not support (is_action_supported)
    is a validation failure, not a silently accepted assignment -- e.g.
    "fire_suppression unsupported: no fire evidence was supplied."

    Returns a list of human-readable error strings; empty means valid.
    """
    if not isinstance(response, dict):
        return ["response must be a JSON object"]

    assignments = response.get("assignments")
    if not isinstance(assignments, list):
        return ["response.assignments must be a list"]

    errors = []
    for index, raw in enumerate(assignments):
        if not isinstance(raw, dict):
            errors.append(f"assignments[{index}] must be an object")
            continue

        agency = raw.get("agency")
        action = raw.get("action")
        units = raw.get("units")

        if agency not in SUPPORTED_AGENCIES:
            errors.append(f"assignments[{index}].agency must be one of {SUPPORTED_AGENCIES}, got {agency!r}")

        if action not in _VALID_ACTIONS:
            errors.append(f"assignments[{index}].action must be one of {_VALID_ACTIONS}, got {action!r}")
        elif agency in _VALID_AGENCY_ACTIONS and action not in _VALID_AGENCY_ACTIONS[agency]:
            errors.append(f"assignments[{index}]: action {action!r} is not valid for agency {agency!r}")
        elif not is_action_supported(action, candidate):
            errors.append(f"{action} unsupported: {_ACTION_EVIDENCE_DESCRIPTION[action]}")

        if not isinstance(units, int) or isinstance(units, bool) or units < 1:
            errors.append(f"assignments[{index}].units must be an integer >= 1")

    return errors


def _normalize_assignments(raw_assignments: list, candidate: dict) -> list:
    """Build B6a-ready internal assignments from one building's
    already-validated raw assignments (agency, action, units -- the model
    is never even asked for footprint_id/label/centroid/task, see
    _ASSIGNMENTS_JSON_SCHEMA). footprint_id/label/centroid always come from
    `candidate`, the SAME authoritative record the request was built from;
    task is ALWAYS B's fixed, deterministic text for the chosen action
    (_ACTION_TASK_TEXT) -- Nano never writes task prose.
    """
    return [
        {
            "agency": raw["agency"],
            "footprint_id": candidate["footprint_id"],
            "label": candidate["label"],
            "centroid": list(candidate["centroid"]),
            "task": _ACTION_TASK_TEXT[raw["action"]],
            "units": raw["units"],
        }
        for raw in raw_assignments
    ]


def _draft_one_building(
    candidate: dict,
    client: AgencyPlanDraftClient,
    fallback_client: AgencyPlanDraftClient,
) -> dict:
    """Times _draft_one_building_inner() (that building's own attempt(s) --
    under concurrent execution, this is that request's own wall-clock span,
    not affected by other buildings running at the same time) and stamps
    the result's diagnostics with elapsed_s, regardless of which of that
    function's return paths was taken.
    """
    start = time.monotonic()
    result = _draft_one_building_inner(candidate, client, fallback_client)
    result["diagnostics"]["elapsed_s"] = time.monotonic() - start
    return result


def _draft_one_building_inner(
    candidate: dict,
    client: AgencyPlanDraftClient,
    fallback_client: AgencyPlanDraftClient,
) -> dict:
    """Run the attempt-1 -> validate -> retry-once -> fallback sequence for
    ONE candidate building. Never raises for a per-building model failure
    (transport error or still-invalid output) -- always falls back to
    fallback_client for THAT building only. Only raises if the fallback
    itself is invalid (a bug in the stub, not a model failure).

    Returns {
        "assignments": list,          # B6a-ready, identity/location attached
        "provenance": "nano" | "stub",
        "diagnostics": {
            "footprint_id": str,
            "attempt_count": int,               # 1 or 2
            "attempt_1_error": str | None,
            "attempt_1_error_category": str | None,
            "attempt_2_error": str | None,
            "attempt_2_error_category": str | None,
            "fallback_reason": str | None,      # None iff provenance == "nano"
            "elapsed_s": float,                 # stamped by _draft_one_building, not here
        },
    }
    """
    diagnostics = {
        "footprint_id": candidate["footprint_id"],
        "attempt_count": 0,
        "attempt_1_error": None,
        "attempt_1_error_category": None,
        "attempt_2_error": None,
        "attempt_2_error_category": None,
        "fallback_reason": None,
    }

    try:
        diagnostics["attempt_count"] = 1
        first_response = client.propose_assignments_for_building(candidate)
        first_errors = _validate_assignments(first_response, candidate)
        if not first_errors:
            assignments = _normalize_assignments(first_response["assignments"], candidate)
            return {"assignments": assignments, "provenance": "nano", "diagnostics": diagnostics}

        validation_error = "; ".join(first_errors)
        diagnostics["attempt_1_error"] = validation_error
        diagnostics["attempt_1_error_category"] = categorize_validation_error(first_errors[0])

        diagnostics["attempt_count"] = 2
        second_response = client.propose_assignments_for_building(candidate, validation_error=validation_error)
        second_errors = _validate_assignments(second_response, candidate)
        if not second_errors:
            assignments = _normalize_assignments(second_response["assignments"], candidate)
            return {"assignments": assignments, "provenance": "nano", "diagnostics": diagnostics}

        second_validation_error = "; ".join(second_errors)
        diagnostics["attempt_2_error"] = second_validation_error
        diagnostics["attempt_2_error_category"] = categorize_validation_error(second_errors[0])
        diagnostics["fallback_reason"] = f"attempt 2 still invalid: {second_validation_error}"
    except NanoClientError as exc:
        category = categorize_client_error(exc)
        if diagnostics["attempt_count"] == 1:
            diagnostics["attempt_1_error"] = str(exc)
            diagnostics["attempt_1_error_category"] = category
            diagnostics["fallback_reason"] = f"attempt 1 failed ({category}): {exc}"
        else:
            diagnostics["attempt_2_error"] = str(exc)
            diagnostics["attempt_2_error_category"] = category
            diagnostics["fallback_reason"] = f"attempt 2 failed ({category}): {exc}"

    fallback_response = fallback_client.propose_assignments_for_building(candidate)
    fallback_errors = _validate_assignments(fallback_response, candidate)
    if fallback_errors:
        # The deterministic stub must always be valid by construction; a
        # failure here is a bug in the stub itself, not a model failure --
        # surfaced loudly rather than silently accepted.
        raise NanoClientError(
            f"deterministic agency-plan fallback produced invalid assignments for "
            f"{candidate['footprint_id']!r}: " + "; ".join(fallback_errors)
        )

    assignments = _normalize_assignments(fallback_response["assignments"], candidate)
    return {"assignments": assignments, "provenance": "stub", "diagnostics": diagnostics}


def draft_agency_plan(
    candidates: list,
    availability: AvailabilityRegistry,
    client: AgencyPlanDraftClient = None,
    fallback_client: AgencyPlanDraftClient = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> dict:
    """Draft a full AgencyPlan (frozen B -> C contract, via B6a's
    build_agency_plan) from `candidates` (see build_planner_candidate) and
    the operator's current `availability`.

    Runs one Nano request per candidate building, with bounded parallelism
    (max_concurrency) and independent per-building recovery (see
    _draft_one_building): a building whose model output is invalid gets
    re-prompted once; a building that still fails, times out, or errors
    falls back to the deterministic stub -- for THAT building only. Other
    buildings' successful results are never discarded. Never mutates
    `candidates`, and never lets any client supply units_available (that
    always comes from `availability`).

    Thin wrapper around draft_agency_plan_with_diagnostics() -- identical
    behavior, return value unchanged (just the plan, no diagnostics) for
    every existing caller.
    """
    return draft_agency_plan_with_diagnostics(candidates, availability, client, fallback_client, max_concurrency)[
        "plan"
    ]


def draft_agency_plan_with_diagnostics(
    candidates: list,
    availability: AvailabilityRegistry,
    client: AgencyPlanDraftClient = None,
    fallback_client: AgencyPlanDraftClient = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> dict:
    """Same behavior as draft_agency_plan() -- not a reimplementation, this
    IS the implementation draft_agency_plan() wraps -- but also returns
    INTERNAL diagnostics making per-building recovery visible. Never added
    to the frozen public AgencyPlan contract.

    candidates are processed with a SHARED thread pool bounded at
    max_concurrency (clamped to [1, len(candidates)] -- never more workers
    than buildings, and this is the only place concurrency is spent; each
    building still gets its own up-to-2 sequential attempts). Results are
    reassembled in the ORIGINAL candidate order regardless of completion
    order.

    Returns {
        "plan": <AgencyPlan, same as draft_agency_plan()'s return value>,
        "diagnostics": {
            "buildings": [
                {
                    "footprint_id": str,
                    "recovery": "nano" | "stub",
                    "attempt_count": int,
                    "attempt_1_error": str | None,
                    "attempt_1_error_category": str | None,
                    "attempt_2_error": str | None,
                    "attempt_2_error_category": str | None,
                    "fallback_reason": str | None,
                    "elapsed_s": float,  # that building's own wall-clock time
                }, ...  # one per candidate, in original order
            ],
            "model_building_count": int,
            "fallback_building_count": int,
        },
    }

    See this module's docstring for how the overall plan's drafted_by is
    chosen when buildings have mixed provenance (a known, unresolved public-
    contract gap -- not invented around here).
    """
    active_client = client if client is not None else RealAgencyPlanDraftClient()
    active_fallback = fallback_client if fallback_client is not None else StubAgencyPlanDraftClient()

    results = [None] * len(candidates)
    if candidates:
        bounded_workers = max(1, min(max_concurrency, len(candidates)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=bounded_workers) as executor:
            future_to_index = {
                executor.submit(_draft_one_building, candidate, active_client, active_fallback): index
                for index, candidate in enumerate(candidates)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                results[future_to_index[future]] = future.result()

    all_assignments = []
    building_diagnostics = []
    model_building_count = 0
    fallback_building_count = 0
    for result in results:
        all_assignments.extend(result["assignments"])
        building_diagnostics.append({**result["diagnostics"], "recovery": result["provenance"]})
        if result["provenance"] == "nano":
            model_building_count += 1
        else:
            fallback_building_count += 1

    # See module docstring's CONTRACT GAP note: conservative choice for a
    # mixed-provenance plan, never overclaiming "nano".
    overall_drafted_by = "nano" if fallback_building_count == 0 else "stub"

    plan = build_agency_plan(all_assignments, drafted_by=overall_drafted_by, availability=availability)

    diagnostics = {
        "buildings": building_diagnostics,
        "model_building_count": model_building_count,
        "fallback_building_count": fallback_building_count,
    }
    return {"plan": plan, "diagnostics": diagnostics}
