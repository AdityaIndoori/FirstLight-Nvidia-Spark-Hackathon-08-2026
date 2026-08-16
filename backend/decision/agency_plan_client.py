"""Client boundary for the B6b agency-plan drafting operation.

PER-BUILDING drafting: each Nano request covers exactly ONE candidate
building. agency_plan_drafter.py runs one request per candidate with
bounded parallelism and per-building recovery, then merges the results.

GROUNDED ACTION ENUM (not free-form task prose): a live run of the earlier
free-form-task schema exposed evidence-grounding failures -- e.g. Nano wrote
"Extinguish structure fire" for a building whose only evidence was dialysis-
facility damage and an obstructed entrance, with no fire mentioned anywhere.
Nano no longer writes prose at all. It picks from a small, closed action
enum (see _VALID_ACTIONS/_VALID_AGENCY_ACTIONS below); B deterministically
maps the chosen action to fixed, pre-written task text
(_ACTION_TASK_TEXT) and -- independently of what Nano picked -- validates
that the action is actually supported by this building's own evidence
(is_action_supported) via small, deterministic, conservative keyword rules.
No second model/NLP classifier: matching is plain substring checks, exactly
like the evidence-based deterministic fallback already used.

Application code (agency_plan_drafter.py) depends only on
AgencyPlanDraftClient.propose_assignments_for_building(candidate,
validation_error). StubAgencyPlanDraftClient is the deterministic fallback
-- it produces the SAME compact {agency, action, units} shape Nano does, so
it is validated and normalized through the exact same
is_action_supported()/_ACTION_TASK_TEXT pipeline, guaranteeing identical
evidence-support rules for both paths. RealAgencyPlanDraftClient is the
HTTP-backed client for the actual Nemotron Nano 9B v2 server (served model
name "nano") behind the OpenAI-compatible vLLM API, reusing
nano_client._post_chat_completion -- the same shared HTTP helper hero
rationale already uses -- rather than duplicating request/timeout/error
handling.

Both implementations return the SAME internal-only, per-building schema:
    {"assignments": [{"agency", "action", "units"}, ...]}
No footprint_id, label, centroid, or free-form task -- the request is
already scoped to one authoritative building, so agency_plan_drafter
re-attaches its footprint_id/label/centroid after parsing, and B (not Nano)
generates the final task text from the chosen action. This is NOT the
public AgencyPlan (B -> C) contract.

Only RealAgencyPlanDraftClient touches the network, and only inside
propose_assignments_for_building() -- constructing it performs no I/O.
"""

import json
from abc import ABC, abstractmethod

from backend.decision.agency_plan import SUPPORTED_AGENCIES
from backend.decision.nano_client import NanoClientError, _post_chat_completion, _resolve_base_url

_DEFAULT_TIMEOUT_S = 10.0
_MAX_TOKENS = 80
"""Cap for ONE building's compact {agency, action, units} schema. Lowered
from 100 (the previous free-form-task schema's starting value) because
action enum values are much shorter than free-form task prose. If a
building's response still hits this cap, nano_client._post_chat_completion
raises (finish_reason == "length" is never accepted as a successful
completion), and agency_plan_drafter falls back to the deterministic stub
for that building only -- other buildings' results are unaffected.
"""

_DAMAGE_SEVERITY = {
    0: "no visible damage",
    1: "minor damage",
    2: "major damage",
    3: "destroyed",
}

_CARE_FACILITY_TYPES = ("nursing_home", "dialysis", "hospital")

# --------------------------------------------------------------------------
# Grounded action enum: the only vocabulary Nano (or the stub) may choose
# from. Closed set, closed agency pairing, closed evidence rules -- see
# is_action_supported() below.
# --------------------------------------------------------------------------

_VALID_AGENCY_ACTIONS = {
    "fire": ("fire_suppression", "collapse_response"),
    "ems": ("medical_support",),
    "police": ("perimeter_control", "road_closure"),
    "public_works": ("debris_clearance",),
}

_VALID_ACTIONS = tuple(action for actions in _VALID_AGENCY_ACTIONS.values() for action in actions)

_ACTION_TASK_TEXT = {
    "fire_suppression": "Suppress structure fire",
    "collapse_response": "Assess structural collapse",
    "medical_support": "Support medical facility",
    "perimeter_control": "Establish scene perimeter",
    "road_closure": "Manage road closure",
    "debris_clearance": "Clear debris from access route",
}
"""Deterministic display task text. These strings come from B, not Nano --
Nano only ever chooses an action code, never writes prose."""

_ACTION_EVIDENCE_DESCRIPTION = {
    "fire_suppression": "no fire evidence was supplied",
    "collapse_response": "no structural collapse evidence was supplied",
    "medical_support": "no care-facility evidence was supplied",
    "perimeter_control": "no perimeter/scene-control evidence was supplied",
    "road_closure": "no road-closure evidence was supplied",
    "debris_clearance": "no debris/access-blockage evidence was supplied",
}

_FALLBACK_UNITS = {
    "fire_suppression": 2,
    "collapse_response": 2,
    "medical_support": 2,
    "perimeter_control": 1,
    "road_closure": 1,
    "debris_clearance": 1,
}

# Conservative, deterministic keyword rules -- plain substring checks, no
# model/NLP classifier. Deliberately narrow: damage_class alone is never
# fire evidence; a bare "closure" doesn't fire both police actions at once
# (road_closure needs "road" + closure language; perimeter_control needs
# its own distinct perimeter/evacuation/scene-control language).
_FIRE_KEYWORDS = ("fire", "flame", "burning")
_COLLAPSE_KEYWORDS = ("collapse", "structural failure")
_MEDICAL_CAPTION_KEYWORDS = ("hospital", "clinic", "dialysis", "nursing home", "medical facility")
_DEBRIS_KEYWORDS = ("debris", "obstruct", "blocked access", "roadway obstruction")
_ROAD_CLOSURE_KEYWORDS = ("road closure", "road block", "route closure")
_PERIMETER_KEYWORDS = ("perimeter", "evacuat", "scene control")


def _caption_has_any(caption: str, keywords: tuple) -> bool:
    lowered = caption.lower()
    return any(keyword in lowered for keyword in keywords)


def is_action_supported(action: str, candidate: dict) -> bool:
    """Deterministic evidence-grounding check: does `candidate`'s own
    evidence (vl_caption, facility_near) actually support `action`? Used to
    validate Nano's proposed actions AND to constrain the deterministic
    fallback -- the SAME rules either way (see module docstring).
    """
    caption = candidate.get("vl_caption") or ""
    facility_near = candidate.get("facility_near")

    if action == "fire_suppression":
        return _caption_has_any(caption, _FIRE_KEYWORDS)
    if action == "collapse_response":
        return _caption_has_any(caption, _COLLAPSE_KEYWORDS)
    if action == "medical_support":
        if facility_near is not None and facility_near.get("type") in _CARE_FACILITY_TYPES:
            return True
        return _caption_has_any(caption, _MEDICAL_CAPTION_KEYWORDS)
    if action == "debris_clearance":
        return _caption_has_any(caption, _DEBRIS_KEYWORDS)
    if action == "road_closure":
        return _caption_has_any(caption, _ROAD_CLOSURE_KEYWORDS)
    if action == "perimeter_control":
        return _caption_has_any(caption, _PERIMETER_KEYWORDS)
    return False


_ASSIGNMENTS_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "building_agency_assignments",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agency": {"type": "string", "enum": list(SUPPORTED_AGENCIES)},
                            "action": {"type": "string", "enum": list(_VALID_ACTIONS)},
                            "units": {"type": "integer", "minimum": 1},
                        },
                        "required": ["agency", "action", "units"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["assignments"],
            "additionalProperties": False,
        },
    },
}


class AgencyPlanDraftClient(ABC):
    """Boundary: proposes assignments (the internal-only, per-building
    action-enum schema) for ONE candidate building.

    candidate is the internal planner-candidate shape (see
    agency_plan_drafter.build_planner_candidate) -- NOT the public RankItem
    or BuildingEvidence contract. validation_error is None on that
    building's first attempt; when the caller re-prompts after an invalid
    first response for that SAME building, this is set to the validator's
    message, exactly once.
    """

    @abstractmethod
    def propose_assignments_for_building(self, candidate: dict, validation_error: str = None) -> dict:
        raise NotImplementedError


class RealAgencyPlanDraftClient(AgencyPlanDraftClient):
    """Real Nemotron Nano 9B v2 client, via the OpenAI-compatible vLLM API
    (served model name "nano"). Agency-plan drafting is a structured
    planning call, so every request is sent with /no_think -- no reasoning
    trace is exposed. Structured output is enforced with
    response_format={"type": "json_schema", ...} (nano_client._post_chat_completion)
    because top-level guided_json is silently ignored on this build. Every
    request also caps completion length at max_tokens=_MAX_TOKENS.

    base_url defaults to the FIRSTLIGHT_NANO_BASE_URL environment variable,
    falling back to http://localhost:8000. Every call has a hard timeout
    (default 10s -- unchanged; per-building requests are what's meant to
    fit inside it, not a longer timeout). Raises NanoClientError on any
    failure (timeout, connection, HTTP, malformed JSON, truncated output
    (finish_reason == "length"), or a response that doesn't even have the
    {"assignments": [...]} shape); this class never falls back to the stub
    itself -- that decision belongs to the caller
    (agency_plan_drafter._draft_one_building) -- so a model failure is
    never silently hidden.

    last_raw_response holds the most recent call's raw assistant text --
    DEVELOPMENT DIAGNOSTIC DATA ONLY, meaningful for sequential single-
    building use. It is NOT safe to read from a client instance shared
    across concurrent per-building requests (agency_plan_drafter's parallel
    drafter does not read it, for exactly that reason) -- a later thread's
    write can race an earlier thread's read. The production
    draft_agency_plan() and its return value never expose it either way.

    usage_log accumulates each response's "usage" object (prompt_tokens/
    completion_tokens), if the server includes one -- unlike
    last_raw_response, list.append is GIL-atomic, so this IS safe to read
    (in aggregate) after concurrent per-building requests through a shared
    client instance (see scripts/agency_plan_parallel_latency_check.py).
    """

    is_stub = False

    def __init__(self, base_url: str = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
        self.base_url = base_url if base_url is not None else _resolve_base_url()
        self.timeout_s = timeout_s
        self.last_raw_response = None
        self.usage_log = []

    def propose_assignments_for_building(self, candidate: dict, validation_error: str = None) -> dict:
        self.last_raw_response = None
        messages = [{"role": "user", "content": _building_drafting_prompt(candidate, validation_error)}]
        content = _post_chat_completion(
            messages,
            thinking=False,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            response_format=_ASSIGNMENTS_JSON_SCHEMA,
            max_tokens=_MAX_TOKENS,
            usage_sink=self.usage_log,
        )
        self.last_raw_response = content

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise NanoClientError(f"agency-plan response was not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict) or not isinstance(parsed.get("assignments"), list):
            raise NanoClientError('agency-plan response must be a JSON object with an "assignments" list')

        return parsed


class StubAgencyPlanDraftClient(AgencyPlanDraftClient):
    """Deterministic fallback. Evidence-based, conservative rules only,
    using the EXACT SAME is_action_supported() grounding checks as Nano's
    output is validated against -- never fabricates casualties, occupancy,
    or resource availability, and never assigns an agency the building's
    own evidence doesn't support. Returns zero, one, or multiple
    assignments for the one building it's given, depending on what its
    evidence actually supports. Never random, never networked.
    """

    is_stub = True

    def propose_assignments_for_building(self, candidate: dict, validation_error: str = None) -> dict:
        return {"assignments": _deterministic_assignments_for_building(candidate)}


def _building_drafting_prompt(candidate: dict, validation_error: str = None) -> str:
    """Build the per-building agency-plan drafting prompt from ONLY the
    supplied candidate fields: footprint_id, damage_class, confidence,
    priority, vl_caption, facility_near (label included as human-readable
    context only) -- no property value, owner names, units_available, or
    invented facts. footprint_id is shown for traceability but the model is
    never asked to echo it (or label/centroid, or free-form task text)
    back -- see _ASSIGNMENTS_JSON_SCHEMA.
    """
    facility_near = candidate.get("facility_near")
    facility_line = (
        f"\nfacility_near: {facility_near['name']} ({facility_near['type']}), "
        f"{facility_near['dist_m']} m away"
        if facility_near is not None
        else ""
    )
    damage_class = candidate["damage_class"]
    severity = _DAMAGE_SEVERITY.get(damage_class, f"damage class {damage_class}")

    rules = (
        "Decide which responding-agency actions this ONE building needs, "
        "based only on the evidence below. Choose only from these exact "
        "agency/action pairs, and ONLY when the evidence actually supports "
        "them:\n"
        "fire: fire_suppression (active fire/flames present), "
        "collapse_response (structural collapse present).\n"
        "ems: medical_support (care-facility evidence present).\n"
        "police: perimeter_control (perimeter/scene-control/evacuation "
        "evidence), road_closure (road/access-closure evidence).\n"
        "public_works: debris_clearance (debris/blocked-access evidence).\n"
        "Never choose an action the evidence does not describe -- damage "
        "class alone is not evidence for any specific action. Return an "
        "empty assignments list if no action is supported. Never invent "
        "casualties, occupancy, or resource availability. units is your "
        "operational recommendation, not a report of availability.\n\n"
    )

    evidence = (
        f"footprint_id: {candidate['footprint_id']}\n"
        f"label: {candidate.get('label', '')}\n"
        f"damage_class: {damage_class} ({severity})\n"
        f"confidence: {candidate['confidence']:.2f}\n"
        f"priority: {candidate['priority']:.5f}\n"
        f"vl_caption: \"{candidate['vl_caption']}\"" + facility_line
    )

    prompt = f"{rules}Building:\n{evidence}"

    if validation_error:
        prompt += f"\n\nYour previous output was invalid: {validation_error}\nCorrect it."

    return prompt


def _deterministic_assignments_for_building(candidate: dict) -> list:
    """Evidence-based conservative rules, no LLM, scoped to ONE candidate --
    tries every valid agency/action pair (in fixed SUPPORTED_AGENCIES
    order) and includes it only if is_action_supported() says this
    building's own evidence actually supports it. Returns zero, one, or
    multiple assignments depending on what the evidence supports. Never
    invents a casualty count or assigns an agency without evidence.
    """
    assignments = []
    for agency in SUPPORTED_AGENCIES:
        for action in _VALID_AGENCY_ACTIONS[agency]:
            if is_action_supported(action, candidate):
                assignments.append(_assignment(agency, action, _FALLBACK_UNITS[action]))
    return assignments


def _assignment(agency: str, action: str, units: int) -> dict:
    """Compact per-building shape -- matches _ASSIGNMENTS_JSON_SCHEMA
    exactly (agency, action, units). No footprint_id/label/centroid/task:
    the request is already scoped to one building, so agency_plan_drafter
    attaches identity/location from the authoritative candidate, and maps
    action -> task text via _ACTION_TASK_TEXT, after parsing.
    """
    return {"agency": agency, "action": action, "units": units}
