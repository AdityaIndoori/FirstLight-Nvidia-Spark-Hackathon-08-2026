"""Client boundary for the flight-plan candidate operation.

Application code depends only on NanoFlightClient.request_flight_plan(planning_input,
validation_error). StubNanoFlightClient is the deterministic fallback;
RealNanoFlightClient is the HTTP-backed client for the actual Nemotron Nano
9B v2 server (served model name "nano") behind the OpenAI-compatible vLLM
API. Both implement the same interface, so flight_planner.py never changes
based on which is active.

REASONING ENABLED, unlike agency planning's /no_think: RealNanoFlightClient
is the first real caller of nano_client._post_chat_completion(thinking=True)
-- next-flight tasking (choosing which ranked candidate gets the next
survey) is a genuine judgment call across priority/damage/facility/
rationale context, not a rote lookup, so it uses "/think" instead of
"/no_think".

GROUNDING BY CONSTRUCTION: Nano's structured output is ONLY
{"target_footprint_id": <str>} -- a single field, schema-enum-constrained
to the footprint_ids actually present in planning_input["candidates"] (see
_target_schema). Nano is never asked for, and structurally cannot express,
a coordinate, a new footprint_id, a facility name, a casualty count, or a
property value. Once a target is selected and validated against the
supplied candidates (defense in depth, even though the schema enum should
already prevent an out-of-set answer), the actual survey GEOMETRY is built
by the EXISTING, UNCHANGED build_deterministic_flight_plan() from that
candidate's own authoritative centroid -- Nano's raw text never becomes a
coordinate in the final FlightPlan, by construction, the same
never-trust-the-model's-copy pattern agency_plan_drafter.py and
archive_write.py already use for identity/location.

Only RealNanoFlightClient touches the network, and only inside
request_flight_plan() -- constructing it performs no I/O.
"""

import json
import math
from abc import ABC, abstractmethod

from backend.decision.nano_client import NanoClientError, _post_chat_completion, _resolve_base_url

_METERS_PER_DEG_LAT = 111_320.0
_CRUISE_SPEED_M_S = 8.0

_DEFAULT_TIMEOUT_S = 10.0
_MAX_TOKENS = 1024
"""Bounded output budget for the flight-tasking call. Reasoning is
ENABLED here (unlike the cheap /no_think structured calls elsewhere in
this project), so the completion includes a chain-of-thought before the
tiny {"target_footprint_id": ...} answer -- 1024 is a sensible starting
default (a genuine reasoning trace over a handful of candidates plus a
short structured answer), not the product of a tuning sweep. Adjust only
if scripts/flight_plan_live_check.py proves truncation (finish_reason ==
"length", raised as NanoClientError by _post_chat_completion and never
silently accepted)."""


class NanoFlightClient(ABC):
    """Boundary: proposes one FlightPlan GeoJSON candidate for planning_input.

    validation_error is None on a request's first attempt. When the runtime
    re-prompts after a schema-invalid first candidate, this is called exactly
    once more, with validation_error set to the validator's message.
    """

    @abstractmethod
    def request_flight_plan(self, planning_input: dict, validation_error: str = None) -> dict:
        raise NotImplementedError


class StubNanoFlightClient(NanoFlightClient):
    """Deterministic fallback standing in for the real Nano client.

    force_invalid_first exists ONLY to deterministically exercise the demo
    self-recovery beat (--demo-force-invalid-first-replan): when set, a
    request's first attempt (validation_error is None) is deliberately
    schema-invalid, and the retry (validation_error is not None) is valid.
    Never randomized.
    """

    is_stub = True

    def __init__(self, force_invalid_first: bool = False):
        self.force_invalid_first = force_invalid_first

    def request_flight_plan(self, planning_input: dict, validation_error: str = None) -> dict:
        if self.force_invalid_first and validation_error is None:
            return _build_invalid_demo_candidate(planning_input)
        return build_deterministic_flight_plan(planning_input)


def build_deterministic_flight_plan(planning_input: dict) -> dict:
    """Pure, deterministic FlightPlan builder. Always produces a valid candidate.

    Used both as the stub client's normal (non-demo-forced) response and as the
    last-resort fallback when both model attempts fail validation, so the
    fallback shape is provably identical to what a successful attempt returns.
    """
    lng, lat = planning_input["centroid"]
    radius_m = planning_input["area_radius_m"]
    altitude_m_agl = planning_input["altitude_m_agl"]
    line_spacing_m = planning_input["line_spacing_m"]

    meters_per_deg_lng = _METERS_PER_DEG_LAT * math.cos(math.radians(lat))
    half_lat = radius_m / _METERS_PER_DEG_LAT
    half_lng = radius_m / meters_per_deg_lng

    survey_area_ring = [
        [lng - half_lng, lat - half_lat],
        [lng + half_lng, lat - half_lat],
        [lng + half_lng, lat + half_lat],
        [lng - half_lng, lat + half_lat],
        [lng - half_lng, lat - half_lat],
    ]

    transects = max(1, round((2 * radius_m) / line_spacing_m))
    row_spacing_lat = line_spacing_m / _METERS_PER_DEG_LAT

    path_coords = []
    for i in range(transects):
        y = lat - half_lat + i * row_spacing_lat
        left, right = [lng - half_lng, y], [lng + half_lng, y]
        path_coords.extend([left, right] if i % 2 == 0 else [right, left])

    total_length_m = transects * (2 * radius_m) + max(0, transects - 1) * line_spacing_m
    est_flight_min = round(total_length_m / _CRUISE_SPEED_M_S / 60, 1)

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"role": "survey-area"},
                "geometry": {"type": "Polygon", "coordinates": [survey_area_ring]},
            },
            {
                "type": "Feature",
                "properties": {
                    "role": "survey-path",
                    "altitude_m_agl": altitude_m_agl,
                    "line_spacing_m": line_spacing_m,
                    "transects": transects,
                    "est_flight_min": est_flight_min,
                },
                "geometry": {"type": "LineString", "coordinates": path_coords},
            },
        ],
    }


def _build_invalid_demo_candidate(planning_input: dict) -> dict:
    """Deterministically invalid candidate: survey-path missing line_spacing_m."""
    plan = build_deterministic_flight_plan(planning_input)
    for feature in plan["features"]:
        if feature["properties"]["role"] == "survey-path":
            del feature["properties"]["line_spacing_m"]
    return plan


def _target_schema(candidates: list) -> dict:
    """json_schema response_format constraining Nano's answer to EXACTLY
    one of the supplied candidates' footprint_id values -- an out-of-set
    answer is structurally unrepresentable via the enum, not just
    rejected after the fact (though request_flight_plan below also checks
    it explicitly, as defense in depth).
    """
    footprint_ids = [candidate["footprint_id"] for candidate in candidates]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "flight_target_selection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"target_footprint_id": {"type": "string", "enum": footprint_ids}},
                "required": ["target_footprint_id"],
                "additionalProperties": False,
            },
        },
    }


def _flight_tasking_prompt(planning_input: dict, validation_error: str = None) -> str:
    """Build the next-flight-tasking prompt from ONLY the supplied
    candidates' footprint_id/label/damage_class/confirmed/facility_near/
    priority/rationale -- no property value, casualty count, or invented
    fact. Nano is told the survey geometry (area_radius_m/altitude_m_agl/
    line_spacing_m) is fixed and it is never asked to write coordinates,
    only to select a footprint_id.
    """
    candidates = planning_input["candidates"]
    lines = []
    for candidate in candidates:
        facility_near = candidate.get("facility_near")
        facility_line = (
            f", facility_near: {facility_near['name']} ({facility_near['type']}), "
            f"{facility_near['dist_m']} m away"
            if facility_near is not None
            else ""
        )
        rationale = candidate.get("rationale")
        rationale_line = f', rationale: "{rationale}"' if rationale else ""
        lines.append(
            f"- footprint_id: {candidate['footprint_id']}, label: {candidate['label']}, "
            f"damage_class: {candidate['damage_class']}, confirmed: {candidate['confirmed']}, "
            f"priority: {candidate['priority']:.5f}{facility_line}{rationale_line}"
        )
    candidate_block = "\n".join(lines)

    prompt = (
        "You are tasking the NEXT drone survey flight. Below is the ranked list of "
        "candidate buildings, already scored and ordered by an existing deterministic "
        "priority system -- you do not recompute priority. Select exactly ONE "
        "footprint_id to survey next, reasoning from the supplied evidence (damage "
        "class, confirmation status, facility proximity, priority, rationale).\n\n"
        "You may choose ONLY a footprint_id that appears in the list below. Never "
        "invent a new building, footprint_id, coordinate, facility name, casualty "
        "count, or property value -- you have no authority to create any of those. "
        "The survey area, altitude, and line spacing are fixed by existing project "
        f"configuration: area_radius_m={planning_input['area_radius_m']}, "
        f"altitude_m_agl={planning_input['altitude_m_agl']}, "
        f"line_spacing_m={planning_input['line_spacing_m']}. You are not asked to write "
        "coordinates or geometry -- only to select the target footprint_id; the survey "
        "geometry is computed deterministically from its already-known authoritative "
        "centroid.\n\n"
        f"Candidates:\n{candidate_block}"
    )

    if validation_error:
        prompt += f"\n\nYour previous output was invalid: {validation_error}\nCorrect it."

    return prompt


def _grounding_invalid_candidate(planning_input: dict) -> dict:
    """Deliberately schema-invalid placeholder returned when Nano's
    target_footprint_id selection fails grounding (not one of the
    supplied candidates -- should not happen given the schema enum, but
    checked anyway). Reuses the SAME missing-property trick as
    _build_invalid_demo_candidate, so the EXISTING validate_flight_plan
    (unchanged) rejects it and flight_planner's existing retry/fallback
    handles it with no new validator code.
    """
    fallback_geometry_input = {
        "centroid": planning_input["centroid"],
        "area_radius_m": planning_input["area_radius_m"],
        "altitude_m_agl": planning_input["altitude_m_agl"],
        "line_spacing_m": planning_input["line_spacing_m"],
    }
    return _build_invalid_demo_candidate(fallback_geometry_input)


class RealNanoFlightClient(NanoFlightClient):
    """Real Nemotron Nano 9B v2 client for next-flight tasking, via the
    OpenAI-compatible vLLM API (served model name "nano"). Reasoning is
    ENABLED (thinking=True -- "/think", not "/no_think") because target
    selection is a genuine judgment call over the supplied candidates, not
    a rote lookup. Structured output is enforced with a json_schema
    response_format whose "target_footprint_id" enum is built from
    planning_input["candidates"] on every call (see _target_schema), so
    Nano cannot express an out-of-set answer even if it tried.

    base_url defaults to the FIRSTLIGHT_NANO_BASE_URL environment variable
    (nano_client._resolve_base_url), falling back to http://localhost:8000
    -- the SAME Nano server flight tasking, hero rationale, and agency
    planning all share. Every call has a hard timeout (default 10s).
    Raises NanoClientError (nano_client's shared exception type, reused
    here rather than inventing a fourth one) on any transport, malformed-
    response, or truncated-output failure; this class never falls back to
    the stub itself -- that decision belongs to flight_planner.request_flight_plan.

    planning_input["candidates"] (a non-empty list) is REQUIRED for this
    client -- without a candidate set there is nothing to ground a
    selection against, so its absence raises NanoClientError immediately,
    handled by the same recovery path as any other client failure.

    force_invalid_first exists ONLY for
    scripts/flight_plan_live_check.py's --force-invalid-first diagnostic
    flag (mirrors StubNanoFlightClient's own force_invalid_first):
    when True and validation_error is None (the first attempt), this
    returns a deterministically invalid candidate WITHOUT calling Nano at
    all, to prove the real retry path deterministically; the retry
    (validation_error set) always calls the real model normally. Default
    False, so production behavior is unaffected.

    last_raw_response/last_selected_footprint_id/last_grounding_error are
    DEVELOPMENT DIAGNOSTIC DATA ONLY (mirrors agency_plan_client's
    last_raw_response) -- never part of any return value contract, not
    safe to read from concurrent calls. usage_log/finish_reason_log
    accumulate each response's "usage" object / finish_reason string
    (list.append is GIL-atomic).
    """

    is_stub = False

    def __init__(
        self,
        base_url: str = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        force_invalid_first: bool = False,
    ):
        self.base_url = base_url if base_url is not None else _resolve_base_url()
        self.timeout_s = timeout_s
        self.force_invalid_first = force_invalid_first
        self.last_raw_response = None
        self.last_selected_footprint_id = None
        self.last_grounding_error = None
        self.usage_log = []
        self.finish_reason_log = []

    def request_flight_plan(self, planning_input: dict, validation_error: str = None) -> dict:
        candidates = planning_input.get("candidates")
        if not candidates:
            raise NanoClientError(
                "planning_input['candidates'] is required for RealNanoFlightClient "
                "(nothing to ground a target selection against)"
            )

        if self.force_invalid_first and validation_error is None:
            self.last_raw_response = None
            self.last_selected_footprint_id = None
            self.last_grounding_error = "force_invalid_first diagnostic flag: skipped the real model call"
            return _grounding_invalid_candidate(planning_input)

        self.last_raw_response = None
        self.last_grounding_error = None

        messages = [{"role": "user", "content": _flight_tasking_prompt(planning_input, validation_error)}]
        content = _post_chat_completion(
            messages,
            thinking=True,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            response_format=_target_schema(candidates),
            max_tokens=_MAX_TOKENS,
            usage_sink=self.usage_log,
            finish_reason_sink=self.finish_reason_log,
        )
        self.last_raw_response = content

        try:
            parsed = json.loads(content)
            target_footprint_id = parsed["target_footprint_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise NanoClientError(f"flight-tasking response was malformed: {exc}") from exc

        candidates_by_id = {candidate["footprint_id"]: candidate for candidate in candidates}
        selected = candidates_by_id.get(target_footprint_id)
        if selected is None:
            self.last_grounding_error = (
                f"target_footprint_id {target_footprint_id!r} is not in the supplied "
                f"candidate set {sorted(candidates_by_id)}"
            )
            return _grounding_invalid_candidate(planning_input)

        self.last_selected_footprint_id = target_footprint_id
        effective_geometry_input = {
            "centroid": list(selected["centroid"]),
            "area_radius_m": planning_input["area_radius_m"],
            "altitude_m_agl": planning_input["altitude_m_agl"],
            "line_spacing_m": planning_input["line_spacing_m"],
        }
        return build_deterministic_flight_plan(effective_geometry_input)
