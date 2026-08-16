"""Client boundary for the Nano rationale operation.

Application code depends only on NanoRationaleClient.generate_rationale(rank_item).
StubNanoRationaleClient is the deterministic fallback; RealNanoRationaleClient
is the HTTP-backed client for the actual Nemotron Nano 9B v2 server (served
model name "nano") behind the OpenAI-compatible vLLM API. Both implement the
same interface, so callers (rationale.py) never change based on which is
active.

Only RealNanoRationaleClient touches the network, and only inside
generate_rationale() -- constructing it performs no I/O.
"""

import json
import os
import re
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

_DAMAGE_SEVERITY = {
    0: "no visible damage",
    1: "minor damage",
    2: "major damage",
    3: "destroyed",
}

_BASE_URL_ENV_VAR = "FIRSTLIGHT_NANO_BASE_URL"
_DEFAULT_BASE_URL = "http://localhost:8000"
_MODEL_NAME = "nano"
_DEFAULT_TIMEOUT_S = 10.0


class NanoClientError(Exception):
    """Raised by RealNanoRationaleClient for any timeout, connection, HTTP, or
    malformed-response failure. A single error type so orchestration code
    (rationale.generate_rationale_with_recovery) can catch one thing and fall
    back to the stub, instead of a model failure being silently hidden.
    """


class NanoRationaleClient(ABC):
    """Boundary: turns one already-ranked RankItem into a concise operator-facing rationale."""

    @abstractmethod
    def generate_rationale(self, rank_item: dict) -> str:
        """Return rationale text derived only from facts present in rank_item.

        Must not mutate rank_item and must not alter rank_item["priority"].
        """
        raise NotImplementedError


class StubNanoRationaleClient(NanoRationaleClient):
    """Deterministic fallback. Labelled is_stub so callers/status code can be honest
    about whether the real Nano model produced this text.
    """

    is_stub = True

    def generate_rationale(self, rank_item: dict) -> str:
        label = rank_item["label"]
        damage_class = rank_item["damage_class"]
        confidence = rank_item["confidence"]
        confirmed = rank_item["confirmed"]
        graded_by = rank_item["graded_by"]
        facility_near = rank_item.get("facility_near")
        inputs = rank_item["inputs"]
        priority = rank_item["priority"]

        severity = _DAMAGE_SEVERITY.get(damage_class, f"damage class {damage_class}")
        grading = "operator-confirmed" if confirmed else f"auto-graded by {graded_by}"

        sentences = [
            f"{label}: {severity} ({grading}, confidence {confidence:.2f}).",
            f"Staleness {inputs['staleness_h']:.1f}h and vulnerable density "
            f"{inputs['vulnerable_density']:.2f} drive urgency; doubt {inputs['doubt']:.2f} "
            "reflects grading uncertainty.",
        ]

        if facility_near is not None:
            sentences.append(
                f"{facility_near['dist_m']}m from {facility_near['name']} ({facility_near['type']})."
            )

        road_cutoff = inputs.get("road_cutoff")
        if road_cutoff is not None:
            sentences.append(f"Access constrained by road cutoff x{road_cutoff:.2f}.")

        sentences.append(f"Priority score: {priority:.5f}.")

        return " ".join(sentences)


class RealNanoRationaleClient(NanoRationaleClient):
    """Real Nemotron Nano 9B v2 client, via the OpenAI-compatible vLLM API
    (served model name "nano"). Hero rationale is a cheap/structured
    operation, so every request is sent with /no_think; reasoning mode
    (/think) is reserved for the future replan beat and is not used here.

    base_url defaults to the FIRSTLIGHT_NANO_BASE_URL environment variable,
    falling back to http://localhost:8000 -- during Mac development this
    reaches the DGX Spark's vLLM server through an SSH tunnel. Every call has
    a hard timeout (default 10s). Raises NanoClientError on any failure; this
    class never falls back to the stub itself -- that decision belongs to the
    caller (see rationale.generate_rationale_with_recovery) -- so a model
    failure is never silently hidden.
    """

    is_stub = False

    def __init__(self, base_url: str = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
        self.base_url = base_url if base_url is not None else _resolve_base_url()
        self.timeout_s = timeout_s

    def generate_rationale(self, rank_item: dict) -> str:
        messages = [{"role": "user", "content": _rationale_prompt(rank_item)}]
        content = _post_chat_completion(
            messages, thinking=False, base_url=self.base_url, timeout_s=self.timeout_s
        )

        violations = _faithfulness_violations(content, rank_item)
        if violations:
            raise NanoClientError(
                "Nano rationale failed faithfulness check: " + "; ".join(violations)
            )
        return content


def _resolve_base_url() -> str:
    return os.environ.get(_BASE_URL_ENV_VAR, _DEFAULT_BASE_URL)


def _rationale_prompt(rank_item: dict) -> str:
    """Build the hero-rationale prompt from ONLY the facts the operator is
    allowed to see cited: label, damage_class, confidence, confirmed,
    graded_by, facility_near, staleness_h, vulnerable_density, doubt,
    road_cutoff, priority. Explicitly teaches the frozen README field
    semantics so the model does not reinterpret scoring factors into
    unsupported real-world quantities (see _faithfulness_violations, which
    checks the response against these same rules).
    """
    damage_class = rank_item["damage_class"]
    severity = _DAMAGE_SEVERITY.get(damage_class, f"damage class {damage_class}")
    inputs = rank_item["inputs"]
    facility_near = rank_item.get("facility_near")
    road_cutoff = inputs.get("road_cutoff")

    facility_line = (
        f"- facility_near: {facility_near['name']} ({facility_near['type']}), {facility_near['dist_m']} m away\n"
        if facility_near is not None
        else "- facility_near: none\n"
    )
    road_line = (
        f"- road_cutoff: {road_cutoff:.2f} (dimensionless multiplier >= 1; NOT a distance, NOT a count)\n"
        if road_cutoff is not None
        else "- road_cutoff: none (road access is clear)\n"
    )

    semantics = (
        "Field semantics you MUST follow exactly:\n"
        "1. staleness_h is hours since the observation; larger values increase priority. "
        "Never say staleness reduces or lowers priority.\n"
        "2. vulnerable_density is a dimensionless scoring factor from the vulnerability join. "
        "Never turn it into a population count or number of people.\n"
        "3. doubt is a dimensionless uncertainty factor; larger doubt raises priority because "
        "uncertain buildings deserve inspection. Never render it as a percentage unless you "
        "explicitly call it a dimensionless score.\n"
        "4. road_cutoff is either null (clear access) or a dimensionless multiplier >= 1 that "
        "raises priority for blocked-road-access buildings. It has NO physical unit -- never "
        "call it meters, miles, or any distance.\n"
        "5. Never append a unit (meters, miles, hours, people, percentages, vehicles, etc.) to "
        "any field unless this list explicitly gives that field a unit.\n"
        "6. facility_near.dist_m is the ONLY field measured in meters.\n"
        "7. confidence is the model's confidence in the damage grade; never reinterpret it as a "
        "probability of casualties or occupancy.\n"
        "8. graded_by identifies who/what assigned the grade; do not imply that source supplied "
        "any other fact.\n"
        "9. priority is already computed and fixed; explain why it is operationally significant, "
        "never calculate or state a different number.\n"
    )

    return (
        "Write a concise, one-to-two sentence operator-facing rationale for this "
        "disaster-triage rank item, using ONLY the facts listed below.\n\n"
        f"{semantics}\n"
        "Facts:\n"
        f"- label: {rank_item['label']}\n"
        f"- damage_class: {damage_class} ({severity})\n"
        f"- confidence: {rank_item['confidence']:.2f}\n"
        f"- confirmed: {rank_item['confirmed']}\n"
        f"- graded_by: {rank_item['graded_by']}\n"
        f"{facility_line}"
        f"- staleness_h: {inputs['staleness_h']:.1f} (hours)\n"
        f"- vulnerable_density: {inputs['vulnerable_density']:.2f} (dimensionless)\n"
        f"- doubt: {inputs['doubt']:.2f} (dimensionless)\n"
        f"{road_line}"
        f"- priority: {rank_item['priority']:.5f} (already computed; do not recompute)\n\n"
        "Do not invent casualties, occupancy, resource availability, or property value."
    )


_UNITLESS_FIELDS = ("road_cutoff", "vulnerable_density", "doubt")

_FORBIDDEN_UNIT_TOKENS = (
    "meters", "metre", "metres", "meter", "km", "kilometers", "kilometres",
    "miles", "mile", "mi", "feet", "foot", "ft", "yards", "yard", "yd",
    "hours", "hour", "hrs", "hr", "people", "persons", "person",
    "vehicles", "vehicle", "m",
)

_DIMENSIONLESS_QUALIFIERS = ("dimensionless", "score", "factor")

_FORBIDDEN_CONTENT_TERMS = (
    "property value", "casualt", "occupant", "resident", "people inside",
    "trapped", "injured", "fatal", "death", "$",
    "ambulance", "rescue team", "personnel available", "resource availab",
    "available resource", "ems unit", "fire truck",
)

_STALENESS_DECREASE_PATTERN = re.compile(
    r"(stale\w*)[^.]{0,40}(reduc\w*|lower\w*|decreas\w*|diminish\w*)"
    r"|(reduc\w*|lower\w*|decreas\w*|diminish\w*)[^.]{0,40}(stale\w*)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Damage-class faithfulness: the rationale must not state/imply a damage
# class that contradicts rank_item["damage_class"]. Conservative on
# purpose (Part 2 of the fix this belongs to): only flags an EXPLICIT
# canonical severity phrase or an explicit "class N" mention for some
# OTHER class than the authoritative one -- omitting damage wording
# entirely is never a violation. Phrases reuse _DAMAGE_SEVERITY verbatim
# (the same canonical wording rationale.py's own stub already produces)
# plus "undamaged" as a reasonable class-0 alias, to avoid inventing a
# parallel vocabulary that could drift from it.
# --------------------------------------------------------------------------

_DAMAGE_CLASS_PHRASES = {
    0: (_DAMAGE_SEVERITY[0], "undamaged"),
    1: (_DAMAGE_SEVERITY[1],),
    2: (_DAMAGE_SEVERITY[2],),
    3: (_DAMAGE_SEVERITY[3],),
}

_CLASS_DIGIT_PATTERN = re.compile(r"\bclass\s*([0-3])\b", re.IGNORECASE)


def _mentioned_damage_classes(text: str) -> set:
    lowered = text.lower()
    mentioned = set()
    for severity_class, phrases in _DAMAGE_CLASS_PHRASES.items():
        for phrase in phrases:
            if re.search(r"\b" + re.escape(phrase) + r"\b", lowered):
                mentioned.add(severity_class)
                break
    for match in _CLASS_DIGIT_PATTERN.finditer(lowered):
        mentioned.add(int(match.group(1)))
    return mentioned


def _damage_class_contradicted(text: str, damage_class: int) -> bool:
    """True iff `text` explicitly claims a severity phrase or "class N"
    for a class OTHER than `damage_class`. A rationale that never mentions
    damage wording at all returns False -- omission is never a violation.
    """
    return any(mentioned != damage_class for mentioned in _mentioned_damage_classes(text))


# --------------------------------------------------------------------------
# Facility faithfulness: the rationale must not invent a named or typed
# nearby facility that contradicts rank_item["facility_near"]. Small
# deterministic checks only -- no full named-entity recognition (Part 3).
# --------------------------------------------------------------------------

_FACILITY_TYPE_PHRASES = {
    "nursing_home": ("nursing home",),
    "dialysis": ("dialysis",),
    "hospital": ("hospital",),
}

# A capitalized name (1-5 words) immediately followed by a facility-type
# suffix word, e.g. "Mercy Hospital", "Riverside Dialysis Center" -- the
# proper-noun heuristic that lets this module compare an ASSERTED facility
# name against the authoritative one without real NER. Deliberately
# case-SENSITIVE on the captured words (a generic, lowercase "the dialysis
# facility" must never match this -- only an apparent proper name does).
_FACILITY_NAME_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,4}\s+(?:Hospital|Dialysis(?:\s+Center)?|Nursing\s+Home))\b"
)


def _mentioned_facility_types(text: str) -> set:
    lowered = text.lower()
    mentioned = set()
    for facility_type, phrases in _FACILITY_TYPE_PHRASES.items():
        for phrase in phrases:
            if re.search(r"\b" + re.escape(phrase) + r"\b", lowered):
                mentioned.add(facility_type)
                break
    return mentioned


def _facility_contradicted(text: str, facility_near) -> bool:
    """True iff `text` asserts a facility type or name that contradicts
    `facility_near` (None, or {"name", "type", "dist_m"}).

    facility_near is None: any mention of a known facility type
    (nursing_home/dialysis/hospital) is an invented facility -> True.

    facility_near present: a mentioned type that differs from the
    authoritative type is a contradiction; so is an apparent proper name
    (via _FACILITY_NAME_PATTERN) that neither contains nor is contained by
    the authoritative name (case-insensitive). A rationale that mentions
    the correct type generically ("a dialysis center") or the authoritative
    name itself is never flagged; omitting facility wording entirely is
    never a violation either.
    """
    mentioned_types = _mentioned_facility_types(text)

    if facility_near is None:
        return bool(mentioned_types)

    authoritative_type = facility_near.get("type")
    if any(mentioned_type != authoritative_type for mentioned_type in mentioned_types):
        return True

    authoritative_name = re.sub(r"\s+", " ", (facility_near.get("name") or "").strip().lower())
    for match in _FACILITY_NAME_PATTERN.finditer(text):
        candidate_name = re.sub(r"\s+", " ", match.group(1).strip().lower())
        if authoritative_name and (candidate_name in authoritative_name or authoritative_name in candidate_name):
            continue
        return True

    return False


def _numeric_variants(value) -> set:
    variants = {str(value)}
    try:
        variants.add(f"{value:.1f}")
        variants.add(f"{value:.2f}")
        variants.add(f"{value:g}")
    except (TypeError, ValueError):
        pass
    return variants


def _value_carries_forbidden_unit(text: str, value) -> bool:
    if value is None:
        return False
    unit_alternation = "|".join(_FORBIDDEN_UNIT_TOKENS)
    for variant in _numeric_variants(value):
        pattern = re.escape(variant) + r"\s?(" + unit_alternation + r")\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _percent_sign_without_dimensionless_label(text: str) -> bool:
    if "%" not in text:
        return False
    lowered = text.lower()
    return not any(qualifier in lowered for qualifier in _DIMENSIONLESS_QUALIFIERS)


def _staleness_described_as_decreasing_priority(text: str) -> bool:
    return bool(_STALENESS_DECREASE_PATTERN.search(text))


def _contains_forbidden_content(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _FORBIDDEN_CONTENT_TERMS)


def _faithfulness_violations(text: str, rank_item: dict) -> list:
    """Check real-model rationale text against FIRST LIGHT's frozen semantics.

    road_cutoff, vulnerable_density, doubt, and priority are dimensionless and
    must never carry a fabricated physical/temporal/headcount unit; only
    facility_near.dist_m legitimately carries meters; staleness must be
    described as increasing (never decreasing) priority; the rationale
    must never invent casualties, occupancy, resource availability, or
    property value; it must never state a damage-class severity that
    contradicts rank_item["damage_class"] (see _damage_class_contradicted);
    and it must never assert a facility name/type that contradicts
    rank_item["facility_near"] (see _facility_contradicted). Returns a list
    of human-readable violation strings -- empty means the text is
    faithful.
    """
    violations = []
    inputs = rank_item["inputs"]

    for field in _UNITLESS_FIELDS:
        value = inputs.get(field)
        if _value_carries_forbidden_unit(text, value):
            violations.append(f"{field}={value} appears with a fabricated physical unit")

    if _value_carries_forbidden_unit(text, rank_item.get("priority")):
        violations.append("priority appears with a fabricated physical unit")

    if _percent_sign_without_dimensionless_label(text):
        violations.append("a '%' sign appears without a dimensionless-score qualifier")

    if _staleness_described_as_decreasing_priority(text):
        violations.append("staleness is described as decreasing priority")

    if _contains_forbidden_content(text):
        violations.append("rationale mentions an unsupported real-world quantity")

    if _damage_class_contradicted(text, rank_item["damage_class"]):
        violations.append(f"rationale contradicts authoritative damage_class={rank_item['damage_class']}")

    if _facility_contradicted(text, rank_item.get("facility_near")):
        violations.append("rationale asserts a facility name/type that contradicts facility_near")

    return violations


def _post_chat_completion(
    messages: list,
    thinking: bool,
    base_url: str,
    timeout_s: float,
    response_format: dict = None,
    max_tokens: int = None,
    usage_sink: list = None,
    finish_reason_sink: list = None,
) -> str:
    """POST one chat-completion request to <base_url>/v1/chat/completions for
    the "nano" served model and return the assistant message content.

    thinking=False prepends a "/no_think" system message (cheap/structured
    operations, used by hero rationale and agency-plan drafting today);
    thinking=True prepends "/think" (the reasoning replan beat -- not wired
    to any caller yet). This helper is intentionally generic so any real
    caller needing structured or reasoning output reuses it instead of
    duplicating HTTP handling.

    response_format, when given, is passed through verbatim (e.g.
    {"type": "json_schema", "json_schema": {...}} -- the vLLM OpenAI-
    compatible structured-output mechanism; top-level guided_json is
    silently ignored on this build, so callers needing schema-constrained
    JSON must use this instead). max_tokens, when given, caps completion
    length. Both are omitted from the payload when None, so existing
    callers that don't pass them (hero rationale) are unaffected.

    usage_sink, when given, gets the response's "usage" object appended to
    it (list.append is GIL-atomic, so this is safe to share across
    concurrent calls, unlike a plain instance-attribute reassignment) --
    purely an optional diagnostic side channel, never required and never
    part of the return value. finish_reason_sink, when given, gets the
    response's "finish_reason" string appended to it the same way -- by
    the time this function returns normally, finish_reason is never
    "length" (that path already raised), so this is diagnostic-only (e.g.
    confirming "stop" for a live-check script), never a second success/
    failure signal callers need to check.

    Raises NanoClientError for any timeout, connection, HTTP, malformed-
    response, or truncated-output (finish_reason == "length") failure --
    a truncated response is never treated as a successful completion,
    regardless of caller.
    """
    directive = "/think" if thinking else "/no_think"
    payload = {
        "model": _MODEL_NAME,
        "messages": [{"role": "system", "content": directive}, *messages],
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw_body = response.read()
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise NanoClientError(f"Nano request to {base_url} failed: {exc}") from exc

    try:
        parsed = json.loads(raw_body)
        choice = parsed["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise NanoClientError(f"Nano response was malformed: {exc}") from exc

    if finish_reason == "length":
        raise NanoClientError("Nano response was truncated by max_tokens (finish_reason=length)")

    if not isinstance(content, str) or not content.strip():
        raise NanoClientError("Nano response contained no usable content")

    if usage_sink is not None:
        usage = parsed.get("usage")
        if usage is not None:
            usage_sink.append(usage)

    if finish_reason_sink is not None:
        finish_reason_sink.append(finish_reason)

    return content.strip()
