"""Client boundary for the Lightning severity-voting operation.

Application code depends only on LightningSeverityClient.sample_severity(
building_context, temperature). StubLightningSeverityClient is the
deterministic fallback; RealLightningSeverityClient is the HTTP-backed
client for the actual Nemotron 3.5 Lightning server (served model name
"lightning") behind the OpenAI-compatible vLLM API. Both implement the same
interface, so lightning_ballot.py never changes based on which is active.

Only RealLightningSeverityClient touches the network, and only inside
sample_severity() -- constructing it performs no I/O.
"""

import json
import os
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

_DEFAULT_TEMPERATURE = 0.7

_BASE_URL_ENV_VAR = "FIRSTLIGHT_LIGHTNING_BASE_URL"
_DEFAULT_BASE_URL = "http://localhost:8001"
_MODEL_NAME = "lightning"
_DEFAULT_TIMEOUT_S = 10.0
_MAX_TOKENS = 4
_VALID_CHOICE_STRINGS = ("0", "1", "2", "3")


class LightningClientError(Exception):
    """Raised by RealLightningSeverityClient for any timeout, connection,
    HTTP, malformed-response, missing-content, or out-of-range-label
    failure. A single error type so callers can catch one thing and fall
    back to the stub, instead of a model failure being silently hidden.
    """


class LightningSeverityClient(ABC):
    """Boundary: returns ONE sampled severity vote (0-3) for one building.

    building_context is the internal ballot input defined in
    lightning_ballot.py -- NOT the frozen A -> B RankItem contract.
    temperature is preserved on the call signature so the real Lightning
    client can sample at temperature=0.7 once it exists; the deterministic
    stub accepts and ignores it.
    """

    @abstractmethod
    def sample_severity(self, building_context: dict, temperature: float = _DEFAULT_TEMPERATURE) -> int:
        """Return exactly one severity label in {0, 1, 2, 3}."""
        raise NotImplementedError


class StubLightningSeverityClient(LightningSeverityClient):
    """Deterministic fallback standing in for the real Lightning client.

    With no configured votes, always returns building_context["grader_class"]
    (unanimous agreement with the VL model's primary grade) -- a safe,
    deterministic default. Tests that need a specific vote distribution
    (e.g. a 6/8 split) pass votes=[...] explicitly; calls cycle through that
    list in call order. Never random, never networked.
    """

    is_stub = True

    def __init__(self, votes: list = None):
        self.votes = list(votes) if votes is not None else None
        self._calls = 0

    def sample_severity(self, building_context: dict, temperature: float = _DEFAULT_TEMPERATURE) -> int:
        self._calls += 1
        if self.votes is not None:
            return self.votes[(self._calls - 1) % len(self.votes)]
        return building_context["grader_class"]


class RealLightningSeverityClient(LightningSeverityClient):
    """Real Nemotron 3.5 Lightning client, via the OpenAI-compatible vLLM API
    (served model name "lightning"). Lightning is TEXT-ONLY -- it never sees
    pixels. It cross-examines the image-capable VL model's (Nemotron Nano
    12B v2, :8002) primary grade, confidence, and independently generated
    caption against GIS context, and is explicitly told not to assume the
    grade is correct. Every request is structured-decoded to exactly one of
    "0"/"1"/"2"/"3" via structured_outputs.choice, and sent with
    chat_template_kwargs.enable_thinking=false -- the natural-language
    prompt is never relied on alone to constrain the output.

    base_url defaults to the FIRSTLIGHT_LIGHTNING_BASE_URL environment
    variable, falling back to http://localhost:8001 -- during Mac
    development this reaches the DGX Spark's vLLM server through an SSH
    tunnel. Every call has a hard timeout (default 10s). Raises
    LightningClientError on any failure (timeout, connection, HTTP,
    malformed JSON, missing content, out-of-range label); this class never
    falls back to the stub itself -- that decision belongs to the caller --
    so a model failure is never silently hidden.

    lightning_ballot.request_lightning_ballot() calls sample_severity()
    exactly k=8 times per ballot to get eight independently sampled
    generations; this class never batches those into a single request.

    usage_log accumulates each response's "usage" object (prompt_tokens/
    completion_tokens), if the server includes one, purely for callers that
    want to introspect token counts (e.g. the batch baseline benchmark). It
    is empty if the server never sends usage, and sample_severity's return
    value/contract is unaffected either way.
    """

    is_stub = False

    def __init__(self, base_url: str = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
        self.base_url = base_url if base_url is not None else _resolve_base_url()
        self.timeout_s = timeout_s
        self.usage_log = []

    def sample_severity(self, building_context: dict, temperature: float = _DEFAULT_TEMPERATURE) -> int:
        label, usage = _post_severity_vote(
            building_context, temperature=temperature, base_url=self.base_url, timeout_s=self.timeout_s
        )
        if usage is not None:
            self.usage_log.append(usage)
        return label


def _resolve_base_url() -> str:
    return os.environ.get(_BASE_URL_ENV_VAR, _DEFAULT_BASE_URL)


def _ballot_prompt(building_context: dict) -> str:
    """Build the one-building severity-voting prompt from the current
    internal ballot inputs (lightning_ballot.py): the VL model's primary
    grade + confidence, its independently generated visual caption, and GIS
    context (footprint area, facility proximity, neighbor damage classes).
    Structured decoding (structured_outputs.choice) is what actually
    constrains the output to 0-3; this prompt gives the model the full
    evidence and explicitly tells it the grade is not ground truth.
    """
    grader_class = building_context["grader_class"]
    grader_confidence = building_context["grader_confidence"]
    vl_caption = building_context["vl_caption"]
    footprint_area_m2 = building_context["footprint_area_m2"]
    facility_context = building_context.get("facility_context")
    neighbor_damage_classes = building_context.get("neighbor_damage_classes", [])

    facility_line = (
        f"- facility_context: {facility_context}\n"
        if facility_context is not None
        else "- facility_context: none\n"
    )

    return (
        "Grade post-disaster damage severity for ONE building. Respond with "
        "exactly one severity label: 0 = no damage, 1 = minor damage, "
        "2 = major damage, 3 = destroyed.\n\n"
        "The class and caption below are two INDEPENDENTLY generated "
        "observations from the image-capable model (Nemotron Nano 12B v2 VL). "
        "Select the severity label most consistent with ALL supplied "
        "evidence. Do not assume the primary grader class is correct -- if "
        "the structured grade conflicts with the visual caption or the "
        "surrounding context, weigh the complete evidence rather than "
        "blindly copying the grader class.\n\n"
        "Primary visual grader:\n"
        f"- class: {grader_class}\n"
        f"- confidence: {grader_confidence:.2f}\n\n"
        "Independent visual description:\n"
        f"- VL caption: \"{vl_caption}\"\n\n"
        "Context:\n"
        f"- footprint area: {footprint_area_m2:.1f} m^2\n"
        f"{facility_line}"
        f"- neighboring building damage classes: {neighbor_damage_classes}\n"
    )


def _post_severity_vote(building_context: dict, temperature: float, base_url: str, timeout_s: float) -> tuple:
    """POST one structured-choice chat-completion request to
    <base_url>/v1/chat/completions for the "lightning" served model and
    return (label: int, usage: dict or None). usage is the response's
    "usage" object verbatim when present (e.g. {"prompt_tokens": ...,
    "completion_tokens": ...}), else None -- never fabricated.

    Raises LightningClientError for any timeout, connection, HTTP,
    malformed-response, missing-content, or out-of-range-label failure.
    """
    payload = {
        "model": _MODEL_NAME,
        "messages": [{"role": "user", "content": _ballot_prompt(building_context)}],
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"choice": list(_VALID_CHOICE_STRINGS)},
        "temperature": temperature,
        "max_tokens": _MAX_TOKENS,
    }
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
        raise LightningClientError(f"Lightning request to {base_url} failed: {exc}") from exc

    try:
        parsed = json.loads(raw_body)
        content = parsed["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LightningClientError(f"Lightning response was malformed: {exc}") from exc

    if not isinstance(content, str) or not content.strip():
        raise LightningClientError("Lightning response contained no usable content")

    label_text = content.strip()
    if label_text not in _VALID_CHOICE_STRINGS:
        raise LightningClientError(
            f"Lightning label must be one of {_VALID_CHOICE_STRINGS}, got {label_text!r}"
        )

    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    return int(label_text), usage


def _post_chat_completion(
    messages: list,
    thinking: bool,
    base_url: str,
    timeout_s: float,
    response_format: dict = None,
    max_tokens: int = None,
    temperature: float = None,
    usage_sink: list = None,
) -> str:
    """POST one chat-completion request to <base_url>/v1/chat/completions
    for the "lightning" served model and return the assistant message
    content. Generic sibling of _post_severity_vote (which is hardcoded to
    the k=8 ballot's single-token structured_outputs.choice shape) and of
    nano_client._post_chat_completion (same idea, different served model)
    -- added here so any other real Lightning caller needing structured
    JSON output (e.g. B7 batch tag extraction, archive_tag_extractor.py)
    reuses this HTTP boilerplate/error handling instead of duplicating it.
    Does not change _post_severity_vote or sample_severity's behavior.

    thinking=False prepends a "/no_think" system message; response_format,
    when given, is passed through verbatim (the vLLM OpenAI-compatible
    structured-output mechanism). max_tokens/temperature, when given, are
    included in the payload; omitted entirely when None so behavior for
    any future caller that doesn't pass them is unaffected.

    Raises LightningClientError for any timeout, connection, HTTP,
    malformed-response, missing-content, or truncated-output
    (finish_reason == "length") failure -- truncation is never treated as
    a successful completion.
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
    if temperature is not None:
        payload["temperature"] = temperature

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
        raise LightningClientError(f"Lightning request to {base_url} failed: {exc}") from exc

    try:
        parsed = json.loads(raw_body)
        choice = parsed["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LightningClientError(f"Lightning response was malformed: {exc}") from exc

    if finish_reason == "length":
        raise LightningClientError("Lightning response was truncated by max_tokens (finish_reason=length)")

    if not isinstance(content, str) or not content.strip():
        raise LightningClientError("Lightning response contained no usable content")

    if usage_sink is not None:
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        if usage is not None:
            usage_sink.append(usage)

    return content.strip()
