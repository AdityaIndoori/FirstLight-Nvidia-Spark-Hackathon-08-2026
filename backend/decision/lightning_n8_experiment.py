"""EXPERIMENTAL: single-request n=8 Lightning ballot, for throughput
comparison against the production 8-separate-requests path ONLY.

Production ballot behavior (lightning_ballot.request_lightning_ballot,
lightning_client.RealLightningSeverityClient) is NOT modified or replaced
by this module. This is a standalone experiment: does vLLM's n parameter
(one ChatCompletionRequest returning n independently sampled choices) beat
eight separate n=1 requests for the same k=8 ballot? Measure first; nothing
here assumes an answer.

Reuses, never duplicates:
- lightning_client._ballot_prompt   -- the EXACT same prompt as production
- lightning_ballot._aggregate_votes -- the EXACT same voted_class/
  vote_agreement/doubt aggregation as production

Only the wire request differs: one POST with "n": 8 instead of eight POSTs
each with an implicit n=1. temperature, chat_template_kwargs.enable_thinking,
structured_outputs.choice, model, and max_tokens are all identical to the
production single-vote request.
"""

import json
import socket
import urllib.error
import urllib.request

from backend.decision.lightning_ballot import K_VOTES, _DEFAULT_TEMPERATURE, _aggregate_votes
from backend.decision.lightning_client import (
    LightningClientError,
    _DEFAULT_TIMEOUT_S,
    _MAX_TOKENS,
    _MODEL_NAME,
    _VALID_CHOICE_STRINGS,
    _ballot_prompt,
    _resolve_base_url,
)


def _post_n8_votes(building_context: dict, temperature: float, base_url: str, timeout_s: float) -> tuple:
    """POST ONE chat-completion request with n=K_VOTES to
    <base_url>/v1/chat/completions for the "lightning" served model, using
    the same prompt/model/temperature/structured-decoding as the production
    single-vote request -- only "n" differs.

    Returns (votes: list[int] length K_VOTES, usage: dict or None).

    Raises LightningClientError for any timeout, connection, HTTP,
    malformed-response, wrong-choice-count, missing-content, or
    out-of-range-label failure. Never silently accepts fewer/extra/garbled
    choices.
    """
    payload = {
        "model": _MODEL_NAME,
        "messages": [{"role": "user", "content": _ballot_prompt(building_context)}],
        "n": K_VOTES,
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
        raise LightningClientError(f"Lightning n={K_VOTES} request to {base_url} failed: {exc}") from exc

    try:
        parsed = json.loads(raw_body)
        choices = parsed["choices"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LightningClientError(f"Lightning n={K_VOTES} response was malformed: {exc}") from exc

    if not isinstance(choices, list) or len(choices) != K_VOTES:
        got = len(choices) if isinstance(choices, list) else choices
        raise LightningClientError(
            f"Lightning n={K_VOTES} response must contain exactly {K_VOTES} choices, got {got!r}"
        )

    votes = []
    for choice in choices:
        try:
            content = choice["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LightningClientError(f"Lightning n={K_VOTES} choice was malformed: {exc}") from exc

        if not isinstance(content, str) or not content.strip():
            raise LightningClientError(f"Lightning n={K_VOTES} choice contained no usable content")

        label_text = content.strip()
        if label_text not in _VALID_CHOICE_STRINGS:
            raise LightningClientError(
                f"Lightning n={K_VOTES} label must be one of {_VALID_CHOICE_STRINGS}, got {label_text!r}"
            )
        votes.append(int(label_text))

    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    return votes, usage


def request_n8_ballot(
    building_context: dict,
    base_url: str = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict:
    """Experimental single-request n=8 ballot for ONE building.

    Identical result contract to lightning_ballot.request_lightning_ballot
    (votes/voted_class/vote_agreement/doubt via the SAME _aggregate_votes),
    but the eight votes come from one HTTP request with n=8 instead of
    eight separate requests. Does not mutate building_context.
    """
    resolved_base_url = base_url if base_url is not None else _resolve_base_url()
    votes, _usage = _post_n8_votes(
        building_context, temperature=temperature, base_url=resolved_base_url, timeout_s=timeout_s
    )
    return _aggregate_votes(votes)
