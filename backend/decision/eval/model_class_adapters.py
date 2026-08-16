"""B8 Part C2: EVAL-ONLY adapter giving Nano an independent damage-class
assessment to compare against Lightning's.

No suitable Nano structured-class helper exists in production
(nano_client.py only ever produces rationale TEXT, never a class label,
and agency_plan_client.py's schema is agency/action/units, not a damage
class) -- per instructions, this adds a small EVAL-ONLY adapter here
rather than modifying nano_client.py/production grading logic. Nothing in
backend/decision/*.py (outside this eval package) imports this module.

"Use the SAME concise text evidence" (Part 6): rather than writing a
second prompt-building function that could silently drift from
Lightning's, this reuses lightning_client._ballot_prompt(building_context)
verbatim for BOTH models -- Nano and Lightning are asked to classify the
literal same prompt text, which is what makes the resulting agreement
number mean anything. Reusing a "private" (`_`-prefixed) function across
modules within the same backend.decision package is a deliberate,
documented exception here, made only for this reason.

This function never writes to any RankItem/BuildingEvidence field and is
called only by B8 evaluation code -- it does not change, and cannot be
reached from, production grading architecture.
"""

import json

from backend.decision.lightning_client import _ballot_prompt
from backend.decision.nano_client import NanoClientError, _post_chat_completion, _resolve_base_url

_VALID_LABELS = ("0", "1", "2", "3")

_CLASS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "eval_damage_class_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"damage_class": {"type": "string", "enum": list(_VALID_LABELS)}},
            "required": ["damage_class"],
            "additionalProperties": False,
        },
    },
}

_MAX_TOKENS = 20
_DEFAULT_TIMEOUT_S = 10.0


def nano_assess_damage_class(building_context: dict, base_url: str = None, timeout_s: float = _DEFAULT_TIMEOUT_S) -> int:
    """Ask Nano (served model "nano", /no_think, json_schema) for an
    independent damage_class assessment in {0,1,2,3} for the SAME
    building_context lightning_ballot.request_lightning_ballot already
    grades, using lightning_client._ballot_prompt as the shared prompt
    text. Raises NanoClientError on any transport/malformed-response/
    out-of-range failure -- callers (eval_lightning_agreement.py) decide
    whether to skip that sample or abort.
    """
    active_base_url = base_url if base_url is not None else _resolve_base_url()
    messages = [{"role": "user", "content": _ballot_prompt(building_context)}]

    content = _post_chat_completion(
        messages,
        thinking=False,
        base_url=active_base_url,
        timeout_s=timeout_s,
        response_format=_CLASS_SCHEMA,
        max_tokens=_MAX_TOKENS,
    )

    try:
        parsed = json.loads(content)
        label = parsed["damage_class"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise NanoClientError(f"Nano class-assessment response was malformed: {exc}") from exc

    if label not in _VALID_LABELS:
        raise NanoClientError(f"Nano class-assessment must be one of {_VALID_LABELS}, got {label!r}")

    return int(label)
