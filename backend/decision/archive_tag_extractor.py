"""B7 Part 8: Lightning batch tag extraction.

extract_tags_batch(captions) -> list[list[str]], one tag list per caption,
same order. Lightning receives ONLY captions (plain strings) -- no pixels,
no damage grades -- and returns a fixed-order JSON array of tag lists in
ONE request for the whole batch (README: "Lightning extracts the tag list
from every caption in one batched sweep... thousands of short structured
generations is precisely its sweet spot"), reusing
lightning_client._post_chat_completion (added for this purpose) rather
than one serial request per caption.

Tags are short, lowercase, factual, and describe structures/terrain/water/
damage concepts only -- never personal identity, never inferred
casualties, never an invented fact. Every real-model tag is passed through
the SAME deterministic post-validation the stub's tags trivially already
satisfy (_normalize_and_validate_tags): normalize (strip/lowercase/dedupe),
drop any tag containing an obviously prohibited person/body/clothing
concept, and drop any tag not actually supported by the caption text --
this mirrors the same grounding discipline agency_plan_client.py's
is_action_supported already applies to Nano's agency assignments, applied
here to Lightning's tags. Tag extraction never reads or writes a damage
class -- TagExtractor's signature has no such parameter, by construction.

Normal pytest never calls LightningTagExtractor -- see
DeterministicStubTagExtractor, a keyword-rule fallback with the same
interface. If Lightning fails, callers use the existing project recovery
convention (try real, fall back to stub, record which one ran) -- this
module does not enforce that policy itself; see
scripts/archive_tagging_live_check.py and archive_write.py for callers
that choose to apply it.
"""

import json
import re
from abc import ABC, abstractmethod

from backend.decision.lightning_client import (
    LightningClientError,
    _post_chat_completion,
    _resolve_base_url,
)

_DEFAULT_TIMEOUT_S = 10.0
_MAX_TOKENS_PER_CAPTION = 100
"""Budget per caption for the batch completion -- a handful of short tags
each, plus JSON array/quote/comma overhead for the WHOLE batch's structured
response; scaled by batch size when building the request's max_tokens. A
live run against the real Lightning server at 40/caption hit
finish_reason=="length" on an 8-caption batch -- see
scripts/archive_tagging_live_check.py; 100/caption was measured to clear
it. finish_reason=="length" is never silently accepted regardless (see
lightning_client._post_chat_completion), so a too-low value fails loudly,
never truncates a tag list unnoticed."""

_TAG_TEMPERATURE = 0.0
"""Deliberately NOT the 0.7 self-consistency-ballot convention
(lightning_client._DEFAULT_TEMPERATURE) -- that value exists to seek
sampling DIVERSITY for k=8 voting. Tag extraction is a factual/structured
task where the whole point is a single deterministic, conservative
answer, so the more natural convention is the one nano_client.py and
agency_plan_client.py already use for their own /no_think structured
calls: no encouragement of variance."""

_MAX_BATCH_SIZE = 32
"""Simple chunking guard so one call never builds an unbounded prompt --
the fixture-sized batches (~10) this project needs today never hit it."""

_PROHIBITED_TAG_TERMS = (
    "person", "people", "man", "woman", "men", "women", "child", "children",
    "kid", "baby", "infant", "adult", "male", "female", "human", "body",
    "corpse", "victim", "casualt", "injured", "resident", "occupant",
    "survivor", "rescuer", "firefighter", "officer", "face", "hand", "arm",
    "leg", "torso", "clothing", "shirt", "pants", "jacket", "helmet",
    "uniform", "shoe", "hat",
)

_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "near", "to", "from"}
)

_TAG_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class TagExtractor(ABC):
    """Boundary: captions in, one tag list per caption out, same order,
    same length. Implementations must never mutate their input list.
    """

    @abstractmethod
    def extract_tags_batch(self, captions: list) -> list:
        raise NotImplementedError


def _tokenize(text: str) -> list:
    return _TAG_TOKEN_PATTERN.findall(text.lower())


def _tag_contains_prohibited_term(tag: str) -> bool:
    lowered = tag.lower()
    return any(term in lowered for term in _PROHIBITED_TAG_TERMS)


def _tag_supported_by_caption(tag: str, caption: str) -> bool:
    """Conservative, deterministic grounding check -- every non-stopword
    word in the tag must appear in the caption, tolerating a trailing
    's'/'d'/'ed' mismatch (e.g. tag "roof collapse" against caption
    "...roof collapsed..."). Never a second model/NLP classifier.
    """
    caption_lower = caption.lower()
    words = [w for w in _tokenize(tag) if w not in _STOPWORDS]
    if not words:
        return False
    for word in words:
        if word in caption_lower:
            continue
        if len(word) > 3 and word[:-1] in caption_lower:
            continue
        if len(word) > 3 and (word + "d") in caption_lower:
            continue
        if len(word) > 3 and (word + "ed") in caption_lower:
            continue
        return False
    return True


def _normalize_and_validate_tags(raw_tags: list, caption: str, check_grounding: bool = True) -> list:
    """Normalize (strip/lowercase), drop prohibited-term tags, optionally
    drop tags not grounded in `caption`, dedupe preserving first-seen
    order. Never raises -- an ungrounded/prohibited/malformed tag is
    silently dropped from THIS caption's list, not a batch-wide failure.

    check_grounding=False is used ONLY by DeterministicStubTagExtractor
    (see its docstring): its tags are grounded by construction (each one
    was emitted because ITS OWN trigger keyword matched the caption), but
    the word-by-word _tag_supported_by_caption check compares the
    CANONICAL tag text, which sometimes differs from the trigger phrase
    (e.g. the "no visible damage" trigger emits the canonical tag
    "undamaged" -- a word that never appears in the caption itself), which
    would incorrectly reject an already-grounded stub tag. Real model
    output (LightningTagExtractor) always uses check_grounding=True, the
    default -- grounding validation there is safety-critical, never
    skipped.
    """
    seen = set()
    result = []
    for raw_tag in raw_tags:
        if not isinstance(raw_tag, str):
            continue
        tag = re.sub(r"\s+", " ", raw_tag.strip().lower())
        if not tag or tag in seen:
            continue
        if _tag_contains_prohibited_term(tag):
            continue
        if check_grounding and not _tag_supported_by_caption(tag, caption):
            continue
        seen.add(tag)
        result.append(tag)
    return result


_KEYWORD_TAG_RULES = (
    (("fire", "flame", "burning"), "fire"),
    (("smoke",), "smoke"),
    (("collapse", "collapsed"), "collapse"),
    (("roof",), "roof damage"),
    (("flood", "flooded", "standing water"), "standing water"),
    (("debris",), "debris"),
    (("crack", "cracking"), "cracking"),
    (("undamaged", "no visible damage", "no structural"), "undamaged"),
    (("dialysis", "hospital", "nursing home", "medical"), "medical facility"),
    (("commercial",), "commercial structure"),
    (("wood structure",), "wood structure"),
)


class DeterministicStubTagExtractor(TagExtractor):
    """Small keyword-rule fallback, no LLM, no network. Iterates a fixed,
    ordered rule table so output is always deterministic. Grounded by
    construction: every emitted tag is only added because ITS OWN trigger
    keyword(s) matched the caption -- so tags pass through
    _normalize_and_validate_tags with check_grounding=False (still
    normalized/deduped, still prohibited-term-filtered) rather than being
    re-checked word-by-word against the caption, which would incorrectly
    reject a canonical tag whose text differs from its trigger phrase
    (e.g. trigger "no visible damage" -> canonical tag "undamaged" -- a
    B8 eval run surfaced this as a real recall defect before this fix).
    """

    is_stub = True

    def extract_tags_batch(self, captions: list) -> list:
        return [self._tags_for_one(caption) for caption in captions]

    def _tags_for_one(self, caption: str) -> list:
        lowered = caption.lower()
        raw_tags = [tag for keywords, tag in _KEYWORD_TAG_RULES if any(k in lowered for k in keywords)]
        return _normalize_and_validate_tags(raw_tags, caption, check_grounding=False)


_TAG_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "caption_tag_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}


def _batch_prompt(captions: list) -> str:
    numbered = "\n".join(f"{i + 1}. \"{caption}\"" for i, caption in enumerate(captions))
    return (
        "For EACH numbered caption below, write 2-4 short factual CONCEPT "
        "tags (1-3 words each) describing only structures, terrain, water, "
        "and damage concepts actually stated in that caption. Never mention "
        "a person, body part, clothing, or any human presence. Never invent "
        "a casualty, occupant, or fact not stated in the caption.\n\n"
        "A tag is a CONCEPT, not a restatement of the sentence -- never "
        "output a determiner, preposition, or filler word (\"the\", \"on\", "
        "\"with\", \"near\", \"shows\", \"visible\") as its own tag, and "
        "never split one caption into one tag per word.\n\n"
        "Example:\n"
        "caption: \"two-storey wood structure with roof collapsed and "
        "standing water in street\"\n"
        'tags: ["wood structure", "roof collapse", "standing water"]\n\n'
        "Return exactly one tag list per caption, in the SAME order, as "
        'JSON: {"results": [[...tags for caption 1...], [...tags for '
        "caption 2...], ...]}.\n\n"
        f"Captions:\n{numbered}"
    )


class LightningTagExtractor(TagExtractor):
    """Real Nemotron 3.5 Lightning client for batch tag extraction, via the
    OpenAI-compatible vLLM API (served model name "lightning"). Every
    request is /no_think and structured-decoded with a json_schema
    response_format constraining the output to {"results": [[str, ...],
    ...]} -- one tag list per input caption, in order. temperature=0.0
    (see _TAG_TEMPERATURE) -- a deliberate departure from the 0.7
    self-consistency-ballot convention, since tagging wants one
    deterministic factual answer, not sampling diversity.

    Batches of more than _MAX_BATCH_SIZE captions are chunked into
    multiple requests (still far fewer than one request per caption).
    Raises LightningClientError (re-raised from
    lightning_client._post_chat_completion, plus this class's own
    response-shape/length-mismatch checks) on any failure -- never falls
    back to the stub itself; that decision belongs to the caller.

    Every real-model tag is passed through _normalize_and_validate_tags
    before being returned, exactly like the stub -- Lightning's raw output
    is never trusted verbatim.
    """

    is_stub = False

    def __init__(self, base_url: str = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
        self.base_url = base_url if base_url is not None else _resolve_base_url()
        self.timeout_s = timeout_s
        self.usage_log = []

    def extract_tags_batch(self, captions: list) -> list:
        results = []
        for start in range(0, len(captions), _MAX_BATCH_SIZE):
            chunk = captions[start : start + _MAX_BATCH_SIZE]
            results.extend(self._extract_chunk(chunk))
        return results

    def _extract_chunk(self, captions: list) -> list:
        if not captions:
            return []

        messages = [{"role": "user", "content": _batch_prompt(captions)}]
        content = _post_chat_completion(
            messages,
            thinking=False,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            response_format=_TAG_SCHEMA,
            max_tokens=_MAX_TOKENS_PER_CAPTION * len(captions),
            temperature=_TAG_TEMPERATURE,
            usage_sink=self.usage_log,
        )

        try:
            parsed = json.loads(content)
            raw_results = parsed["results"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise LightningClientError(f"Lightning tag batch response was malformed: {exc}") from exc

        if not isinstance(raw_results, list) or len(raw_results) != len(captions):
            raise LightningClientError(
                f"Lightning tag batch returned {len(raw_results) if isinstance(raw_results, list) else 'non-list'} "
                f"result(s) for {len(captions)} caption(s)"
            )

        return [
            _normalize_and_validate_tags(raw_tags if isinstance(raw_tags, list) else [], caption)
            for raw_tags, caption in zip(raw_results, captions)
        ]
