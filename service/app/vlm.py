"""Thin OpenAI-compatible client for the three local vLLM servers.

WHY stdlib urllib and not a client library: this box is offline by policy and the
only sanctioned destinations are localhost:8000/8001/8002. A dependency that can
be pointed at a hostname by a hostile caption is a dependency we do not want.
The opener below is built with an empty ProxyHandler so an inherited proxy
environment variable can never redirect a "localhost" call off the box.

WHY the odd structured-output dance: verified on this build (README section 4),
a top-level `guided_json` parameter is SILENTLY IGNORED. Pass objects as
`response_format={"type": "json_schema", "json_schema": {"name": n, "schema": s}}`
and enumerated picks as `guided_choice=[...]`. Never use `guided_json`.

WHY every call returns a `how` label: A3 and technique 6 of the plan. A wedged
endpoint must degrade to a deterministic labelled fallback with an identical
signature, never stall ingest and never pretend a stub was a model.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from PIL import Image

from . import config

# PUBLIC API
# Endpoint dataclass: .url .model .timeout .name
# nano() -> Endpoint
# lightning() -> Endpoint
# vl() -> Endpoint
# chat(endpoint, messages, *, schema=None, schema_name="response", choice=None,
#      max_tokens=512, temperature=0.0, timeout=None) -> tuple[str, str]
#     returns (text, how) where how is "model" or "stub"
# caption_and_grade(image_path_or_pil, *, crop_box=None, endpoint=None, timeout=None)
#     -> {"class": 0-3, "caption": str, "conf": float, "how": "model"|"stub"}
# stub_grade(image_path_or_pil, *, crop_box=None) -> same dict, always how="stub"
# caption_mentions_person(text) -> bool          # A6 caption post-filter
# stats() -> {"model": int, "stub": int}         # HUD "model recovered" vs "stub engaged"
# GRADE_SCHEMA, STUB_CAPTION, GRADE_HOW_MODEL, GRADE_HOW_STUB

log = logging.getLogger("firstlight.vlm")

GRADE_HOW_MODEL = "model"
GRADE_HOW_STUB = "stub"

# A neutral placeholder. It must read as "we did not look", never as an observation.
STUB_CAPTION = "no model caption available, automated pixel-statistic grade only"

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_STATS = {GRADE_HOW_MODEL: 0, GRADE_HOW_STUB: 0}
_STATS_LOCK = threading.Lock()

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

MAX_CAPTION_CHARS = 240
# One structure crop does not need full resolution, and token count is latency.
VL_MAX_SIDE = 768


# --------------------------------------------------------------------- endpoints
@dataclass(frozen=True)
class Endpoint:
    """One vLLM server. `name` is what the status bar prints."""

    url: str
    model: str
    timeout: float
    name: str


def _base(url: str) -> str:
    return url.rstrip("/")


def nano() -> Endpoint:
    """Nemotron Nano 9B v2, the decision-maker. Read at call time so tests can patch config."""
    return Endpoint(_base(config.NANO_URL), config.NANO_MODEL, config.LLM_TIMEOUT_S, "nano")


def lightning() -> Endpoint:
    """Nemotron 3.5 Lightning, text-only. It never sees pixels, it cross-examines."""
    return Endpoint(
        _base(config.LIGHTNING_URL), config.LIGHTNING_MODEL, config.LLM_TIMEOUT_S, "lightning"
    )


def vl() -> Endpoint:
    """Nemotron Nano 12B v2 VL, the only model here that sees an image."""
    return Endpoint(_base(config.VL_URL), config.VL_MODEL, config.VL_TIMEOUT_S, "vl")


# ------------------------------------------------------------- circuit breaker
# WHY: a wedged endpoint costs one full timeout PER CALL, and a tile grades up to
# a dozen crops. Twelve consecutive 20 s timeouts is four minutes on a tile whose
# budget is ten seconds, repeated for every tile in the card dump. After
# BREAKER_TRIP consecutive failures an endpoint is skipped outright for
# BREAKER_COOLDOWN_S, so ingest degrades to instant labelled stubs instead of
# stalling, and one probe call after the cooldown lets it heal by itself.
BREAKER_TRIP = 3
BREAKER_COOLDOWN_S = 30.0

_BREAKER: dict[str, tuple[int, float]] = {}
_BREAKER_LOCK = threading.Lock()


def _breaker_open(key: str) -> bool:
    with _BREAKER_LOCK:
        fails, until = _BREAKER.get(key, (0, 0.0))
    return fails >= BREAKER_TRIP and time.monotonic() < until


def _breaker_note(key: str, ok: bool) -> None:
    with _BREAKER_LOCK:
        if ok:
            _BREAKER.pop(key, None)
            return
        fails, _until = _BREAKER.get(key, (0, 0.0))
        fails += 1
        _BREAKER[key] = (fails, time.monotonic() + BREAKER_COOLDOWN_S)
        if fails == BREAKER_TRIP:
            log.warning(
                "%s unreachable %d times, skipping it for %.0fs", key, fails, BREAKER_COOLDOWN_S
            )


def reset_breakers() -> None:
    """Clear the skip state. Used by tests and by an operator-triggered retry."""
    with _BREAKER_LOCK:
        _BREAKER.clear()


# ------------------------------------------------------------------- transport
def _record(how: str) -> None:
    with _STATS_LOCK:
        _STATS[how] = _STATS.get(how, 0) + 1


def stats() -> dict:
    """Model-versus-stub counts since process start, for the HUD honesty indicator."""
    with _STATS_LOCK:
        return dict(_STATS)


def _post(url: str, payload: dict, timeout: float) -> dict:
    """POST one JSON body with a hard deadline.

    WHY the wall-clock check: urllib's timeout is per socket operation, so a
    slow-dripping response can outlive it. Ingest cannot afford that.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.monotonic()
    with _OPENER.open(req, timeout=timeout) as resp:
        raw = resp.read()
    elapsed = time.monotonic() - started
    if elapsed > timeout:
        raise TimeoutError(f"{url} took {elapsed:.1f}s over a {timeout:.1f}s budget")
    return json.loads(raw.decode("utf-8"))


def _strip_think(text: str) -> str:
    """Lightning ignores /no_think (that is Nano syntax) and emits a preamble as content."""
    out = _THINK_RE.sub("", text)
    if "</think>" in out:  # unclosed opener, keep whatever followed the close tag
        out = out.split("</think>")[-1]
    return _FENCE_RE.sub("", out).strip()


def _schema_stub(schema: Any) -> Any:
    """Smallest shape-valid instance of a JSON schema.

    This is a SHAPE, not a judgement: callers that need a meaningful fallback
    value (grading does) must supply their own, see stub_grade.
    """
    if not isinstance(schema, dict):
        return None
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = kind[0] if kind else "null"
    if kind == "object":
        props = schema.get("properties") or {}
        wanted = schema.get("required") or list(props)
        return {k: _schema_stub(props.get(k, {})) for k in wanted}
    if kind == "array":
        return []
    if kind == "integer":
        return int(schema.get("minimum", 0))
    if kind == "number":
        return float(schema.get("minimum", 0.0))
    if kind == "boolean":
        return False
    if kind == "string":
        return ""
    return None


def _fallback_text(schema: Optional[dict], choice: Optional[Sequence[Any]]) -> str:
    if choice:
        return str(choice[0])
    if schema is not None:
        return json.dumps(_schema_stub(schema))
    return ""


def _coerce_choice(text: str, choice: Sequence[Any]) -> Optional[str]:
    """Accept an exact guided pick, else the first allowed token found in the text."""
    allowed = [str(c) for c in choice]
    if text in allowed:
        return text
    for c in allowed:
        if re.search(rf"(?<![\w.-]){re.escape(c)}(?![\w.-])", text):
            return c
    return None


def chat(
    endpoint: Endpoint,
    messages: Sequence[dict],
    *,
    schema: Optional[dict] = None,
    schema_name: str = "response",
    choice: Optional[Sequence[Any]] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> tuple[str, str]:
    """One chat completion against a local vLLM server.

    Returns (text, how). `how` is "model" when the server answered usefully and
    "stub" for the deterministic fallback, so callers can label the path they
    actually took. Never raises: ingest threads must not die on a wedged port.
    """
    if _breaker_open(endpoint.name):
        log.debug("%s breaker open, stub without a round trip", endpoint.name)
        _record(GRADE_HOW_STUB)
        return _fallback_text(schema, choice), GRADE_HOW_STUB
    deadline = float(endpoint.timeout if timeout is None else timeout)
    payload: dict[str, Any] = {
        "model": endpoint.model,
        "messages": list(messages),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    if schema is not None:
        # Verified syntax for this build. Do NOT use top-level guided_json.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }
    if choice is not None:
        payload["guided_choice"] = [str(c) for c in choice]

    try:
        data = _post(f"{endpoint.url}/chat/completions", payload, deadline)
        message = data["choices"][0]["message"]
        text = _strip_think(message.get("content") or "")
        if not text:
            raise ValueError("empty content")
        if choice is not None:
            picked = _coerce_choice(text, choice)
            if picked is None:
                raise ValueError(f"off-menu pick {text[:40]!r}")
            text = picked
    except Exception as exc:  # noqa: BLE001 - any failure is a labelled stub, by design
        log.warning("%s stub engaged: %s: %s", endpoint.name, type(exc).__name__, exc)
        _breaker_note(endpoint.name, ok=False)
        _record(GRADE_HOW_STUB)
        return _fallback_text(schema, choice), GRADE_HOW_STUB
    _breaker_note(endpoint.name, ok=True)
    _record(GRADE_HOW_MODEL)
    return text, GRADE_HOW_MODEL


# ----------------------------------------------------------------- image helpers
def _as_image(src: Union[str, Path, Image.Image]) -> Image.Image:
    """Never mutate a caller's image: every path here yields our own RGB copy."""
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    with Image.open(src) as im:
        return im.convert("RGB")


def _prepare(src: Union[str, Path, Image.Image], crop_box: Optional[Sequence[int]]) -> Image.Image:
    img = _as_image(src)
    if crop_box is not None:
        box = _sane_box(crop_box, img.width, img.height)
        img = img.crop(box)
    return img


def _sane_box(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = (int(round(float(v))) for v in box)
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def _data_url(img: Image.Image, max_side: int = VL_MAX_SIDE) -> str:
    scaled = img
    longest = max(img.width, img.height)
    if longest > max_side:
        ratio = max_side / float(longest)
        scaled = img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.BILINEAR
        )
    buf = io.BytesIO()
    scaled.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# -------------------------------------------------------------- caption + grade
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "class": {"type": "integer", "minimum": 0, "maximum": 3},
        "caption": {"type": "string", "maxLength": MAX_CAPTION_CHARS},
    },
    "required": ["class", "caption"],
    "additionalProperties": False,
}

_GRADE_SYSTEM = (
    "You grade aerial disaster imagery for an emergency operations centre. "
    "You see one nadir crop containing one structure. "
    "Damage scale: 0 no damage, 1 minor damage such as lost shingles or scattered debris, "
    "2 major damage such as partial collapse, a large roof hole or burn scarring, "
    "3 destroyed, meaning collapsed, washed away or burned out. "
    "Describe structures, terrain and water only. "
    "Never mention people, bodies, faces, clothing or vehicle occupants. "
    "If no structure is visible, grade 0 and say so. Answer with JSON only."
)
_GRADE_USER = "Grade this structure and write one factual caption of at most 30 words."

# A6 caption post-filter vocabulary. Over-withholding costs one operator review
# click; under-withholding costs the whole privacy claim, so this list is broad.
_PERSON_WORDS = (
    "person",
    "persons",
    "people",
    "pedestrian",
    "pedestrians",
    "human",
    "humans",
    "man",
    "men",
    "woman",
    "women",
    "child",
    "children",
    "kid",
    "kids",
    "boy",
    "girl",
    "body",
    "bodies",
    "corpse",
    "victim",
    "victims",
    "survivor",
    "survivors",
    "crowd",
    "crowds",
    "someone",
    "somebody",
    "individual",
    "individuals",
    "figure",
    "figures",
    "face",
    "faces",
    "arm",
    "leg",
    "hand",
    "clothing",
    "clothes",
    "shirt",
    "jacket",
    "hat",
    "shoes",
    "backpack",
    "occupant",
    "occupants",
    "resident",
    "residents",
    "worker",
    "workers",
    "rescuer",
    "rescuers",
)
_PERSON_RE = re.compile(r"(?<!\w)(" + "|".join(_PERSON_WORDS) + r")(?!\w)", re.IGNORECASE)
# "body of water" is exactly the terrain vocabulary we asked for, not a person.
_BODY_OF_WATER_RE = re.compile(r"\bbod(?:y|ies)\s+of\s+water\b", re.IGNORECASE)


def caption_mentions_person(text: Optional[str]) -> bool:
    """True when a caption may describe a person, so the archive re-withholds it.

    WHY here and not in the archive: the vocabulary belongs beside the prompt that
    is supposed to prevent it, so the two never drift apart.
    """
    if not text:
        return False
    return bool(_PERSON_RE.search(_BODY_OF_WATER_RE.sub("water", text)))


def _clean_caption(text: str) -> str:
    out = _WS_RE.sub(" ", text).strip().strip('"')
    return out[:MAX_CAPTION_CHARS]


# Caption vocabulary bands, used only to cross-check the structured grade against
# the free-text caption of the SAME response. See _grade_confidence.
_BAND_WORDS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        3,
        (
            "destroyed",
            "collapsed",
            "collapse",
            "rubble",
            "razed",
            "flattened",
            "levelled",
            "leveled",
            "burned out",
            "burnt out",
            "gutted",
            "washed away",
            "obliterated",
        ),
    ),
    (
        2,
        (
            "partial collapse",
            "partially collapsed",
            "roof hole",
            "hole in the roof",
            "missing roof",
            "roof missing",
            "caved",
            "charred",
            "scorched",
            "burn",
            "burning",
            "on fire",
            "breached",
            "severe",
            "major damage",
        ),
    ),
    (
        1,
        (
            "debris",
            "shingle",
            "shingles",
            "tarp",
            "cracked",
            "minor damage",
            "dented",
            "scattered",
            "damaged",
        ),
    ),
    (
        0,
        (
            "no damage",
            "no visible damage",
            "undamaged",
            "intact",
            "sound",
            "unaffected",
            "appears intact",
        ),
    ),
)


def _caption_band(caption: str) -> Optional[int]:
    low = caption.lower()
    for band, words in _BAND_WORDS:
        if any(w in low for w in words):
            return band
    return None


def _grade_confidence(cls: int, caption: str) -> float:
    """How much the structured grade and the caption of one VL response agree.

    NOT a probability, and deliberately not a constant. The plan (B3) warns that a
    column of identical doubt floors reads as decoration, and `doubt` falls back to
    1 - grader_confidence until Lightning's ballot is wired. Two channels of the
    same response disagreeing is real evidence that a human should re-check, which
    is the same argument Lightning's k=8 ballot makes at higher resolution.
    """
    band = _caption_band(caption)
    if band is None:
        return 0.7
    gap = abs(int(cls) - band)
    if gap == 0:
        return 0.85
    if gap == 1:
        return 0.6
    return 0.4


def _pixel_stats(img: Image.Image) -> tuple[int, float]:
    """Cheap deterministic damage ORDERING from pixels, for the labelled fallback.

    WHY this and not a constant: intact roofs are large flat regions, while rubble,
    debris fields and burn scars raise local texture energy and darken the crop.
    That correlation is weak but real and it is stable for identical pixels, so a
    stub run still produces a list an operator can walk down. It is explicitly not
    a damage assessment, which is why everything it grades is labelled
    stub-pixelstat-v1 on the wire.
    """
    import numpy as np

    small = img.convert("L").resize((64, 64), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    texture = float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean())
    contrast = float(arr.std())
    dark = float((arr < 0.25).mean())
    score = 2.2 * texture + 0.9 * contrast + 0.5 * dark
    score = max(0.0, min(1.0, score))
    if score >= 0.72:
        cls = 3
    elif score >= 0.52:
        cls = 2
    elif score >= 0.32:
        cls = 1
    else:
        cls = 0
    return cls, round(score, 4)


def stub_grade(
    src: Union[str, Path, Image.Image], *, crop_box: Optional[Sequence[int]] = None
) -> dict:
    """The labelled fallback grade. Never calls the network, never raises."""
    try:
        img = _prepare(src, crop_box)
        cls, score = _pixel_stats(img)
    except Exception as exc:  # noqa: BLE001 - an unreadable crop is still gradable as unknown
        log.warning("pixel-stat grade failed, defaulting to class 1: %s", exc)
        cls, score = 1, 0.0
    # Confidence is low on purpose: doubt = 1 - conf until Lightning votes, and a
    # pixel statistic deserves to be re-checked before anything the model looked at.
    conf = round(0.3 + 0.05 * math.tanh(score), 3)
    return {"class": int(cls), "caption": STUB_CAPTION, "conf": conf, "how": GRADE_HOW_STUB}


def caption_and_grade(
    src: Union[str, Path, Image.Image],
    *,
    crop_box: Optional[Sequence[int]] = None,
    endpoint: Optional[Endpoint] = None,
    timeout: Optional[float] = None,
) -> dict:
    """ONE VL pass per crop: the damage grade and the archive caption together.

    A6 forbids calling the VLM twice per crop, so the class and the caption come
    out of a single generation. Returns
    {"class": 0-3, "caption": str, "conf": float, "how": "model"|"stub"} and is
    safe to call from an ingest thread: the hard timeout comes from
    config.VL_TIMEOUT_S and every failure path lands on stub_grade.
    """
    try:
        img = _prepare(src, crop_box)
    except Exception as exc:  # noqa: BLE001
        log.warning("crop unreadable, stub grade: %s", exc)
        return stub_grade(src)

    # A crop this small carries no structure detail worth a VL round trip.
    if img.width < 8 or img.height < 8:
        return stub_grade(img)

    ep = endpoint or vl()
    messages = [
        {"role": "system", "content": _GRADE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _GRADE_USER},
                {"type": "image_url", "image_url": {"url": _data_url(img)}},
            ],
        },
    ]
    text, how = chat(
        ep,
        messages,
        schema=GRADE_SCHEMA,
        schema_name="damage_grade",
        max_tokens=220,
        temperature=0.0,
        timeout=timeout,
    )
    if how != GRADE_HOW_MODEL:
        return stub_grade(img)

    try:
        parsed = json.loads(text)
        cls = int(parsed["class"])
        caption = _clean_caption(str(parsed.get("caption") or ""))
        if cls not in (0, 1, 2, 3) or not caption:
            raise ValueError(f"class={cls} caption={caption[:30]!r}")
    except Exception as exc:  # noqa: BLE001
        log.warning("VL answered off-contract, stub grade: %s", exc)
        return stub_grade(img)

    return {
        "class": cls,
        "caption": caption,
        "conf": _grade_confidence(cls, caption),
        "how": GRADE_HOW_MODEL,
    }


__all__ = [
    "Endpoint",
    "GRADE_HOW_MODEL",
    "GRADE_HOW_STUB",
    "GRADE_SCHEMA",
    "STUB_CAPTION",
    "caption_and_grade",
    "caption_mentions_person",
    "chat",
    "lightning",
    "nano",
    "stats",
    "stub_grade",
    "vl",
]
