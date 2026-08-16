"""B7 Part 4: deliberately small, deterministic structured-filter parser.

No LLM turns q into SQL -- a fixed set of recognized `key:value` tokens is
extracted from the raw query text with simple string splitting, each
producing one parameterized SQL fragment (never a string-interpolated
value). Recognized tokens are removed from the residual text so the
remaining natural-language words are what location/semantic resolution
sees (Part 6).

Recognized tokens:
    class:0 | class:1 | class:2 | class:3
    after:<ISO date/datetime or bare HH:MM>
    before:<ISO date/datetime or bare HH:MM>
    key:true | key:false
    needs_geo:true | needs_geo:false
    sector:<value>            (only meaningful because archive_store.py's
                                schema already carries an internal `sector`
                                column -- see that module's docstring;
                                never added to the public ArchiveItem)

A bare "HH:MM" after:/before: value (no date) is combined with
_BARE_TIME_REFERENCE_DATE, a fixed, documented reference day -- FIRST
LIGHT's fixture/demo scenario is a single incident day, so a bare
time-of-day has no other unambiguous meaning without inventing a "now"
that would make parsing non-deterministic across test runs. Use a full
ISO date/datetime (e.g. "2026-08-15T06:00") for anything outside that day.

A token whose prefix matches a recognized key but whose value is invalid
(e.g. "class:5", "class:abc", "key:maybe") raises ArchiveSearchError --
never silently ignored, never executed as malformed SQL-like input. A
token whose prefix does NOT match any recognized key (e.g. "buildings",
"on:fire") is not a filter at all -- it stays in the residual text for
location/semantic resolution.
"""

import re
from datetime import datetime, timezone

_BARE_TIME_REFERENCE_DATE = "2026-08-15"
_BARE_TIME_PATTERN = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")

_VALID_CLASS_VALUES = ("0", "1", "2", "3")
_VALID_BOOL_VALUES = ("true", "false")

_TOKEN_PATTERN = re.compile(
    r"\b(class|after|before|key|needs_geo|sector):(\S+)", re.IGNORECASE
)


class ArchiveSearchError(ValueError):
    """Raised for an invalid SearchRequest (q/limit) or malformed structured
    filter syntax -- a contract/input failure, never silently corrected or
    executed as-is.
    """


def _parse_time_token(raw_value: str) -> float:
    value = raw_value
    if _BARE_TIME_PATTERN.match(value):
        value = f"{_BARE_TIME_REFERENCE_DATE}T{value}"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArchiveSearchError(
            f"invalid after:/before: value {raw_value!r}: expected an ISO date/datetime "
            "(e.g. 2026-08-15 or 2026-08-15T06:00) or a bare HH:MM time"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _parse_bool_token(key: str, raw_value: str) -> bool:
    lowered = raw_value.lower()
    if lowered not in _VALID_BOOL_VALUES:
        raise ArchiveSearchError(f"invalid {key}: value {raw_value!r}: must be 'true' or 'false'")
    return lowered == "true"


def parse_structured_filters(q: str) -> tuple:
    """Extract recognized structured tokens from `q`.

    Returns (filters, residual_text):
        filters: list of (sql_fragment: str, param) pairs, e.g.
                 [("class_max = ?", 3), ("captured_at >= ?", 1755234000.0)]
                 -- always bound as SQL parameters by archive_store.py,
                 never interpolated into the query text.
        residual_text: `q` with every recognized token removed, whitespace
                 collapsed -- the input to location/semantic resolution.

    Raises ArchiveSearchError for a recognized-prefix token with an invalid
    value. Multiple after:/before: tokens are all applied (AND'd together);
    the parser does not attempt to detect a contradictory range.
    """
    filters = []
    residual = q

    for match in _TOKEN_PATTERN.finditer(q):
        key = match.group(1).lower()
        raw_value = match.group(2)

        if key == "class":
            if raw_value not in _VALID_CLASS_VALUES:
                raise ArchiveSearchError(
                    f"invalid class: value {raw_value!r}: must be one of {_VALID_CLASS_VALUES}"
                )
            filters.append(("class_max = ?", int(raw_value)))
        elif key == "after":
            filters.append(("captured_at >= ?", _parse_time_token(raw_value)))
        elif key == "before":
            filters.append(("captured_at <= ?", _parse_time_token(raw_value)))
        elif key == "key":
            filters.append(("key_evidence = ?", 1 if _parse_bool_token("key", raw_value) else 0))
        elif key == "needs_geo":
            filters.append(("needs_geo = ?", 1 if _parse_bool_token("needs_geo", raw_value) else 0))
        elif key == "sector":
            filters.append(("sector = ?", raw_value))

        residual = residual.replace(match.group(0), " ", 1)

    residual = re.sub(r"\s+", " ", residual).strip()
    return filters, residual
