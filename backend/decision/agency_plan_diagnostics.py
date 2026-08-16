"""Diagnostic-only error categorization for B6b Nano failures/recovery.

Makes visible WHY a real Nano attempt failed and whether the existing
one-reprompt policy actually ran -- without changing that policy, the
prompt, the JSON schema, or the fallback rules. Purely observational: these
functions never raise, never affect control flow, and are never added to
the frozen public AgencyPlan contract. Only consumed by
agency_plan_drafter.draft_agency_plan_with_diagnostics() (an internal
wrapper) and the opt-in live-check script.
"""

import socket
import urllib.error

CATEGORY_TIMEOUT = "timeout"
CATEGORY_CONNECTION_FAILURE = "connection_failure"
CATEGORY_HTTP_FAILURE = "http_failure"
CATEGORY_MALFORMED_JSON = "malformed_json"
CATEGORY_SCHEMA_PARSING_FAILURE = "schema_parsing_failure"
CATEGORY_TRUNCATED_OUTPUT = "truncated_output"
CATEGORY_UNSUPPORTED_AGENCY = "unsupported_agency"
CATEGORY_UNKNOWN_FOOTPRINT_ID = "unknown_footprint_id"
CATEGORY_INVALID_UNITS = "invalid_units"
CATEGORY_MISSING_REQUIRED_FIELD = "missing_required_field"
CATEGORY_FAITHFULNESS_FAILURE = "faithfulness_failure"
CATEGORY_OTHER_VALIDATION_ERROR = "other_validation_error"


def categorize_client_error(exc: Exception) -> str:
    """Best-effort categorization of a NanoClientError -- inspecting the
    original exception it was chained from (`raise ... from exc`, already
    preserved by nano_client.py/agency_plan_client.py without any change
    here) plus its message -- into one diagnostic category. Never changes
    what was raised or how retry/fallback behaves.
    """
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, (socket.timeout, TimeoutError)):
        return CATEGORY_TIMEOUT
    if isinstance(cause, urllib.error.HTTPError):
        return CATEGORY_HTTP_FAILURE
    if isinstance(cause, (urllib.error.URLError, ConnectionError, OSError)):
        return CATEGORY_CONNECTION_FAILURE

    message = str(exc).lower()
    if "truncated" in message or "finish_reason=length" in message:
        return CATEGORY_TRUNCATED_OUTPUT
    if "not valid json" in message or "was malformed" in message:
        return CATEGORY_MALFORMED_JSON
    if "must be a json object" in message or "no usable content" in message:
        return CATEGORY_SCHEMA_PARSING_FAILURE
    return CATEGORY_OTHER_VALIDATION_ERROR


def categorize_validation_error(message: str) -> str:
    """Best-effort categorization of one agency_plan_drafter._validate_assignments()
    error string into a diagnostic category.
    """
    lowered = message.lower()
    if "agency must be one of" in lowered:
        return CATEGORY_UNSUPPORTED_AGENCY
    if "is not one of the supplied candidates" in lowered:
        return CATEGORY_UNKNOWN_FOOTPRINT_ID
    if "units must be an integer" in lowered:
        return CATEGORY_INVALID_UNITS
    if "task must be a non-empty string" in lowered:
        return CATEGORY_MISSING_REQUIRED_FIELD
    if "must be a json object" in lowered or "assignments must be a list" in lowered:
        return CATEGORY_SCHEMA_PARSING_FAILURE
    return CATEGORY_OTHER_VALIDATION_ERROR
