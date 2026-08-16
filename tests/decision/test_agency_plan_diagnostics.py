import socket
import urllib.error

from backend.decision.agency_plan_diagnostics import (
    CATEGORY_CONNECTION_FAILURE,
    CATEGORY_HTTP_FAILURE,
    CATEGORY_INVALID_UNITS,
    CATEGORY_MALFORMED_JSON,
    CATEGORY_MISSING_REQUIRED_FIELD,
    CATEGORY_OTHER_VALIDATION_ERROR,
    CATEGORY_SCHEMA_PARSING_FAILURE,
    CATEGORY_TIMEOUT,
    CATEGORY_TRUNCATED_OUTPUT,
    CATEGORY_UNKNOWN_FOOTPRINT_ID,
    CATEGORY_UNSUPPORTED_AGENCY,
    categorize_client_error,
    categorize_validation_error,
)
from backend.decision.nano_client import NanoClientError

# This module (agency_plan_diagnostics.py) is unchanged by the per-building
# parallel drafting redesign -- its categorization functions are pure and
# reused as-is by agency_plan_drafter._draft_one_building(). Integration
# coverage of per-building diagnostics (attempt_count, fallback_reason,
# mixed provenance, etc.) lives in test_agency_plan_drafter.py; these tests
# cover the categorization functions directly.


def test_categorize_client_error_timeout():
    exc = NanoClientError("wrapped")
    exc.__cause__ = TimeoutError("timed out")
    assert categorize_client_error(exc) == CATEGORY_TIMEOUT


def test_categorize_client_error_socket_timeout():
    exc = NanoClientError("wrapped")
    exc.__cause__ = socket.timeout("timed out")
    assert categorize_client_error(exc) == CATEGORY_TIMEOUT


def test_categorize_client_error_connection_failure():
    exc = NanoClientError("Nano request to http://localhost:8000 failed: connection refused")
    exc.__cause__ = urllib.error.URLError(OSError("connection refused"))
    assert categorize_client_error(exc) == CATEGORY_CONNECTION_FAILURE


def test_categorize_client_error_http_failure():
    exc = NanoClientError("Nano request to http://localhost:8000 failed: HTTP 500")
    exc.__cause__ = urllib.error.HTTPError("http://localhost:8000", 500, "Internal Server Error", {}, None)
    assert categorize_client_error(exc) == CATEGORY_HTTP_FAILURE


def test_categorize_client_error_malformed_json():
    exc = NanoClientError("agency-plan response was not valid JSON: Expecting value: line 1 column 1")
    assert categorize_client_error(exc) == CATEGORY_MALFORMED_JSON


def test_categorize_client_error_schema_parsing_failure():
    exc = NanoClientError('agency-plan response must be a JSON object with an "assignments" list')
    assert categorize_client_error(exc) == CATEGORY_SCHEMA_PARSING_FAILURE


def test_categorize_client_error_truncated_output():
    exc = NanoClientError("Nano response was truncated by max_tokens (finish_reason=length)")
    assert categorize_client_error(exc) == CATEGORY_TRUNCATED_OUTPUT


def test_categorize_client_error_falls_back_to_other():
    exc = NanoClientError("something unexpected happened")
    assert categorize_client_error(exc) == CATEGORY_OTHER_VALIDATION_ERROR


def test_categorize_validation_error_unsupported_agency():
    message = "assignments[0].agency must be one of ('fire', 'ems', 'police', 'public_works'), got 'hazmat'"
    assert categorize_validation_error(message) == CATEGORY_UNSUPPORTED_AGENCY


def test_categorize_validation_error_unknown_footprint_id():
    message = "assignments[0].footprint_id 'fp-999' is not one of the supplied candidates"
    assert categorize_validation_error(message) == CATEGORY_UNKNOWN_FOOTPRINT_ID


def test_categorize_validation_error_invalid_units():
    message = "assignments[0].units must be an integer >= 1"
    assert categorize_validation_error(message) == CATEGORY_INVALID_UNITS


def test_categorize_validation_error_missing_required_field():
    message = "assignments[0].task must be a non-empty string"
    assert categorize_validation_error(message) == CATEGORY_MISSING_REQUIRED_FIELD


def test_categorize_validation_error_schema_parsing_failure():
    assert categorize_validation_error("response must be a JSON object") == CATEGORY_SCHEMA_PARSING_FAILURE
    assert categorize_validation_error("response.assignments must be a list") == CATEGORY_SCHEMA_PARSING_FAILURE


def test_categorize_validation_error_falls_back_to_other():
    assert categorize_validation_error("assignments[0] must be an object") == CATEGORY_OTHER_VALIDATION_ERROR
