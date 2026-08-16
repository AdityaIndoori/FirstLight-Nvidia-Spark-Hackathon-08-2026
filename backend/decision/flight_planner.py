"""Flight-plan request orchestration: validate, retry once with the validation
error, fall back deterministically if the retry also fails schema validation
or the client raises a transport/model failure.

request flight plan -> client candidate -> validate
    valid   -> return candidate, recovery=None
    invalid -> capture error -> re-prompt client ONCE with that error -> validate retry
        valid   -> return retry,   recovery="model"
        invalid -> deterministic fallback, recovery="stub"
    client raises NanoClientError (either attempt) -> deterministic fallback, recovery="stub"

Application code calls only request_flight_plan(planning_input). Swapping the
stub for the real Nano client (localhost:8000, reasoning enabled, guided
JSON -- see flight_client.RealNanoFlightClient) changes only which
NanoFlightClient is constructed as the default/passed-in client here; this
orchestration function is otherwise unchanged from the stub-only version,
with try/except added around both client calls (the stub never raises, so
this is purely additive for the real client's failure modes -- transport
timeout, malformed response, etc. -- following the same
try-real-except-NanoClientError-fall-back-to-stub convention already used
by rationale.generate_rationale_with_recovery and
building_evidence.process_building_evidence).
"""

from backend.decision.flight_client import (
    NanoFlightClient,
    StubNanoFlightClient,
    build_deterministic_flight_plan,
)
from backend.decision.nano_client import NanoClientError

_REQUIRED_SURVEY_PATH_PROPERTIES = ("altitude_m_agl", "line_spacing_m", "transects", "est_flight_min")

_default_client: NanoFlightClient = StubNanoFlightClient()


def request_flight_plan(planning_input: dict, client: NanoFlightClient = None) -> dict:
    """Return {"flight_plan": <GeoJSON FeatureCollection>, "recovery": None|"model"|"stub"}."""
    active_client = client if client is not None else _default_client

    try:
        first_candidate = active_client.request_flight_plan(planning_input)
    except NanoClientError:
        return {"flight_plan": build_deterministic_flight_plan(planning_input), "recovery": "stub"}

    first_errors = validate_flight_plan(first_candidate)
    if not first_errors:
        return {"flight_plan": first_candidate, "recovery": None}

    validation_error = "; ".join(first_errors)
    try:
        retry_candidate = active_client.request_flight_plan(planning_input, validation_error=validation_error)
    except NanoClientError:
        return {"flight_plan": build_deterministic_flight_plan(planning_input), "recovery": "stub"}

    retry_errors = validate_flight_plan(retry_candidate)
    if not retry_errors:
        return {"flight_plan": retry_candidate, "recovery": "model"}

    fallback = build_deterministic_flight_plan(planning_input)
    return {"flight_plan": fallback, "recovery": "stub"}


def validate_flight_plan(candidate) -> list:
    """Return a list of validation error strings; empty means candidate is valid."""
    errors = []

    if not isinstance(candidate, dict) or candidate.get("type") != "FeatureCollection":
        return ["candidate is not a GeoJSON FeatureCollection"]

    features = candidate.get("features")
    if not isinstance(features, list):
        return ["FeatureCollection is missing a features list"]

    survey_areas = [f for f in features if isinstance(f, dict) and f.get("properties", {}).get("role") == "survey-area"]
    survey_paths = [f for f in features if isinstance(f, dict) and f.get("properties", {}).get("role") == "survey-path"]

    if len(survey_areas) != 1:
        errors.append("expected exactly one survey-area feature")
    if len(survey_paths) != 1:
        errors.append("expected exactly one survey-path feature")

    if survey_areas:
        errors.extend(_validate_geometry(survey_areas[0], "survey-area", "Polygon", min_ring_points=4))

    if survey_paths:
        path = survey_paths[0]
        errors.extend(_validate_geometry(path, "survey-path", "LineString", min_ring_points=2))
        props = path.get("properties", {})
        for field in _REQUIRED_SURVEY_PATH_PROPERTIES:
            if field not in props:
                errors.append(f"survey-path missing required property {field}")

    return errors


def _validate_geometry(feature: dict, role: str, expected_type: str, min_ring_points: int) -> list:
    errors = []
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return [f"{role} feature has no geometry"]

    geom_type = geometry.get("type")
    if geom_type != expected_type:
        errors.append(f"{role} geometry must be {expected_type}, got {geom_type!r}")
        return errors

    coordinates = geometry.get("coordinates")
    if expected_type == "Polygon":
        rings = coordinates if isinstance(coordinates, list) else None
        if not rings or not isinstance(rings[0], list):
            errors.append(f"{role} Polygon coordinates are malformed")
            return errors
        point_lists = rings[:1]
    else:
        if not isinstance(coordinates, list):
            errors.append(f"{role} LineString coordinates are malformed")
            return errors
        point_lists = [coordinates]

    for points in point_lists:
        if not isinstance(points, list) or len(points) < min_ring_points:
            errors.append(f"{role} geometry does not have enough coordinate points")
            continue
        for point in points:
            if not _is_valid_lng_lat_point(point):
                errors.append(f"{role} has a malformed or out-of-range [lng,lat] coordinate: {point!r}")
                break

    return errors


def _is_valid_lng_lat_point(point) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False
    lng, lat = point
    if not isinstance(lng, (int, float)) or not isinstance(lat, (int, float)):
        return False
    return -180 <= lng <= 180 and -90 <= lat <= 90
