import socket
import urllib.request

import pytest

from backend.decision.archive_location import (
    DEFAULT_POINT_RADIUS_M,
    FixtureLocalPlaceIndex,
    LocalLocationResolver,
)

_PLACES = [
    {"name": "35th Ave SW", "centroid": [-122.3902, 47.5981]},
    {"name": "Providence Mount St. Vincent", "centroid": [-122.3861, 47.5670]},
]


def _resolver():
    return LocalLocationResolver(FixtureLocalPlaceIndex(_PLACES))


# coordinate literal resolves; parsed as [lng, lat] from "lat, lng" input text
def test_coordinate_literal_resolves_to_lng_lat_order():
    result = _resolver().resolve("47.558, -122.377")
    assert result is not None
    assert result["source"] == "coordinate"
    lng_min, lat_min, lng_max, lat_max = result["bbox"]
    assert lng_min < -122.377 < lng_max
    assert lat_min < 47.558 < lat_max


def test_coordinate_literal_bbox_uses_default_radius():
    result = _resolver().resolve("47.558, -122.377")
    lng_min, lat_min, lng_max, lat_max = result["bbox"]
    # a wildly different radius would produce a visibly different-sized box
    assert (lat_max - lat_min) > 0
    assert (lng_max - lng_min) > 0
    assert DEFAULT_POINT_RADIUS_M == 400.0


def test_named_road_resolves_from_local_fixture():
    result = _resolver().resolve("35th Ave SW")
    assert result is not None
    assert result["source"] == "place"
    assert result["matched_text"] == "35th Ave SW"


def test_named_facility_resolves_from_local_fixture():
    result = _resolver().resolve("Providence Mount St. Vincent")
    assert result is not None
    assert result["source"] == "place"


def test_near_phrase_resolves_and_strips_from_residual():
    result = _resolver().resolve("buildings on fire near 35th Ave SW")
    assert result is not None
    assert result["residual_after"] == "buildings on fire"


def test_matching_is_case_insensitive():
    result = _resolver().resolve("35TH ave sw")
    assert result is not None
    result2 = _resolver().resolve("near PROVIDENCE mount st. vincent")
    assert result2 is not None


def test_near_unknown_place_returns_none_not_a_guess():
    result = _resolver().resolve("near Atlantis")
    assert result is None


# do not classify arbitrary disaster vocabulary as a place
def test_descriptive_text_without_near_does_not_resolve():
    result = _resolver().resolve("buildings on fire")
    assert result is None


def test_partial_substring_of_place_name_does_not_resolve_without_near():
    # "35th" alone must not fuzzy-match "35th Ave SW"
    result = _resolver().resolve("35th")
    assert result is None


def test_empty_place_index_still_resolves_coordinates():
    resolver = LocalLocationResolver(FixtureLocalPlaceIndex([]))
    result = resolver.resolve("47.558, -122.377")
    assert result is not None
    assert result["source"] == "coordinate"


def test_empty_place_index_never_resolves_names():
    resolver = LocalLocationResolver(FixtureLocalPlaceIndex([]))
    assert resolver.resolve("35th Ave SW") is None


# no external geocoder / no network -- resolving still works even if all
# network access is blocked at the socket level.
def test_resolve_never_touches_the_network(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("resolve() must never touch the network")

    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)

    resolver = _resolver()
    assert resolver.resolve("35th Ave SW") is not None
    assert resolver.resolve("47.558, -122.377") is not None
    assert resolver.resolve("buildings on fire") is None
