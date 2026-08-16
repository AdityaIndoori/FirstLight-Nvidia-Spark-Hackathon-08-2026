"""B7 Part 5: local location resolution -- turns a location expression
inside SearchRequest.q into a geographic narrowing bbox. No external
geocoder, no network, ever.

STATUS: no real local OSM road/facility dataset exists in this repository
yet (same gap backend/routing/osm_loader.py already documented for B4 --
checked again for B7: still no *.osm/*.pbf/*.geojson road or facility
source, no SQLite table of place names). This module is therefore built
against a clean adapter interface, LocalPlaceIndex, with a
FixtureLocalPlaceIndex implementation for tests and the live-check script.
REAL LOCAL OSM ASSET INTEGRATION IS PENDING -- when Member A's real
road/facility export lands, implement one more LocalPlaceIndex (e.g.
backed by backend.routing.osm_loader's already-loaded RoadGraph node names,
or a fresh table), and nothing in archive_search.py changes.

Resolution order (first match wins):
    A. coordinate literal, e.g. "47.558, -122.377"  (human lat,lng order in
       the query text; the resolved point becomes GeoJSON [lng, lat],
       matching the ArchiveItem.centroid convention everywhere else)
    B/C. local road/facility name lookup via LocalPlaceIndex.lookup()
       ("35th Ave SW", "near Providence Mount", ...) -- both roads and
       facilities are just named points in this fixture-sized adapter; a
       real implementation could distinguish "kind" if it mattered
       downstream, but nothing downstream reads it today.

To avoid classifying arbitrary disaster vocabulary as a place (Part 6's
explicit non-negotiable), a bare place-name match (no "near " prefix) is
only accepted if it matches the ENTIRE input text exactly -- never a fuzzy
substring of a longer descriptive sentence. "near <phrase>" is more
permissive (trimming trailing words) because "near " is itself an explicit
signal the words that follow name a place, not a disaster description.

The matched point (or coordinate literal) becomes a bbox via a simple
meters-based square around it -- reusing the SAME equirectangular
approximation backend/routing/geo_math.py already uses elsewhere in this
project (not geodesically precise over long distances, fine for a small
demo AOI, avoids adding pyproj). No PostGIS, no spatial service --
centroid-bounds filtering is explicitly acceptable at this dataset size
(Part 5).
"""

import math
import re
from abc import ABC, abstractmethod

from backend.routing.geo_math import METERS_PER_DEGREE_LAT

DEFAULT_POINT_RADIUS_M = 400.0
"""Default half-width of the narrowing bbox around a resolved point --
roughly a short block's worth of buildings either side, a sensible default
for this project's small demo AOI. Callers needing a tighter/looser
narrowing pass LocalLocationResolver(radius_m=...) explicitly."""

_COORD_PATTERN = re.compile(r"(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)")
_NEAR_PATTERN = re.compile(r"\bnear\s+(.+)$", re.IGNORECASE)


class LocalPlaceIndex(ABC):
    """Boundary: case-insensitive local name -> resolved point. NOT an
    external geocoder -- implementations must never make a network call.
    """

    @abstractmethod
    def lookup(self, name: str) -> dict:
        """Return {"name": str, "centroid": [lng, lat]} for an exact
        (case-insensitive, whitespace-normalized) local name match, or
        None if nothing matches.
        """
        raise NotImplementedError


class FixtureLocalPlaceIndex(LocalPlaceIndex):
    """Deterministic in-memory place index over a fixed list of
    {"name": str, "centroid": [lng, lat]} rows, standing in for the real
    local OSM road/facility source (see module docstring). Matching is
    exact after case-folding and whitespace-collapsing -- no fuzzy
    matching, no network, no external geocoder.
    """

    def __init__(self, places: list):
        self._by_normalized_name = {}
        for place in places:
            key = _normalize(place["name"])
            self._by_normalized_name[key] = {"name": place["name"], "centroid": list(place["centroid"])}

    def lookup(self, name: str) -> dict:
        return self._by_normalized_name.get(_normalize(name))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _bbox_around_point(lng: float, lat: float, radius_m: float) -> tuple:
    lat_rad = math.radians(lat)
    meters_per_deg_lng = METERS_PER_DEGREE_LAT * max(math.cos(lat_rad), 1e-6)
    d_lat = radius_m / METERS_PER_DEGREE_LAT
    d_lng = radius_m / meters_per_deg_lng
    return (lng - d_lng, lat - d_lat, lng + d_lng, lat + d_lat)


def _try_coordinate_literal(text: str) -> tuple:
    """Returns (matched_substring, [lng, lat]) or (None, None)."""
    match = _COORD_PATTERN.search(text)
    if match is None:
        return None, None
    lat = float(match.group(1))
    lng = float(match.group(2))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None, None
    return match.group(0), [lng, lat]


def _try_near_phrase(text: str, place_index: LocalPlaceIndex) -> tuple:
    """Returns (matched_substring, place_record) or (None, None). Trims
    trailing words from the "near <phrase>" capture until a local place
    matches, or gives up (no location match rather than a guess) if none
    of the trimmed candidates resolve.
    """
    match = _NEAR_PATTERN.search(text)
    if match is None:
        return None, None

    phrase = match.group(1).strip()
    words = phrase.split()
    for end in range(len(words), 0, -1):
        candidate = " ".join(words[:end])
        record = place_index.lookup(candidate)
        if record is not None:
            matched_substring = f"{match.group(0)[: match.group(0).lower().index('near')]}near {candidate}"
            return matched_substring, record
    return None, None


def _try_whole_text_place_name(text: str, place_index: LocalPlaceIndex) -> tuple:
    """Only accepts a match against the FULL residual text -- never a
    substring of a longer sentence -- so disaster vocabulary is never
    misread as a place (Part 6's non-negotiable).
    """
    stripped = text.strip()
    if not stripped:
        return None, None
    record = place_index.lookup(stripped)
    if record is not None:
        return stripped, record
    return None, None


class LocalLocationResolver:
    """Resolves a location expression within residual query text into a
    narrowing bbox, using ONLY local data (LocalPlaceIndex + coordinate
    literal parsing). Never touches the network.
    """

    def __init__(self, place_index: LocalPlaceIndex = None, radius_m: float = DEFAULT_POINT_RADIUS_M):
        self.place_index = place_index if place_index is not None else FixtureLocalPlaceIndex([])
        self.radius_m = radius_m

    def resolve(self, text: str) -> dict:
        """Returns None if nothing local resolves, else:
            {
                "bbox": (lng_min, lat_min, lng_max, lat_max),
                "matched_text": str,   # the substring consumed
                "source": "coordinate" | "place",
                "residual_after": str, # `text` with matched_text removed
            }
        """
        matched_text, point = _try_coordinate_literal(text)
        if point is not None:
            return self._build_result(text, matched_text, point, "coordinate")

        matched_text, record = _try_near_phrase(text, self.place_index)
        if record is not None:
            return self._build_result(text, matched_text, record["centroid"], "place")

        matched_text, record = _try_whole_text_place_name(text, self.place_index)
        if record is not None:
            return self._build_result(text, matched_text, record["centroid"], "place")

        return None

    def _build_result(self, text: str, matched_text: str, point: list, source: str) -> dict:
        lng, lat = point
        bbox = _bbox_around_point(lng, lat, self.radius_m)
        residual_after = re.sub(r"\s+", " ", text.replace(matched_text, " ", 1)).strip()
        return {"bbox": bbox, "matched_text": matched_text, "source": source, "residual_after": residual_after}
