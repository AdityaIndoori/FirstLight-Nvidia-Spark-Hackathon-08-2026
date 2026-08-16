"""Shared meters-aware geometry math for the routing package.

Local equirectangular approximation, consistent with the meters-per-degree
convention already used elsewhere in FIRST LIGHT (flight_planner.py's
transect sizing, routing/blockages.py's buffer conversion). Not
geodesically precise over long distances -- fine for a small demo AOI, and
avoids adding pyproj (not currently a project dependency) for that reason.
"""

import math

METERS_PER_DEGREE_LAT = 111_320.0


def point_distance_m(p1, p2) -> float:
    """Approximate meters between two [lng, lat] points."""
    lng1, lat1 = p1
    lng2, lat2 = p2
    avg_lat_rad = math.radians((lat1 + lat2) / 2.0)
    meters_per_deg_lng = METERS_PER_DEGREE_LAT * math.cos(avg_lat_rad)
    dx = (lng2 - lng1) * meters_per_deg_lng
    dy = (lat2 - lat1) * METERS_PER_DEGREE_LAT
    return math.hypot(dx, dy)


def linestring_length_m(coordinates: list) -> float:
    """Sum of consecutive point_distance_m() along a LineString's coordinates."""
    return sum(point_distance_m(coordinates[i], coordinates[i + 1]) for i in range(len(coordinates) - 1))
