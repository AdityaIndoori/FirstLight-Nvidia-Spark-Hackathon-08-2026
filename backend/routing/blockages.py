"""Frozen C -> B Road block contract and active-blockage state.

    {
        road_name: str,
        geometry: LineString,   # GeoJSON [[lng, lat], ...]
        blocked: bool,
        operator: str,
    }

An edge is forbidden if EITHER:
    A. its normalized road_name matches an active block's road_name, OR
    B. its geometry intersects the active block's geometry (buffered by a
       small tolerance) -- real polyline geometry, never string matching.
Both are checked independently; neither is required alongside the other.

Road-name matching is conservative and deterministic: trim whitespace,
lowercase. No fuzzy/NLP matching.

This module only tracks CURRENT active blockage state in memory
(set_block(blocked=True/False)). It does not touch the decision log --
logging integration is separate.
"""

from dataclasses import dataclass

from shapely.geometry import LineString

_METERS_PER_DEGREE_LAT = 111_320.0


@dataclass(frozen=True)
class RoadBlock:
    road_name: str
    geometry: list  # GeoJSON LineString coordinates [[lng, lat], ...]
    blocked: bool
    operator: str


def normalize_road_name(road_name: str) -> str:
    """Conservative, deterministic road-name normalization: trim + lowercase."""
    return road_name.strip().lower()


def _meters_to_degrees(meters: float) -> float:
    """Crude local equirectangular approximation for a small buffer
    tolerance -- consistent with the meters-per-degree convention used
    elsewhere in FIRST LIGHT (e.g. flight_planner.py). Not geodesically
    precise, and not meant to be for a tolerance of a few meters.
    """
    return meters / _METERS_PER_DEGREE_LAT


def geometries_intersect(edge_geometry: list, block_geometry: list, buffer_m: float) -> bool:
    """True if edge_geometry (a LineString) intersects block_geometry
    buffered by buffer_m meters -- real geometric intersection, computed
    with shapely, never a name comparison.
    """
    edge_line = LineString(edge_geometry)
    block_line = LineString(block_geometry)
    buffered = block_line.buffer(_meters_to_degrees(buffer_m))
    return edge_line.intersects(buffered)


class BlockageRegistry:
    """Tracks CURRENT active road blocks in memory, keyed by normalized
    road_name.
    """

    def __init__(self):
        self._active_by_name: dict = {}

    def set_block(self, road_block: RoadBlock) -> None:
        """blocked=True activates/replaces the block for this road_name;
        blocked=False removes/deactivates the matching active block, if any.
        """
        key = normalize_road_name(road_block.road_name)
        if road_block.blocked:
            self._active_by_name[key] = road_block
        else:
            self._active_by_name.pop(key, None)

    def active_blocks(self) -> list:
        return list(self._active_by_name.values())

    def is_edge_forbidden(self, edge, buffer_m: float):
        """Return the road_name of the first active block forbidding edge
        (by name OR geometry), else None. An edge is never required to
        match on both.
        """
        edge_name = normalize_road_name(edge.road_name)
        for block in self._active_by_name.values():
            if normalize_road_name(block.road_name) == edge_name:
                return block.road_name
            if geometries_intersect(edge.geometry, block.geometry, buffer_m):
                return block.road_name
        return None
