"""Adapter: local OSM-derived road source -> the existing B4a RoadGraph.

STATUS AS OF B4b: no real local OSM road/facility data source exists in this
repository yet. Checked: no *.osm/*.pbf/*.geojson/*.json/*.gpkg/*.sqlite/
*.db/*.graphml files anywhere in the repo, no SQLite database with a road
table, and tests/fixtures/ was present but empty. This loader is therefore
built against a DOCUMENTED, ASSUMED local-source interface -- a GeoJSON
FeatureCollection of road LineStrings, the format most local OSM export
tools (osmnx, ogr2ogr from a .pbf/.osm extract, etc.) already produce, and
consistent with every other GeoJSON shape already used in this project
(flight_planner.py, the frozen Route contract itself). Swap in the exact
export Member A provides by writing a second `load_road_graph_from_*`
function with the same RoadGraph-shaped return value -- nothing downstream
(router.py, blockages.py) needs to change.

Expected per-feature shape:
    {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[lng, lat], ...]},
        "properties": {
            "name": str | null,          # missing/null -> "Unnamed road"
            "drivable": bool | null,     # TRUSTED DIRECTLY if present
            "highway": str | null,       # OSM tag, used only if "drivable" absent
            "oneway": bool | str | null, # missing/null -> bidirectional (documented default)
            "length_m": float | null,    # TRUSTED DIRECTLY if present and > 0
        },
    }

Only LineString features are considered; anything else is skipped. Never
mutates the input FeatureCollection.

Road filtering (deliberately simple -- no OSM access-policy engine):
  - properties.drivable, if present, is trusted as-is.
  - otherwise, properties.highway is checked against a small denylist of
    obviously non-drivable tags (footway, steps, pedestrian, path, track,
    cycleway, bridleway, corridor).
  - a feature with neither field is treated as drivable rather than dropped
    -- unknown metadata is not a reason to discard road data.

A feature with no usable name is preserved (not dropped) as "Unnamed road".

Node identity: OSM-derived road networks share exact coordinate values at
intersections, so each LineString endpoint is keyed by its rounded
[lng, lat] -- two features sharing an endpoint coordinate become the same
graph node, which is what makes the loaded graph routable end to end.
"""

from backend.routing.geo_math import linestring_length_m
from backend.routing.graph import RoadGraph

_NON_DRIVABLE_HIGHWAY_TAGS = frozenset(
    {"footway", "steps", "pedestrian", "path", "track", "cycleway", "bridleway", "corridor"}
)
_UNNAMED_ROAD_FALLBACK = "Unnamed road"
_COORD_PRECISION = 7  # ~1.1 cm at these latitudes -- dedupes floating-point noise, not real distinctness


def _is_drivable(properties: dict) -> bool:
    drivable = properties.get("drivable")
    if drivable is not None:
        return bool(drivable)
    highway = str(properties.get("highway") or "").strip().lower()
    if not highway:
        return True
    return highway not in _NON_DRIVABLE_HIGHWAY_TAGS


def _road_name(properties: dict) -> str:
    name = properties.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return _UNNAMED_ROAD_FALLBACK


def _is_oneway(properties: dict):
    """True/False if the source states oneway; None if absent -- B4b treats
    an absent oneway field as bidirectional, documented rather than silently
    assumed (see module docstring). Does not handle OSM's oneway=-1
    (reversed-direction) convention -- out of scope for this simple pass.
    """
    value = properties.get("oneway")
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _edge_distance_m(properties: dict, coordinates: list) -> float:
    length_m = properties.get("length_m")
    if isinstance(length_m, (int, float)) and length_m > 0:
        return float(length_m)
    return linestring_length_m(coordinates)


def _node_key(lng: float, lat: float) -> str:
    return f"{round(lng, _COORD_PRECISION)},{round(lat, _COORD_PRECISION)}"


def load_road_graph_from_geojson(feature_collection: dict) -> RoadGraph:
    """Build a RoadGraph (backend/routing/graph.py -- the same type the
    existing Dijkstra engine already consumes) from a local GeoJSON
    FeatureCollection of road LineStrings. See this module's docstring for
    the exact expected per-feature shape.

    Never mutates feature_collection. Skips non-LineString and non-drivable
    features. Unnamed roads are preserved with a deterministic fallback
    name, never dropped.
    """
    graph = RoadGraph()
    seen_nodes = set()

    for feature in feature_collection.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue

        properties = feature.get("properties") or {}
        if not _is_drivable(properties):
            continue

        road_name = _road_name(properties)
        distance_m = _edge_distance_m(properties, coordinates)

        from_lng, from_lat = coordinates[0]
        to_lng, to_lat = coordinates[-1]
        from_node = _node_key(from_lng, from_lat)
        to_node = _node_key(to_lng, to_lat)

        for node_id, lng, lat in ((from_node, from_lng, from_lat), (to_node, to_lng, to_lat)):
            if node_id not in seen_nodes:
                graph.add_node(node_id, lng, lat)
                seen_nodes.add(node_id)

        oneway = _is_oneway(properties)
        graph.add_edge(
            from_node,
            to_node,
            distance_m,
            road_name,
            coordinates,
            bidirectional=not (oneway is True),
        )

    return graph
