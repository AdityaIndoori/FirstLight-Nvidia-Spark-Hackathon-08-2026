"""FIRST LIGHT routing engine.

B4a: Dijkstra over an in-memory, injectable road graph (graph.py), with
active road blocks banned at edge level by name and real polyline geometry
(blockages.py). See router.route() for the frozen Route contract.

B4b: adapter from a local OSM-derived road source into that same RoadGraph
(osm_loader.py -- no real source exists in this repo yet, see its module
docstring), a nearest-routable-node spatial index (spatial_index.py), and
an internal coordinate-based entry point (router.route_between_coordinates)
that snaps [lng, lat] pairs to graph nodes before delegating to the
unmodified route(). No network calls, no OSM download, no UI here.
"""

from backend.routing.blockages import BlockageRegistry, RoadBlock
from backend.routing.graph import Edge, Node, RoadGraph
from backend.routing.osm_loader import load_road_graph_from_geojson
from backend.routing.router import (
    DEFAULT_BLOCKAGE_BUFFER_M,
    DEFAULT_MAX_SNAP_DISTANCE_M,
    EMERGENCY_ROUTING_SPEED_KMH,
    route,
    route_between_coordinates,
)
from backend.routing.spatial_index import NodeSpatialIndex

__all__ = [
    "BlockageRegistry",
    "RoadBlock",
    "Edge",
    "Node",
    "RoadGraph",
    "route",
    "route_between_coordinates",
    "load_road_graph_from_geojson",
    "NodeSpatialIndex",
    "EMERGENCY_ROUTING_SPEED_KMH",
    "DEFAULT_BLOCKAGE_BUFFER_M",
    "DEFAULT_MAX_SNAP_DISTANCE_M",
]
