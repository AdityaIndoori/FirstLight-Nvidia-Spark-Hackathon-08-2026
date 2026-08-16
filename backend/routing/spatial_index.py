"""Nearest-routable-node spatial index.

Wraps scipy.spatial.cKDTree (scipy is already a project dependency) over a
RoadGraph's node coordinates so nearest_node() lookups are O(log n) instead
of a linear scan over every node per request. Built ONCE from a loaded
graph; router.route_between_coordinates() reuses the same index across
every routing request against that graph -- the graph/index are never
rebuilt per request.
"""

from scipy.spatial import cKDTree

from backend.routing.geo_math import point_distance_m


class NodeSpatialIndex:
    """Nearest-routable-node lookup for one loaded RoadGraph."""

    def __init__(self, graph):
        self._node_ids = list(graph.nodes.keys())
        self._coordinates = [(graph.nodes[node_id].lng, graph.nodes[node_id].lat) for node_id in self._node_ids]
        self._tree = cKDTree(self._coordinates) if self._coordinates else None

    def nearest_node(self, lng: float, lat: float):
        """Return (node_id, snap_distance_m) for the routable node nearest
        to (lng, lat). Returns (None, None) if the graph has no nodes.
        """
        if self._tree is None:
            return None, None
        _, index = self._tree.query([lng, lat])
        nearest_coord = self._coordinates[index]
        snap_distance_m = point_distance_m((lng, lat), nearest_coord)
        return self._node_ids[index], snap_distance_m
