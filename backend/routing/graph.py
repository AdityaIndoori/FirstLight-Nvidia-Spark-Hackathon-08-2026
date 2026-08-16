"""Minimal in-memory road graph.

Plain stdlib adjacency-list representation -- NetworkX is not already a
project dependency, so this stays a small dataclass-based graph instead of
coupling the routing algorithm to it. Injectable: router.py takes a
RoadGraph instance, so tests can exercise Dijkstra against small synthetic
graphs before the real offline OSM dataset is wired in.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    node_id: str
    lng: float
    lat: float


@dataclass(frozen=True)
class Edge:
    """A directed edge. geometry is a GeoJSON-compatible LineString
    coordinate list ([[lng, lat], ...]) running from_node -> to_node.
    """

    from_node: str
    to_node: str
    distance_m: float
    road_name: str
    geometry: list


class RoadGraph:
    """Directed adjacency-list road graph: node_id -> list[Edge] (outgoing
    edges, insertion order preserved -- Dijkstra's edge iteration order,
    and therefore its tie-breaking, depends on this being stable).
    """

    def __init__(self):
        self.nodes: dict = {}
        self.edges: dict = {}

    def add_node(self, node_id: str, lng: float, lat: float) -> None:
        self.nodes[node_id] = Node(node_id=node_id, lng=lng, lat=lat)
        self.edges.setdefault(node_id, [])

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        distance_m: float,
        road_name: str,
        geometry: list,
        bidirectional: bool = False,
    ) -> None:
        """geometry must run from_node -> to_node coordinate order. If
        bidirectional, a reverse edge is also added with geometry reversed.
        """
        self.edges.setdefault(from_node, []).append(
            Edge(
                from_node=from_node,
                to_node=to_node,
                distance_m=distance_m,
                road_name=road_name,
                geometry=list(geometry),
            )
        )
        if bidirectional:
            self.edges.setdefault(to_node, []).append(
                Edge(
                    from_node=to_node,
                    to_node=from_node,
                    distance_m=distance_m,
                    road_name=road_name,
                    geometry=list(reversed(geometry)),
                )
            )

    def outgoing_edges(self, node_id: str) -> list:
        return self.edges.get(node_id, [])
