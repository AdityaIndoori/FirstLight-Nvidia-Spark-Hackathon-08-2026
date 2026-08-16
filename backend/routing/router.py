"""Deterministic Dijkstra routing over an in-memory RoadGraph (graph.py),
with active road blocks (blockages.py) banned at edge level. Returns
exactly the frozen Route contract. Never returns a route that crosses an
active blockage -- if no blockage-free path exists, returns ok=false with
an explicit warning instead of silently routing through one.

    Route = {
        ok: bool,
        geometry: LineString,          # GeoJSON, coordinates always [lng, lat]
        steps: [{text: str, dist_m: int}],
        distance_m: int,
        eta_min: float,
        crosses_blockage: bool,        # always False here -- a blockage-
                                        # crossing route is never selected
        blocked_roads_avoided: [str],
        warning: str | None,
    }
"""

import heapq

from backend.routing.blockages import BlockageRegistry

EMERGENCY_ROUTING_SPEED_KMH = 40.0
"""B4a placeholder ETA speed: a flat 40 km/h assumed average emergency-
vehicle road speed. Not derived from real speed limits or road class yet --
that is later work. Named here so eta_min's assumption is never a hidden
magic number.
"""

DEFAULT_BLOCKAGE_BUFFER_M = 5.0
"""Geometry-intersection tolerance in meters applied to blocked-road
geometry before testing edge intersection (see blockages.geometries_intersect)."""

_NO_CLEAN_ROUTE_WARNING = "No blockage-free route is available."

DEFAULT_MAX_SNAP_DISTANCE_M = 250.0
"""B4b: how far (meters) a requested [lng, lat] may be from the nearest
routable node before route_between_coordinates() refuses to snap it, rather
than silently snapping an absurd distance. Named so it is never a hidden
magic number.
"""


def route(
    graph,
    start_node: str,
    end_node: str,
    blockage_registry: BlockageRegistry = None,
    speed_kmh: float = EMERGENCY_ROUTING_SPEED_KMH,
    buffer_m: float = DEFAULT_BLOCKAGE_BUFFER_M,
) -> dict:
    """Shortest (by distance_m) blockage-free route from start_node to
    end_node in graph. Never mutates graph or blockage_registry.

    blockage_registry defaults to an empty registry (no active blocks) if
    not supplied. Returns exactly the frozen Route contract described in
    this module's docstring.
    """
    active_registry = blockage_registry if blockage_registry is not None else BlockageRegistry()

    dist, prev_edge, blocked_roads_avoided = _dijkstra(graph, start_node, end_node, active_registry, buffer_m)

    if end_node not in dist:
        return {
            "ok": False,
            "geometry": {"type": "LineString", "coordinates": []},
            "steps": [],
            "distance_m": 0,
            "eta_min": 0,
            "crosses_blockage": False,
            "blocked_roads_avoided": sorted(blocked_roads_avoided),
            "warning": _NO_CLEAN_ROUTE_WARNING,
        }

    edges = _reconstruct_path(prev_edge, start_node, end_node)
    coordinates = _build_geometry(edges)
    steps = [{"text": f"Continue on {edge.road_name}", "dist_m": int(round(edge.distance_m))} for edge in edges]
    distance_m = int(round(sum(edge.distance_m for edge in edges)))

    return {
        "ok": True,
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "steps": steps,
        "distance_m": distance_m,
        "eta_min": _compute_eta_min(distance_m, speed_kmh),
        "crosses_blockage": False,
        "blocked_roads_avoided": sorted(blocked_roads_avoided),
        "warning": None,
    }


def route_between_coordinates(
    graph,
    spatial_index,
    start: list,
    end: list,
    blockage_registry: BlockageRegistry = None,
    max_snap_distance_m: float = DEFAULT_MAX_SNAP_DISTANCE_M,
    speed_kmh: float = EMERGENCY_ROUTING_SPEED_KMH,
    buffer_m: float = DEFAULT_BLOCKAGE_BUFFER_M,
) -> dict:
    """B4b INTERNAL routing entry point -- NOT the public C -> B API yet.

    start/end are [lng, lat] coordinates. Snaps each to the nearest
    routable node in `graph` via `spatial_index` (a
    spatial_index.NodeSpatialIndex built once from `graph` and reused
    across calls -- neither `graph` nor `spatial_index` is rebuilt here),
    then delegates to the existing, unmodified route() for the actual
    pathfinding. Returns exactly the frozen Route contract route() already
    returns.

    If either coordinate cannot be reasonably snapped (nearer than
    max_snap_distance_m to any routable node, or the graph has no nodes at
    all), returns ok=false with an explicit warning instead of silently
    snapping an absurd distance.
    """
    start_node, start_snap_m = spatial_index.nearest_node(start[0], start[1])
    if start_node is None or start_snap_m > max_snap_distance_m:
        return _snap_failure_result("start", max_snap_distance_m)

    end_node, end_snap_m = spatial_index.nearest_node(end[0], end[1])
    if end_node is None or end_snap_m > max_snap_distance_m:
        return _snap_failure_result("end", max_snap_distance_m)

    return route(graph, start_node, end_node, blockage_registry, speed_kmh=speed_kmh, buffer_m=buffer_m)


def _snap_failure_result(label: str, max_snap_distance_m: float) -> dict:
    return {
        "ok": False,
        "geometry": {"type": "LineString", "coordinates": []},
        "steps": [],
        "distance_m": 0,
        "eta_min": 0,
        "crosses_blockage": False,
        "blocked_roads_avoided": [],
        "warning": (
            f"Could not snap {label} coordinate to a routable road within {max_snap_distance_m:.0f} m."
        ),
    }


def _dijkstra(graph, start_node: str, end_node: str, blockage_registry: BlockageRegistry, buffer_m: float):
    """Deterministic Dijkstra by distance_m. Heap entries are (distance,
    node_id): ties are broken by node_id string comparison, edges are
    considered in the graph's insertion order, and relaxation uses strict
    '<' so the first-found equal-cost path always wins -- identical inputs
    always produce an identical result across runs.

    Early-exits once end_node is popped (settled), so a blocked edge only
    counts toward blocked_roads_avoided if it was actually examined while
    reaching that point -- blocks elsewhere in the graph are never reported.

    Returns (dist: {node_id: float}, prev_edge: {node_id: Edge},
    blocked_roads_avoided: set[str]).
    """
    dist = {start_node: 0}
    prev_edge = {}
    visited = set()
    blocked_roads_avoided = set()
    heap = [(0, start_node)]

    while heap:
        current_distance, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == end_node:
            break

        for edge in graph.outgoing_edges(node):
            blocking_road_name = blockage_registry.is_edge_forbidden(edge, buffer_m)
            if blocking_road_name is not None:
                blocked_roads_avoided.add(blocking_road_name)
                continue

            neighbor = edge.to_node
            new_distance = current_distance + edge.distance_m
            if neighbor not in dist or new_distance < dist[neighbor]:
                dist[neighbor] = new_distance
                prev_edge[neighbor] = edge
                heapq.heappush(heap, (new_distance, neighbor))

    return dist, prev_edge, blocked_roads_avoided


def _reconstruct_path(prev_edge: dict, start_node: str, end_node: str) -> list:
    edges = []
    node = end_node
    while node != start_node:
        edge = prev_edge[node]
        edges.append(edge)
        node = edge.from_node
    edges.reverse()
    return edges


def _build_geometry(edges: list) -> list:
    """Concatenate edge geometries into one LineString's coordinates,
    dropping each edge's duplicate leading point (shared with the previous
    edge's trailing point at the junction node).
    """
    coordinates = []
    for index, edge in enumerate(edges):
        points = edge.geometry
        coordinates.extend(points if index == 0 else points[1:])
    return coordinates


def _compute_eta_min(distance_m: int, speed_kmh: float) -> float:
    hours = (distance_m / 1000.0) / speed_kmh
    return round(hours * 60.0, 2)
