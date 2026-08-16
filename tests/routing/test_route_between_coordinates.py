import json
from pathlib import Path

from backend.routing.osm_loader import load_road_graph_from_geojson
from backend.routing.router import DEFAULT_MAX_SNAP_DISTANCE_M, route_between_coordinates
from backend.routing.spatial_index import NodeSpatialIndex

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "routing" / "sample_roads.geojson"

START_COORD = [-122.400, 47.600]
END_COORD = [-122.380, 47.600]


def load_fixture_graph():
    with open(FIXTURE_PATH) as f:
        feature_collection = json.load(f)
    return load_road_graph_from_geojson(feature_collection)


def test_route_between_coordinates_snaps_and_routes():
    graph = load_fixture_graph()
    index = NodeSpatialIndex(graph)

    result = route_between_coordinates(graph, index, START_COORD, END_COORD)

    assert result["ok"] is True
    assert result["distance_m"] > 0
    assert result["geometry"]["coordinates"][0] == START_COORD
    assert result["geometry"]["coordinates"][-1] == END_COORD


# 8: excessive snap distance fails loudly
def test_excessive_snap_distance_fails_loudly():
    graph = load_fixture_graph()
    index = NodeSpatialIndex(graph)

    far_away_coord = [10.0, 10.0]  # nowhere near the fixture's road network
    result = route_between_coordinates(graph, index, far_away_coord, END_COORD)

    assert result["ok"] is False
    assert result["geometry"] == {"type": "LineString", "coordinates": []}
    assert result["steps"] == []
    assert result["distance_m"] == 0
    assert result["eta_min"] == 0
    assert result["crosses_blockage"] is False
    assert isinstance(result["warning"], str) and "start" in result["warning"].lower()


def test_excessive_snap_distance_on_end_coordinate_also_fails_loudly():
    graph = load_fixture_graph()
    index = NodeSpatialIndex(graph)

    far_away_coord = [10.0, 10.0]
    result = route_between_coordinates(graph, index, START_COORD, far_away_coord)

    assert result["ok"] is False
    assert "end" in result["warning"].lower()


def test_max_snap_distance_is_configurable():
    graph = load_fixture_graph()
    index = NodeSpatialIndex(graph)

    # A point a bit off of START -- succeeds with a generous max, fails with a tiny one.
    nearby_coord = [-122.4005, 47.6005]

    generous = route_between_coordinates(graph, index, nearby_coord, END_COORD, max_snap_distance_m=1000.0)
    strict = route_between_coordinates(graph, index, nearby_coord, END_COORD, max_snap_distance_m=0.001)

    assert generous["ok"] is True
    assert strict["ok"] is False


def test_default_max_snap_distance_is_a_named_constant():
    assert DEFAULT_MAX_SNAP_DISTANCE_M > 0


# 10: graph is loaded once / reused across requests
def test_graph_and_index_loaded_once_and_reused_across_requests():
    graph = load_fixture_graph()
    index = NodeSpatialIndex(graph)  # built exactly once

    first = route_between_coordinates(graph, index, START_COORD, END_COORD)
    second = route_between_coordinates(graph, index, START_COORD, END_COORD)

    assert first["ok"] is True and second["ok"] is True
    assert first == second  # same loaded graph/index -> identical deterministic result
