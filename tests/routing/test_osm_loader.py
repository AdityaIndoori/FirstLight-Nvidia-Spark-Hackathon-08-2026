import copy
import json
from pathlib import Path

from backend.routing.geo_math import linestring_length_m
from backend.routing.graph import Edge, Node, RoadGraph
from backend.routing.osm_loader import load_road_graph_from_geojson

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "routing" / "sample_roads.geojson"


def load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _find_node_id(graph: RoadGraph, lng: float, lat: float) -> str:
    for node_id, node in graph.nodes.items():
        if abs(node.lng - lng) < 1e-9 and abs(node.lat - lat) < 1e-9:
            return node_id
    raise AssertionError(f"no node at ({lng}, {lat})")


def _edge(graph: RoadGraph, from_lng, from_lat, to_lng, to_lat):
    from_id = _find_node_id(graph, from_lng, from_lat)
    to_id = _find_node_id(graph, to_lng, to_lat)
    for edge in graph.outgoing_edges(from_id):
        if edge.to_node == to_id:
            return edge
    return None


# 1: real-source adapter creates existing graph Node/Edge types
def test_adapter_creates_existing_graph_node_edge_types():
    graph = load_road_graph_from_geojson(load_fixture())

    assert isinstance(graph, RoadGraph)
    assert graph.nodes  # non-empty
    assert all(isinstance(node, Node) for node in graph.nodes.values())
    all_edges = [edge for edges in graph.edges.values() for edge in edges]
    assert all_edges
    assert all(isinstance(edge, Edge) for edge in all_edges)


# 2: geometry remains [lng, lat]
def test_loaded_edge_geometry_remains_lng_lat_order():
    graph = load_road_graph_from_geojson(load_fixture())
    edge = _edge(graph, -122.400, 47.600, -122.395, 47.605)  # START -> B

    assert edge.geometry[0] == [-122.400, 47.600]
    assert edge.geometry[-1] == [-122.395, 47.605]


# 3: distances are meters
def test_distances_are_meters_trusted_when_present_else_computed():
    graph = load_road_graph_from_geojson(load_fixture())

    # START -> D has an explicit length_m=999.0 in the fixture -- must be
    # trusted directly, NOT recomputed from geometry (which would be ~1.1km,
    # not 999.0 -- deliberately different so the test is unambiguous).
    start_to_d = _edge(graph, -122.400, 47.600, -122.395, 47.595)
    assert start_to_d.distance_m == 999.0

    # D -> E has no length_m -- must be computed from geometry in meters.
    d_to_e = _edge(graph, -122.395, 47.595, -122.385, 47.595)
    expected_m = linestring_length_m([[-122.395, 47.595], [-122.385, 47.595]])
    assert d_to_e.distance_m == expected_m
    assert d_to_e.distance_m > 100  # sanity: meters, not raw degrees (~0.01)


# 4: unnamed roads are preserved
def test_unnamed_roads_preserved_with_fallback_name():
    graph = load_road_graph_from_geojson(load_fixture())
    c_to_end = _edge(graph, -122.385, 47.605, -122.380, 47.600)  # name: null in fixture

    assert c_to_end is not None
    assert c_to_end.road_name == "Unnamed road"


# 5: bidirectional roads create both directed edges
def test_bidirectional_road_creates_both_directed_edges():
    graph = load_road_graph_from_geojson(load_fixture())

    forward = _edge(graph, -122.400, 47.600, -122.395, 47.605)  # START -> B
    backward = _edge(graph, -122.395, 47.605, -122.400, 47.600)  # B -> START

    assert forward is not None
    assert backward is not None
    assert forward.road_name == backward.road_name == "Start to B Rd"


# 6: oneway roads create only one directed edge when source supports it
def test_oneway_road_creates_only_one_directed_edge():
    graph = load_road_graph_from_geojson(load_fixture())

    forward = _edge(graph, -122.395, 47.605, -122.385, 47.605)  # B -> C, oneway: true
    backward = _edge(graph, -122.385, 47.605, -122.395, 47.605)  # C -> B

    assert forward is not None
    assert backward is None


def test_non_drivable_features_are_excluded():
    graph = load_road_graph_from_geojson(load_fixture())
    all_names = {edge.road_name for edges in graph.edges.values() for edge in edges}

    assert "Diamond Footpath" not in all_names  # highway=footway
    assert "Closed Service Road" not in all_names  # drivable: false overrides highway=service


# 9: real-data adapter does not mutate input records
def test_loader_does_not_mutate_input_feature_collection():
    feature_collection = load_fixture()
    snapshot = copy.deepcopy(feature_collection)

    load_road_graph_from_geojson(feature_collection)

    assert feature_collection == snapshot
