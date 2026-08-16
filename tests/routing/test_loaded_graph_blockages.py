import json
from pathlib import Path

from backend.routing.blockages import BlockageRegistry, RoadBlock
from backend.routing.osm_loader import load_road_graph_from_geojson
from backend.routing.router import _NO_CLEAN_ROUTE_WARNING, route

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "routing" / "sample_roads.geojson"

START_COORD = [-122.400, 47.600]
END_COORD = [-122.380, 47.600]


def load_fixture_graph():
    with open(FIXTURE_PATH) as f:
        feature_collection = json.load(f)
    return load_road_graph_from_geojson(feature_collection)


def _find_node_id(graph, lng, lat) -> str:
    for node_id, node in graph.nodes.items():
        if abs(node.lng - lng) < 1e-9 and abs(node.lat - lat) < 1e-9:
            return node_id
    raise AssertionError(f"no node at ({lng}, {lat})")


# 11: existing B4a blockage logic still works on loaded edges (by name)
def test_name_blockage_still_works_on_loaded_graph():
    graph = load_fixture_graph()
    start = _find_node_id(graph, *START_COORD)
    end = _find_node_id(graph, *END_COORD)

    clean = route(graph, start, end)
    assert clean["ok"] is True

    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )
    blocked = route(graph, start, end, registry)

    assert blocked["ok"] is True
    assert "Continue on B to C Ave" not in [s["text"] for s in blocked["steps"]]
    assert blocked["distance_m"] != clean["distance_m"]  # detoured
    assert blocked["blocked_roads_avoided"] == ["B to C Ave"]


# 12: geometric blockage still works on loaded edges, even when road names differ
def test_geometric_blockage_still_works_on_loaded_graph():
    graph = load_fixture_graph()
    start = _find_node_id(graph, *START_COORD)
    end = _find_node_id(graph, *END_COORD)

    registry = BlockageRegistry()
    # Blocks the shorter path (via B/C) by name...
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )
    # ...and the longer path's D->E edge purely by GEOMETRY, under an
    # unrelated name -- a vertical line crossing D->E's horizontal line.
    registry.set_block(
        RoadBlock(
            road_name="Unrelated Detour Rd",
            geometry=[[-122.390, 47.590], [-122.390, 47.600]],
            blocked=True,
            operator="ops-2",
        )
    )

    result = route(graph, start, end, registry)

    # If geometry-only blocking did not work on the loaded graph, this
    # would incorrectly succeed via D->E.
    assert result["ok"] is False
    assert set(result["blocked_roads_avoided"]) == {"B to C Ave", "Unrelated Detour Rd"}


# 13: blocked route never crosses the blocked geometry
def test_blocked_route_never_crosses_blocked_geometry():
    graph = load_fixture_graph()
    start = _find_node_id(graph, *START_COORD)
    end = _find_node_id(graph, *END_COORD)

    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )
    result = route(graph, start, end, registry)

    coordinates = result["geometry"]["coordinates"]
    assert [-122.395, 47.605] not in coordinates  # B
    assert [-122.385, 47.605] not in coordinates  # C
    assert result["crosses_blockage"] is False


# 14: no-clean-route behavior remains unchanged on a loaded graph
def test_no_clean_route_behavior_unchanged_on_loaded_graph():
    graph = load_fixture_graph()
    start = _find_node_id(graph, *START_COORD)
    end = _find_node_id(graph, *END_COORD)

    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )
    registry.set_block(
        RoadBlock(
            road_name="D to E Ave",
            geometry=[[-122.395, 47.595], [-122.385, 47.595]],
            blocked=True,
            operator="ops-2",
        )
    )

    result = route(graph, start, end, registry)

    assert result["ok"] is False
    assert result["warning"] == _NO_CLEAN_ROUTE_WARNING
    assert result["geometry"] == {"type": "LineString", "coordinates": []}
    assert result["steps"] == []
    assert result["distance_m"] == 0
    assert result["eta_min"] == 0
    assert result["crosses_blockage"] is False
