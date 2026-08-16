from backend.routing.blockages import BlockageRegistry, RoadBlock
from backend.routing.graph import RoadGraph
from backend.routing.router import _NO_CLEAN_ROUTE_WARNING, route

# Test graph (section 9):
#
#         B ---- C
#        /        \
#  START            END
#        \        /
#         D ---- E
#
# START-B-C-END = 100+100+100 = 300 m  (shorter)
# START-D-E-END = 150+150+150 = 450 m  (longer)

_NODES = [
    ("START", -122.400, 47.600),
    ("B", -122.395, 47.605),
    ("C", -122.385, 47.605),
    ("D", -122.395, 47.595),
    ("E", -122.385, 47.595),
    ("END", -122.380, 47.600),
]


def build_test_graph() -> RoadGraph:
    graph = RoadGraph()
    for node_id, lng, lat in _NODES:
        graph.add_node(node_id, lng, lat)

    graph.add_edge("START", "B", 100, "Start to B Rd", [[-122.400, 47.600], [-122.395, 47.605]])
    graph.add_edge("B", "C", 100, "B to C Ave", [[-122.395, 47.605], [-122.385, 47.605]])
    graph.add_edge("C", "END", 100, "C to End Rd", [[-122.385, 47.605], [-122.380, 47.600]])

    graph.add_edge("START", "D", 150, "Start to D Rd", [[-122.400, 47.600], [-122.395, 47.595]])
    graph.add_edge("D", "E", 150, "D to E Ave", [[-122.395, 47.595], [-122.385, 47.595]])
    graph.add_edge("E", "END", 150, "E to End Rd", [[-122.385, 47.595], [-122.380, 47.600]])

    return graph


def _step_texts(result: dict) -> list:
    return [step["text"] for step in result["steps"]]


# 1: shortest path selected with no blockages
def test_shortest_path_selected_with_no_blockages():
    graph = build_test_graph()
    result = route(graph, "START", "END")

    assert result["ok"] is True
    assert result["distance_m"] == 300
    assert _step_texts(result) == ["Continue on Start to B Rd", "Continue on B to C Ave", "Continue on C to End Rd"]
    assert result["crosses_blockage"] is False
    assert result["warning"] is None


# 2: road-name blockage excludes matching edges
def test_road_name_blockage_excludes_matching_edges():
    graph = build_test_graph()
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )

    result = route(graph, "START", "END", registry)

    assert result["ok"] is True
    assert result["distance_m"] == 450
    assert "Continue on B to C Ave" not in _step_texts(result)


# 3: matching is case-insensitive
def test_road_name_matching_is_case_insensitive():
    graph = build_test_graph()
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="  b TO c AVE  ",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )

    result = route(graph, "START", "END", registry)

    assert result["ok"] is True
    assert result["distance_m"] == 450  # B->C excluded despite case/whitespace differences


# 4: geometry blockage excludes an edge even when names differ -- proven by
# forcing "no clean route" using two DIFFERENTLY-named blocks: one matches
# by name, the other matches only by geometry crossing the other path.
def test_geometry_blockage_excludes_edge_even_when_names_differ():
    graph = build_test_graph()
    registry = BlockageRegistry()

    # Blocks the shorter path by NAME.
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )
    # Blocks the longer path's D->E edge purely by GEOMETRY -- a vertical
    # segment crossing D->E's horizontal line at lat=47.595, lng=-122.390
    # (strictly between D's -122.395 and E's -122.385). The road_name below
    # shares nothing with "D to E Ave".
    registry.set_block(
        RoadBlock(
            road_name="Unrelated Detour Rd",
            geometry=[[-122.390, 47.590], [-122.390, 47.600]],
            blocked=True,
            operator="ops-2",
        )
    )

    result = route(graph, "START", "END", registry)

    # If geometry-only blocking did not work, D->E would still be usable
    # and this would incorrectly succeed via the longer path.
    assert result["ok"] is False
    assert "D to E Ave" not in _step_texts(result)  # (steps is empty anyway, but explicit)
    assert set(result["blocked_roads_avoided"]) == {"B to C Ave", "Unrelated Detour Rd"}


# 5: blocked edge never appears in returned geometry
def test_blocked_edge_never_appears_in_returned_geometry():
    graph = build_test_graph()
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )

    result = route(graph, "START", "END", registry)
    coordinates = result["geometry"]["coordinates"]

    b_coord = [-122.395, 47.605]
    c_coord = [-122.385, 47.605]
    assert b_coord not in coordinates
    assert c_coord not in coordinates


# 6: longer clean path is selected when shorter path is blocked
def test_longer_clean_path_selected_when_shorter_path_blocked():
    graph = build_test_graph()
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )

    result = route(graph, "START", "END", registry)

    assert result["ok"] is True
    assert _step_texts(result) == ["Continue on Start to D Rd", "Continue on D to E Ave", "Continue on E to End Rd"]
    assert result["distance_m"] == 450


# 7: blocked_roads_avoided contains the relevant blocked road
def test_blocked_roads_avoided_contains_relevant_road():
    graph = build_test_graph()
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )

    result = route(graph, "START", "END", registry)
    assert result["blocked_roads_avoided"] == ["B to C Ave"]


# 8: unrelated road blocks are not reported as avoided
def test_unrelated_road_blocks_are_not_reported():
    graph = build_test_graph()
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="Totally Unrelated St",
            geometry=[[10.0, 10.0], [11.0, 11.0]],  # nowhere near this graph
            blocked=True,
            operator="ops-1",
        )
    )

    result = route(graph, "START", "END", registry)

    assert result["ok"] is True
    assert result["distance_m"] == 300  # shortest path unaffected
    assert result["blocked_roads_avoided"] == []


# 9: unblock restores the original shortest route
def test_unblock_restores_original_shortest_route():
    graph = build_test_graph()
    registry = BlockageRegistry()
    block = RoadBlock(
        road_name="B to C Ave",
        geometry=[[-122.395, 47.605], [-122.385, 47.605]],
        blocked=True,
        operator="ops-1",
    )
    registry.set_block(block)
    assert route(graph, "START", "END", registry)["distance_m"] == 450

    registry.set_block(RoadBlock(road_name="B to C Ave", geometry=block.geometry, blocked=False, operator="ops-1"))
    result = route(graph, "START", "END", registry)

    assert result["ok"] is True
    assert result["distance_m"] == 300
    assert result["blocked_roads_avoided"] == []


# 10, 11, 12, 13: no clean route -- both paths blocked by name
def test_no_clean_route_when_both_paths_blocked():
    graph = build_test_graph()
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

    result = route(graph, "START", "END", registry)

    # 10
    assert result["ok"] is False
    # 11
    assert result["warning"] == _NO_CLEAN_ROUTE_WARNING
    assert isinstance(result["warning"], str) and result["warning"]
    # 12: no blockage-crossing fallback
    assert result["steps"] == []
    assert result["geometry"] == {"type": "LineString", "coordinates": []}
    assert result["distance_m"] == 0
    assert result["eta_min"] == 0
    # 13
    assert result["crosses_blockage"] is False


# 14: distance_m equals selected edge-distance sum
def test_distance_m_equals_selected_edge_distance_sum():
    graph = build_test_graph()
    result = route(graph, "START", "END")
    assert result["distance_m"] == 100 + 100 + 100


# 15: eta calculation is deterministic
def test_eta_calculation_is_deterministic():
    graph = build_test_graph()
    first = route(graph, "START", "END")
    second = route(graph, "START", "END")

    assert first["eta_min"] == second["eta_min"]
    # distance_m=300 -> 0.3 km / 40 km/h * 60 = 0.45 min, at the stated
    # EMERGENCY_ROUTING_SPEED_KMH=40.0 default
    assert first["eta_min"] == 0.45


# 16: coordinates remain [lng, lat]
def test_coordinates_remain_lng_lat_order():
    graph = build_test_graph()
    result = route(graph, "START", "END")
    coordinates = result["geometry"]["coordinates"]

    start_node = graph.nodes["START"]
    end_node = graph.nodes["END"]
    assert coordinates[0] == [start_node.lng, start_node.lat]
    assert coordinates[-1] == [end_node.lng, end_node.lat]


# 17: equal-cost routes resolve deterministically
def test_equal_cost_routes_resolve_deterministically():
    graph = RoadGraph()
    graph.add_node("S", 0.0, 0.0)
    graph.add_node("X", 1.0, 1.0)
    graph.add_node("Y", 1.0, -1.0)
    graph.add_node("T", 2.0, 0.0)
    graph.add_edge("S", "X", 100, "S to X Rd", [[0.0, 0.0], [1.0, 1.0]])
    graph.add_edge("X", "T", 100, "X to T Rd", [[1.0, 1.0], [2.0, 0.0]])
    graph.add_edge("S", "Y", 100, "S to Y Rd", [[0.0, 0.0], [1.0, -1.0]])
    graph.add_edge("Y", "T", 100, "Y to T Rd", [[1.0, -1.0], [2.0, 0.0]])

    results = [route(graph, "S", "T") for _ in range(5)]

    assert all(r["distance_m"] == 200 for r in results)
    assert all(r == results[0] for r in results)  # every run picks the identical route
    # documented tie-break: heap ties broken by node_id, "X" < "Y"
    assert _step_texts(results[0]) == ["Continue on S to X Rd", "Continue on X to T Rd"]


# 18: input graph is not mutated by route calculation
def test_input_graph_not_mutated():
    graph = build_test_graph()
    nodes_before = dict(graph.nodes)
    edges_before = {node_id: list(edges) for node_id, edges in graph.edges.items()}

    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="B to C Ave",
            geometry=[[-122.395, 47.605], [-122.385, 47.605]],
            blocked=True,
            operator="ops-1",
        )
    )
    route(graph, "START", "END", registry)

    assert graph.nodes == nodes_before
    assert graph.edges == edges_before
