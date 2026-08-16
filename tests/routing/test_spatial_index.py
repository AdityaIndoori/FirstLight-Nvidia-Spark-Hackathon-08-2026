from backend.routing.graph import RoadGraph
from backend.routing.spatial_index import NodeSpatialIndex


def _build_small_graph() -> RoadGraph:
    graph = RoadGraph()
    graph.add_node("START", -122.400, 47.600)
    graph.add_node("B", -122.395, 47.605)
    graph.add_node("END", -122.380, 47.600)
    graph.add_edge("START", "B", 100, "Start to B Rd", [[-122.400, 47.600], [-122.395, 47.605]])
    return graph


# 7: nearest-node snapping chooses the correct node
def test_nearest_node_chooses_correct_node():
    graph = _build_small_graph()
    index = NodeSpatialIndex(graph)

    # A tiny offset from START -- must still snap to START, not B or END.
    node_id, snap_distance_m = index.nearest_node(-122.4001, 47.6001)

    assert node_id == "START"
    assert snap_distance_m < 50.0  # a ~0.0001 deg offset is a handful of meters


def test_nearest_node_picks_closest_of_multiple_candidates():
    graph = _build_small_graph()
    index = NodeSpatialIndex(graph)

    # Roughly equidistant-ish query point, nudged toward END.
    node_id, _ = index.nearest_node(-122.381, 47.600)
    assert node_id == "END"


def test_nearest_node_on_empty_graph_returns_none():
    index = NodeSpatialIndex(RoadGraph())
    node_id, snap_distance_m = index.nearest_node(-122.4, 47.6)
    assert node_id is None
    assert snap_distance_m is None
