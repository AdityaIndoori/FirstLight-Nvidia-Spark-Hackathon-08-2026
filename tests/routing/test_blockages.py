from backend.routing.blockages import (
    BlockageRegistry,
    RoadBlock,
    geometries_intersect,
    normalize_road_name,
)
from backend.routing.graph import Edge


# Direct spatial intersection assertions -- never road-step text.
def test_geometries_intersect_true_when_segments_cross():
    edge_geometry = [[-122.395, 47.595], [-122.385, 47.595]]  # horizontal
    block_geometry = [[-122.390, 47.590], [-122.390, 47.600]]  # vertical, crosses it

    assert geometries_intersect(edge_geometry, block_geometry, buffer_m=1.0) is True


def test_geometries_intersect_false_when_far_apart():
    edge_geometry = [[-122.395, 47.595], [-122.385, 47.595]]
    block_geometry = [[10.0, 10.0], [11.0, 11.0]]  # nowhere near

    assert geometries_intersect(edge_geometry, block_geometry, buffer_m=5.0) is False


def test_geometries_intersect_false_when_parallel_and_separated():
    edge_geometry = [[-122.395, 47.595], [-122.385, 47.595]]
    block_geometry = [[-122.395, 47.605], [-122.385, 47.605]]  # same longitudes, different latitude

    assert geometries_intersect(edge_geometry, block_geometry, buffer_m=1.0) is False


def test_geometries_intersect_respects_buffer_tolerance():
    # Two parallel segments ~11 m apart in latitude (0.0001 deg ~= 11.1 m).
    edge_geometry = [[-122.395, 47.595], [-122.385, 47.595]]
    block_geometry = [[-122.395, 47.5951], [-122.385, 47.5951]]

    assert geometries_intersect(edge_geometry, block_geometry, buffer_m=1.0) is False
    assert geometries_intersect(edge_geometry, block_geometry, buffer_m=50.0) is True


def test_normalize_road_name_trims_and_lowercases():
    assert normalize_road_name("  Harbor Ave SW  ") == "harbor ave sw"
    assert normalize_road_name("HARBOR AVE SW") == normalize_road_name("harbor ave sw")


def test_registry_set_and_unset_block():
    registry = BlockageRegistry()
    assert registry.active_blocks() == []

    block = RoadBlock(road_name="Harbor Ave SW", geometry=[[0.0, 0.0], [1.0, 1.0]], blocked=True, operator="ops-1")
    registry.set_block(block)
    assert [b.road_name for b in registry.active_blocks()] == ["Harbor Ave SW"]

    registry.set_block(RoadBlock(road_name="Harbor Ave SW", geometry=block.geometry, blocked=False, operator="ops-1"))
    assert registry.active_blocks() == []


def test_is_edge_forbidden_by_name_match():
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(road_name="Harbor Ave SW", geometry=[[0.0, 0.0], [1.0, 1.0]], blocked=True, operator="ops-1")
    )
    edge = Edge(
        from_node="A", to_node="B", distance_m=100, road_name="harbor ave sw", geometry=[[50.0, 50.0], [51.0, 51.0]]
    )

    assert registry.is_edge_forbidden(edge, buffer_m=1.0) == "Harbor Ave SW"


def test_is_edge_forbidden_by_geometry_match_with_different_name():
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(
            road_name="Unrelated Detour Rd",
            geometry=[[-122.390, 47.590], [-122.390, 47.600]],
            blocked=True,
            operator="ops-1",
        )
    )
    edge = Edge(
        from_node="D",
        to_node="E",
        distance_m=150,
        road_name="D to E Ave",
        geometry=[[-122.395, 47.595], [-122.385, 47.595]],
    )

    assert registry.is_edge_forbidden(edge, buffer_m=1.0) == "Unrelated Detour Rd"


def test_is_edge_forbidden_returns_none_when_clear():
    registry = BlockageRegistry()
    registry.set_block(
        RoadBlock(road_name="Harbor Ave SW", geometry=[[0.0, 0.0], [1.0, 1.0]], blocked=True, operator="ops-1")
    )
    edge = Edge(
        from_node="D",
        to_node="E",
        distance_m=150,
        road_name="D to E Ave",
        geometry=[[-122.395, 47.595], [-122.385, 47.595]],
    )

    assert registry.is_edge_forbidden(edge, buffer_m=1.0) is None
