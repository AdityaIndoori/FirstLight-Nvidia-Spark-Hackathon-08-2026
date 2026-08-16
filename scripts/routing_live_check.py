#!/usr/bin/env python3
"""B4b routing smoke check -- LOCAL DATA ONLY, no network calls, no download.

As of this script's writing, no real local OSM road/facility data source
exists anywhere in this repository (checked: no *.osm/*.pbf/*.geojson/
*.json/*.gpkg/*.sqlite/*.db/*.graphml files, no SQLite database with a road
table, tests/fixtures/ was present but empty).

This script therefore:
  1. Looks for a real dataset at the path in FIRSTLIGHT_ROAD_DATA_PATH (a
     GeoJSON FeatureCollection matching backend/routing/osm_loader.py's
     documented interface).
  2. If that env var is unset or the file doesn't exist, it LOUDLY says so
     and falls back to the small synthetic fixture at
     tests/fixtures/routing/sample_roads.geojson -- every print below is
     labeled accordingly so this is never mistaken for a real-data run.

Usage:
    python scripts/routing_live_check.py
    FIRSTLIGHT_ROAD_DATA_PATH=/path/to/real_roads.geojson python scripts/routing_live_check.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.routing.blockages import BlockageRegistry, RoadBlock
from backend.routing.osm_loader import load_road_graph_from_geojson
from backend.routing.router import route_between_coordinates
from backend.routing.spatial_index import NodeSpatialIndex

_ROAD_DATA_ENV_VAR = "FIRSTLIGHT_ROAD_DATA_PATH"
_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "routing" / "sample_roads.geojson"

# Deterministic coordinates inside the fixture's demo AOI (the B4a diamond).
START_COORD = [-122.400, 47.600]
END_COORD = [-122.380, 47.600]


def _resolve_data_path() -> tuple:
    """Returns (path, is_real_data: bool)."""
    configured = os.environ.get(_ROAD_DATA_ENV_VAR)
    if configured and Path(configured).exists():
        return Path(configured), True

    print(f"NOTE: {_ROAD_DATA_ENV_VAR} is not set (or does not point to an existing file).")
    print("NO REAL LOCAL OSM ROAD SOURCE WAS FOUND.")
    print(f"Falling back to SYNTHETIC FIXTURE DATA at {_FIXTURE_PATH} -- NOT real OSM data.\n")
    return _FIXTURE_PATH, False


def main():
    data_path, is_real_data = _resolve_data_path()
    label = "REAL DATA" if is_real_data else "FIXTURE DATA (not real OSM)"

    with open(data_path) as f:
        feature_collection = json.load(f)

    graph = load_road_graph_from_geojson(feature_collection)
    index = NodeSpatialIndex(graph)

    node_count = len(graph.nodes)
    edge_count = sum(len(edges) for edges in graph.edges.values())
    named_edge_count = sum(
        1 for edges in graph.edges.values() for edge in edges if edge.road_name != "Unnamed road"
    )

    print(f"[{label}] Loaded {data_path}")
    print(f"node count: {node_count}")
    print(f"edge count: {edge_count}")
    print(f"named-road edge count: {named_edge_count}\n")

    print(f"[{label}] Routing {START_COORD} -> {END_COORD}, no blockages...")
    before = route_between_coordinates(graph, index, START_COORD, END_COORD)
    print(f"ok: {before['ok']}")
    print(f"distance_m: {before['distance_m']}")
    print(f"eta_min: {before['eta_min']}")
    print(f"route edges/steps: {len(before['steps'])}")
    for step in before["steps"]:
        print(f"  - {step['text']} ({step['dist_m']} m)")
    print()

    if not before["ok"] or not before["steps"]:
        print("No clean baseline route was found -- cannot demonstrate a blockage on it. Stopping.")
        return

    # Pick a "middle" edge of the route rather than the first: an edge
    # touching the start (or end) node shares that exact junction
    # coordinate with other roads leaving the same junction, and the
    # frozen B4a geometry check (LineString.intersects(), even before any
    # buffer) treats a shared endpoint as an intersection -- so blocking a
    # first/last edge can incorrectly also exclude an unrelated road at the
    # same junction. A middle edge avoids that and cleanly demonstrates a
    # detour. This is a property of the existing, unmodified blockage
    # algorithm, not something this script works around silently.
    demo_step_index = len(before["steps"]) // 2
    blocked_road_name = before["steps"][demo_step_index]["text"].removeprefix("Continue on ")
    # Recover that step's edge geometry from the graph to block it precisely.
    blocked_geometry = None
    for edges in graph.edges.values():
        for edge in edges:
            if edge.road_name == blocked_road_name:
                blocked_geometry = edge.geometry
                break
        if blocked_geometry is not None:
            break

    print(f"[{label}] Activating a blockage on: {blocked_road_name!r}")
    registry = BlockageRegistry()
    registry.set_block(RoadBlock(road_name=blocked_road_name, geometry=blocked_geometry, blocked=True, operator="ops-smoke-check"))

    after = route_between_coordinates(graph, index, START_COORD, END_COORD, blockage_registry=registry)
    print(f"ok: {after['ok']}")
    print(f"distance_m: {after['distance_m']}")
    print(f"eta_min: {after['eta_min']}")
    print(f"warning: {after['warning']}")
    print(f"blocked_roads_avoided: {after['blocked_roads_avoided']}\n")

    print(f"Before distance_m: {before['distance_m']}   After distance_m: {after['distance_m']}")

    if after["ok"] and after["distance_m"] != before["distance_m"]:
        print("RESULT: a different (longer) clean route was returned around the blockage. (A)")
    elif not after["ok"]:
        print(f"RESULT: no clean route is available -- warning was explicit: {after['warning']!r} (B)")
    else:
        print("RESULT: UNEXPECTED -- route unchanged despite an active blockage on one of its own roads.")
        sys.exit(1)


if __name__ == "__main__":
    main()
