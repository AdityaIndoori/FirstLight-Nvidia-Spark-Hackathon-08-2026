#!/usr/bin/env python3
"""B4 bench: 50 routes at demo tempo, with and without closures.

Fifty is the demo's number (fifty buildings in the rank), so this measures what
the console actually asks for in one beat. Uses an isolated db so it cannot touch
the box's real rows.

Usage: python scripts/routing_bench.py
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["FIRSTLIGHT_DB"] = str(Path(tempfile.gettempdir()) / "firstlight-b4-bench.db")
_db = Path(os.environ["FIRSTLIGHT_DB"])
if _db.exists():
    _db.unlink()

from app import db, routing, scorer  # noqa: E402

N = 50


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> int:
    db.init()
    t0 = time.perf_counter()
    g = routing.build_graph(force=True)
    build_ms = (time.perf_counter() - t0) * 1000.0
    if not g.edges:
        print("NO ROAD GRAPH: nothing to bench")
        return 1
    print(f"graph build (cold)     {build_ms:.0f} ms   {len(g.nodes)} nodes  {len(g.edges)} edges")
    t0 = time.perf_counter()
    routing.build_graph()
    print(f"graph build (cached)   {(time.perf_counter() - t0) * 1000:.3f} ms")

    random.seed(7)
    pairs = [(random.choice(g.nodes), random.choice(g.nodes)) for _ in range(N)]

    def sweep(label: str) -> tuple[list[float], int, int, int]:
        lat: list[float] = []
        ok = cross = named = 0
        dists: list[int] = []
        steps: list[int] = []
        for a, b in pairs:
            t = time.perf_counter()
            r = routing.route(list(a), list(b))
            lat.append((time.perf_counter() - t) * 1000.0)
            if r["ok"]:
                ok += 1
                dists.append(r["distance_m"])
                steps.append(len(r["steps"]))
                if r["crosses_blockage"]:
                    cross += 1
                if r["blocked_roads_avoided"]:
                    named += 1
        print(f"\n{label}")
        print(f"  ok                   {ok}/{N}")
        print(f"  latency p50          {statistics.median(lat):.1f} ms")
        print(f"  latency mean         {statistics.mean(lat):.1f} ms")
        print(f"  latency p95          {pct(lat, 0.95):.1f} ms")
        print(f"  latency max          {max(lat):.1f} ms")
        print(f"  all {N} sequentially  {sum(lat):.0f} ms")
        if dists:
            print(
                f"  distance median      {statistics.median(dists):.0f} m, "
                f"steps median {statistics.median(steps):.0f}"
            )
        return lat, ok, cross, named

    sweep("50 routes, no closures")

    blocked = []
    for eid in (len(g.edges) // 3, len(g.edges) // 2, 2 * len(g.edges) // 3):
        ed = g.edges[eid]
        name = ed.name or f"unnamed segment {eid}"
        line = {"type": "LineString", "coordinates": [list(g.nodes[ed.u]), list(g.nodes[ed.v])]}
        scorer.set_road_block(name, line, True, "bench")
        blocked.append((name, line))
    routing.reset()
    routing.build_graph()

    stats = routing.graph_stats()
    banned = stats["blocked_edges_excluded"]
    print(f"\n3 closures declared -> {banned} edges banned")
    t0 = time.perf_counter()
    routing.graph_stats()
    print(f"  ban recompute cached {(time.perf_counter() - t0) * 1000:.3f} ms")

    _, ok2, cross, named = sweep("50 routes, 3 closures standing")
    print(f"  crosses_blockage     {cross} of {ok2} ok routes (must be 0)")
    print(f"  named a closure      {named} of {ok2} ok routes")

    worst = 0.0
    for a, b in pairs:
        r = routing.route(list(a), list(b))
        if not r["ok"]:
            continue
        for _, line in blocked:
            worst = max(worst, routing.shared_length_m(r["geometry"], line))
    print(f"\nworst metres along ANY closure over every ok route: {worst} m (tolerance 30)")
    verdict = "OK" if cross == 0 and worst <= 30.0 else "FAILED"
    print(verdict)
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
