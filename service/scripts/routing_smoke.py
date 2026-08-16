#!/usr/bin/env python3
"""B4 smoke: drive the real endpoints, then measure on whatever roads are local.

Not a test. This is the "run the thing" proof: it boots the FastAPI app, calls
GET /api/route and GET /api/plan through the real HTTP surface, then reports the
measured graph size, a real route, its latency and a real before/after detour.

Usage:  python scripts/routing_smoke.py
Env:    FIRSTLIGHT_DATA points at the data dir to measure against.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import config, datasets, db, routing, scorer  # noqa: E402
from app.main import app  # noqa: E402


def hr(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> int:
    db.init()
    hr("environment")
    print(f"dataset dir      {config.DATASET_DIR}")
    print(f"AOI              {config.AOI}  ({config.AOI_NAME})")
    counts = datasets.available()
    print(f"dataset features {counts}")

    hr("graph")
    t0 = time.perf_counter()
    routing.build_graph(force=True)
    build_ms = (time.perf_counter() - t0) * 1000.0
    stats = routing.graph_stats()
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"measured build   {build_ms:.0f} ms")
    if not routing.available():
        print("\nNO ROAD GRAPH. Everything below reports the honest refusal path.")

    # A real origin and destination: the two graph nodes furthest apart along the
    # first road, so the route is not a one-edge triviality.
    g = routing.build_graph()
    hr("one real route")
    if len(g.nodes) >= 2:
        origin = g.nodes[0]
        dest = max(g.nodes, key=lambda p: routing.haversine_m(origin, p))
        print(f"origin {list(origin)}  dest {list(dest)}")
        samples = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = routing.route(list(origin), list(dest))
            samples.append((time.perf_counter() - t0) * 1000.0)
        print(f"ok               {r['ok']}")
        print(f"distance_m       {r['distance_m']}")
        print(f"eta_min          {r['eta_min']}")
        print(f"steps            {len(r['steps'])}")
        print(f"coords           {len((r['geometry'] or {}).get('coordinates', []))}")
        print(f"crosses_blockage {r['crosses_blockage']}")
        print(f"warning          {r['warning']}")
        print(
            f"latency          min {min(samples):.1f} ms  median "
            f"{statistics.median(samples):.1f} ms  max {max(samples):.1f} ms  (n=10)"
        )
        for s in r["steps"][:12]:
            print(f"   {s['dist_m']:>6} m  {s['text']}")
        if len(r["steps"]) > 12:
            print(f"   ... {len(r['steps']) - 12} more")
    else:
        print("no nodes, so no route to measure")
        r = routing.route([config.AOI[0], config.AOI[1]], [config.AOI[2], config.AOI[3]])
        print(f"refusal          {r['warning']}")

    hr("before / after a closure")
    if r.get("ok") and len(g.nodes) >= 2:
        # Close a segment the route ACTUALLY DRIVES, halfway along it. Picking the
        # edge nearest the midpoint coordinate is not the same thing: on a dense
        # network the nearest edge is often a side street the route never touches,
        # which closes something irrelevant and reports a +0 m detour that says
        # nothing. So walk the returned line and take the middle segment of it.
        coords = r["geometry"]["coordinates"]
        i = max(1, len(coords) // 2)
        a, b = coords[i - 1], coords[i]
        line = {"type": "LineString", "coordinates": [list(a), list(b)]}
        # Name it after the road the route says it is on at that point, purely for
        # the log line: the ban is geometric either way.
        label = ""
        for eid, ed in enumerate(g.edges):
            if routing.haversine_m(g.nodes[ed.u], a) < 1.0 and routing.haversine_m(g.nodes[ed.v], b) < 1.0:
                label = ed.name
                break
            if routing.haversine_m(g.nodes[ed.v], a) < 1.0 and routing.haversine_m(g.nodes[ed.u], b) < 1.0:
                label = ed.name
                break
        label = label or "unnamed segment mid route"
        print(f"closed           {label}")
        print(f"                 {json.dumps(line['coordinates'])}")
        print(f"segment length   {routing.haversine_m(a, b):.1f} m")
        scorer.set_road_block(label, line, True, "smoke")
        stats2 = routing.graph_stats()
        print(f"edges banned     {stats2['blocked_edges_excluded']}")
        t0 = time.perf_counter()
        after = routing.route(list(origin), list(dest))
        after_ms = (time.perf_counter() - t0) * 1000.0
        print(f"before           {r['distance_m']} m, {len(r['steps'])} steps")
        if after["ok"]:
            delta = after["distance_m"] - r["distance_m"]
            print(f"after            {after['distance_m']} m, {len(after['steps'])} steps")
            print(f"detour cost      {delta:+d} m ({after_ms:.1f} ms)")
            print(f"avoided          {after['blocked_roads_avoided']}")
            print(f"crosses_blockage {after['crosses_blockage']}")
            before_shared = routing.shared_length_m(r["geometry"], line)
            shared = routing.shared_length_m(after["geometry"], line)
            print(f"before drove     {before_shared} m of the closed segment")
            print(f"after  drives    {shared} m of it (tolerance 30 m)")
            if delta == 0 and before_shared > 30.0:
                print("SUSPECT: the route changed nothing yet it used to drive the closure")
        else:
            print(f"after            REFUSED: {after['warning']} ({after_ms:.1f} ms)")
        scorer.set_road_block(label, line, False, "smoke")
        routing.reset()
    else:
        print("no clean route to detour from")

    hr("HTTP surface")
    with TestClient(app) as client:
        t0 = time.perf_counter()
        resp = client.get("/api/route", params={"footprint_id": "", "agency": "fire"})
        http_ms = (time.perf_counter() - t0) * 1000.0
        print(f"GET /api/route (no footprint) -> {resp.status_code} in {http_ms:.1f} ms")
        print(f"   {json.dumps(resp.json())[:400]}")

        row = db.q1("SELECT footprint_id FROM buildings WHERE centroid_json IS NOT NULL LIMIT 1")
        if row is not None:
            t0 = time.perf_counter()
            resp = client.get("/api/route", params={"footprint_id": row["footprint_id"]})
            http_ms = (time.perf_counter() - t0) * 1000.0
            body = resp.json()
            print(
                f"GET /api/route?footprint_id={row['footprint_id']} -> "
                f"{resp.status_code} in {http_ms:.1f} ms"
            )
            print(
                f"   ok={body['ok']} distance_m={body['distance_m']} "
                f"steps={len(body['steps'])} warning={body['warning']}"
            )
        else:
            print("no graded buildings in the db, so no footprint route to call")

        keys = {
            "ok", "geometry", "steps", "distance_m", "eta_min",
            "crosses_blockage", "blocked_roads_avoided", "warning",
        }
        assert set(resp.json()) == keys, f"contract drift: {set(resp.json()) ^ keys}"
        print("contract keys    exactly the frozen section 7 Route shape")

        t0 = time.perf_counter()
        plan = client.get("/api/plan").json()
        plan_ms = (time.perf_counter() - t0) * 1000.0
        print(f"GET /api/plan -> {plan_ms:.0f} ms, drafted_by={plan['drafted_by']}")
        for a in plan["agencies"]:
            has = "route" in a
            geom = (a.get("route") or {}).get("geometry") or {}
            print(
                f"   {a['agency']:<13} steps={len(a['steps']):<3} route key "
                f"{'PRESENT' if has else 'absent ':<8} "
                f"coords={len(geom.get('coordinates') or [])}"
            )
            if a["agency"] == "police":
                assert not has, "police must never carry a route"
            if has:
                assert a["route"]["ok"] is True, "an attached route must be ok:true"
                assert len(geom.get("coordinates") or []) > 1, "attached route needs geometry"
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
