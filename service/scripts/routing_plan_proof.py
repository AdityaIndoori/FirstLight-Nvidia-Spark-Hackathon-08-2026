#!/usr/bin/env python3
"""B4 seam 2 proof: /api/plan attaches a real route, and police never get one.

Runs against an ISOLATED db (FIRSTLIGHT_DB is repointed before app import) so it
cannot disturb the box's real rows, and seeds buildings on real Pinellas roads.

The seam under proof: `route` is present ONLY when routing returns ok:true with a
real geometry, ABSENT otherwise, and never present for police.

Usage: python scripts/routing_plan_proof.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["FIRSTLIGHT_DB"] = str(Path(tempfile.gettempdir()) / "firstlight-b4-proof.db")
_db = Path(os.environ["FIRSTLIGHT_DB"])
if _db.exists():
    _db.unlink()

from fastapi.testclient import TestClient  # noqa: E402

from app import db, routing, scorer  # noqa: E402
from app.main import app  # noqa: E402


def seed(fid: str, centroid: list[float], cls: int, *, facility: bool = False) -> None:
    db.run(
        """INSERT INTO buildings
             (footprint_id, label, centroid_json, geom_json, damage_class, confidence,
              graded_by, confirmed, doubt, facility_json, svi, area_m2, last_seen_at)
           VALUES (?,?,?,?,?,?,?,0,?,?,?,?,?)""",
        (
            fid,
            f"{fid} on a real Pinellas road",
            json.dumps(centroid),
            json.dumps({"type": "Point", "coordinates": centroid}),
            cls,
            0.8,
            "nemotron-vl",
            0.2,
            json.dumps({"name": "Bayview Care", "type": "nursing_home", "dist_m": 40})
            if facility
            else None,
            0.7,
            140.0,
            time.time() - 3600.0,
        ),
    )


def main() -> int:
    db.init()
    g = routing.build_graph()
    if not g.edges:
        print("NO ROAD GRAPH: cannot prove the seam without roads")
        return 1
    print(f"graph: {len(g.nodes)} nodes, {len(g.edges)} edges")

    # Pick well-separated real graph nodes so the stops are genuinely routable and
    # the chain has to actually drive between them.
    origin = g.nodes[0]
    ordered = sorted(g.nodes, key=lambda p: routing.haversine_m(origin, p))
    picks = [ordered[len(ordered) // 4], ordered[len(ordered) // 2], ordered[-1]]

    # class 3 -> fire, and one with a facility -> ems. Two fire stops so the fire
    # chain has more than one leg.
    seed("b_fire_1", list(picks[0]), 3)
    seed("b_fire_2", list(picks[1]), 3)
    seed("b_ems_1", list(picks[0]), 2, facility=True)
    seed("b_ems_2", list(picks[2]), 2, facility=True)

    # A police closure post, which build_plan adds from road_blocks.
    ed = g.edges[len(g.edges) // 2]
    line = {"type": "LineString", "coordinates": [list(g.nodes[ed.u]), list(g.nodes[ed.v])]}
    scorer.set_road_block(ed.name or "unnamed closure", line, True, "proof")
    routing.reset()

    with TestClient(app) as client:
        t0 = time.perf_counter()
        plan = client.get("/api/plan").json()
        ms = (time.perf_counter() - t0) * 1000.0
    print(f"GET /api/plan -> {ms:.0f} ms, drafted_by={plan['drafted_by']}\n")

    failures = []
    routed = 0
    for a in plan["agencies"]:
        present = "route" in a
        r = a.get("route")
        coords = ((r or {}).get("geometry") or {}).get("coordinates") or []
        print(
            f"{a['agency']:<13} steps={len(a['steps']):<3} route="
            f"{'PRESENT' if present else 'absent':<8} coords={len(coords):<4}"
            + (f" {r['distance_m']} m, {len(r['steps'])} turns" if present else "")
        )
        if present:
            routed += 1
            if r["ok"] is not True:
                failures.append(f"{a['agency']}: attached a route with ok={r['ok']}")
            if len(coords) < 2:
                failures.append(f"{a['agency']}: attached a route with no geometry")
            if r["crosses_blockage"]:
                failures.append(f"{a['agency']}: attached a route that crosses a blockage")
            if r["blocked_roads_avoided"]:
                print(f"              avoids {r['blocked_roads_avoided']}")
            for s in r["steps"][:4]:
                print(f"              {s['dist_m']:>6} m  {s['text']}")
        if a["agency"] == "police":
            if present:
                failures.append("police carries a route, which the console must never draw")
            if not a["steps"]:
                failures.append("police has no closure post, so the exemption is untested")

    # The key must be genuinely ABSENT, not null: a null reads as "routed, empty".
    raw = json.dumps(plan)
    if '"route": null' in raw or '"route":null' in raw:
        failures.append("a route key is null somewhere, which the map reads as routed-but-empty")

    print()
    if not routed:
        failures.append("no agency got a route at all, so the attachment is unproven")
    for f in failures:
        print(f"FAIL: {f}")
    print("OK" if not failures else "FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
