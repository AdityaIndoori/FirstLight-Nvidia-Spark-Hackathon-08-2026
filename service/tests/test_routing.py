"""B4 routing tests. Every claim is geometric.

WHY no assertion on step text: the strings are a rendering choice and they will be
reworded. The contract this feature exists to keep is that the LINE follows roads
and stays off closures, and that is what these tests measure, using the same
`shared_length_m` the module uses to decide `crosses_blockage`. If the wording of
"turn left" changes, nothing here goes red; if a route starts cutting through a
closed road, everything does.

The fixture is a synthetic grid, deliberately. A real county extract cannot prove
"the detour is longer AND clear" because the answer depends on which streets that
county happens to have. On the grid the correct answer is computable by hand: see
GRID below.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, datasets, db, routing  # noqa: E402

# ------------------------------------------------------------------- fixture
#
# Six streets over Pinellas ground, a 3x3 lattice on a 0.01 degree pitch, which
# is about 1110 m north-south and 985 m east-west at this latitude.
#
#   lat 27.79   A st ---+--------+--------+     (north row)
#                       |        |        |
#   lat 27.78   B st ---+--------+--------+     (middle row)
#                       |        |        |
#   lat 27.77   C st ---+--------+--------+     (south row)
#                      1st      2nd      3rd
#            lng    -82.76   -82.75   -82.74
#
# Note that NO road here shares a coordinate with a crossing road: the avenues
# run as single two-point LineStrings from lat 27.77 to 27.79 and the streets run
# west to east. That is on purpose, because county exports are usually not noded,
# and it means these tests only pass if the planarizer inserts the intersections.
LATS = {"A": 27.79, "B": 27.78, "C": 27.77}
LNGS = {"1st": -82.76, "2nd": -82.75, "3rd": -82.74}
STREET_NAMES = {"A": "A St N", "B": "B St N", "C": "C St N"}
AVE_NAMES = {"1st": "1st Ave", "2nd": "2nd Ave", "3rd": "3rd Ave"}


def _fc(features: list[dict]) -> str:
    return json.dumps({"type": "FeatureCollection", "features": features})


def _line(coords: list[list[float]], **props) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": props,
    }


def _grid_features() -> list[dict]:
    feats = []
    for key, lat in LATS.items():
        feats.append(
            _line(
                [[LNGS["1st"], lat], [LNGS["3rd"], lat]],
                name=STREET_NAMES[key],
                highway="residential",
            )
        )
    for key, lng in LNGS.items():
        feats.append(
            _line(
                [[lng, LATS["C"]], [lng, LATS["A"]]],
                name=AVE_NAMES[key],
                highway="residential",
            )
        )
    return feats


def _node(street: str, ave: str) -> list[float]:
    return [LNGS[ave], LATS[street]]


@pytest.fixture
def grid(tmp_path, monkeypatch):
    """A road table on disk, the loaders and the routing graph both reset."""
    d = tmp_path / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATASET_DIR", d)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "routing.db")
    monkeypatch.setattr(db, "_local", type(db._local)())
    db.init()
    (d / "roads.geojson").write_text(_fc(_grid_features()), encoding="utf-8")
    datasets.reset_cache()
    routing.reset()
    yield d
    datasets.reset_cache()
    routing.reset()


def _write_roads(dirpath: Path, features: list[dict]) -> None:
    (dirpath / "roads.geojson").write_text(_fc(features), encoding="utf-8")
    datasets.reset_cache()
    routing.reset()


def _block(name: str, coords: list[list[float]]) -> None:
    db.run(
        """INSERT INTO road_blocks (road_name, geom_json, blocked, operator, ts)
           VALUES (?,?,1,'tester',?)
           ON CONFLICT(road_name) DO UPDATE SET
             geom_json = excluded.geom_json, blocked = 1, ts = excluded.ts""",
        (name, json.dumps({"type": "LineString", "coordinates": coords}), time.time()),
    )


def _coords(route: dict) -> list[list[float]]:
    geom = route["geometry"] or {}
    return list(geom.get("coordinates") or [])


def _length_m(route: dict) -> float:
    pts = _coords(route)
    return sum(routing.haversine_m(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _has_contract_keys(route: dict) -> None:
    """The frozen section 7 Route keys, plus `offroad`.

    `offroad` carries the snap segments: the stretches between the road network and
    a building, which are real distance a crew must cover but are not on any mapped
    road. It is additive and always present, so a consumer that ignores it still
    reads a valid Route, while the map can draw those stretches as what they are
    instead of implying a driveway the OSM extract never had.
    """
    assert set(route) == {
        "ok",
        "geometry",
        "offroad",
        "steps",
        "distance_m",
        "eta_min",
        "crosses_blockage",
        "blocked_roads_avoided",
        "warning",
    }
    assert isinstance(route["offroad"], list)
    for seg in route["offroad"]:
        # Two points and a real distance, or it is not a drawable segment.
        assert len(seg["coordinates"]) == 2
        assert seg["metres"] >= 0


# --------------------------------------------------------------------- graph
def test_graph_joins_crossing_streets_that_share_no_coordinate(grid):
    """The whole feature rests on this: unnoded county roads must still connect.

    Six two-point LineStrings, no shared coordinate anywhere. A naive coordinate
    graph gives 12 endpoints, 6 disconnected edges and no route at all.

    The arithmetic, so the numbers below are checkable rather than recorded:
    nine lattice junctions. The four CORNERS are endpoint against endpoint, so
    welding within SNAP_M joins those and no split is needed. The other five, the
    four edge midpoints of the lattice plus the centre, are endpoint against
    interior or interior against interior, so five splits get inserted. Nine
    nodes remain, and each of the six streets is cut into two spans, giving
    twelve edges.
    """
    g = routing.build_graph()
    stats = routing.graph_stats()
    assert stats["roads_features"] == 6
    assert stats["nodes"] == 9, "the 3x3 lattice has nine distinct junctions"
    assert stats["edges"] == 12, "six streets, each cut once at the middle junction"
    assert stats["junctions_added"] == 5, "four corners weld, five interior hits split"
    assert routing.available() is True
    assert len(g.adj) == 9
    # Connectivity is the claim, and it is what a route needs: every junction has
    # to be reachable from every other, or the county is in pieces.
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v, _ in g.adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    assert len(seen) == 9, "the lattice is not one connected component"


def test_graph_refuses_to_weld_a_bridge_to_the_road_beneath_it(grid):
    """A crossing where one side is a bridge is not an intersection.

    Welding it would invent a ramp, and a crew told to turn onto a road they
    cannot reach from an overpass is worse off than one told there is no route.
    """
    _write_roads(
        grid,
        [
            _line(
                [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]],
                name="B St N",
                highway="residential",
            ),
            _line(
                [[LNGS["2nd"], LATS["C"]], [LNGS["2nd"], LATS["A"]]],
                name="Overpass Blvd",
                highway="primary",
                bridge="yes",
            ),
        ],
    )
    assert routing.graph_stats()["junctions_added"] == 0
    r = routing.route(_node("B", "1st"), _node("A", "2nd"))
    assert r["ok"] is False
    assert r["warning"].startswith(routing.NO_PATH_PREFIX)


def test_reset_rebuilds_after_a_dataset_swap(grid):
    """The librarian swaps files atomically, so the graph must be droppable."""
    assert routing.graph_stats()["edges"] == 12
    _write_roads(grid, [])
    assert routing.available() is False
    assert routing.graph_stats()["edges"] == 0
    r = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert r["ok"] is False
    assert r["warning"].startswith(routing.UNAVAILABLE_PREFIX)
    _write_roads(grid, _grid_features())
    assert routing.graph_stats()["edges"] == 12


def test_a_new_closure_takes_effect_without_a_reset(grid):
    """Closures are cached against the closure table, so the cache must invalidate.

    Nobody calls reset() when an operator closes a road: they POST /api/roadblock
    and the next Navigate has to see it. A ban set cached past its table would
    route a crew down the road the operator just closed.
    """
    r = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert r["ok"] is True
    assert r["blocked_roads_avoided"] == []

    _block("B St N", [])  # no reset(), exactly as the endpoint behaves
    after = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert after["ok"] is True
    interior = [p for p in _coords(after) if LNGS["1st"] < p[0] < LNGS["3rd"]]
    assert interior and all(abs(p[1] - LATS["B"]) > 1e-6 for p in interior)

    # And re-opening it takes effect the same way, or a cleared road stays shut.
    db.run("UPDATE road_blocks SET blocked = 0 WHERE road_name = 'B St N'")
    reopened = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert reopened["blocked_roads_avoided"] == []
    for p in _coords(reopened):
        assert p[1] == pytest.approx(LATS["B"], abs=1e-6), "the reopened street is the direct run"


def test_a_rebuilt_graph_never_serves_stale_bans(grid):
    """The ban set indexes EDGE IDS, so it must die with the graph that made it.

    Serving a ban set built against a previous graph would ban edge numbers that
    now mean different roads, which is silent and arbitrary. The generation
    counter is what prevents it, and a rebuilt graph must always get fresh bans.
    """
    _block("B St N", [])
    assert routing.route(_node("B", "1st"), _node("B", "3rd"))["ok"] is True
    first = routing.build_graph()
    banned_first = routing.graph_stats()["blocked_edges_excluded"]
    assert banned_first > 0

    # Same closure, a graph where B St N is the ONLY road: the old edge ids are
    # meaningless against it, so a stale ban set would leave some other edge shut.
    _write_roads(
        grid,
        [
            _line(
                [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]],
                name="B St N",
                highway="residential",
            ),
            _line(
                [[LNGS["1st"], LATS["A"]], [LNGS["3rd"], LATS["A"]]],
                name="A St N",
                highway="residential",
            ),
            _line(
                [[LNGS["1st"], LATS["A"]], [LNGS["1st"], LATS["B"]]],
                name="1st Ave",
                highway="residential",
            ),
            _line(
                [[LNGS["3rd"], LATS["A"]], [LNGS["3rd"], LATS["B"]]],
                name="3rd Ave",
                highway="residential",
            ),
        ],
    )
    second = routing.build_graph()
    assert second.generation != first.generation, "a rebuild must bump the generation"

    # B St N is still closed by name, so the detour is over A St N, and every
    # OTHER road must be open: a stale ban would have shut one of them.
    r = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert r["ok"] is True, r["warning"]
    assert any(p[1] == pytest.approx(LATS["A"], abs=1e-6) for p in _coords(r))
    # This lattice is a rectangle, so B St N has no interior junction and stays a
    # single edge: exactly one ban, and it is the closed street, not a leftover id.
    assert routing.graph_stats()["blocked_edges_excluded"] == 1
    assert r["blocked_roads_avoided"] == ["B St N"]
    # The three open roads are all still usable, which is the real claim.
    for a, b in (
        (_node("A", "1st"), _node("A", "3rd")),
        (_node("A", "1st"), _node("B", "1st")),
        (_node("A", "3rd"), _node("B", "3rd")),
    ):
        leg = routing.route(a, b)
        assert leg["ok"] is True, f"a stale ban closed a road it should not have: {leg['warning']}"


def test_a_librarian_swap_is_seen_without_anyone_calling_reset(grid):
    """The librarian swaps the road file and never tells this module.

    librarian._invalidate_caches() walks a FIXED list of (module, function) pairs
    that does not include routing, so nothing calls reset() after a dataset swap.
    The graph therefore has to notice the file changed by itself, or the box keeps
    routing over the county it was started with until someone restarts it.

    Deliberately does NOT call routing.reset(), because that is the whole point.
    """
    assert routing.graph_stats()["edges"] == 12

    # Exactly what a librarian swap looks like from here: the file changes and the
    # dataset loader cache is dropped. No routing call whatsoever.
    (grid / "roads.geojson").write_text(
        _fc(
            [
                _line(
                    [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]],
                    name="B St N",
                    highway="residential",
                )
            ]
        ),
        encoding="utf-8",
    )
    datasets.reset_cache()

    stats = routing.graph_stats()
    assert stats["roads_features"] == 1, "the swapped file was never picked up"
    assert stats["edges"] == 1

    # And a route now reflects the new county, not the old one: A St N is gone, so
    # a trip that needed it has to fail rather than silently use a stale edge.
    assert routing.route(_node("B", "1st"), _node("B", "3rd"))["ok"] is True
    gone = routing.route(_node("A", "1st"), _node("A", "3rd"))
    assert gone["ok"] is False, "A St N no longer exists, so this cannot route"


def test_a_swap_under_a_standing_closure_rebans_against_the_new_graph(grid):
    """A closure plus a swap is where a generation-blind ban cache breaks.

    The closure table has not changed, so its fingerprint is identical, and only
    the graph generation tells the ban cache that its edge ids are now garbage.
    Without that check the stale set is served against a brand new graph and the
    bans land on arbitrary roads.
    """
    _block("B St N", [])
    assert routing.graph_stats()["blocked_edges_excluded"] > 0

    # Same closure table. New road file, written so B St N is the LAST feature and
    # so the ids the old ban set holds now belong to other streets.
    (grid / "roads.geojson").write_text(
        _fc(
            [
                _line(
                    [[LNGS["1st"], LATS["A"]], [LNGS["3rd"], LATS["A"]]],
                    name="A St N",
                    highway="residential",
                ),
                _line(
                    [[LNGS["1st"], LATS["A"]], [LNGS["1st"], LATS["B"]]],
                    name="1st Ave",
                    highway="residential",
                ),
                _line(
                    [[LNGS["3rd"], LATS["A"]], [LNGS["3rd"], LATS["B"]]],
                    name="3rd Ave",
                    highway="residential",
                ),
                _line(
                    [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]],
                    name="B St N",
                    highway="residential",
                ),
            ]
        ),
        encoding="utf-8",
    )
    datasets.reset_cache()

    # The ban must be recomputed against the NEW graph: one edge, and it is B St N.
    assert routing.graph_stats()["blocked_edges_excluded"] == 1
    r = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert r["ok"] is True, r["warning"]
    assert r["blocked_roads_avoided"] == ["B St N"]
    # The detour has to leave B St N entirely and run the far side of the block.
    closed = [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]]
    assert routing.shared_length_m(r["geometry"], closed) < 30.0
    assert any(p[1] == pytest.approx(LATS["A"], abs=1e-6) for p in _coords(r))
    # A St N carries the detour, so it must not have inherited a stale ban.
    assert routing.route(_node("A", "1st"), _node("A", "3rd"))["ok"] is True


# -------------------------------------------------------------- clean routing
def test_straight_through_route_when_nothing_is_blocked(grid):
    """B St N runs straight from 1st to 3rd, so the line must BE B St N."""
    r = routing.route(_node("B", "1st"), _node("B", "3rd"))
    _has_contract_keys(r)
    assert r["ok"] is True
    assert r["crosses_blockage"] is False
    assert r["blocked_roads_avoided"] == []
    pts = _coords(r)
    assert len(pts) >= 2
    for p in pts:
        assert p[1] == pytest.approx(LATS["B"], abs=1e-6), "the line never leaves B St N"
    assert min(p[0] for p in pts) == pytest.approx(LNGS["1st"], abs=1e-6)
    assert max(p[0] for p in pts) == pytest.approx(LNGS["3rd"], abs=1e-6)
    # Two 0.01 degree spans of longitude at 27.78 N, about 1970 m.
    assert r["distance_m"] == pytest.approx(1970, rel=0.02)
    assert r["eta_min"] > 0
    assert r["warning"] is None


def test_route_follows_the_graph_and_never_cuts_the_diagonal(grid):
    """The claim this module exists to make. C-1st to A-3rd is an L, not a hypotenuse."""
    r = routing.route(_node("C", "1st"), _node("A", "3rd"))
    assert r["ok"] is True
    straight = routing.haversine_m(_node("C", "1st"), _node("A", "3rd"))
    assert _length_m(r) > straight * 1.3, "a diagonal shortcut would be near the straight line"
    for p in _coords(r):
        on_street = any(abs(p[1] - lat) < 1e-6 for lat in LATS.values())
        on_ave = any(abs(p[0] - lng) < 1e-6 for lng in LNGS.values())
        assert on_street or on_ave, f"{p} is off the road lattice"


def test_crosses_blockage_is_false_on_every_ok_route(grid):
    """Swept across the lattice with a closure standing, and with none."""
    _block("2nd Ave", [[LNGS["2nd"], LATS["C"]], [LNGS["2nd"], LATS["A"]]])
    checked = 0
    for s_from in LATS:
        for a_from in LNGS:
            for s_to in LATS:
                for a_to in LNGS:
                    r = routing.route(_node(s_from, a_from), _node(s_to, a_to))
                    _has_contract_keys(r)
                    if r["ok"]:
                        assert r["crosses_blockage"] is False
                        checked += 1
    assert checked > 50, "the sweep has to actually route, not just fail everywhere"


def test_snapping_picks_the_nearest_node(grid):
    """An off-road origin snaps to its nearest junction, not to any other.

    Placed 60 m north east of C-1st: the next nearest junction is 985 m away, so
    a wrong snap changes the first coordinate visibly.
    """
    off = [LNGS["1st"] + 0.0006, LATS["C"] + 0.0005]
    r = routing.route(off, _node("A", "3rd"))
    assert r["ok"] is True
    pts = _coords(r)
    assert pts[0] == pytest.approx(off, abs=1e-9), "the line starts at the point asked for"
    assert pts[1] == pytest.approx(_node("C", "1st"), abs=1e-6), "then joins the nearest junction"
    assert r["warning"] and "from the nearest mapped road" in r["warning"]

    # And the snap distance is charged, not hidden: the leg is in the total.
    on_grid = routing.route(_node("C", "1st"), _node("A", "3rd"))
    assert r["distance_m"] > on_grid["distance_m"]


def test_a_point_beyond_the_snap_limit_is_a_named_failure(grid):
    """Off-network is "no route", never "routing unavailable": the graph is fine."""
    r = routing.route([LNGS["1st"] - 0.5, LATS["C"]], _node("A", "3rd"))
    assert r["ok"] is False
    assert r["geometry"] is None
    assert r["warning"].startswith(routing.NO_PATH_PREFIX)
    assert "nearest mapped road" in r["warning"]


# ------------------------------------------------------------------- detours
def test_detour_when_a_mid_path_road_is_blocked_is_longer_and_geometrically_clear(grid):
    """The headline case. B-1st to B-3rd with the middle of B St N closed.

    Blocking only the SEGMENT between 1st and 2nd, drawn as its own line, so the
    route cannot simply avoid a whole named street: it has to leave B St N, work
    around through A or C, and come back.
    """
    before = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert before["ok"] is True
    blocked_line = [[LNGS["1st"], LATS["B"]], [LNGS["2nd"], LATS["B"]]]
    _block("B St N mid block", blocked_line)

    after = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert after["ok"] is True, after["warning"]
    assert after["crosses_blockage"] is False
    assert _length_m(after) > _length_m(before) * 1.5, "a detour has to cost something"

    shared = routing.shared_length_m(after["geometry"], blocked_line)
    assert shared < 30.0, f"the detour shares {shared} m with the closed segment"

    # And the closure the operator declared is named on the answer.
    assert "B St N mid block" in after["blocked_roads_avoided"]

    # Sanity on the fixture itself: the BEFORE route did drive the closed segment,
    # so the assertion above is measuring a real change and not a tautology.
    assert routing.shared_length_m(before["geometry"], blocked_line) > 500.0


def test_blocking_by_name_closes_every_segment_of_that_street(grid):
    """An operator who types the street name has closed all of it.

    Declared with NO geometry at all, so only the name ban can act. B St N gone
    entirely means B-1st to B-3rd must route through A or C, never along lat B.
    """
    _block("B St N", [])
    r = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert r["ok"] is True
    interior = [p for p in _coords(r) if LNGS["1st"] < p[0] < LNGS["3rd"]]
    assert interior, "the route has to go somewhere in between"
    for p in interior:
        assert abs(p[1] - LATS["B"]) > 1e-6, f"{p} is still on the closed street"


def test_a_blockage_on_an_unnamed_segment_still_bans_geometrically(grid):
    """The reason the geometric ban exists.

    The middle rung of the ladder here carries no name at all, so there is nothing
    to match on. Only the drawn line can close it. Without the geometric ban the
    route would take the short middle rung; with it the route must use one of the
    named outer rungs, and the line must clear the drawn closure.
    """
    _write_roads(
        grid,
        [
            _line(
                [[LNGS["1st"], LATS["A"]], [LNGS["1st"], LATS["C"]]],
                name="1st Ave",
                highway="residential",
            ),
            _line(
                [[LNGS["3rd"], LATS["A"]], [LNGS["3rd"], LATS["C"]]],
                name="3rd Ave",
                highway="residential",
            ),
            _line(
                [[LNGS["1st"], LATS["A"]], [LNGS["3rd"], LATS["A"]]],
                name="A St N",
                highway="residential",
            ),
            _line(
                [[LNGS["1st"], LATS["C"]], [LNGS["3rd"], LATS["C"]]],
                name="C St N",
                highway="residential",
            ),
            # The unnamed alley: the short way across, and nameless.
            _line(
                [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]],
                highway="service",
            ),
        ],
    )
    alley = [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]]
    before = routing.route([LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]])
    assert before["ok"] is True
    assert routing.shared_length_m(before["geometry"], alley) > 900.0, "the alley is the short way"

    # An operator drawing across an alley with no name at all: name is a label for
    # the log, the geometry is what closes it.
    _block("unnamed alley behind the strip mall", alley)
    after = routing.route([LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]])
    assert after["ok"] is True, after["warning"]
    assert after["crosses_blockage"] is False
    shared = routing.shared_length_m(after["geometry"], alley)
    assert shared < 30.0, f"the reroute still shares {shared} m with the closed alley"
    assert _length_m(after) > _length_m(before)


def test_a_closure_far_from_the_route_is_not_listed_as_avoided(grid):
    """blocked_roads_avoided describes THIS route, or the field is decoration."""
    _block("A St N", [[LNGS["1st"], LATS["A"]], [LNGS["3rd"], LATS["A"]]])
    r = routing.route(_node("C", "1st"), _node("C", "3rd"))
    assert r["ok"] is True
    assert r["blocked_roads_avoided"] == [], "A St N is 2.2 km from a route along C St N"


# ---------------------------------------------------------------- loud failure
def test_no_route_when_the_blockage_fully_severs_the_graph(grid):
    """Close every north-south link and say so loudly.

    A ladder: two rungs, one rail. Closing the rail leaves the two rungs
    disconnected, so there is no clean path and the honest answer is ok:false with
    a warning naming the closure. Never a straight line.
    """
    _write_roads(
        grid,
        [
            _line(
                [[LNGS["1st"], LATS["A"]], [LNGS["2nd"], LATS["A"]]],
                name="A St N",
                highway="residential",
            ),
            _line(
                [[LNGS["1st"], LATS["C"]], [LNGS["2nd"], LATS["C"]]],
                name="C St N",
                highway="residential",
            ),
            _line(
                [[LNGS["2nd"], LATS["A"]], [LNGS["2nd"], LATS["C"]]],
                name="Only Link Ave",
                highway="residential",
            ),
        ],
    )
    assert routing.route(_node("A", "1st"), _node("C", "1st"))["ok"] is True

    _block("Only Link Ave", [[LNGS["2nd"], LATS["A"]], [LNGS["2nd"], LATS["C"]]])
    r = routing.route(_node("A", "1st"), _node("C", "1st"))
    _has_contract_keys(r)
    assert r["ok"] is False
    assert r["geometry"] is None, "never a straight line when there is no path"
    assert r["steps"] == []
    assert r["distance_m"] == 0
    assert r["warning"].startswith(routing.NO_CLEAN_PREFIX)
    assert "Only Link Ave" in r["warning"], "the operator has to know WHAT stopped it"
    assert "Only Link Ave" in r["blocked_roads_avoided"]


def test_no_path_and_routing_unavailable_are_distinguishable(grid):
    """A caller must tell "we have no roads" from "the roads do not connect"."""
    severed = [
        _line(
            [[LNGS["1st"], LATS["A"]], [LNGS["2nd"], LATS["A"]]],
            name="A St N",
            highway="residential",
        ),
        _line(
            [[LNGS["1st"], LATS["C"]], [LNGS["2nd"], LATS["C"]]],
            name="C St N",
            highway="residential",
        ),
    ]
    _write_roads(grid, severed)
    no_path = routing.route(_node("A", "1st"), _node("C", "1st"))
    assert no_path["ok"] is False
    assert no_path["warning"].startswith(routing.NO_PATH_PREFIX)

    _write_roads(grid, [])
    unavailable = routing.route(_node("A", "1st"), _node("C", "1st"))
    assert unavailable["ok"] is False
    assert unavailable["warning"].startswith(routing.UNAVAILABLE_PREFIX)
    assert no_path["warning"] != unavailable["warning"]


def test_a_path_that_measures_as_crossing_comes_back_not_ok(grid):
    """The output check has to be able to overrule the input ban.

    Contrived on purpose: the ONLY path runs along a segment the operator declared
    closed, and the declaration is placed so the name ban misses it (the drawn
    line has a different name than the street). If `crosses_blockage` were read
    off the ban list this would return ok:true with a route down a closed road.
    The measurement on the returned line is what catches it.
    """
    _write_roads(
        grid,
        [
            _line(
                [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]],
                name="Sole Route Blvd",
                highway="residential",
            )
        ],
    )
    # Wide enough that the ban buffer misses the edge but the route still runs
    # inside the crossing tolerance of the declared line: 0.0009 degrees of
    # latitude is about 100 m, well past BUFFER_M of 15 m.
    off_by_100_m = [
        [LNGS["1st"], LATS["B"] + 0.0009],
        [LNGS["3rd"], LATS["B"] + 0.0009],
    ]
    _block("Sole Route Blvd shoulder", off_by_100_m)
    assert routing.graph_stats()["blocked_edges_excluded"] == 0, "the ban must have missed it"

    r = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert r["ok"] is True, "at 100 m clearance this is genuinely a clean route"
    assert r["crosses_blockage"] is False

    # Now move the declaration onto the road, still under a name that matches no
    # street, and confirm the geometric ban closes it and the answer says so.
    _block("Sole Route Blvd shoulder", [[LNGS["1st"], LATS["B"]], [LNGS["3rd"], LATS["B"]]])
    assert routing.graph_stats()["blocked_edges_excluded"] > 0
    r2 = routing.route(_node("B", "1st"), _node("B", "3rd"))
    assert r2["ok"] is False
    assert r2["warning"].startswith(routing.NO_CLEAN_PREFIX)


def test_the_output_check_catches_a_crossing_the_edge_ban_cannot_see(grid):
    """The reason crosses_blockage is MEASURED and not read off the ban list.

    The ban acts on graph edges. The returned line also contains the two snap
    legs from the origin and the destination to the network, and those are not
    edges, so no edge ban can ever see them. Here the destination sits 300 m off
    the network on the far side of a closure, so the drive from the last junction
    to the door runs straight down the closed road.

    Every edge on the path is clean, the ban list is empty for this route, and the
    only thing standing between an operator and a route down a closed road is the
    measurement on the line we are about to hand back.
    """
    _write_roads(
        grid,
        [
            _line(
                [[LNGS["1st"], LATS["C"]], [LNGS["2nd"], LATS["C"]]],
                name="C St N",
                highway="residential",
            )
        ],
    )
    # 0.0027 degrees of latitude is about 300 m: inside MAX_SNAP_M, so the
    # destination is routable, and the snap leg runs due north.
    door = [LNGS["2nd"], LATS["C"] + 0.0027]
    clean = routing.route(_node("C", "1st"), door)
    assert clean["ok"] is True
    assert clean["crosses_blockage"] is False

    # The closure lies along that snap leg, not along any edge.
    driveway = [[LNGS["2nd"], LATS["C"]], door]
    _block("washed out driveway", driveway)
    assert routing.graph_stats()["blocked_edges_excluded"] == 0, (
        "the closure must ban no edge, or this is not testing the output check"
    )

    r = routing.route(_node("C", "1st"), door)
    assert r["ok"] is False, "a line running down a closure is never ok"
    assert r["crosses_blockage"] is True
    assert r["geometry"] is not None, "the operator needs to SEE why it was refused"
    assert r["warning"].startswith(routing.NO_CLEAN_PREFIX)
    assert "washed out driveway" in r["warning"]
    assert routing.shared_length_m(r["geometry"], driveway) > 30.0


def test_a_route_request_with_no_coordinates_fails_without_raising(grid):
    for bad in (None, [], [0.0, 0.0], ["x", "y"], {"centroid": None}):
        r = routing.route(bad, _node("B", "3rd"))
        _has_contract_keys(r)
        assert r["ok"] is False
        assert r["warning"]


# ------------------------------------------------------------------ multistop
def test_agency_chain_visits_stops_in_order(grid):
    """The map draws one solid line, so the concatenation has to be in crew order.

    Three stops down the west rail: C-1st, B-1st, A-1st. The latitude along the
    returned line must rise monotonically, and each stop must appear on it in
    turn. A reordered chain would show up as a latitude that goes back down.
    """
    stops = [
        {"footprint_id": "b1", "label": "1100 C St", "centroid": _node("C", "1st")},
        {"footprint_id": "b2", "label": "1100 B St", "centroid": _node("B", "1st")},
        {"footprint_id": "b3", "label": "1100 A St", "centroid": _node("A", "1st")},
    ]
    r = routing.route_for_agency(stops)
    _has_contract_keys(r)
    assert r["ok"] is True
    pts = _coords(r)
    lats = [p[1] for p in pts]
    assert lats == sorted(lats), "the chain doubled back, so the stop order was lost"
    for stop in stops:
        assert any(
            routing.haversine_m(p, stop["centroid"]) < 1.0 for p in pts
        ), f"{stop['label']} is not on the drawn line"
    assert pts[0] == pytest.approx(_node("C", "1st"), abs=1e-9)
    assert pts[-1] == pytest.approx(_node("A", "1st"), abs=1e-9)

    # The chain is the sum of its legs, not one long straight.
    leg1 = routing.route(_node("C", "1st"), _node("B", "1st"))
    leg2 = routing.route(_node("B", "1st"), _node("A", "1st"))
    assert r["distance_m"] == leg1["distance_m"] + leg2["distance_m"]
    assert r["steps"], "a printed packet needs the turns for every leg"


def test_agency_chain_reversed_order_produces_a_different_line(grid):
    """Order is data, not decoration: the same three stops reversed must differ."""
    fwd = routing.route_for_agency([_node("C", "1st"), _node("B", "2nd"), _node("A", "3rd")])
    rev = routing.route_for_agency([_node("A", "3rd"), _node("B", "2nd"), _node("C", "1st")])
    assert fwd["ok"] and rev["ok"]
    assert _coords(fwd)[0] != _coords(rev)[0]
    assert _coords(fwd) == list(reversed(_coords(rev))) or _coords(fwd) != _coords(rev)


def test_agency_chain_takes_bare_coordinate_pairs_and_skips_unplaceable_stops(grid):
    """Plan steps carry a null centroid for an operator-added task."""
    r = routing.route_for_agency(
        [
            {"label": "typed by hand", "centroid": None},
            _node("C", "1st"),
            {"label": "placed", "centroid": _node("A", "1st")},
        ]
    )
    assert r["ok"] is True
    assert _coords(r)[0] == pytest.approx(_node("C", "1st"), abs=1e-9)


def test_agency_chain_needs_two_stops(grid):
    r = routing.route_for_agency([_node("C", "1st")])
    _has_contract_keys(r)
    assert r["ok"] is False
    assert r["geometry"] is None
    assert r["warning"].startswith(routing.NO_PATH_PREFIX)


def test_agency_chain_reports_a_severed_leg_rather_than_bridging_it(grid):
    """One unroutable leg must not silently vanish into a straight connector."""
    _write_roads(
        grid,
        [
            _line(
                [[LNGS["1st"], LATS["C"]], [LNGS["1st"], LATS["B"]]],
                name="1st Ave",
                highway="residential",
            ),
            _line(
                [[LNGS["3rd"], LATS["B"]], [LNGS["3rd"], LATS["A"]]],
                name="3rd Ave",
                highway="residential",
            ),
        ],
    )
    r = routing.route_for_agency([_node("C", "1st"), _node("B", "1st"), _node("A", "3rd")])
    assert r["ok"] is False, "a chain with a severed leg is not a usable route"
    assert r["warning"] and routing.NO_PATH_PREFIX in r["warning"]


# ------------------------------------------------------- the geometric measure
def test_shared_length_measures_along_ness_not_mere_proximity(grid):
    """The tool the other assertions lean on, checked against a hand computation.

    "Along" is the whole point. A street CROSSING a closure passes through its
    buffer at the junction, and if that counted as shared length then one closure
    across an arterial would ban every cross street and sever the county at a
    single closure. The alignment term is what makes crossing measure zero.
    """
    on_top = [[LNGS["1st"], LATS["B"]], [LNGS["2nd"], LATS["B"]]]
    span_m = routing.haversine_m(on_top[0], on_top[1])
    assert routing.shared_length_m(on_top, on_top) == pytest.approx(span_m, rel=0.02)

    # 0.001 degrees of latitude is about 110 m, far outside the 15 m buffer.
    parallel = [[LNGS["1st"], LATS["B"] + 0.001], [LNGS["2nd"], LATS["B"] + 0.001]]
    assert routing.shared_length_m(parallel, on_top) == 0.0

    # A perpendicular street is 90 degrees off, so no part of it runs ALONG the
    # closure even though it passes right through the buffer.
    crossing = [[LNGS["2nd"], LATS["C"]], [LNGS["2nd"], LATS["A"]]]
    assert routing.shared_length_m(crossing, on_top) == 0.0
    # Relaxing alignment to 90 degrees turns it back into a pure proximity test,
    # which proves the zero above came from alignment and not from the buffer
    # missing: a few metres of the cross street really are inside it.
    proximity = routing.shared_length_m(crossing, on_top, align_deg=90.0)
    assert 0.0 < proximity < 45.0


def test_a_closure_on_half_a_street_leaves_the_other_half_open(grid):
    """A closure endpoint TOUCHES the clean half, and touching is not driving.

    B St N from 1st to 2nd is closed. The half from 2nd to 3rd shares exactly one
    point with that closure. If the ban were a proximity test it would take the
    clean half too, and a route from B-2nd to B-3rd, which is entirely clear
    road, would fail for no reason.
    """
    _block("B St N west half", [[LNGS["1st"], LATS["B"]], [LNGS["2nd"], LATS["B"]]])
    r = routing.route(_node("B", "2nd"), _node("B", "3rd"))
    assert r["ok"] is True, r["warning"]
    assert r["crosses_blockage"] is False
    for p in _coords(r):
        assert p[1] == pytest.approx(LATS["B"], abs=1e-6), "the open half is a straight run"
        assert p[0] >= LNGS["2nd"] - 1e-9, f"{p} strayed onto the closed half"


def test_shared_length_tolerates_geometry_it_cannot_use(grid):
    """A closure drawn as a point closes no road, and must not raise."""
    for junk in (None, {}, {"type": "Point", "coordinates": [1, 2]}, [[1]], "nonsense"):
        assert routing.shared_length_m(junk, [[0, 0], [1, 1]]) == 0.0
        assert routing.shared_length_m([[0, 0], [1, 1]], junk) == 0.0
