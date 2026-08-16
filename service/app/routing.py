"""B4: offline routing over the local road graph. Dijkstra, with real turns.

WHY this module exists: a straight diagonal from a staging point to a collapsed
building destroys the one claim routing makes, which is that a crew can drive the
line on the paper. Every metre of a returned route here is a real segment out of
the county road table, and when no road-following path exists the answer says so
loudly instead of inventing one.

Five things are load-bearing.

1. Blocked roads are banned at EDGE level, by name AND geometrically. An operator
   who draws a closure across an unnamed alley has closed that alley even though
   there is no name to match, so name matching alone is not enough.
2. The geometric ban measures how far an edge runs ALONG a closure, not whether
   it comes near one. Touching is not driving: the clean half of a street whose
   other half is closed shares one endpoint with the closure, and a proximity
   test would ban it and cut the network in half for no reason.
3. `crosses_blockage` is measured on the LINE WE RETURN, never read back off the
   ban list. Reading the ban list would only restate the input. Measuring the
   output catches whatever the ban missed, and any path that measures as crossing
   comes back ok:false with the geometry attached so an operator can see why.
4. Nothing here ever degrades to a straight line. `ok:false` carries a warning
   naming what stopped it, and the three warning prefixes below let a caller tell
   "no road network loaded" apart from "the closure severed the graph".
5. County road files are frequently NOT noded at junctions: a side street ends in
   the middle of an arterial and the two share no coordinate, so a coordinate
   graph over them is a pile of disconnected sticks and every route fails. We
   split at proper crossings AND weld an endpoint that lands on another feature's
   interior, skipping pairs where either side is a bridge or a tunnel or the two
   carry different `layer` values, because welding an overpass to the road
   beneath it invents a ramp that is not there. `graph_stats()` reports how many
   junctions were inserted, so the number is never hidden.
"""
from __future__ import annotations

# PUBLIC API
# route(origin: list[float], dest: list[float]) -> dict
#     The frozen section 7 Route contract, exactly these keys:
#     {ok, geometry, steps: [{text, dist_m}], distance_m, eta_min,
#      crosses_blockage, blocked_roads_avoided, warning}
#     geometry is a GeoJSON LineString dict or None. Coordinates are [lng, lat].
#     Callers distinguish the three failure modes on the `warning` PREFIX:
#       UNAVAILABLE_PREFIX  "routing unavailable"  no road graph at all
#       NO_PATH_PREFIX      "no route"             graph has no path, or off-network
#       NO_CLEAN_PREFIX     "no clean route"       every path crosses a closure
#     ok:true may still carry a warning (a long snap to the road network).
#     ok:true always means crosses_blockage is False.
# route_for_agency(steps) -> dict
#     One Route chaining an agency's ordered stops. `steps` is the plan's step
#     list ([{centroid, label, ...}]) or a bare list of [lng, lat] pairs. The
#     geometry is the concatenation of the legs, for the map's solid routed line.
# build_graph(force: bool = False) -> Graph      cached; force rebuilds
# reset() -> None                                drop the graph after a dataset swap
# available() -> bool                            a routable graph exists right now
# graph_stats() -> dict
#     {nodes, edges, blocked_edges_excluded, roads_features, junctions_added,
#      planarize_truncated, build_ms, snap_m, buffer_m}
# shared_length_m(line, blocked, *, buffer_m=BUFFER_M, step_m=SAMPLE_M,
#                 align_deg=ALIGN_DEG) -> float
#     Metres of `line` running along `blocked`, within buffer_m and within
#     align_deg of its local bearing. THE geometric test behind the ban and
#     behind crosses_blockage, exposed so a test can assert on geometry.
# Graph, Edge                                    the graph types
# Constants: SNAP_M, BUFFER_M, ALIGN_DEG, CROSS_TOLERANCE_M, NEIGHBOURHOOD_M,
#            MAX_SNAP_M, SNAP_NOTE_M, SAMPLE_M, DEFAULT_SPEED_KMH

import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Iterable, Optional, Sequence

from . import db

log = logging.getLogger("firstlight.routing")

# One degree of latitude, and of longitude at the equator, in metres. Same
# constants datasets.py uses, kept local so neither module reaches into the
# other's privates.
_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LNG = 111320.0
_R_EARTH_M = 6371008.8

# How close two coordinates have to be to become ONE graph node. County exports
# are full of endpoints that miss each other by a metre or two, and every one of
# those is a road that does not connect. 3 m welds them without welding anything
# real: the closest genuinely distinct centrelines, the two halves of a divided
# arterial, are 10 m apart or more.
SNAP_M = float(os.environ.get("FIRSTLIGHT_ROUTE_SNAP_M", "3"))
_SNAP_DEG = SNAP_M / _M_PER_DEG_LAT

# Spatial index cell, about 220 m of latitude. Same bucket size the dataset join
# settled on, for the same reason: county-scale nearest-neighbour without a tree.
CELL_DEG = 0.002

# How close an edge has to run to a blocked segment to be banned. Wide enough to
# absorb an operator drawing a closure freehand on a map, narrow enough not to
# take the parallel street with it.
BUFFER_M = float(os.environ.get("FIRSTLIGHT_ROUTE_BUFFER_M", "15"))
# And how nearly parallel it has to run. A cross street passing through a
# closure's buffer at a junction is crossing the closed road, not driving down
# it, and banning every cross street would sever the county at one closure.
ALIGN_DEG = float(os.environ.get("FIRSTLIGHT_ROUTE_ALIGN_DEG", "40"))
# Metres of a returned line running along a closure before we call it a crossing.
CROSS_TOLERANCE_M = float(os.environ.get("FIRSTLIGHT_ROUTE_CROSS_M", "30"))
# Sampling step for the shared-length measurement. 5 m against a 30 m tolerance
# leaves six samples of headroom, and the whole measurement is a few hundred
# distance calls on a county-sized route.
SAMPLE_M = 5.0
# A closure counts as avoided by THIS route only if it sits this close to the
# line. Otherwise every closure in the county would be listed on every route.
NEIGHBOURHOOD_M = float(os.environ.get("FIRSTLIGHT_ROUTE_NEIGHBOURHOOD_M", "250"))
# Furthest a point may be from the road network and still be routable at all.
MAX_SNAP_M = float(os.environ.get("FIRSTLIGHT_ROUTE_MAX_SNAP_M", "800"))
# Only for the "how far off the network are you" number in a refusal message.
# Never routable: past MAX_SNAP_M the answer is still no.
REPORT_SNAP_M = float(os.environ.get("FIRSTLIGHT_ROUTE_REPORT_SNAP_M", "50000"))
# Past this the snap is worth saying out loud, because the first leg of the drive
# is not on a mapped road.
SNAP_NOTE_M = 25.0

# Free-flow speeds by road class, km/h, for the ETA only. Never used as a Dijkstra
# weight: the plan says metres, so the chosen path is the shortest one and the ETA
# is a separate read over it.
DEFAULT_SPEED_KMH = float(os.environ.get("FIRSTLIGHT_ROUTE_SPEED_KMH", "35"))
_SPEED_KMH = {
    "motorway": 90.0, "motorway_link": 60.0, "trunk": 80.0, "trunk_link": 55.0,
    "primary": 60.0, "primary_link": 45.0, "secondary": 50.0, "secondary_link": 40.0,
    "tertiary": 45.0, "tertiary_link": 35.0, "unclassified": 35.0, "residential": 30.0,
    "living_street": 15.0, "service": 20.0, "track": 20.0, "road": 35.0,
    "pedestrian": 10.0, "footway": 5.0, "path": 5.0, "cycleway": 12.0,
}
_CLASS_KEYS = ("highway", "HIGHWAY", "road_class", "ROADCLASS", "roadtype", "ROADTYPE",
               "FUNCTIONAL", "fclass", "TYPE", "type")
_BRIDGE_KEYS = ("bridge", "BRIDGE", "tunnel", "TUNNEL")
_LAYER_KEYS = ("layer", "LAYER", "level", "LEVEL")

# Junction insertion is bounded so a pathological file cannot hang startup.
PLANARIZE = os.environ.get("FIRSTLIGHT_ROUTE_PLANARIZE", "1") not in ("0", "false", "no")
MAX_PAIR_TESTS = int(os.environ.get("FIRSTLIGHT_ROUTE_MAX_PAIRS", "6000000"))

UNAVAILABLE_PREFIX = "routing unavailable"
NO_PATH_PREFIX = "no route"
NO_CLEAN_PREFIX = "no clean route"

_ARRIVE = "arrive at the destination"

_lock = threading.Lock()
_graph: Optional["Graph"] = None
_graph_token: Optional[tuple] = None
_bans_cache: tuple[Optional[int], str, Optional["_Bans"]] = (None, "", None)
# Never reset. A recycled address can collide, a monotonic counter cannot.
_generation = 0


# ------------------------------------------------------------------- geometry
def haversine_m(a: Sequence[float], b: Sequence[float]) -> float:
    """Great-circle metres between two [lng, lat] pairs."""
    lat1, lat2 = math.radians(a[1]), math.radians(b[1])
    dlat = lat2 - lat1
    dlng = math.radians(b[0] - a[0])
    h = math.sin(dlat * 0.5) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng * 0.5) ** 2
    return 2.0 * _R_EARTH_M * math.asin(min(1.0, math.sqrt(h)))


def _kx(lat: float) -> float:
    return _M_PER_DEG_LNG * math.cos(math.radians(lat))


def _pt_seg_dist(p, a, b) -> float:
    """Distance from p to segment ab, all three in the same planar metre frame."""
    ax, ay = a
    vx, vy = b[0] - ax, b[1] - ay
    wx, wy = p[0] - ax, p[1] - ay
    den = vx * vx + vy * vy
    t = 0.0 if den <= 0.0 else max(0.0, min(1.0, (wx * vx + wy * vy) / den))
    return math.hypot(wx - t * vx, wy - t * vy)


def _straddles(p1, p2, q1, q2) -> bool:
    """True when segments p1p2 and q1q2 properly cross, in one metre frame."""

    def side(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = side(q1, q2, p1), side(q1, q2, p2)
    d3, d4 = side(p1, p2, q1), side(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _seg_seg_m(a1, a2, b1, b2) -> float:
    """Metres between two [lng, lat] segments. Zero when they cross.

    Either pair may be degenerate, so a point-to-segment distance is just this
    with a1 == a2, which is how the sampler uses it.
    """
    lat0 = (a1[1] + a2[1] + b1[1] + b2[1]) * 0.25
    kx, ky = _kx(lat0), _M_PER_DEG_LAT
    ox, oy = a1[0], a1[1]
    p1 = ((a1[0] - ox) * kx, (a1[1] - oy) * ky)
    p2 = ((a2[0] - ox) * kx, (a2[1] - oy) * ky)
    q1 = ((b1[0] - ox) * kx, (b1[1] - oy) * ky)
    q2 = ((b2[0] - ox) * kx, (b2[1] - oy) * ky)
    if _straddles(p1, p2, q1, q2):
        return 0.0
    return min(
        _pt_seg_dist(p1, q1, q2),
        _pt_seg_dist(p2, q1, q2),
        _pt_seg_dist(q1, p1, p2),
        _pt_seg_dist(q2, p1, p2),
    )


def _bearing(a: Sequence[float], b: Sequence[float]) -> float:
    """Compass bearing in degrees, 0 = north, from a to b."""
    lat1, lat2 = math.radians(a[1]), math.radians(b[1])
    dlng = math.radians(b[0] - a[0])
    y = math.sin(dlng) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlng)
    return math.degrees(math.atan2(y, x)) % 360.0


def _bearing_gap(a: float, b: float) -> float:
    """Undirected angle between two bearings, 0 to 90. Direction is irrelevant:
    a closure drawn against the traffic still closes the road."""
    d = abs(a - b) % 180.0
    return 180.0 - d if d > 90.0 else d


_COMPASS = ("north", "north east", "east", "south east", "south", "south west", "west", "north west")


def _compass(bearing: float) -> str:
    return _COMPASS[int((bearing + 22.5) % 360.0 // 45.0)]


def _lines_of(geom: Any) -> list[list[tuple[float, float]]]:
    """Every LineString in a geometry, as lists of (lng, lat).

    Accepts a GeoJSON dict, a bare coordinate list, or None. Anything that is not
    a line contributes nothing, because a closure drawn as a point closes no road
    and a raise here would take the whole route down.
    """
    if not geom:
        return []
    if isinstance(geom, dict):
        kind = str(geom.get("type") or "")
        coords = geom.get("coordinates")
        if kind == "LineString":
            return _lines_of(coords)
        if kind == "MultiLineString":
            out: list[list[tuple[float, float]]] = []
            for part in coords or []:
                out.extend(_lines_of(part))
            return out
        if kind in ("Polygon", "MultiPolygon"):
            # A drawn polygon closure is treated as its boundary rings.
            rings = coords or []
            if kind == "MultiPolygon":
                rings = [r for poly in rings for r in poly]
            out = []
            for ring in rings:
                out.extend(_lines_of(ring))
            return out
        return []
    if isinstance(geom, (list, tuple)):
        pts: list[tuple[float, float]] = []
        for p in geom:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError):
                    return []
        return [pts] if len(pts) >= 2 else []
    return []


def _segments_with_bearing(geom: Any) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    out = []
    for pts in _lines_of(geom):
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a != b:
                out.append((a, b, _bearing(a, b)))
    return out


def shared_length_m(
    line: Any,
    blocked: Any,
    *,
    buffer_m: float = BUFFER_M,
    step_m: float = SAMPLE_M,
    align_deg: float = ALIGN_DEG,
) -> float:
    """Metres of `line` running ALONG `blocked`.

    A sample counts when it sits within `buffer_m` of a blocked segment AND its
    own bearing is within `align_deg` of that segment's. The alignment term is
    what separates driving down a closed road from crossing it at a junction, and
    without it one closure across an arterial would ban every cross street and
    sever the county. Pass align_deg=90 to measure pure proximity.

    This is THE geometric blockage test, and it is public so a test can assert on
    geometry rather than on step text. Accurate to roughly one `step_m`.
    """
    route_lines = _lines_of(line)
    block_segs = _segments_with_bearing(blocked)
    if not route_lines or not block_segs:
        return 0.0
    total = 0.0
    for pts in route_lines:
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            seg_m = haversine_m(a, b)
            if seg_m <= 0.0:
                continue
            here = _bearing(a, b)
            candidates = [
                (q1, q2) for q1, q2, qb in block_segs if _bearing_gap(here, qb) <= align_deg
            ]
            if not candidates:
                continue
            n = max(1, int(math.ceil(seg_m / step_m)))
            piece = seg_m / n
            for k in range(n):
                t = (k + 0.5) / n
                mid = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                for q1, q2 in candidates:
                    if _seg_seg_m(mid, mid, q1, q2) <= buffer_m:
                        total += piece
                        break
    return round(total, 2)


# ---------------------------------------------------------------- road records
@dataclass(frozen=True)
class _Road:
    """What the graph needs off one source feature, resolved once."""

    name: str
    speed_kmh: float
    layer: str
    separated: bool


def _prop(props: dict, keys: Sequence[str]) -> str:
    for k in keys:
        v = props.get(k)
        if v not in (None, ""):
            return str(v)
    return ""


def _road_of(feature: Any) -> _Road:
    props = getattr(feature, "props", None) or {}
    cls = _prop(props, _CLASS_KEYS).strip().lower()
    bridge = _prop(props, _BRIDGE_KEYS).strip().lower()
    return _Road(
        name=str(getattr(feature, "name", "") or "").strip(),
        speed_kmh=_SPEED_KMH.get(cls, DEFAULT_SPEED_KMH),
        layer=_prop(props, _LAYER_KEYS).strip(),
        separated=bridge not in ("", "no", "false", "0"),
    )


def _norm_name(name: str) -> str:
    """Names match exactly after normalizing case and whitespace.

    Deliberately not fuzzy: substring matching on "Ave" would close half a county.
    The geometric ban is what covers a name the operator spelled differently, and
    the console sends the line out of the same road table, so it usually does.
    """
    return " ".join(str(name or "").strip().lower().split())


# ---------------------------------------------------------------------- graph
@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    metres: float
    seconds: float
    name: str
    feature: int


@dataclass
class Graph:
    nodes: list[tuple[float, float]] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    adj: list[list[tuple[int, int]]] = field(default_factory=list)
    node_cells: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    edge_cells: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    roads_features: int = 0
    junctions_added: int = 0
    planarize_truncated: bool = False
    build_ms: int = 0
    # Monotonic build number. The ban cache keys on this and NOT on id(), because
    # CPython recycles addresses: a rebuilt graph can land on the freed one's id
    # and a stale ban set would then index edges that no longer exist.
    generation: int = 0

    def nearest_node(self, point: Sequence[float], max_m: float = MAX_SNAP_M):
        """Nearest graph node to [lng, lat] within max_m, expanding-ring search."""
        if not self.node_cells:
            return None, float("inf")
        lng, lat = float(point[0]), float(point[1])
        radius = CELL_DEG * _M_PER_DEG_LAT
        best: Optional[int] = None
        best_d = float("inf")
        while True:
            span = max(1, int(math.ceil(radius / (CELL_DEG * _M_PER_DEG_LAT))))
            cx, cy = _cell(lng, lat)
            for dx in range(-span, span + 1):
                for dy in range(-span, span + 1):
                    for i in self.node_cells.get((cx + dx, cy + dy), ()):
                        d = haversine_m((lng, lat), self.nodes[i])
                        if d < best_d:
                            best, best_d = i, d
            if best is not None and best_d <= radius:
                return best, best_d
            if radius >= max_m:
                return (best, best_d) if best is not None and best_d <= max_m else (None, best_d)
            radius = min(max_m, radius * 3.0)

    def edges_near(self, a: Sequence[float], b: Sequence[float], pad_m: float) -> set[int]:
        """Edge ids whose cells touch the padded bbox of segment ab."""
        w, e = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
        s, n = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
        kx = max(1.0, _kx((s + n) * 0.5))
        pw, ph = pad_m / kx, pad_m / _M_PER_DEG_LAT
        out: set[int] = set()
        for cell in _cells_for_bbox(w - pw, s - ph, e + pw, n + ph):
            out.update(self.edge_cells.get(cell, ()))
        return out

    def edge_line(self, eid: int) -> list[tuple[float, float]]:
        ed = self.edges[eid]
        return [self.nodes[ed.u], self.nodes[ed.v]]

    def stats(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "roads_features": self.roads_features,
            "junctions_added": self.junctions_added,
            "planarize_truncated": self.planarize_truncated,
            "build_ms": self.build_ms,
        }


def _cell(lng: float, lat: float) -> tuple[int, int]:
    return (int(math.floor(lng / CELL_DEG)), int(math.floor(lat / CELL_DEG)))


def _cells_for_bbox(w: float, s: float, e: float, n: float) -> Iterable[tuple[int, int]]:
    x0, y0 = _cell(w, s)
    x1, y1 = _cell(e, n)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            yield (x, y)


class _NodeIndex:
    """Coordinate to node id, welding anything within SNAP_M.

    Rounding to a grid alone is not enough: two points 1 m apart either side of a
    cell boundary round to different cells, which is exactly the case welding
    exists to fix. So we scan the eight neighbouring cells and take the nearest
    existing node inside the tolerance before minting a new one.
    """

    def __init__(self) -> None:
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.pts: list[tuple[float, float]] = []

    def _key(self, lng: float, lat: float) -> tuple[int, int]:
        return (int(math.floor(lng / _SNAP_DEG)), int(math.floor(lat / _SNAP_DEG)))

    def intern(self, lng: float, lat: float) -> int:
        gx, gy = self._key(lng, lat)
        best: Optional[int] = None
        best_d = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in self.cells.get((gx + dx, gy + dy), ()):
                    d = haversine_m((lng, lat), self.pts[i])
                    if d < best_d:
                        best, best_d = i, d
        if best is not None and best_d <= SNAP_M:
            return best
        i = len(self.pts)
        self.pts.append((lng, lat))
        self.cells.setdefault((gx, gy), []).append(i)
        return i


def _raw_segments(features: Sequence[Any]) -> tuple[list[tuple[int, tuple, tuple]], list[_Road]]:
    segs: list[tuple[int, tuple, tuple]] = []
    roads: list[_Road] = []
    for fi, f in enumerate(features):
        roads.append(_road_of(f))
        for pts in _lines_of(getattr(f, "geom", None)):
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                if a != b:
                    segs.append((fi, a, b))
    return segs, roads


def _cross_point(a, b, c, d) -> Optional[tuple[float, float, float, float]]:
    """Proper crossing of ab and cd as (x, y, t, u), or None.

    Solved in degrees: the transform to metres is affine at this scale, so the
    parameters come out the same and the extra multiplications buy nothing.
    Endpoint touches are rejected, because welding already joined those.
    """
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    den = r[0] * s[1] - r[1] * s[0]
    if den == 0.0:
        return None
    qp = (c[0] - a[0], c[1] - a[1])
    t = (qp[0] * s[1] - qp[1] * s[0]) / den
    u = (qp[0] * r[1] - qp[1] * r[0]) / den
    eps = 1e-9
    if not (eps < t < 1.0 - eps and eps < u < 1.0 - eps):
        return None
    return (a[0] + r[0] * t, a[1] + r[1] * t, t, u)


def _interior_hit(seg_a, seg_b, point) -> Optional[float]:
    """Parameter t where `point` meets the interior of seg_a-seg_b, or None.

    "Interior" means further than SNAP_M from either end, because an endpoint that
    close is already the same node and splitting there would only mint a duplicate.
    """
    lat0 = (seg_a[1] + seg_b[1]) * 0.5
    kx, ky = _kx(lat0), _M_PER_DEG_LAT
    ax, ay = 0.0, 0.0
    bx, by = (seg_b[0] - seg_a[0]) * kx, (seg_b[1] - seg_a[1]) * ky
    px, py = (point[0] - seg_a[0]) * kx, (point[1] - seg_a[1]) * ky
    den = bx * bx + by * by
    if den <= 0.0:
        return None
    t = (px * bx + py * by) / den
    if not (0.0 < t < 1.0):
        return None
    length = math.sqrt(den)
    if t * length <= SNAP_M or (1.0 - t) * length <= SNAP_M:
        return None
    if math.hypot(px - t * bx, py - t * by) > SNAP_M:
        return None
    return t


def _planarize(
    segs: list[tuple[int, tuple, tuple]], roads: Sequence[_Road]
) -> tuple[dict[int, list[tuple[float, tuple[float, float]]]], int, bool]:
    """Split points per segment index, from crossings and from T junctions.

    Grade-separated pairs are skipped: welding a bridge to the road under it
    invents a ramp, which is a worse lie than a missing connection.

    A T junction splits the through segment at the OTHER feature's endpoint
    coordinate, not at the projection of it, so that the node index interns the
    exact same coordinate both features carry and the two really do join. The
    resulting metre-scale kink in the through road is below the noise in the
    source data.
    """
    grid: dict[tuple[int, int], list[int]] = {}
    for i, (_, a, b) in enumerate(segs):
        w, e = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
        s, n = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
        for cell in _cells_for_bbox(w, s, e, n):
            grid.setdefault(cell, []).append(i)

    splits: dict[int, list[tuple[float, tuple[float, float]]]] = {}
    seen: set[tuple[int, int]] = set()
    tests = 0
    truncated = False
    added = 0

    def record(idx: int, t: float, pt: tuple[float, float]) -> None:
        splits.setdefault(idx, []).append((t, pt))

    for ids in grid.values():
        if len(ids) < 2:
            continue
        for pos, i in enumerate(ids):
            fi, a, b = segs[i]
            for j in ids[pos + 1:]:
                fj, c, d = segs[j]
                if fi == fj:
                    continue
                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)
                tests += 1
                if tests > MAX_PAIR_TESTS:
                    truncated = True
                    break
                ri, rj = roads[fi], roads[fj]
                if ri.separated or rj.separated or ri.layer != rj.layer:
                    continue
                hit = _cross_point(a, b, c, d)
                if hit is not None:
                    x, y, t, u = hit
                    record(i, t, (x, y))
                    record(j, u, (x, y))
                    added += 1
                    continue
                # No proper crossing: the common county case is a side street
                # ending part way along an arterial.
                for end in (c, d):
                    t = _interior_hit(a, b, end)
                    if t is not None:
                        record(i, t, end)
                        added += 1
                for end in (a, b):
                    u = _interior_hit(c, d, end)
                    if u is not None:
                        record(j, u, end)
                        added += 1
            if truncated:
                break
        if truncated:
            break
    if truncated:
        log.warning("routing junction insertion truncated at %d pair tests", MAX_PAIR_TESTS)
    return splits, added, truncated


def _roads_token() -> tuple:
    """Cheap identity of the road file on disk: name, size and mtime per candidate.

    WHY: the librarian swaps a dataset atomically and calls datasets.reset_cache(),
    but its invalidation list is a fixed tuple of module names that cannot know
    about this module. Rather than reach into somebody else's file, the graph
    notices the swap itself. A stat per road file is microseconds against a build
    measured in seconds, and it makes a stale graph structurally impossible.
    """
    from . import config

    out = []
    for stem in ("roads", "osm_roads"):
        for ext in (".geojson", ".json", ".csv"):
            p = config.DATASET_DIR / f"{stem}{ext}"
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((p.name, st.st_size, st.st_mtime_ns))
    return tuple(out)


def build_graph(force: bool = False) -> Graph:
    """Build (or return the cached) routing graph from datasets.roads().

    Rebuilds by itself when the road file on disk changes, so a librarian swap is
    visible without a restart and without a caller remembering to call reset().
    """
    global _graph, _graph_token, _generation
    with _lock:
        token = _roads_token()
        if _graph is not None and not force and token == _graph_token:
            return _graph
        started = time.time()
        try:
            from . import datasets

            features = datasets.roads()
        except Exception as exc:  # noqa: BLE001 - a missing dataset is not a crash
            log.warning("road table unavailable: %s", exc)
            features = []

        segs, roads = _raw_segments(features)
        splits: dict[int, list[tuple[float, tuple[float, float]]]] = {}
        added, truncated = 0, False
        if PLANARIZE and segs:
            splits, added, truncated = _planarize(segs, roads)

        index = _NodeIndex()
        _generation += 1
        g = Graph(
            roads_features=len(features),
            junctions_added=added,
            planarize_truncated=truncated,
            generation=_generation,
        )
        for i, (fi, a, b) in enumerate(segs):
            chain = [a]
            for _, pt in sorted(splits.get(i, ()), key=lambda x: x[0]):
                if pt != chain[-1]:
                    chain.append(pt)
            if b != chain[-1]:
                chain.append(b)
            road = roads[fi]
            for k in range(len(chain) - 1):
                p, q = chain[k], chain[k + 1]
                u = index.intern(p[0], p[1])
                v = index.intern(q[0], q[1])
                if u == v:
                    continue
                metres = haversine_m(p, q)
                if metres <= 0.0:
                    continue
                g.edges.append(
                    Edge(
                        u=u,
                        v=v,
                        metres=metres,
                        seconds=metres / (max(1.0, road.speed_kmh) * 1000.0 / 3600.0),
                        name=road.name,
                        feature=fi,
                    )
                )

        g.nodes = index.pts
        g.adj = [[] for _ in g.nodes]
        for eid, ed in enumerate(g.edges):
            g.adj[ed.u].append((ed.v, eid))
            g.adj[ed.v].append((ed.u, eid))
        for i, pt in enumerate(g.nodes):
            g.node_cells.setdefault(_cell(pt[0], pt[1]), []).append(i)
        for eid, ed in enumerate(g.edges):
            a, b = g.nodes[ed.u], g.nodes[ed.v]
            w, e = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
            s, n = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
            for cell in _cells_for_bbox(w, s, e, n):
                g.edge_cells.setdefault(cell, []).append(eid)

        g.build_ms = int(round((time.time() - started) * 1000))
        _graph = g
        _graph_token = token
        log.info(
            "routing graph: %d nodes, %d edges, %d junctions inserted, %d ms",
            len(g.nodes), len(g.edges), added, g.build_ms,
        )
        return g


def reset() -> None:
    """Drop the cached graph and ban set.

    The graph also self-invalidates when the road file changes, so this is for a
    test that rewrites a fixture within one mtime tick, and for any caller that
    wants the rebuild to happen now rather than on the next route.
    """
    global _graph, _graph_token, _bans_cache
    with _lock:
        _graph = None
        _graph_token = None
        _bans_cache = (None, "", None)


def available() -> bool:
    """True when there is a graph with edges in it, so a route is even possible."""
    return bool(build_graph().edges)


def graph_stats() -> dict:
    g = build_graph()
    out = g.stats()
    try:
        out["blocked_edges_excluded"] = len(_bans(g).edge_ids)
    except Exception as exc:  # noqa: BLE001 - no db yet is not a routing failure
        log.debug("blocked edge count unavailable: %s", exc)
        out["blocked_edges_excluded"] = 0
    out["snap_m"] = SNAP_M
    out["buffer_m"] = BUFFER_M
    return out


# ------------------------------------------------------------------ blockages
@dataclass
class _Bans:
    edge_ids: frozenset[int]
    by_name: dict[str, set[int]]
    lines_by_name: dict[str, list[list[tuple[float, float]]]]


def _blocked_rows() -> list[tuple[str, Any, float]]:
    rows = db.q("SELECT road_name, geom_json, ts FROM road_blocks WHERE blocked = 1")
    return [(str(r["road_name"] or ""), r["geom_json"], float(r["ts"] or 0)) for r in rows]


def _fingerprint(rows: Sequence[tuple[str, Any, float]]) -> str:
    payload = json.dumps(sorted((n, str(gj or ""), ts) for n, gj, ts in rows), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runs_along(g: Graph, eid: int, line: list[tuple[float, float]]) -> bool:
    """True when driving edge `eid` means driving along the closure `line`.

    Threshold is the crossing tolerance or half the edge, whichever is smaller, so
    a 20 m alley wholly inside a closure is banned while a 1 km arterial that only
    clips one is not. Merely touching a closure's end, which is what the clean half
    of a half-closed street does, measures zero and stays open.
    """
    ed = g.edges[eid]
    shared = shared_length_m(g.edge_line(eid), line)
    return shared > min(CROSS_TOLERANCE_M, ed.metres * 0.5)


def _bans(g: Graph) -> _Bans:
    """Edges banned by the current closures, cached against the road_blocks table.

    Rebuilding the whole graph per closure would cost seconds on a county
    network, so the graph is built once and the ban set is recomputed only when
    the closure table actually changes.
    """
    global _bans_cache
    rows = _blocked_rows()
    fp = _fingerprint(rows)
    cached_gen, cached_fp, cached = _bans_cache
    if cached is not None and cached_gen == g.generation and cached_fp == fp:
        return cached

    by_name: dict[str, set[int]] = {}
    lines_by_name: dict[str, list[list[tuple[float, float]]]] = {}
    wanted: dict[str, str] = {}
    for name, geom_json, _ in rows:
        wanted[_norm_name(name)] = name
        by_name.setdefault(name, set())
        lines_by_name[name] = _lines_of(db.jload(geom_json, None))

    # By name: an operator who typed the street closed all of it.
    if wanted:
        for eid, ed in enumerate(g.edges):
            display = wanted.get(_norm_name(ed.name))
            if display is not None:
                by_name[display].add(eid)

    # Geometrically: a line drawn across an unnamed alley still closes the alley.
    for name, lines in lines_by_name.items():
        for pts in lines:
            candidates: set[int] = set()
            for i in range(len(pts) - 1):
                candidates |= g.edges_near(pts[i], pts[i + 1], BUFFER_M)
            for eid in candidates:
                if eid in by_name[name]:
                    continue
                if _runs_along(g, eid, pts):
                    by_name[name].add(eid)

    bans = _Bans(
        edge_ids=frozenset().union(*by_name.values()) if by_name else frozenset(),
        by_name=by_name,
        lines_by_name=lines_by_name,
    )
    _bans_cache = (g.generation, fp, bans)
    return bans


def _avoided(g: Graph, bans: _Bans, line: list[tuple[float, float]]) -> list[str]:
    """Closure names actually excluded from the neighbourhood of THIS line.

    A closure qualifies only when it banned at least one edge (so it really did
    change the graph) and either its own geometry or one of those banned edges
    lies within NEIGHBOURHOOD_M of the route. Listing every closure in the county
    on every route would make the field decoration.
    """
    if not bans.by_name or len(line) < 2:
        return []
    segs = [(line[i], line[i + 1]) for i in range(len(line) - 1)]
    out: list[str] = []
    for name, eids in bans.by_name.items():
        if not eids:
            continue
        probes: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for pts in bans.lines_by_name.get(name, ()):
            probes.extend((pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        for eid in eids:
            probes.append(tuple(g.edge_line(eid)))  # type: ignore[arg-type]
        near = False
        for q1, q2 in probes:
            for a, b in segs:
                if _seg_seg_m(a, b, q1, q2) <= NEIGHBOURHOOD_M:
                    near = True
                    break
            if near:
                break
        if near:
            out.append(name)
    return sorted(out)


# --------------------------------------------------------------------- search
def _dijkstra(g: Graph, src: int, dst: int, banned: frozenset[int]):
    """Shortest path in metres. Returns (node ids, edge ids) or None."""
    if src == dst:
        return [src], []
    dist = {src: 0.0}
    prev: dict[int, tuple[int, int]] = {}
    heap: list[tuple[float, int]] = [(0.0, src)]
    done: set[int] = set()
    while heap:
        d, u = heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == dst:
            break
        for v, eid in g.adj[u]:
            if v in done or eid in banned:
                continue
            nd = d + g.edges[eid].metres
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = (u, eid)
                heappush(heap, (nd, v))
    if dst not in dist:
        return None
    nodes = [dst]
    edges: list[int] = []
    cur = dst
    while cur != src:
        pu, eid = prev[cur]
        edges.append(eid)
        nodes.append(pu)
        cur = pu
    nodes.reverse()
    edges.reverse()
    return nodes, edges


# ---------------------------------------------------------------------- steps
def _turn(delta: float) -> tuple[str, str]:
    """(verb, side) for a signed bearing change. Positive delta turns right."""
    mag = abs(delta)
    side = "right" if delta > 0 else "left"
    if mag < 25.0:
        return "continue", ""
    if mag < 60.0:
        return "bear", side
    if mag < 150.0:
        return "turn", side
    return "u-turn", side


def _steps_for(
    g: Graph,
    nodes: list[int],
    edges: list[int],
    lead_m: float,
    lead_name: str,
    tail_m: float,
    arrive_text: Optional[str],
) -> list[dict]:
    """Turn-by-turn from bearing changes between consecutive road legs.

    The text carries the instruction and NOT the distance: the console and the
    printed packet both render `text + " (" + dist_m + " m)"`, so a distance
    inside the text would print twice.
    """
    legs: list[dict] = []
    for i, eid in enumerate(edges):
        ed = g.edges[eid]
        a, b = g.nodes[nodes[i]], g.nodes[nodes[i + 1]]
        heading = _bearing(a, b)
        if legs and _norm_name(legs[-1]["name"]) == _norm_name(ed.name):
            legs[-1]["metres"] += ed.metres
            legs[-1]["end"] = heading
            continue
        legs.append({"name": ed.name, "metres": ed.metres, "start": heading, "end": heading})

    steps: list[dict] = []
    if lead_m >= SNAP_NOTE_M:
        onto = f" to reach {lead_name}" if lead_name else " to reach the road network"
        steps.append({"text": f"leave the start point{onto}", "dist_m": int(round(lead_m))})
    for i, leg in enumerate(legs):
        name = leg["name"]
        if i == 0:
            where = f" on {name}" if name else ""
            text = f"head {_compass(leg['start'])}{where}"
        else:
            delta = ((leg["start"] - legs[i - 1]["end"] + 540.0) % 360.0) - 180.0
            verb, side = _turn(delta)
            onto = f" onto {name}" if name else ""
            if verb == "continue":
                text = f"continue{onto}" if name else "continue straight"
            elif verb == "u-turn":
                text = f"make a u-turn{onto}"
            else:
                text = f"{verb} {side}{onto}"
        steps.append({"text": text, "dist_m": int(round(leg["metres"]))})
    if tail_m >= SNAP_NOTE_M:
        steps.append(
            {"text": "leave the road network for the destination", "dist_m": int(round(tail_m))}
        )
    if arrive_text:
        steps.append({"text": arrive_text, "dist_m": 0})
    return steps


# ---------------------------------------------------------------------- route
def _fail(warning: str, geometry: Any = None, crosses: bool = False, avoided=None) -> dict:
    return {
        "ok": False,
        "geometry": geometry,
        "steps": [],
        "distance_m": 0,
        "eta_min": 0.0,
        "crosses_blockage": crosses,
        "blocked_roads_avoided": list(avoided or []),
        "warning": warning,
    }


def _point(value: Any) -> Optional[tuple[float, float]]:
    if isinstance(value, dict):
        value = value.get("centroid") or value.get("coordinates")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        lng, lat = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if math.isnan(lng) or math.isnan(lat) or (lng == 0.0 and lat == 0.0):
        return None
    return (lng, lat)


def _leg(
    g: Graph,
    bans: _Bans,
    origin: tuple[float, float],
    dest: tuple[float, float],
    *,
    arrive_text: Optional[str] = _ARRIVE,
) -> dict:
    """One origin-to-destination Route. The core both public entry points share."""
    src, src_m = g.nearest_node(origin, MAX_SNAP_M)
    dst, dst_m = g.nearest_node(dest, MAX_SNAP_M)
    if src is None or dst is None:
        which = "start point" if src is None else "destination"
        point = origin if src is None else dest
        far = src_m if src is None else dst_m
        if far == float("inf"):
            # Nothing inside the snap limit, so the limited search never measured
            # anything. Re-ask without the limit purely to report a real number:
            # "1400 m off the network" tells an operator their coordinate is in
            # the next county, where "an unmeasurable distance" tells them nothing.
            _, far = g.nearest_node(point, REPORT_SNAP_M)
        far_txt = f"{int(round(far))} m" if far != float("inf") else f"over {int(REPORT_SNAP_M)} m"
        return _fail(
            f"{NO_PATH_PREFIX}: the {which} is {far_txt} from the nearest mapped road, "
            f"further than the {int(MAX_SNAP_M)} m limit"
        )

    found = _dijkstra(g, src, dst, bans.edge_ids)
    if found is None:
        if bans.edge_ids:
            clear = _dijkstra(g, src, dst, frozenset())
            if clear is not None:
                line = [origin] + [g.nodes[i] for i in clear[0]] + [dest]
                names = _avoided(g, bans, line) or sorted(n for n, e in bans.by_name.items() if e)
                named = ", ".join(names) if names else "the declared closures"
                return _fail(
                    f"{NO_CLEAN_PREFIX}: every path from the start point to the destination "
                    f"crosses {named}, so no route is offered",
                    avoided=names,
                )
        return _fail(
            f"{NO_PATH_PREFIX}: the local road network has no connected path between "
            "these two points"
        )

    nodes, edges = found
    coords: list[tuple[float, float]] = [origin]
    for i in nodes:
        pt = g.nodes[i]
        if pt != coords[-1]:
            coords.append(pt)
    if dest != coords[-1]:
        coords.append(dest)
    if len(coords) < 2:
        coords = [origin, dest]

    on_road_m = sum(g.edges[e].metres for e in edges)
    seconds = sum(g.edges[e].seconds for e in edges)
    per_s = max(1.0, DEFAULT_SPEED_KMH) * 1000.0 / 3600.0
    total_m = on_road_m + src_m + dst_m
    seconds += (src_m + dst_m) / per_s

    geometry = {"type": "LineString", "coordinates": [[c[0], c[1]] for c in coords]}
    avoided = _avoided(g, bans, coords)

    # Measured on the line we are about to hand back, not read off the ban list.
    # If the ban missed something, this is what catches it.
    crossed: list[str] = []
    worst = 0.0
    for name, lines in bans.lines_by_name.items():
        for pts in lines:
            shared = shared_length_m(coords, pts)
            if shared > CROSS_TOLERANCE_M:
                worst = max(worst, shared)
                if name not in crossed:
                    crossed.append(name)
    if crossed:
        return _fail(
            f"{NO_CLEAN_PREFIX}: the only path found runs {int(round(worst))} m along "
            f"{', '.join(sorted(crossed))}, which is closed",
            geometry=geometry,
            crosses=True,
            avoided=avoided,
        )

    lead_name = g.edges[edges[0]].name if edges else ""
    warning = None
    if src_m >= SNAP_NOTE_M or dst_m >= SNAP_NOTE_M:
        warning = (
            f"the start point is {int(round(src_m))} m and the destination "
            f"{int(round(dst_m))} m from the nearest mapped road, so those two legs are "
            "straight lines"
        )
    return {
        "ok": True,
        "geometry": geometry,
        "steps": _steps_for(g, nodes, edges, src_m, lead_name, dst_m, arrive_text),
        "distance_m": int(round(total_m)),
        "eta_min": round(seconds / 60.0, 1),
        "crosses_blockage": False,
        "blocked_roads_avoided": avoided,
        "warning": warning,
    }


def _ready() -> tuple[Optional[Graph], Optional[_Bans], Optional[dict]]:
    """(graph, bans, failure). Exactly one of graph and failure is set."""
    g = build_graph()
    if not g.edges:
        return None, None, _fail(
            f"{UNAVAILABLE_PREFIX}: no local road network is loaded, so no route can be "
            "computed offline"
        )
    try:
        bans = _bans(g)
    except Exception as exc:  # noqa: BLE001 - an unreadable closure table must not
        log.warning("closure table unreadable, routing without bans: %s", exc)
        bans = _Bans(frozenset(), {}, {})
    return g, bans, None


def route(origin: list[float], dest: list[float]) -> dict:
    """Turn-by-turn from origin to dest over the local road graph.

    Returns the frozen section 7 Route contract. Never raises, never returns a
    straight line dressed up as a route.
    """
    o, d = _point(origin), _point(dest)
    if o is None or d is None:
        return _fail(f"{NO_PATH_PREFIX}: a route needs two [lng, lat] points")
    g, bans, failure = _ready()
    if failure is not None:
        return failure
    return _leg(g, bans, o, d)


def route_for_agency(steps: Sequence[Any]) -> dict:
    """One Route through an agency's ordered stops, for the map's solid line.

    `steps` is the plan's step list or a bare list of [lng, lat]. The geometry is
    the concatenation of the legs, in the order the crew works them, so the map
    draws the actual drive rather than a fan of straight connectors. One
    unroutable leg makes the whole chain ok:false and says which leg: a chain
    silently bridged over a severed leg is a lie on paper.
    """
    stops: list[tuple[tuple[float, float], str]] = []
    for s in steps or ():
        pt = _point(s)
        if pt is None:
            continue
        label = ""
        if isinstance(s, dict):
            label = str(s.get("label") or s.get("task") or "")
        stops.append((pt, label))
    if len(stops) < 2:
        return _fail(
            f"{NO_PATH_PREFIX}: an agency route needs at least two stops with coordinates"
        )
    g, bans, failure = _ready()
    if failure is not None:
        return failure

    coords: list[list[float]] = []
    out_steps: list[dict] = []
    distance = 0
    eta = 0.0
    avoided: list[str] = []
    crosses = False
    warnings: list[str] = []
    ok = True
    for i in range(len(stops) - 1):
        (a, _), (b, label) = stops[i], stops[i + 1]
        stop_n = i + 2
        arrive = f"arrive at stop {stop_n}" + (f", {label}" if label else "")
        leg = _leg(g, bans, a, b, arrive_text=arrive)
        for name in leg["blocked_roads_avoided"]:
            if name not in avoided:
                avoided.append(name)
        crosses = crosses or bool(leg["crosses_blockage"])
        if leg["warning"]:
            warnings.append(f"leg {i + 1} to {stop_n}: {leg['warning']}")
        if not leg["ok"]:
            ok = False
            continue
        distance += int(leg["distance_m"])
        eta += float(leg["eta_min"])
        out_steps.extend(leg["steps"])
        for c in (leg["geometry"] or {}).get("coordinates", []):
            if not coords or coords[-1] != c:
                coords.append(c)
    geometry = {"type": "LineString", "coordinates": coords} if len(coords) > 1 else None
    return {
        "ok": bool(ok and geometry),
        "geometry": geometry,
        "steps": out_steps,
        "distance_m": distance,
        "eta_min": round(eta, 1),
        "crosses_blockage": crosses,
        "blocked_roads_avoided": sorted(avoided),
        "warning": "; ".join(warnings) if warnings else None,
    }


__all__ = [
    "ALIGN_DEG",
    "BUFFER_M",
    "CROSS_TOLERANCE_M",
    "DEFAULT_SPEED_KMH",
    "Edge",
    "Graph",
    "MAX_SNAP_M",
    "NEIGHBOURHOOD_M",
    "NO_CLEAN_PREFIX",
    "NO_PATH_PREFIX",
    "SAMPLE_M",
    "SNAP_M",
    "SNAP_NOTE_M",
    "UNAVAILABLE_PREFIX",
    "available",
    "build_graph",
    "graph_stats",
    "haversine_m",
    "reset",
    "route",
    "route_for_agency",
    "shared_length_m",
]
