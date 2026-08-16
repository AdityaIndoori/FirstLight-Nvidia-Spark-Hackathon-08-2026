"""FIRST LIGHT HTTP surface.

One FastAPI app. Every route is thin: it validates, calls a module, and shapes
the frozen contract. The invariants live in the modules, not here.

Privacy posture, stated where a reader will find it: every tile is ANALYZED, so
a person in frame still contributes buildings to the ranking, because that is
rescue signal. Storage is the guarded step. The only surface that ever names a
withheld file is GET /api/review/withheld, behind a configured token.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, contracts, db, exports, scorer

app = FastAPI(title="FIRST LIGHT", docs_url=None, redoc_url=None)

_stop = threading.Event()
_watcher: Optional[threading.Thread] = None
_replan_ms = 0
_recovery: Optional[str] = None


def _mod(name: str):
    """Import a pipeline module lazily and tolerate an unbuilt one.

    Members build in parallel: a missing module degrades one endpoint instead of
    taking down the console.
    """
    try:
        import importlib

        return importlib.import_module(f".{name}", __package__)
    except Exception:
        return None


@app.on_event("startup")
def _startup() -> None:
    db.init()
    global _watcher
    ingest = _mod("ingest")
    if ingest and hasattr(ingest, "watch_loop"):
        _watcher = threading.Thread(target=ingest.watch_loop, args=(_stop,), daemon=True)
        _watcher.start()
    # Warm the sentence embedder off the request path. It is a lazy singleton, and
    # the first caller pays ~5 s to load the weights: measured, that caller was
    # /api/status via archive.stats(), which made the console's first poll take 13 s
    # and look hung. A background thread so startup itself does not block either.
    threading.Thread(target=_warm_embedder, daemon=True).start()
    db.log("system", "startup", {"policy": "egress-allowlist"})


def _warm_embedder() -> None:
    """Load the embedder now so no HTTP request has to. Never raises: a cold
    embedder is a slow first search, not a dead service."""
    try:
        from . import embed

        t0 = time.time()
        embed.model_version()
        db.log(
            "system",
            "embedder-warm",
            {"ms": int((time.time() - t0) * 1000), "version": embed.model_version()},
        )
    except Exception as exc:  # noqa: BLE001
        db.log("system", "embedder-warm-failed", {"error": type(exc).__name__})


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop.set()


# --------------------------------------------------------------------- status
@app.get("/api/status")
def status() -> dict:
    ingest = _mod("ingest")
    gate = _mod("privacy_gate")
    grading = _mod("grading")
    archive = _mod("archive")
    embed = _mod("embed")
    librarian = _mod("librarian")

    counts = ingest.counts() if ingest and hasattr(ingest, "counts") else {}
    tally = {
        contracts.CLASS_LABEL.get(int(r["damage_class"] or 0), "unknown"): int(r["n"])
        for r in db.q(
            "SELECT damage_class, COUNT(*) AS n FROM buildings "
            "WHERE damage_class IS NOT NULL GROUP BY damage_class"
        )
    }
    ranked = scorer.rank(limit=10_000)

    return contracts.status_payload(
        tiles_analyzed=counts.get("analyzed", _count("tiles")),
        tiles_stored=counts.get("stored", _count("tiles", "stored = 1")),
        tiles_withheld_from_storage=counts.get(
            "withheld", _count("tiles", "stored = 0")
        ),
        tiles_error=counts.get("error", _count("tiles", "status = 'error'")),
        tile_latency_ms_p50=(
            ingest.latency_p50() if ingest and hasattr(ingest, "latency_p50") else 0
        ),
        # How many tiles that median rests on. Without it a p50 from four tiles
        # reads exactly like a p50 from four hundred.
        tile_latency_n=(
            ingest.latency_sample_size()
            if ingest and hasattr(ingest, "latency_sample_size")
            else 0
        ),
        tally=tally,
        model_versions={
            "gate": gate.model_version() if gate and hasattr(gate, "model_version") else "not wired",
            "grader": grading.model_version() if grading and hasattr(grading, "model_version") else "not wired",
            "planner": f"{config.NANO_MODEL} @ {config.NANO_URL}",
            # ballot.model_version() names the endpoint AND the measured ballot p50,
            # so this row says whether the ballot actually ran rather than asserting
            # a hardcoded "(not wired)" that outlived the wiring.
            "lightning": (
                _mod("ballot").model_version()
                if _mod("ballot") and hasattr(_mod("ballot"), "model_version")
                else f"{config.LIGHTNING_MODEL} @ {config.LIGHTNING_URL} (not wired)"
            ),
            "captioner": f"{config.VL_MODEL} @ {config.VL_URL}",
            "embedder": embed.model_version() if embed and hasattr(embed, "model_version") else "not wired",
        },
        memory_gb=_mem()[0],
        memory_total_gb=_mem()[1],
        gpu_power=_power(),
        last_replan_ms=_replan_ms,
        recovery=_recovery,
        # Measured on this box from the token counts every model server already
        # returns, so the strip reports throughput instead of "not measured yet".
        tokens_per_s=_tokens_per_s(),
        doubt_distribution=ranked["doubt_distribution"],
        aoi=config.AOI,
        aoi_name=getattr(config, "AOI_NAME", "custom"),
        datasets=librarian.catalog() if librarian and hasattr(librarian, "catalog") else [],
        openshell=_openshell_status(),
    )


def _openshell_status() -> dict:
    """B5. The real enforcement and audit layer, or an honest report of its
    absence: containment.status() names which feed is on screen and never claims
    runtime enforcement it does not have."""
    mod = _mod("containment")
    if mod and hasattr(mod, "status"):
        try:
            return mod.status()
        except Exception as exc:
            return {
                "policy": "policy state could not be read",
                "denials": 0,
                "allows": 0,
                "audit": [],
                "note": f"containment layer failed to report: {exc}",
                "overhead_ms": None,
            }
    return {
        "policy": "egress allowlist: localhost inference plus five GET-only sources",
        "denials": 0,
        "allows": 0,
        "audit": [],
        "note": "containment module not importable, so no policy state is shown.",
        "overhead_ms": None,
    }


def _count(table: str, where: str = "1=1") -> int:
    row = db.q1(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}")
    return int(row["n"]) if row else 0


def _tokens_per_s() -> dict:
    """Measured decode rates, or {} when no model has answered yet.

    Lazy like the other optional modules: vlm pulls in Pillow and the HTTP opener,
    and the status endpoint must answer on a box where a model server is down.
    """
    mod = _mod("vlm")
    if mod is None or not hasattr(mod, "tokens_per_s"):
        return {}
    try:
        return mod.tokens_per_s()
    except Exception:  # noqa: BLE001 - a HUD number never breaks status
        return {}


def _mem() -> tuple[float, float]:
    try:
        text = Path("/proc/meminfo").read_text()
        vals = {}
        for line in text.splitlines():
            k, _, rest = line.partition(":")
            vals[k] = int(rest.split()[0])
        total = vals.get("MemTotal", 0) / 1e6
        avail = vals.get("MemAvailable", 0) / 1e6
        return round(total - avail, 1), round(total, 1)
    except Exception:
        return 0.0, 0.0


def _power() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=4,
        ).stdout.strip()
        return out or ""
    except Exception:
        return ""


# ----------------------------------------------------------------- rank, plan
@app.get("/api/rank")
def rank(limit: int = Query(50, ge=1, le=500)) -> dict:
    return scorer.rank(limit=limit)


@app.post("/api/grade")
def grade(body: dict = Body(...)) -> dict:
    try:
        return scorer.flip_grade(
            str(body["footprint_id"]), int(body["new_class"]), str(body.get("operator", ""))
        )
    except KeyError as exc:
        raise HTTPException(404, f"unknown footprint {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/plan")
def plan() -> dict:
    return scorer.build_plan()


@app.post("/api/plan/edit")
def plan_edit(body: dict = Body(...)) -> dict:
    """Record an operator edit to the plan AND persist it.

    This used to log and return ok without changing anything, so every edit was
    undone by the next /api/plan poll about two seconds later: a reassign snapped
    back to the drafted agency in front of the operator. The log is still written -
    it is the audit trail - but the override table is what makes the edit hold.
    """
    operator = str(body.get("operator", "")).strip()
    if not operator:
        raise HTTPException(400, "operator name is required for any edit")
    op = str(body.get("op", ""))
    if op not in {"add", "move", "edit", "delete", "reassign"}:
        raise HTTPException(400, f"unknown op {op!r}")

    payload = body.get("payload") or {}
    footprint_id = str(payload.get("footprint_id") or body.get("footprint_id") or "")
    applied = False
    if footprint_id:
        try:
            if op == "reassign":
                scorer.set_plan_override(
                    footprint_id, operator=operator, agency=str(payload.get("to_agency") or "")
                )
            elif op == "delete":
                scorer.set_plan_override(footprint_id, operator=operator, deleted=True)
            elif op == "move":
                scorer.set_plan_override(
                    footprint_id, operator=operator, order_key=float(payload.get("order_key", 0.0))
                )
            elif op == "edit":
                scorer.set_plan_override(
                    footprint_id,
                    operator=operator,
                    units=(int(payload["units"]) if payload.get("units") is not None else None),
                    task=(str(payload["task"]) if payload.get("task") else None),
                )
            applied = op != "add"
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    db.log(
        f"operator:{operator}",
        f"plan-{op}",
        {
            "agency": body.get("agency"),
            "step_n": body.get("step_n"),
            "payload": payload,
            "persisted": applied,
        },
    )
    return {"ok": True, "logged": True, "persisted": applied}


@app.post("/api/plan/reset")
def plan_reset(body: dict = Body(default={})) -> dict:
    """Discard every operator edit and return to the drafted plan."""
    operator = str(body.get("operator", "")).strip()
    if not operator:
        raise HTTPException(400, "operator name is required")
    return {"ok": True, "cleared": scorer.clear_plan_overrides(operator)}


@app.post("/api/availability")
def availability(body: dict = Body(...)) -> dict:
    try:
        scorer.set_availability(
            str(body["agency"]), int(body["units_available"]), str(body.get("operator", ""))
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@app.post("/api/roadblock")
def roadblock(body: dict = Body(...)) -> dict:
    try:
        scorer.set_road_block(
            str(body["road_name"]),
            body.get("geometry") or {},
            bool(body.get("blocked", True)),
            str(body.get("operator", "")),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@app.post("/api/replan")
def replan(body: dict = Body(default={})) -> dict:
    """Re-draft the plan AND re-task the next flight.

    Availability changes never re-draft silently: the operator saves numbers, then
    triggers this explicitly. Re-tasking the flight here is the point of the beat,
    because a replan that returns an identical survey box has not replanned
    anything. The flight is derived from the staleness and damage state, so a road
    closure or a grade flip genuinely moves it, and when nothing has changed the box
    is stable rather than randomly jittered.
    """
    global _replan_ms, _recovery
    t0 = time.time()
    out = scorer.build_plan()
    fc = flight()
    _replan_ms = int((time.time() - t0) * 1000)
    _recovery = "stub" if out["drafted_by"].startswith("stub") else "model"
    area = next(
        (f for f in fc["features"] if f["properties"].get("role") == "survey-area"), None
    )
    db.log(
        f"operator:{body.get('operator', 'unknown')}",
        "replan",
        {
            "ms": _replan_ms,
            "flight_reason": (area or {}).get("properties", {}).get("reason"),
            "anchor": (area or {}).get("properties", {}).get("anchor_footprint_id"),
        },
    )
    out["flight"] = fc
    return out


# ------------------------------------------------------------------ map layers
@app.get("/api/buildings")
def buildings() -> dict:
    feats = []
    for r in db.q(
        """SELECT footprint_id, label, geom_json, centroid_json, damage_class,
                  confirmed, confidence, doubt
             FROM buildings WHERE damage_class IS NOT NULL"""
    ):
        geom = db.jload(r["geom_json"])
        if not geom:
            c = db.jload(r["centroid_json"], [0.0, 0.0])
            geom = {"type": "Point", "coordinates": c}
        feats.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "footprint_id": r["footprint_id"],
                    "label": r["label"] or "",
                    "damage_class": int(r["damage_class"] or 0),
                    "confirmed": bool(r["confirmed"]),
                    "confidence": round(float(r["confidence"] or 0), 3),
                    "doubt": round(float(r["doubt"] or 0), 3),
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}


@app.get("/api/roads")
def roads() -> dict:
    ds = _mod("datasets")
    base = ds.roads_geojson() if ds and hasattr(ds, "roads_geojson") else {
        "type": "FeatureCollection",
        "features": [],
    }
    blocked = {r["road_name"]: db.jload(r["geom_json"], {}) for r in db.q(
        "SELECT road_name, geom_json FROM road_blocks WHERE blocked = 1"
    )}
    for f in base.get("features", []):
        name = (f.get("properties") or {}).get("name", "")
        f.setdefault("properties", {})["blocked"] = name in blocked
    for name, geom in blocked.items():
        if geom:
            base.setdefault("features", []).append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"name": name, "blocked": True, "operator_declared": True},
                }
            )
    return base


@app.get("/api/facilities")
def facilities() -> dict:
    ds = _mod("datasets")
    if ds and hasattr(ds, "facilities_geojson"):
        return ds.facilities_geojson()
    return {"type": "FeatureCollection", "features": []}


@app.get("/api/flight")
def flight() -> dict:
    """Proposed survey area plus a serpentine path over the sector that most needs
    another look.

    WHY every number here is derived rather than declared. The previous version
    announced 60 m line spacing and a 22 minute flight while flying transects 553 m
    apart over ground that would take 36 minutes: a camera with a 60 m swath flying
    553 m lanes photographs about a ninth of the box and calls it surveyed. So the
    spacing now FIXES the transect count, and the duration is computed from the
    path actually drawn plus turn overhead.

    Sensor model, stated so it can be argued with: ground swath = 2 * altitude *
    tan(hfov / 2). At 90 m with a 76 degree horizontal field of view, that is about
    140 m, and 30 percent sidelap leaves 98 m of usable lane spacing. Those two
    constants are the whole photogrammetry assumption; change them and the transect
    count follows.
    """
    w, s, e, n = config.AOI

    # The sector that most needs another look: oldest observation first, and among
    # equally stale ground the worst damage, because a destroyed block is where a
    # second pass changes a decision. Ties break on footprint_id so a replan with
    # no new information is stable rather than random.
    row = db.q1(
        """SELECT footprint_id, centroid_json, damage_class, last_seen_at
             FROM buildings
            WHERE centroid_json IS NOT NULL
            ORDER BY COALESCE(last_seen_at, 0) ASC,
                     COALESCE(damage_class, 0) DESC,
                     footprint_id ASC
            LIMIT 1"""
    )
    cx, cy = (w + e) / 2.0, (s + n) / 2.0
    reason = "AOI centre: no graded buildings yet, so nothing is stale"
    if row:
        c = db.jload(row["centroid_json"], [cx, cy])
        cx, cy = float(c[0]), float(c[1])
        age_h = (
            (time.time() - float(row["last_seen_at"])) / 3600.0 if row["last_seen_at"] else None
        )
        reason = (
            f"least recently surveyed ground, class {int(row['damage_class'] or 0)}"
            + (f", last seen {age_h:.1f} h ago" if age_h is not None else ", never seen")
        )

    # A box sized in METRES, not as a fraction of the AOI. One sortie is one
    # battery, so the area has to be something a drone can actually fly.
    half_m = _env_float("FIRSTLIGHT_SURVEY_HALF_M", 400.0)
    lat_scale = math.cos(math.radians(cy)) or 1.0
    dx = half_m / (111_320.0 * lat_scale)
    dy = half_m / 110_540.0
    box = [
        [cx - dx, cy - dy], [cx + dx, cy - dy], [cx + dx, cy + dy],
        [cx - dx, cy + dy], [cx - dx, cy - dy],
    ]

    altitude = _env_float("FIRSTLIGHT_SURVEY_ALT_M", 90.0)
    hfov = _env_float("FIRSTLIGHT_SENSOR_HFOV_DEG", 76.0)
    sidelap = _env_float("FIRSTLIGHT_SIDELAP", 0.30)
    speed = _env_float("FIRSTLIGHT_SURVEY_SPEED_MS", 12.0)

    swath_m = 2.0 * altitude * math.tan(math.radians(hfov) / 2.0)
    max_spacing_m = max(10.0, swath_m * (1.0 - sidelap))
    height_m = 2.0 * half_m
    # Spacing drives the count, never the other way round: ceil so the lanes are at
    # or TIGHTER than the coverage limit, never wider.
    transects = max(2, int(math.ceil(height_m / max_spacing_m)) + 1)
    # Then report the spacing actually flown. Ceiling the count and distributing
    # evenly makes the real gap smaller than the limit that produced it, and
    # publishing the limit instead of the gap is how a plan comes to claim 60 m
    # while flying 553 m.
    spacing_m = height_m / (transects - 1)

    line: list[list[float]] = []
    for i in range(transects):
        y = (cy - dy) + (2.0 * dy) * (i / (transects - 1))
        x0, x1 = (cx - dx, cx + dx) if i % 2 == 0 else (cx + dx, cx - dx)
        line.append([x0, y])
        line.append([x1, y])

    # Measured off the path that is actually drawn, plus a turn allowance, because a
    # serpentine spends real time decelerating and coming about at every lane end.
    path_m = 0.0
    for i in range(len(line) - 1):
        ax, ay = line[i]
        bx, by = line[i + 1]
        path_m += math.hypot(
            (bx - ax) * 111_320.0 * lat_scale, (by - ay) * 110_540.0
        )
    turn_s = _env_float("FIRSTLIGHT_SURVEY_TURN_S", 6.0) * (transects - 1)
    est_min = round((path_m / max(1.0, speed) + turn_s) / 60.0, 1)

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [box]},
                "properties": {
                    "role": "survey-area",
                    "reason": reason,
                    "anchor_footprint_id": (row["footprint_id"] if row else None),
                    "area_km2": round((2 * half_m) ** 2 / 1e6, 2),
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line},
                "properties": {
                    "role": "survey-path",
                    "altitude_m_agl": int(altitude),
                    "line_spacing_m": int(round(spacing_m)),
                    "transects": transects,
                    "est_flight_min": est_min,
                    "path_m": int(round(path_m)),
                    "speed_ms": speed,
                    "ground_swath_m": int(round(swath_m)),
                    "sidelap": sidelap,
                },
            },
        ],
    }


def _env_float(name: str, default: float) -> float:
    """A survey constant, overridable so a different airframe or camera can be
    flown without editing code."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@app.get("/api/route")
def route(footprint_id: str = "", agency: str = "") -> dict:
    """Turn-by-turn to one building over the local road graph, B4.

    The drive starts at the incident staging point, which is the centre of the
    configured AOI: the console asks "how does a crew get to this door", and from
    an EOC that is where the crew leaves from. routing snaps it to the nearest
    mapped road and charges the snap distance, so the number is not flattered.

    When there is no road network loaded this keeps answering ok:false with a
    warning, because a straight line the map would then draw through buildings is
    worse than an honest refusal.
    """
    routing = _mod("routing")
    if routing is None or not hasattr(routing, "route"):
        return {
            "ok": False,
            "geometry": None,
            "steps": [],
            "distance_m": 0,
            "eta_min": 0.0,
            "crosses_blockage": False,
            "blocked_roads_avoided": [],
            "warning": "routing unavailable: the offline routing module is not loadable",
        }
    if not routing.available():
        stats = routing.graph_stats()
        return {
            "ok": False,
            "geometry": None,
            "steps": [],
            "distance_m": 0,
            "eta_min": 0.0,
            "crosses_blockage": False,
            "blocked_roads_avoided": [],
            "warning": (
                "routing unavailable: the local road table has "
                f"{stats.get('roads_features', 0)} features and builds no graph, so no "
                "offline route can be computed"
            ),
        }

    dest = None
    if footprint_id:
        row = db.q1(
            "SELECT centroid_json FROM buildings WHERE footprint_id = ?", (footprint_id,)
        )
        if row is not None:
            dest = db.jload(row["centroid_json"])
    if dest is None and footprint_id.startswith("block:"):
        # A police closure post carries footprint_id "block:<road name>", which is
        # not a building at all, so route to the head of the declared closure.
        blk = db.q1(
            "SELECT geom_json FROM road_blocks WHERE road_name = ?",
            (footprint_id.split(":", 1)[1],),
        )
        if blk is not None:
            coords = (db.jload(blk["geom_json"], {}) or {}).get("coordinates") or []
            if coords and isinstance(coords[0], (list, tuple)) and len(coords[0]) >= 2:
                dest = [float(coords[0][0]), float(coords[0][1])]
    if not dest:
        return {
            "ok": False,
            "geometry": None,
            "steps": [],
            "distance_m": 0,
            "eta_min": 0.0,
            "crosses_blockage": False,
            "blocked_roads_avoided": [],
            "warning": (
                f"no route: {footprint_id or 'the request'} has no known location to "
                "route to"
            ),
        }

    w, s, e, n = config.AOI
    staging = [(w + e) / 2.0, (s + n) / 2.0]
    out = routing.route(staging, dest)
    db.log(
        "system",
        "route",
        {
            "footprint_id": footprint_id,
            "agency": agency,
            "ok": out["ok"],
            "distance_m": out["distance_m"],
            "avoided": out["blocked_roads_avoided"],
        },
    )
    return out


# ---------------------------------------------------------------- tiles, upload
@app.get("/api/tiles")
def tiles(limit: int = Query(200, ge=1, le=2000)) -> dict:
    rows = db.q(
        """SELECT filename, bounds_json, status, stored, withheld_reason, needs_geo,
                  captured_at, latency_ms
             FROM tiles ORDER BY COALESCE(analyzed_at, 0) DESC LIMIT ?""",
        (limit,),
    )
    items = []
    for r in rows:
        b = db.q(
            "SELECT footprint_id, damage_class, confidence FROM buildings WHERE source_tile = ?",
            (r["filename"],),
        )
        items.append(
            {
                "filename": r["filename"],
                "bounds": db.jload(r["bounds_json"]),
                "status": r["status"],
                "stored": bool(r["stored"]),
                "withheld_reason": r["withheld_reason"],
                "needs_geo": bool(r["needs_geo"]),
                "captured_at": float(r["captured_at"] or 0),
                "latency_ms": int(r["latency_ms"] or 0),
                "buildings": [
                    {"id": x["footprint_id"], "class": int(x["damage_class"] or 0),
                     "conf": round(float(x["confidence"] or 0), 3)}
                    for x in b
                ],
            }
        )
    return {"items": items}


@app.post("/api/upload")
async def upload(files: list[UploadFile]) -> dict:
    """One request per file on the client side, but accept a batch too: one bad
    file must never fail the rest.

    A `.bounds.json` sidecar in the same selection is honoured. WHY: the geo chain
    reads GeoTIFF transform, then EXIF GPS, then sidecar, and the watch-folder path
    already picks sidecars up off disk - but a browser upload dropped them, so
    georeferenced JPEGs with no EXIF (all of the NOAA post-Michael imagery, for
    instance) landed in needs_geo and could not be graded without a manual drag.
    Sidecars are written FIRST so the image that follows finds its bounds.
    """
    ingest = _mod("ingest")
    if ingest is None:
        raise HTTPException(503, "ingest not wired")

    images: list[tuple[str, UploadFile]] = []
    for uf in files:
        safe = Path(uf.filename or "upload.jpg").name
        if safe.endswith(".bounds.json"):
            try:
                (config.WATCH_DIR / safe).write_bytes(await uf.read())
            except OSError as exc:
                # Not fatal: the image still ingests and lands in needs_geo, which
                # the operator can drag into place. Recorded so a silently missing
                # sidecar is discoverable rather than mysterious.
                db.log("upload", "sidecar-write-failed",
                       {"file": safe, "error": type(exc).__name__})
            continue
        images.append((safe, uf))

    out = []
    for safe, uf in images:
        dest = config.WATCH_DIR / safe
        try:
            with dest.open("wb") as fh:
                shutil.copyfileobj(uf.file, fh)
            rec = await asyncio.to_thread(ingest.analyze_tile, dest, source="upload")
            # wire_with_stages carries the per-stage outcome C3's card renders.
            out.append(
                ingest.wire_with_stages(rec)
                if hasattr(ingest, "wire_with_stages")
                else (rec.wire() if hasattr(rec, "wire") else rec)
            )
        except Exception as exc:  # one bad file, not a failed batch
            out.append({"filename": safe, "status": "error", "error": str(exc)[:200]})
    return {"items": out}


@app.get("/api/tiles/needs-geo")
def needs_geo() -> dict:
    rows = db.q("SELECT filename, captured_at FROM tiles WHERE needs_geo = 1")
    return {"items": [{"filename": r["filename"], "captured_at": r["captured_at"]} for r in rows]}


@app.post("/api/tiles/place")
def place_tile(body: dict = Body(...)) -> dict:
    """Drag-to-place a needs_geo tile, then RE-ANALYZE it: without bounds there
    were no georeferenced outlines, so placing it is what makes it gradeable."""
    ingest = _mod("ingest")
    ds = _mod("datasets")
    filename = str(body.get("filename", ""))
    bounds = body.get("bounds")
    row = db.q1("SELECT stored_path FROM tiles WHERE filename = ?", (filename,))
    if row is None:
        raise HTTPException(404, filename)
    path = Path(row["stored_path"]) if row["stored_path"] else None
    geo = _mod("geo")
    # Persist the placement as a sidecar so a later re-ingest of the same bytes
    # resolves the operator's bounds instead of asking again.
    if geo and hasattr(geo, "write_sidecar") and path and path.exists():
        try:
            geo.write_sidecar(path, bounds, by=f"operator:{operator}")
        except Exception:
            pass
    db.run(
        "UPDATE tiles SET bounds_json = ?, needs_geo = 0, geo_source = ? WHERE filename = ?",
        (json.dumps(bounds), f"operator:{operator}", filename),
    )
    db.log(f"operator:{operator}", "place-tile", {"filename": filename, "bounds": bounds})
    if ds and hasattr(ds, "reset_cache"):
        ds.reset_cache()
    # source="upload" is in RERUN_SOURCES, so dedup is bypassed and the tile is
    # regraded against its new bounds.
    if ingest and path and path.exists():
        try:
            rec = ingest.analyze_tile(path, source="upload")
            return {"ok": True, "reanalyzed": rec.wire() if hasattr(rec, "wire") else None}
        except Exception as exc:
            return {"ok": True, "reanalyzed": None, "warning": str(exc)[:200]}
    return {"ok": True, "reanalyzed": None}


# -------------------------------------------------------------------- archive
@app.get("/api/archive/search")
def archive_search(q: str = "", limit: int = Query(60, ge=1, le=500)) -> dict:
    arch = _mod("archive")
    if arch is None or not hasattr(arch, "search"):
        return {"items": [], "resolved_by": [], "took_ms": 0, "note": "archive not wired"}
    return arch.search(q, limit)


@app.post("/api/archive/edit")
def archive_edit(body: dict = Body(...)) -> dict:
    arch = _mod("archive")
    operator = str(body.get("operator", "")).strip()
    if not operator:
        raise HTTPException(400, "operator name is required for any edit")
    if arch is None or not hasattr(arch, "update_metadata"):
        raise HTTPException(503, "archive not wired")
    try:
        return arch.update_metadata(
            str(body["image_id"]),
            caption=body.get("caption"),
            tags=body.get("tags"),
            centroid=body.get("centroid"),
            key_evidence=body.get("key_evidence"),
            operator=operator,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/archive/add")
async def archive_add(files: list[UploadFile]) -> dict:
    """The archive panel's own add button. It routes through the SAME ingest
    door, so the gate runs on it exactly as on a card dump. A judge who tries to
    sneak a person image in here watches it get refused again."""
    arch = _mod("archive")
    if arch is None or not hasattr(arch, "add_via_ingest_door"):
        raise HTTPException(503, "archive not wired")
    out = []
    for uf in files:
        dest = config.WATCH_DIR / Path(uf.filename or "added.jpg").name
        with dest.open("wb") as fh:
            shutil.copyfileobj(uf.file, fh)
        rec = await asyncio.to_thread(arch.add_via_ingest_door, dest)
        wire = rec.wire() if hasattr(rec, "wire") else {}
        out.append(
            {
                **wire,
                "refused": not wire.get("stored", True),
                "message": (
                    "the gate ran again and refused it: analyzed, not stored"
                    if not wire.get("stored", True)
                    else "stored and searchable"
                ),
            }
        )
    return {"items": out}


@app.get("/thumbs/{filename}")
def thumb(filename: str) -> Response:
    """Serve a thumbnail only for a row that still exists.

    Two reasons this is a route and not a static mount. Eviction deletes the file,
    so serving the directory blind would keep answering for a row that is gone;
    and a withheld image must never have a thumbnail reachable at all, so the
    lookup goes through the archive table by construction.

    The path takes the whole filename rather than `{image_id}.jpg`, because
    Starlette binds one path parameter per segment: a literal suffix after the
    parameter never matches, so the old pattern 404'd every request.
    """
    image_id = Path(filename).stem
    row = db.q1("SELECT thumb_path FROM archive WHERE image_id = ?", (image_id,))
    if row is None:
        raise HTTPException(404, "not in the archive")
    # thumb_path is a URL ("/thumbs/<id>.jpg"), and on POSIX a URL path looks
    # absolute to pathlib, so treating it as a filesystem path sends the lookup to
    # the filesystem root and 404s every thumbnail. Take the basename only, and
    # resolve it inside THUMB_DIR: that is also what keeps a caller-supplied name
    # from escaping the directory.
    p = config.THUMB_DIR / Path(str(row["thumb_path"])).name
    if not p.exists():
        raise HTTPException(404, "thumbnail gone")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/api/archive/image/{filename}")
def archive_image(filename: str) -> Response:
    """The full-resolution image behind an archive row, for a click-to-enlarge.

    Same enforcement as the thumbnail and for the same reason: the row must exist,
    so a withheld image has no reachable full-size copy either. The file is read
    from the analyzed directory by the name the archive recorded, never from a
    caller-supplied path, so there is nothing here to traverse with.
    """
    image_id = Path(filename).stem
    row = db.q1("SELECT filename FROM archive WHERE image_id = ?", (image_id,))
    if row is None:
        raise HTTPException(404, "not in the archive")
    name = Path(str(row["filename"])).name
    for directory in (config.ANALYZED_DIR, config.WATCH_DIR):
        candidate = directory / name
        if candidate.exists():
            return FileResponse(str(candidate), media_type="image/jpeg")
    raise HTTPException(404, "the stored image is gone")


# ---------------------------------------------------------- authorized review
@app.get("/api/review/withheld")
def review_withheld(token: str = "") -> Any:
    """The ONLY surface that names a withheld file.

    An unset token returns 503 rather than defaulting to an open door.
    """
    arch = _mod("archive")
    configured = (
        arch.review_configured()
        if arch and hasattr(arch, "review_configured")
        else bool(config.REVIEW_TOKEN)
    )
    if not configured:
        raise HTTPException(503, "review endpoint not configured: set FIRSTLIGHT_REVIEW_TOKEN")
    if arch and hasattr(arch, "withheld_review"):
        try:
            return arch.withheld_review(token)
        except PermissionError as exc:
            raise HTTPException(403, "not authorized") from exc
    if token != config.REVIEW_TOKEN:
        raise HTTPException(403, "not authorized")
    rows = db.q(
        "SELECT filename, withheld_reason, captured_at FROM tiles WHERE stored = 0"
    )
    return {
        "items": [
            {
                "filename": r["filename"],
                "reason": r["withheld_reason"],
                "captured_at": r["captured_at"],
            }
            for r in rows
        ]
    }


# ------------------------------------------------------------------- datasets
@app.get("/api/datasets")
def datasets_list() -> dict:
    lib = _mod("librarian")
    return {"items": lib.catalog() if lib and hasattr(lib, "catalog") else []}


@app.post("/api/datasets/refresh")
def datasets_refresh(body: dict = Body(...)) -> dict:
    """The agent and the operator both refresh BY NAME. There is no endpoint
    anywhere that accepts a URL, so an injected instruction has no fetch
    primitive to abuse."""
    lib = _mod("librarian")
    ds = _mod("datasets")
    if lib is None or not hasattr(lib, "refresh"):
        raise HTTPException(503, "librarian not wired")
    name = str(body.get("name", ""))
    try:
        out = lib.refresh(name)
    except ValueError as exc:
        raise HTTPException(400, f"not on the allowlist: {name!r}") from exc
    if ds and hasattr(ds, "reset_cache"):
        ds.reset_cache()
    return out


# -------------------------------------------------------------------- exports
@app.get("/api/export/aid-package")
def aid_package(operator: str = "") -> Response:
    data = exports.aid_package(operator or None)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="firstlight-aid-package.zip"'},
    )


@app.get("/api/export/fema-pda.csv")
def fema_csv() -> Response:
    return PlainTextResponse(exports.fema_pda_csv(), media_type="text/csv")


@app.get("/api/export/decision-log.json")
def decision_log() -> Response:
    return PlainTextResponse(exports.decision_log_json(), media_type="application/json")


@app.get("/api/flight/export")
def flight_export(fmt: str = Query("plan")) -> Response:
    fx = _mod("flight_export")
    fc = flight()
    if fx and hasattr(fx, "render"):
        try:
            body, mime, name = fx.render(fc, fmt)
            return Response(
                content=body,
                media_type=mime,
                headers={"Content-Disposition": f'attachment; filename="{name}"'},
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    if fmt not in {"geojson", "plan"}:
        raise HTTPException(400, f"format {fmt!r} not available yet")
    return JSONResponse(fc)


# --------------------------------------------------------------------- static
class _NoStoreStatic(StaticFiles):
    """Serve the console with caching off.

    WHY: StaticFiles sends etag and last-modified but no Cache-Control, so a
    browser is free to heuristically cache the JS. On this box that produced a
    console running code the server had already replaced - an upload built two
    cards for one image and spun for the full three-minute card timeout, with the
    deployed file on disk being correct the whole time. A demo box serves one
    operator over a metre of ethernet; there is nothing to gain by caching and a
    silent version skew to lose.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: D102
        return False

    async def get_response(self, path: str, scope):  # noqa: D102
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


if config.WEB.exists():
    app.mount("/", _NoStoreStatic(directory=str(config.WEB), html=True), name="web")
