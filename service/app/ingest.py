"""A1: streaming ingest, the single chokepoint every image enters through.

THE PIVOT lives here, in the ORDER of the stages. Every tile is analyzed -
outlines, grade, join - because a person in frame is rescue signal, and throwing
that tile away would discard exactly the information triage needs. The privacy
gate runs LAST, inside archive.try_store, because it guards STORAGE, not
analysis. There is one storage door and this function is the only thing that
walks up to it.

Stage failures never cascade. A wedged VL endpoint, a missing dataset file or a
crashed archive writer each degrade one stage and leave the rest of the record
intact, because an operator who loses a whole tile to a caption timeout stops
believing the machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config, contracts, db, geo

# PUBLIC API
# ---------------------------------------------------------------------------
# analyze_tile(path: Path | str, *, source: str = "watch") -> contracts.TileRecord
#     The chokepoint. geo -> grade -> join -> buildings upsert -> archive.try_store
#     (the gate) -> tiles row -> decision log -> move the file. Never raises.
#     Returns a TileRecord carrying extra runtime attributes: `.stages`
#     (TileStages), `.dedup` (bool), `.geo_source` (str), `.stored_path` (str|None).
# wire_with_stages(rec: contracts.TileRecord) -> dict
#     rec.wire() plus {"stages": {...}, "stage_error": str|None, "dedup": bool,
#     "geo_source": str}. Use this for the POST /api/upload response so C3's
#     per-image card can name the stage that failed.
# stages_of(rec: contracts.TileRecord) -> TileStages
# watch_loop(stop_event: threading.Event, interval_s: float = 2.0) -> None
#     Blocking poll of config.WATCH_DIR. Run it in a daemon thread.
# scan_once() -> list[contracts.TileRecord]
#     One settled-file sweep, for tests and for a manual "rescan" button.
# latency_p50() -> int          median per-tile end-to-end latency in ms
# latency_percentile(p: float) -> int
# counts() -> dict              status_payload's tiles_* keys, straight from SQL
# sha256_of(path) -> str
# place_result(path, stored) -> Path        ANALYZED_DIR vs WITHHELD_DIR mover
# RERUN_SOURCES: frozenset      sources that bypass dedup so the gate re-runs
# is_in_flight(path) -> bool    True while a tile is mid-pipeline
# in_flight() -> int            how many tiles are mid-pipeline, for the HUD
# reset_watch_state() -> None   forget settle bookkeeping (reseed script, tests)
#
# NOTE for the route layer: `status` is the ANALYSIS outcome only, one of
# "processed", "needs_geo" or "error". Withholding is NOT an analysis outcome
# under the pivot, so it never appears in `status`; read `stored` plus
# `withheld_reason` for the storage decision, and counts() for the HUD.
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".jp2"}

# Adding an image through the archive's own button MUST re-run the gate (gates 3
# and 7), so those sources skip the content-hash dedup and walk the full path.
RERUN_SOURCES = frozenset({"archive-add", "upload", "review", "replay"})

_STAGE_OK = "ok"
_STAGE_SKIPPED = "skipped"

_inited: set[str] = set()
_init_lock = threading.Lock()

# Two callers reach this module: the watch poller and the upload/archive door,
# and both look at the same folder. Grading takes seconds, so without this the
# poller's settle check passes mid-flight and the same frame is graded twice.
# analyze_tile serializes on the path rather than refusing, because refusing
# would mean a raise, and callers are promised a record. Entries are refcounted
# and dropped when idle: every downlink frame is a fresh filename, so a table
# that only grows would leak for the length of a deployment.
_inflight: dict[str, tuple[threading.Lock, list[int]]] = {}
_inflight_lock = threading.Lock()


def _claim_key(path: Path) -> str:
    """Normalized absolute path, computed WITHOUT touching the filesystem.

    The file is moved out of the watch folder before the entry is released, so a
    key derived from `exists()` or `resolve()` would not match on release.
    """
    return os.path.normcase(os.path.abspath(str(path)))


@contextmanager
def _serialized(key: str):
    with _inflight_lock:
        entry = _inflight.get(key)
        if entry is None:
            entry = _inflight[key] = (threading.Lock(), [0])
        lock, waiters = entry
        waiters[0] += 1
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _inflight_lock:
            waiters[0] -= 1
            if waiters[0] <= 0:
                _inflight.pop(key, None)


def is_in_flight(path: Path | str) -> bool:
    """True while a tile sits mid-pipeline, so the poller can leave it alone."""
    key = _claim_key(Path(path))
    with _inflight_lock:
        return key in _inflight


def in_flight() -> int:
    """How many tiles are mid-pipeline right now, for the HUD's backlog line."""
    with _inflight_lock:
        return len(_inflight)


@dataclass
class TileStages:
    """Per-stage outcome, so C3's card can say which stage an image cleared.

    Console mapping: `grade` is card stage 1 (outlines + grades), `join` is
    stage 2 (indexing), `store` is stage 3 (storage decision).
    """

    geo: str = _STAGE_OK
    grade: str = _STAGE_OK
    join: str = _STAGE_OK
    store: str = _STAGE_OK

    def wire(self) -> dict:
        return asdict(self)

    def failed(self) -> Optional[str]:
        for name, value in asdict(self).items():
            if value not in (_STAGE_OK, _STAGE_SKIPPED):
                return f"{name}: {value}"
        return None


def _ensure_db() -> None:
    """Idempotent per database file, so a test that repoints config.DB_PATH gets
    its schema without every call paying for the script."""
    key = str(config.DB_PATH)
    if key in _inited:
        return
    with _init_lock:
        db.init()
        _inited.add(key)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stages_of(rec: contracts.TileRecord) -> TileStages:
    return getattr(rec, "stages", TileStages())


def wire_with_stages(rec: contracts.TileRecord) -> dict:
    stages = stages_of(rec)
    payload = rec.wire()
    payload.update(
        {
            "stages": stages.wire(),
            "stage_error": stages.failed(),
            "dedup": bool(getattr(rec, "dedup", False)),
            "geo_source": getattr(rec, "geo_source", geo.SOURCE_NONE),
        }
    )
    return payload


# ------------------------------------------------------------------ placement
def _unique(dest_dir: Path, name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    for i in range(1, 10_000):
        cand = dest_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
    return dest_dir / f"{stem}_{int(time.time() * 1000)}{suffix}"


def place_result(path: Path, stored: bool) -> Path:
    """Move the image out of the watch folder once the decision is made.

    Sidecars travel with the image, otherwise an operator-placed bbox stops
    resolving the moment the file lands in its final directory.
    """
    dest_dir = config.ANALYZED_DIR if stored else config.WITHHELD_DIR
    path = Path(path)
    try:
        if path.parent.resolve() == dest_dir.resolve():
            return path
        target = _unique(dest_dir, path.name)
        shutil.move(str(path), str(target))
        for sidecar in geo.sidecar_paths(path):
            if sidecar.is_file():
                shutil.move(str(sidecar), str(target.with_suffix(".bounds.json")))
        return target
    except OSError:
        # A locked file is a placement problem, not an analysis problem. The
        # record already exists; the watcher's `seen` set stops a reprocess.
        return path


# ------------------------------------------------------------------- db rows
def _facility_json(building: Any) -> Optional[str]:
    fac = getattr(building, "facility_near", None)
    if fac is None:
        return None
    wire = fac.wire() if hasattr(fac, "wire") else fac
    return _dumps(wire) if isinstance(wire, dict) else None


def _dumps(value: Any) -> Optional[str]:
    """JSON or nothing. Geometry may arrive as a dict or as a shapely object, and
    a value we cannot serialize is a missing column, never a lost tile."""
    if value is None:
        return None
    if hasattr(value, "__geo_interface__"):
        value = value.__geo_interface__
    try:
        return json.dumps(value, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return None


def _upsert_buildings(buildings: Iterable[Any], tile_name: str, now: float) -> int:
    """Write the graded buildings, and never overwrite an operator's call.

    A re-flight of the same block must refresh geometry and the vulnerability
    join, but a confirmed grade belongs to the named human who confirmed it.
    When the grade IS refreshed, the stale ballot is cleared: a doubt figure
    computed against the previous class would silently misprice the new one.
    """
    written = 0
    for b in buildings:
        fid = getattr(b, "footprint_id", None)
        if not fid:
            continue
        centroid = _dumps(getattr(b, "centroid", None))
        geom = _dumps(getattr(b, "geom", None))
        label = getattr(b, "label", None)
        facility = _facility_json(b)
        svi = getattr(b, "svi", None)
        area = getattr(b, "area_m2", None)
        row = db.q1("SELECT confirmed FROM buildings WHERE footprint_id = ?", (fid,))
        if row is None:
            db.run(
                """INSERT INTO buildings
                     (footprint_id, label, centroid_json, geom_json, damage_class,
                      confidence, graded_by, confirmed, facility_json, svi, area_m2,
                      last_seen_at, source_tile)
                   VALUES (?,?,?,?,?,?,?,0,?,?,?,?,?)""",
                (
                    fid,
                    label,
                    centroid,
                    geom,
                    int(getattr(b, "cls", 0) or 0),
                    float(getattr(b, "conf", 0.0) or 0.0),
                    getattr(b, "graded_by", "unknown"),
                    facility,
                    svi,
                    area,
                    now,
                    tile_name,
                ),
            )
        elif int(row["confirmed"] or 0):
            db.run(
                """UPDATE buildings
                      SET label = COALESCE(?, label),
                          centroid_json = COALESCE(?, centroid_json),
                          geom_json = COALESCE(?, geom_json),
                          facility_json = COALESCE(?, facility_json),
                          svi = COALESCE(?, svi),
                          area_m2 = COALESCE(?, area_m2),
                          last_seen_at = ?,
                          source_tile = ?
                    WHERE footprint_id = ?""",
                (label, centroid, geom, facility, svi, area, now, tile_name, fid),
            )
        else:
            db.run(
                """UPDATE buildings
                      SET label = COALESCE(?, label),
                          centroid_json = COALESCE(?, centroid_json),
                          geom_json = COALESCE(?, geom_json),
                          damage_class = ?,
                          confidence = ?,
                          graded_by = ?,
                          facility_json = COALESCE(?, facility_json),
                          svi = COALESCE(?, svi),
                          area_m2 = COALESCE(?, area_m2),
                          doubt = NULL,
                          votes_json = NULL,
                          vote_agreement = NULL,
                          last_seen_at = ?,
                          source_tile = ?
                    WHERE footprint_id = ?""",
                (
                    label,
                    centroid,
                    geom,
                    int(getattr(b, "cls", 0) or 0),
                    float(getattr(b, "conf", 0.0) or 0.0),
                    getattr(b, "graded_by", "unknown"),
                    facility,
                    svi,
                    area,
                    now,
                    tile_name,
                ),
            )
        written += 1
    return written


def _write_tile_row(
    rec: contracts.TileRecord,
    sha: str,
    geo_source: str,
    stored_path: Optional[str],
    now: float,
) -> None:
    db.run(
        """INSERT INTO tiles
             (filename, sha256, status, stored, withheld_reason, needs_geo, bounds_json,
              captured_at, analyzed_at, latency_ms, geo_source, stored_path)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(filename) DO UPDATE SET
              sha256 = excluded.sha256,
              status = excluded.status,
              stored = excluded.stored,
              withheld_reason = excluded.withheld_reason,
              needs_geo = excluded.needs_geo,
              bounds_json = excluded.bounds_json,
              analyzed_at = excluded.analyzed_at,
              latency_ms = excluded.latency_ms,
              geo_source = excluded.geo_source,
              stored_path = excluded.stored_path""",
        (
            rec.filename,
            sha,
            rec.status,
            1 if rec.stored else 0,
            rec.withheld_reason,
            1 if rec.needs_geo else 0,
            _dumps(rec.bounds),
            rec.captured_at,
            now,
            int(rec.latency_ms),
            geo_source,
            stored_path,
        ),
    )


def _record_from_row(row: Any) -> contracts.TileRecord:
    buildings = [
        contracts.Building(
            id=r["footprint_id"],
            cls=int(r["damage_class"] or 0),
            conf=float(r["confidence"] or 0.0),
        )
        for r in db.q(
            "SELECT footprint_id, damage_class, confidence FROM buildings WHERE source_tile = ?",
            (row["filename"],),
        )
    ]
    rec = contracts.TileRecord(
        filename=row["filename"],
        bounds=db.jload(row["bounds_json"]),
        status=row["status"],
        captured_at=float(row["captured_at"] or 0.0),
        latency_ms=int(row["latency_ms"] or 0),
        buildings=buildings,
        stored=bool(row["stored"]),
        withheld_reason=row["withheld_reason"],
        needs_geo=bool(row["needs_geo"]),
    )
    rec.stages = TileStages()
    rec.dedup = True
    rec.geo_source = row["geo_source"] or geo.SOURCE_NONE
    rec.stored_path = row["stored_path"]
    return rec


# ------------------------------------------------------------- the chokepoint
def analyze_tile(path: Path | str, *, source: str = "watch") -> contracts.TileRecord:
    """Analyze one tile end to end and return its record. Never raises.

    Order is the contract: geo (before the file moves, so sidecars resolve),
    outlines and grade, vulnerability join, buildings upsert, then the archive
    writer LAST because that is where the privacy gate lives.

    One tile is analyzed once at a time. The upload door and the watch poller both
    reach this function with the same path, and grading is slow enough that they
    would otherwise overlap on it.
    """
    t0 = time.perf_counter()
    p = Path(path)
    with _serialized(_claim_key(p)):
        # The second arrival re-reads state inside the lock, so it sees the first
        # arrival's tiles row and takes the dedup exit instead of regrading.
        return _analyze(p, source, t0)


def _analyze(p: Path, source: str, t0: float) -> contracts.TileRecord:
    stages = TileStages()
    _ensure_db()

    try:
        captured_at = p.stat().st_mtime
    except OSError:
        captured_at = time.time()

    # Dedup on content, not on name: a card dump and a downlink frame of the
    # same scene arrive with different filenames.
    sha = ""
    try:
        sha = sha256_of(p)
    except OSError as exc:
        stages.geo = f"unreadable file ({type(exc).__name__})"
    if sha and source not in RERUN_SOURCES:
        prior = db.q1("SELECT * FROM tiles WHERE sha256 = ?", (sha,))
        if prior is not None:
            rec = _record_from_row(prior)
            # Identical bytes get the identical storage decision, so a duplicate
            # of a withheld tile never lands beside the stored ones.
            place_result(p, bool(prior["stored"]))
            db.log("ingest", "tile-duplicate", {"source": source, "of_status": rec.status})
            return rec

    # ---- stage: geo. Read before any move so a sidecar beside the file counts.
    result = geo.extract(p)
    if result.needs_geo:
        stages.geo = result.detail or "no location"

    # ---- stage: grade. The only stage whose failure makes the tile an error.
    graded: list[Any] = []
    grade_failed = False
    try:
        from . import grading

        graded = list(grading.outline_and_grade(p, result.bounds) or [])
        if not graded and result.bounds is None:
            stages.grade = _STAGE_SKIPPED
    except Exception as exc:  # noqa: BLE001 - a dead grader must not lose the tile
        grade_failed = True
        stages.grade = f"{type(exc).__name__}: {exc}"[:160]

    # ---- stage: join. Address labels, facilities, SVI. Never fatal.
    if graded:
        try:
            from . import datasets

            datasets.join(graded, result.bounds)
        except Exception as exc:  # noqa: BLE001
            stages.join = f"{type(exc).__name__}: {exc}"[:160]
    else:
        stages.join = _STAGE_SKIPPED

    now = time.time()
    try:
        _upsert_buildings(graded, p.name, now)
    except Exception as exc:  # noqa: BLE001 - the tile row still gets written
        stages.join = f"buildings write failed: {type(exc).__name__}"

    status: contracts.TileStatus = "processed"
    if grade_failed:
        status = "error"
    elif result.needs_geo:
        status = "needs_geo"

    rec = contracts.TileRecord(
        filename=p.name,
        bounds=result.bounds,
        status=status,
        captured_at=float(captured_at),
        latency_ms=0,
        buildings=[
            contracts.Building(
                id=getattr(b, "footprint_id", ""),
                cls=int(getattr(b, "cls", 0) or 0),
                conf=float(getattr(b, "conf", 0.0) or 0.0),
            )
            for b in graded
            if getattr(b, "footprint_id", None)
        ],
        stored=False,
        withheld_reason=None,
        needs_geo=bool(result.needs_geo),
    )

    # ---- B3 ballot (BallotLightning) ----------------------------------------
    # The ballot runs after the join, so the vote sees the facility and area
    # context, and before the storage decision, so a tile that will be WITHHELD
    # still contributes its doubt: analysis is not what the gate guards.
    #
    # Bounded twice, because the plan's per-tile budget is 10 s and VL grading
    # has already spent most of it. When more buildings arrived than the budget
    # can vote on, the least certain go first (caption contradicting the grade,
    # then rising grader confidence) and the rest keep their grader-confidence
    # doubt. Which of the two happened is recorded, never inferred.
    #
    # A ballot failure is not a tile failure: `stages` is untouched on the happy
    # path and the tile keeps every grade it earned either way.
    if graded:
        try:
            from . import ballot

            cap = ballot.tile_max_buildings()
            results = ballot.vote_batch(
                graded,
                budget_s=ballot.tile_budget_s(),
                uncertain_only=cap if len(graded) > cap else None,
            )
            ballot.persist_all(results)
            spread = ballot.spread_check(results)
            rec.ballot = {
                "voted": spread["total"],
                "of_buildings": len(graded),
                "selection": ballot.last_sweep()["selection"],
                "model": spread["model"],
                "stub": spread["stub"],
                "at_floor": spread["at_floor"],
                "contested": spread["contested"],
                "mean_doubt": spread["mean_doubt"],
                "ms": ballot.last_sweep()["wall_ms"],
            }
            db.log("ballot", "tile-ballot", rec.ballot)
        except Exception as exc:  # noqa: BLE001 - no ballot is a missing column, not a lost tile
            rec.ballot = {"error": f"{type(exc).__name__}: {exc}"[:160]}
            db.log("ballot", "tile-ballot-failed", {"error": rec.ballot["error"]})
    # ---- end B3 ballot -------------------------------------------------------

    # ---- stage: store. The gate runs inside try_store. Fail CLOSED: if the
    # writer itself breaks we withhold, because "stored" must never be the
    # outcome of an unproven check.
    try:
        from . import archive

        stored, reason = archive.try_store(p, rec, graded)
        rec.stored = bool(stored)
        rec.withheld_reason = None if stored else (reason or "withheld")
    except Exception as exc:  # noqa: BLE001
        rec.stored = False
        rec.withheld_reason = "storage error"
        stages.store = f"{type(exc).__name__}: {exc}"[:160]

    final = place_result(p, rec.stored)
    rec.latency_ms = int((time.perf_counter() - t0) * 1000)
    rec.stages = stages
    rec.dedup = False
    rec.geo_source = result.source
    rec.stored_path = str(final) if rec.stored else None

    try:
        _write_tile_row(rec, sha, result.source, rec.stored_path, now)
    except Exception:  # noqa: BLE001 - a locked DB must not raise into the watcher
        pass

    # The log is exported unauthenticated, so a withheld tile contributes no
    # filename and no gate reason. Absence is all the log is allowed to say.
    db.log(
        "ingest",
        "tile-analyzed",
        {
            "source": source,
            "status": rec.status,
            "stored": rec.stored,
            "buildings": len(rec.buildings),
            "needs_geo": rec.needs_geo,
            "geo_source": result.source,
            "latency_ms": rec.latency_ms,
            "stage_error": stages.failed(),
            **({"filename": rec.filename} if rec.stored else {}),
        },
    )
    return rec


# ----------------------------------------------------------------- the watcher
@dataclass
class _WatchState:
    """Settle bookkeeping. `seen` is only added to after a file is dealt with,
    so a transient failure retries instead of being silently abandoned."""

    seen: set[str] = field(default_factory=set)
    sizes: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)


MAX_ATTEMPTS = 3
_state = _WatchState()


def _candidates() -> list[Path]:
    try:
        entries = sorted(config.WATCH_DIR.iterdir())
    except OSError:
        return []
    out = []
    for p in entries:
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue  # sidecars and .part files are not tiles
        if p.name in _state.seen or is_in_flight(p):
            continue  # the upload door already has this one
        out.append(p)
    return out


def _settled(p: Path) -> bool:
    """Two reads, same size. A frame still being written must not be graded."""
    try:
        size = p.stat().st_size
    except OSError:
        return False
    prev = _state.sizes.get(p.name)
    _state.sizes[p.name] = size
    return prev is not None and prev == size and size > 0


def scan_once(source: str = "watch") -> list[contracts.TileRecord]:
    """Process every settled file in the watch folder once."""
    out: list[contracts.TileRecord] = []
    for p in _candidates():
        if not _settled(p):
            continue
        try:
            out.append(analyze_tile(p, source=source))
        except Exception as exc:  # noqa: BLE001 - analyze_tile is not supposed to
            n = _state.failures.get(p.name, 0) + 1
            _state.failures[p.name] = n
            if n >= MAX_ATTEMPTS:
                _state.seen.add(p.name)
                db.log("ingest", "tile-abandoned", {"attempts": n, "error": type(exc).__name__})
            continue
        _state.seen.add(p.name)
        _state.sizes.pop(p.name, None)
        _state.failures.pop(p.name, None)
    return out


def watch_loop(stop_event: threading.Event, interval_s: float = 2.0) -> None:
    """Poll the watch folder until stopped. Run in a daemon thread.

    Polling beats inotify here: the SD-card path mounts and unmounts, and a
    watcher that dies with its mount point is a watcher an operator cannot trust.
    """
    _ensure_db()
    while not stop_event.is_set():
        scan_once()
        stop_event.wait(interval_s)


def reset_watch_state() -> None:
    """Forget the settle bookkeeping, for the reseed script and for tests."""
    _state.seen.clear()
    _state.sizes.clear()
    _state.failures.clear()


# ---------------------------------------------------------------------- stats
def _latencies() -> list[int]:
    _ensure_db()
    rows = db.q("SELECT latency_ms FROM tiles WHERE latency_ms IS NOT NULL AND latency_ms > 0")
    return sorted(int(r["latency_ms"]) for r in rows)


def latency_p50() -> int:
    vals = _latencies()
    return int(statistics.median(vals)) if vals else 0


def latency_percentile(p: float = 95.0) -> int:
    """Nearest-rank percentile. With a few hundred tiles interpolation would be
    a precision we have not earned."""
    vals = _latencies()
    if not vals:
        return 0
    idx = min(len(vals) - 1, max(0, int(round((p / 100.0) * len(vals) + 0.5)) - 1))
    return vals[idx]


def counts() -> dict:
    """The tiles_* half of contracts.status_payload, straight from SQL."""
    _ensure_db()
    row = db.q1(
        """SELECT COUNT(*) AS analyzed,
                  COALESCE(SUM(stored), 0) AS stored,
                  COALESCE(SUM(1 - stored), 0) AS withheld,
                  COALESCE(SUM(status = 'error'), 0) AS errors,
                  COALESCE(SUM(needs_geo), 0) AS needs_geo
             FROM tiles"""
    )
    return {
        "tiles_analyzed": int(row["analyzed"] or 0),
        "tiles_stored": int(row["stored"] or 0),
        "tiles_withheld_from_storage": int(row["withheld"] or 0),
        "tiles_error": int(row["errors"] or 0),
        "tiles_needs_geo": int(row["needs_geo"] or 0),
        "tile_latency_ms_p50": latency_p50(),
    }


__all__ = [
    "IMAGE_SUFFIXES",
    "MAX_ATTEMPTS",
    "RERUN_SOURCES",
    "TileStages",
    "analyze_tile",
    "counts",
    "in_flight",
    "is_in_flight",
    "latency_p50",
    "latency_percentile",
    "place_result",
    "reset_watch_state",
    "scan_once",
    "sha256_of",
    "stages_of",
    "watch_loop",
    "wire_with_stages",
]
