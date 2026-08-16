"""SQLite. The decision log is append-only, enforced by SQL triggers.

Thread-local connections: the watcher, the downlink and request handlers all
write. WAL plus the default busy timeout makes that workable.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Iterable, Optional

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
  filename        TEXT PRIMARY KEY,
  sha256          TEXT,
  status          TEXT NOT NULL,
  stored          INTEGER NOT NULL DEFAULT 1,
  withheld_reason TEXT,
  needs_geo       INTEGER NOT NULL DEFAULT 0,
  bounds_json     TEXT,
  captured_at     REAL,
  analyzed_at     REAL,
  latency_ms      INTEGER,
  geo_source      TEXT,
  stored_path     TEXT,
  -- The grading settings this tile was processed under (VL budget and concurrency),
  -- so a published latency percentile never averages tiles graded different ways.
  grading_profile TEXT
);

CREATE TABLE IF NOT EXISTS buildings (
  footprint_id  TEXT PRIMARY KEY,
  label         TEXT,
  centroid_json TEXT,
  geom_json     TEXT,
  damage_class  INTEGER,
  confidence    REAL,
  graded_by     TEXT,
  confirmed     INTEGER NOT NULL DEFAULT 0,
  doubt         REAL,
  votes_json    TEXT,
  vote_agreement REAL,
  facility_json TEXT,
  svi           REAL,
  area_m2       REAL,
  last_seen_at  REAL,
  source_tile   TEXT
);

CREATE TABLE IF NOT EXISTS archive (
  image_id       TEXT PRIMARY KEY,
  filename       TEXT,
  thumb_path     TEXT,
  captured_at    REAL,
  centroid_json  TEXT,
  needs_geo      INTEGER NOT NULL DEFAULT 0,
  caption        TEXT,
  caption_by     TEXT,
  tags_json      TEXT,
  class_max      INTEGER DEFAULT 0,
  key_evidence   INTEGER NOT NULL DEFAULT 0,
  embedding      BLOB,
  footprints_json TEXT,
  -- The footprint the caption describes. A tile holds tens of buildings and the
  -- caption is about one of them, so the console can say which.
  caption_anchor TEXT
);

-- Operator edits to the drafted plan. WHY a table: /api/plan recomputes the draft
-- from the current ranking on every poll, so an edit that lived only in the log was
-- discarded ~2 s later - a reassign visibly snapped back, and add/reorder/delete
-- did nothing at all. These rows are replayed over each fresh draft, so an operator
-- decision outlives a re-rank without freezing the ranking underneath it.
CREATE TABLE IF NOT EXISTS plan_overrides (
  footprint_id TEXT PRIMARY KEY,
  agency       TEXT,          -- reassigned owner, NULL to keep the drafted one
  order_key    REAL,          -- operator ordering within the agency, NULL for drafted
  deleted      INTEGER NOT NULL DEFAULT 0,
  units        INTEGER,       -- operator-set crew count, NULL for drafted
  task         TEXT,          -- operator-edited task text, NULL for drafted
  operator     TEXT,
  ts           REAL
);

CREATE TABLE IF NOT EXISTS road_blocks (
  road_name  TEXT PRIMARY KEY,
  geom_json  TEXT,
  blocked    INTEGER NOT NULL DEFAULT 1,
  operator   TEXT,
  ts         REAL
);

CREATE TABLE IF NOT EXISTS availability (
  agency          TEXT PRIMARY KEY,
  units_available INTEGER NOT NULL,
  operator        TEXT,
  ts              REAL
);

CREATE TABLE IF NOT EXISTS datasets (
  name          TEXT PRIMARY KEY,
  source        TEXT,
  last_refreshed REAL,
  sha256        TEXT,
  bytes         INTEGER,
  note          TEXT
);

CREATE TABLE IF NOT EXISTS decision_log (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        REAL NOT NULL,
  actor     TEXT NOT NULL,
  action    TEXT NOT NULL,
  payload   TEXT
);

CREATE INDEX IF NOT EXISTS idx_buildings_class ON buildings(damage_class);
CREATE INDEX IF NOT EXISTS idx_archive_captured ON archive(captured_at);

-- Append-only, enforced by SQL, not by convention. Both UPDATE and DELETE abort.
CREATE TRIGGER IF NOT EXISTS decision_log_no_update
BEFORE UPDATE ON decision_log
BEGIN SELECT RAISE(ABORT, 'decision_log is append-only'); END;

CREATE TRIGGER IF NOT EXISTS decision_log_no_delete
BEFORE DELETE ON decision_log
BEGIN SELECT RAISE(ABORT, 'decision_log is append-only'); END;
"""


def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(config.DB_PATH), timeout=10.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        # REPLACE deletes rows without firing delete triggers unless recursive
        # triggers are on. Close that hole explicitly.
        c.execute("PRAGMA recursive_triggers=ON")
        c.execute("PRAGMA foreign_keys=ON")
        _local.conn = c
    return c


# Columns added after the first box was already running. CREATE TABLE IF NOT EXISTS
# silently does nothing to an existing table, so a schema edit alone would leave a
# populated DB one column short and every read of it would raise. Additive only:
# nothing here drops or rewrites operator data.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("archive", "caption_anchor", "TEXT"),
    # The grading settings that produced this tile's latency, so a published
    # percentile never averages tiles graded under different settings.
    ("tiles", "grading_profile", "TEXT"),
)


def init() -> None:
    c = conn()
    c.executescript(SCHEMA)
    for table, column, decl in _ADDED_COLUMNS:
        have = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
        if not have:
            continue  # the table itself is absent; SCHEMA above owns creating it
        if column not in have:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    c.commit()


def log(actor: str, action: str, payload: Optional[dict] = None) -> None:
    """Append to the decision log. Never include withheld filenames or reasons:
    that would leak through the unauthenticated log export."""
    c = conn()
    c.execute(
        "INSERT INTO decision_log (ts, actor, action, payload) VALUES (?,?,?,?)",
        (time.time(), actor, action, json.dumps(payload or {}, separators=(",", ":"))),
    )
    c.commit()


def q(sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return list(conn().execute(sql, tuple(args)).fetchall())


def q1(sql: str, args: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    return conn().execute(sql, tuple(args)).fetchone()


def run(sql: str, args: Iterable[Any] = ()) -> None:
    c = conn()
    c.execute(sql, tuple(args))
    c.commit()


def jload(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
