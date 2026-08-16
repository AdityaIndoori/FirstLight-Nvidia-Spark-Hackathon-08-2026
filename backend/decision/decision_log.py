"""Append-only decision/audit log.

Generic logging mechanism only - nothing here wires into Grade Flip, Road
Block, Plan Edit, or Set Availability yet. Those call append_decision() when
they're built.

Append-only is enforced by SQLite triggers (BEFORE UPDATE / BEFORE DELETE that
RAISE(ABORT, ...)), not by convention in this module's Python code. The
triggers protect the table even if something executes SQL directly against the
same database file, bypassing append_decision()/read_decisions() entirely.
"""

import json
import os
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS decision_log_no_update
BEFORE UPDATE ON decision_log
BEGIN
    SELECT RAISE(ABORT, 'decision_log is append-only: UPDATE is not permitted');
END;

CREATE TRIGGER IF NOT EXISTS decision_log_no_delete
BEFORE DELETE ON decision_log
BEGIN
    SELECT RAISE(ABORT, 'decision_log is append-only: DELETE is not permitted');
END;
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open (creating if needed) the local SQLite decision log at db_path,
    with schema and append-only triggers in place.
    """
    if db_path != ":memory:":
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def append_decision(conn: sqlite3.Connection, actor: str, action: str, entity_id: str, details: dict, ts: float = None) -> int:
    """Insert one immutable decision-log row. Returns the new row's id.

    details is serialized deterministically (sorted keys, fixed separators) so
    identical details always produce identical stored text.
    """
    if ts is None:
        ts = time.time()
    payload = json.dumps(details, sort_keys=True, separators=(",", ":"))

    cursor = conn.execute(
        "INSERT INTO decision_log (ts, actor, action, entity_id, details) VALUES (?, ?, ?, ?, ?)",
        (ts, actor, action, entity_id, payload),
    )
    conn.commit()
    return cursor.lastrowid


def read_decisions(conn: sqlite3.Connection) -> list:
    """Return all decision-log rows in chronological (insertion) order."""
    rows = conn.execute(
        "SELECT id, ts, actor, action, entity_id, details FROM decision_log ORDER BY id ASC"
    ).fetchall()

    return [
        {
            "id": row[0],
            "ts": row[1],
            "actor": row[2],
            "action": row[3],
            "entity_id": row[4],
            "details": json.loads(row[5]),
        }
        for row in rows
    ]
