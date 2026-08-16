"""B7: local SQLite storage for the searchable image archive.

STATUS: Member A owns the real archive/ingest pipeline and the privacy gate
that decides which images are cleared (see README section 5.3 / A6). That
table does not exist in this repository yet. This module defines the
smallest internal SQLite structure B7's search needs, built as a clean,
swappable storage layer -- once Member A's real archive table exists,
either point get_connection() at that same database file (schema is
additive and namespaced to its own table) or replace this module's
functions with equivalent ones over the real table; nothing in
archive_search.py/archive_write.py needs to change as long as the same
function signatures are honored. This is the ONE integration seam pending
real A-side data (see also archive_location.py's docstring for the second,
independent pending seam: real local OSM road/facility data).

No vector DB (Chroma/FAISS/Pinecone/Qdrant/pgvector) -- SQLite only, plus
NumPy for the actual cosine ranking (archive_search.py). This module never
ranks; it only stores and narrows.

--------------------------------------------------------------------------
SCHEMA
--------------------------------------------------------------------------

One table, `archive_items`, storing exactly the fields the frozen public
ArchiveItem contract needs, plus two purely internal columns that are NEVER
exposed as new ArchiveItem fields:
  - embedding / embedding_dim: the caption's normalized vector (Part 2/3)
  - sector: optional internal metadata some ingest pipelines may already
    produce (used only by the structured `sector:<value>` filter token,
    see archive_filter_parser.py)

    image_id       TEXT PRIMARY KEY
    thumb_path     TEXT NOT NULL
    captured_at    REAL NOT NULL
    centroid_lng   REAL              -- NULL iff centroid is null
    centroid_lat   REAL              -- NULL iff centroid is null
    needs_geo      INTEGER NOT NULL  -- 0/1
    caption        TEXT NOT NULL
    tags           TEXT NOT NULL     -- JSON array of strings
    class_max      INTEGER NOT NULL  -- 0-3
    key_evidence   INTEGER NOT NULL  -- 0/1
    sector         TEXT              -- internal only, never in ArchiveItem
    embedding      BLOB              -- internal only, see ENCODING below
    embedding_dim  INTEGER           -- internal only, cross-checked on read

ENCODING (embedding): raw float32, little-endian, no header --
`vector.astype("<f4").tobytes()`. embedding_dim is stored alongside and
cross-checked against `len(blob) // 4` on every read; a mismatch (a
corrupted or hand-edited row) causes that row's embedding to be treated as
absent -- excluded from semantic ranking, never raised as a crash (see
Part 3's "missing embedding" requirement, honored uniformly for "never
had one" and "unreadable").

Only rows that Member A's privacy gate has already cleared may ever be
INSERTed here -- enforced in archive_write.py's writer, not in this
storage layer (see that module's docstring); this module has no concept of
"withheld" at all because a withheld record should never reach it.
"""

import json
import os
import sqlite3

import numpy as np

EMBEDDING_DIM = 384

_SCHEMA = """
CREATE TABLE IF NOT EXISTS archive_items (
    image_id TEXT PRIMARY KEY,
    thumb_path TEXT NOT NULL,
    captured_at REAL NOT NULL,
    centroid_lng REAL,
    centroid_lat REAL,
    needs_geo INTEGER NOT NULL,
    caption TEXT NOT NULL,
    tags TEXT NOT NULL,
    class_max INTEGER NOT NULL,
    key_evidence INTEGER NOT NULL,
    sector TEXT,
    embedding BLOB,
    embedding_dim INTEGER
);

CREATE INDEX IF NOT EXISTS idx_archive_items_class_max ON archive_items(class_max);
CREATE INDEX IF NOT EXISTS idx_archive_items_captured_at ON archive_items(captured_at);
CREATE INDEX IF NOT EXISTS idx_archive_items_centroid ON archive_items(centroid_lng, centroid_lat);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open (creating if needed) the local SQLite archive index at db_path,
    with schema in place. db_path=":memory:" is fine for tests.
    """
    if db_path != ":memory:":
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _encode_embedding(vector: np.ndarray) -> bytes:
    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(f"embedding must have shape ({EMBEDDING_DIM},), got {vector.shape}")
    return np.asarray(vector, dtype="<f4").tobytes()


def _decode_embedding(blob, embedding_dim) -> np.ndarray:
    """Returns None (never raises) if blob/embedding_dim are absent or
    inconsistent -- a corrupt/missing embedding excludes that row from
    semantic ranking rather than crashing the search (Part 3).
    """
    if blob is None or embedding_dim is None:
        return None
    vector = np.frombuffer(blob, dtype="<f4")
    if vector.shape != (EMBEDDING_DIM,) or embedding_dim != EMBEDDING_DIM or len(blob) != EMBEDDING_DIM * 4:
        return None
    return vector


def _tags_to_json(tags: list) -> str:
    return json.dumps(list(tags), separators=(",", ":"))


def _tags_from_json(raw: str) -> list:
    return json.loads(raw)


def upsert_item(conn: sqlite3.Connection, item: dict, embedding: np.ndarray = None, sector: str = None) -> None:
    """Pure storage: INSERT OR REPLACE one archive item row. NO eligibility
    policy here -- see archive_write.index_cleared_archive_item, the only
    caller allowed to reach this for a NEW row. `item` is the frozen public
    ArchiveItem shape (already validated by the caller); `embedding`, if
    given, must be a (EMBEDDING_DIM,) float array -- validated here (raises
    ValueError on a dimension mismatch, per Part 1's "validate dimensions
    on read/write"). Every value is bound as a parameter -- never
    interpolated into the SQL text.
    """
    centroid = item.get("centroid")
    centroid_lng, centroid_lat = (centroid[0], centroid[1]) if centroid is not None else (None, None)

    embedding_blob = None
    embedding_dim = None
    if embedding is not None:
        embedding_blob = _encode_embedding(embedding)
        embedding_dim = EMBEDDING_DIM

    conn.execute(
        """
        INSERT INTO archive_items
            (image_id, thumb_path, captured_at, centroid_lng, centroid_lat,
             needs_geo, caption, tags, class_max, key_evidence, sector,
             embedding, embedding_dim)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            thumb_path=excluded.thumb_path,
            captured_at=excluded.captured_at,
            centroid_lng=excluded.centroid_lng,
            centroid_lat=excluded.centroid_lat,
            needs_geo=excluded.needs_geo,
            caption=excluded.caption,
            tags=excluded.tags,
            class_max=excluded.class_max,
            key_evidence=excluded.key_evidence,
            sector=excluded.sector,
            embedding=excluded.embedding,
            embedding_dim=excluded.embedding_dim
        """,
        (
            item["image_id"],
            item["thumb_path"],
            float(item["captured_at"]),
            centroid_lng,
            centroid_lat,
            1 if item["needs_geo"] else 0,
            item["caption"],
            _tags_to_json(item["tags"]),
            int(item["class_max"]),
            1 if item["key_evidence"] else 0,
            sector,
            embedding_blob,
            embedding_dim,
        ),
    )
    conn.commit()


def update_item_fields(
    conn: sqlite3.Connection,
    image_id: str,
    caption: str = None,
    tags: list = None,
    embedding: np.ndarray = None,
    key_evidence: bool = None,
) -> None:
    """Partial update of an ALREADY-INDEXED row -- never creates a new row
    (archive_write.reindex_archive_item checks existence first). Only the
    explicitly-supplied fields change; every other column is left alone.
    Used by Part 10 (edit/reindex): a caption edit may come paired with a
    freshly-recomputed embedding from the same call.
    """
    sets = []
    params = []
    if caption is not None:
        sets.append("caption = ?")
        params.append(caption)
    if tags is not None:
        sets.append("tags = ?")
        params.append(_tags_to_json(tags))
    if embedding is not None:
        sets.append("embedding = ?")
        params.append(_encode_embedding(embedding))
        sets.append("embedding_dim = ?")
        params.append(EMBEDDING_DIM)
    if key_evidence is not None:
        sets.append("key_evidence = ?")
        params.append(1 if key_evidence else 0)

    if not sets:
        return

    params.append(image_id)
    conn.execute(f"UPDATE archive_items SET {', '.join(sets)} WHERE image_id = ?", params)
    conn.commit()


def _row_to_public_item(row: dict) -> dict:
    centroid = (
        [row["centroid_lng"], row["centroid_lat"]]
        if row["centroid_lng"] is not None and row["centroid_lat"] is not None
        else None
    )
    return {
        "image_id": row["image_id"],
        "thumb_path": row["thumb_path"],
        "captured_at": row["captured_at"],
        "centroid": centroid,
        "needs_geo": bool(row["needs_geo"]),
        "caption": row["caption"],
        "tags": _tags_from_json(row["tags"]),
        "class_max": row["class_max"],
        "key_evidence": bool(row["key_evidence"]),
    }


def get_item(conn: sqlite3.Connection, image_id: str) -> dict:
    """Public-shaped ArchiveItem, or None if not indexed."""
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM archive_items WHERE image_id = ?", (image_id,))
    row = cursor.fetchone()
    return _row_to_public_item(row) if row is not None else None


def fetch_candidate_rows(conn: sqlite3.Connection, filters: list = None, bbox: tuple = None) -> list:
    """Fetch candidate rows for search, narrowed by `filters` (a list of
    (sql_fragment, param) pairs from archive_filter_parser.parse_structured_filters
    -- always parameterized, never string-interpolated) AND, if given, a
    `bbox` = (lng_min, lat_min, lng_max, lat_max) centroid bound (Part 5).
    An item with centroid=null never matches a bbox (excluded by the
    `centroid_lng IS NOT NULL` guard below), per Part 5's invariant.

    Returns INTERNAL rows: the public ArchiveItem fields plus one
    underscore-prefixed diagnostic key, "_embedding" (a (EMBEDDING_DIM,)
    np.ndarray or None) -- callers (archive_search.py) strip "_embedding"
    before constructing public ArchiveItem objects; it exists so semantic
    ranking doesn't need a second round-trip to fetch vectors.
    """
    conn.row_factory = sqlite3.Row
    where_clauses = []
    params = []

    for fragment, value in filters or []:
        where_clauses.append(fragment)
        params.append(value)

    if bbox is not None:
        lng_min, lat_min, lng_max, lat_max = bbox
        where_clauses.append(
            "(centroid_lng IS NOT NULL AND centroid_lat IS NOT NULL "
            "AND centroid_lng BETWEEN ? AND ? AND centroid_lat BETWEEN ? AND ?)"
        )
        params.extend([lng_min, lng_max, lat_min, lat_max])

    sql = "SELECT * FROM archive_items"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    rows = conn.execute(sql, params).fetchall()

    results = []
    for row in rows:
        item = _row_to_public_item(row)
        item["_embedding"] = _decode_embedding(row["embedding"], row["embedding_dim"])
        results.append(item)
    return results


def count_items(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("SELECT COUNT(*) FROM archive_items")
    return cursor.fetchone()[0]
