"""B7 Parts 3, 6, 7, 11, 16: the one search entry point, FILTER THEN RANK.

    search_archive(request, conn, embedder=None, location_resolver=None) -> SearchResult

--------------------------------------------------------------------------
FROZEN PUBLIC CONTRACTS (do not change these shapes -- see also
archive_store.py's ArchiveItem docstring)
--------------------------------------------------------------------------

SearchRequest (in):   {q: str, limit: int}
SearchResult (out):   {items: [ArchiveItem], resolved_by: [...], took_ms: int}
resolved_by contains only resolvers that actually fired, always emitted in
the fixed order location, semantic, filter (regardless of internal
execution order -- Part 6's explicit requirement). No search scores are
ever placed on the public contract; internal scores exist only for
diagnostics/tests (see _semantic_rank's return shape).

--------------------------------------------------------------------------
ALGORITHM -- FILTER, THEN RANK (never the reverse)
--------------------------------------------------------------------------

1. validate q/limit (ArchiveSearchError on failure)
2. parse_structured_filters(q) -> filters, residual_after_filters
   (archive_filter_parser.py -- a small deterministic parser, never an LLM
   turning text into SQL; every filter value is a bound SQL parameter)
3. location_resolver.resolve(residual_after_filters) -> bbox or None
   (archive_location.py -- local-only, no external geocoder)
4. archive_store.fetch_candidate_rows(conn, filters, bbox) -- SQL narrows
   the candidate set FIRST; nothing is semantically ranked yet
5. if meaningful residual text remains after removing structured/location
   terms: embed it and cosine-rank ONLY the narrowed candidates (NumPy,
   matrix @ query_vector, both sides L2-normalized -- see _semantic_rank)
6. else: deterministic non-semantic ordering (captured_at desc, image_id
   asc -- the same stable tie-break semantic ranking itself uses)
7. apply request.limit AFTER ranking/narrowing, never before
8. build frozen ArchiveItem objects (drop the internal "_embedding"/score
   diagnostics), set resolved_by, measure took_ms (time.perf_counter)

A candidate row missing its embedding is excluded from semantic ranking,
never crashes the search (Part 3) -- archive_store.fetch_candidate_rows
already returns None for a missing/corrupt embedding; _semantic_rank
simply skips those rows.

Pure filter and pure location searches NEVER construct or call an
embedder -- the embedder is only touched when semantic residual text
actually exists (see the `embedder is None` lazy-default below), so a
caller who never wants BGE loaded for a filter/location-only deployment
never pays that cost.

Part 11 (API/service boundary): this repository has no FastAPI/HTTP layer
anywhere yet (checked: no fastapi/APIRouter usage in the codebase), so per
instructions this IS the service boundary -- search_archive is a plain,
directly callable function taking/returning exactly the frozen wire
shapes, following the same "plain business function" convention already
used by backend/decision/agent_tools.py. Wire it into a real HTTP layer
later without changing this function's contract.
"""

import time

import numpy as np

from backend.decision import archive_store
from backend.decision.archive_embedder import BgeSmallEmbedder, Embedder
from backend.decision.archive_filter_parser import ArchiveSearchError, parse_structured_filters
from backend.decision.archive_location import LocalLocationResolver

MAX_LIMIT = 200
"""Safe maximum for SearchRequest.limit -- prevents an absurd request from
forcing an unbounded response; large enough that no realistic operator
query in this project's demo-sized archive would ever need more."""


def _validate_request(q, limit) -> None:
    if not isinstance(q, str):
        raise ArchiveSearchError(f"q must be a string, got {q!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= MAX_LIMIT):
        raise ArchiveSearchError(f"limit must be an integer in [1, {MAX_LIMIT}], got {limit!r}")


def _deterministic_sort_key(row: dict) -> tuple:
    """captured_at DESCENDING, then image_id ASCENDING -- the stable
    tie-break used both as the non-semantic ordering (step 6) and as the
    tie-break for equal cosine scores (_semantic_rank), so ordering is
    reproducible across repeated identical searches.
    """
    return (-row["captured_at"], row["image_id"])


def _deterministic_order(rows: list) -> list:
    return sorted(rows, key=_deterministic_sort_key)


def _semantic_rank(rows: list, query_text: str, embedder: Embedder) -> list:
    """Cosine-rank `rows` by similarity to `query_text`, NumPy matrix-vector
    multiply (both sides L2-normalized, so cosine similarity IS the plain
    dot product -- Part 3). Rows missing an embedding are excluded from
    ranking (never crash the search) but nothing else about the candidate
    set changes -- this function is only ever called with the ALREADY
    SQL-narrowed candidate rows (filter-then-rank), never the whole
    archive.

    Returns rows in descending-score order, ties broken by
    _deterministic_sort_key -- a fully deterministic top-k regardless of
    how many rows tie at the same score.
    """
    scoreable = [row for row in rows if row["_embedding"] is not None]
    if not scoreable:
        return []

    matrix = np.stack([row["_embedding"] for row in scoreable]).astype(np.float32)
    query_vector = np.asarray(embedder.embed_text(query_text), dtype=np.float32)
    scores = matrix @ query_vector

    order = sorted(
        range(len(scoreable)),
        key=lambda i: (-float(scores[i]),) + _deterministic_sort_key(scoreable[i]),
    )
    return [scoreable[i] for i in order]


def _to_public_archive_item(row: dict) -> dict:
    return {
        "image_id": row["image_id"],
        "thumb_path": row["thumb_path"],
        "captured_at": row["captured_at"],
        "centroid": row["centroid"],
        "needs_geo": row["needs_geo"],
        "caption": row["caption"],
        "tags": list(row["tags"]),
        "class_max": row["class_max"],
        "key_evidence": row["key_evidence"],
    }


def search_archive(
    request: dict,
    conn,
    embedder: Embedder = None,
    location_resolver: LocalLocationResolver = None,
) -> dict:
    """Run the full B7 filter-then-rank search for one SearchRequest
    against the already-indexed archive (conn, an archive_store
    connection -- only cleared items can ever be rows there, see
    archive_write.py). Never mutates `request`.

    `embedder` defaults to BgeSmallEmbedder() -- constructed LAZILY, only
    if semantic ranking is actually needed for this request, so a pure
    filter/location search never loads it. Tests always pass an explicit
    DeterministicStubEmbedder(). `location_resolver` defaults to an empty
    LocalLocationResolver (coordinate literals still resolve; named
    road/facility lookups do not, since no fixture data is wired by
    default) -- callers with real local place data pass their own.
    """
    started_at = time.perf_counter()

    q = request.get("q")
    limit = request.get("limit")
    _validate_request(q, limit)

    active_resolver = location_resolver if location_resolver is not None else LocalLocationResolver()

    filters, residual_after_filters = parse_structured_filters(q)
    fired_filter = bool(filters)

    location_result = active_resolver.resolve(residual_after_filters)
    fired_location = location_result is not None
    residual_after_location = (
        location_result["residual_after"] if location_result is not None else residual_after_filters
    )

    semantic_query = residual_after_location.strip()
    fired_semantic = bool(semantic_query)

    bbox = location_result["bbox"] if location_result is not None else None
    candidate_rows = archive_store.fetch_candidate_rows(conn, filters, bbox)

    if fired_semantic:
        active_embedder = embedder if embedder is not None else BgeSmallEmbedder()
        ranked_rows = _semantic_rank(candidate_rows, semantic_query, active_embedder)
    else:
        ranked_rows = _deterministic_order(candidate_rows)

    limited_rows = ranked_rows[:limit]
    items = [_to_public_archive_item(row) for row in limited_rows]

    resolved_by = [
        name
        for name, fired in (("location", fired_location), ("semantic", fired_semantic), ("filter", fired_filter))
        if fired
    ]

    took_ms = int(round((time.perf_counter() - started_at) * 1000))
    return {"items": items, "resolved_by": resolved_by, "took_ms": took_ms}


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Part 16: tiny future-B8-evaluator hook, NOT the full 20-query
    benchmark. |retrieved_ids[:k] ∩ relevant_ids| / |relevant_ids|; 0.0 if
    relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """|retrieved_ids[:k] ∩ relevant_ids| / k (over the actual number
    returned, if fewer than k); 0.0 if nothing was retrieved.
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & set(relevant_ids)) / len(top_k)
