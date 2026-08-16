"""B8 Part E2: search precision@k / recall@k over 20 held-out queries with
known-relevant image IDs, against the real B7 search_archive().

Reuses the existing B7 fixture archive verbatim
(tests.decision.archive_fixture_data.build_fixture_db/FIXTURE_ELIGIBLE_ITEMS)
rather than inventing a second held-out dataset -- the 20 queries below are
written against that SAME 13-item archive, with known_relevant_image_ids
hand-labeled from the fixture's actual captions/locations/class_max, not
copied from whatever a particular embedder happens to return (see the
per-query "reason" field). Query composition spans every mix Part 9
requires: semantic only, structured only, location only, filter+semantic,
location+semantic, location+filter+semantic.

--------------------------------------------------------------------------
TWO MODES (BGE is not available on this Mac)
--------------------------------------------------------------------------
offline / evaluate_search_recall_precision_offline(): indexes and queries
    with DeterministicStubEmbedder. Runs in pytest. Verifies the metric
    ARITHMETIC and the filter-then-rank composition machinery -- explicitly
    labeled in details as stub-quality, NEVER published as real-BGE
    retrieval quality (Part 9's explicit instruction). The 8 structured-
    only and location-only queries in this set are genuine even here,
    since neither resolver touches the embedder at all.

live / evaluate_search_recall_precision_live(): attempts BgeSmallEmbedder()
    first; if unavailable (not cached locally -- this function NEVER
    downloads one), returns a single DEFERRED metric with that reason
    instead of silently falling back to the stub and mislabeling it as
    real quality.

precision@k convention (documented per Part 9's instruction to pick and
document one): precision@k = |relevant ∩ retrieved[:k]| / len(retrieved[:k])
-- i.e. divided by the number ACTUALLY returned when fewer than k exist
(a query with only 2 true candidates and k=10 is judged on those 2, not
artificially penalized for k - 2 "misses" that were never possible to
retrieve).
"""

import time

from backend.decision.archive_embedder import BgeSmallEmbedder, DeterministicStubEmbedder
from backend.decision.archive_search import search_archive
from backend.decision.eval.report import STATUS_DEFERRED, STATUS_MEASURED, deferred_metric, make_metric
from tests.decision.archive_fixture_data import build_fixture_db

# --------------------------------------------------------------------------
# 20 held-out queries. resolver_mix documents the intended composition,
# purely descriptive (the metric itself derives resolved_by from the real
# search_archive() call, never trusts this label blindly).
# --------------------------------------------------------------------------

SEARCH_QUERY_FIXTURE = [
    # -- semantic only (4) --
    {
        "q": "buildings on fire",
        "k": 5,
        "relevant_image_ids": {"img-001", "img-008", "img-013"},
        "resolver_mix": "semantic",
        "reason": "the only three captions containing fire/flame/burning wording",
    },
    {
        "q": "collapsed roof",
        "k": 5,
        "relevant_image_ids": {"img-003", "img-011"},
        "resolver_mix": "semantic",
        "reason": "the only two captions describing a collapsed roof",
    },
    {
        "q": "flooding and standing water",
        "k": 5,
        "relevant_image_ids": {"img-002", "img-010"},
        "resolver_mix": "semantic",
        "reason": "the only two captions describing flooding/standing water",
    },
    {
        "q": "undamaged structure no visible concerns",
        "k": 5,
        "relevant_image_ids": {"img-005", "img-012"},
        "resolver_mix": "semantic",
        "reason": "the only two captions describing no damage",
    },
    # -- structured only (4) --
    {
        "q": "class:3",
        "k": 10,
        "relevant_image_ids": {"img-001", "img-003", "img-008", "img-009", "img-011", "img-013"},
        "resolver_mix": "filter",
        "reason": "fixture items with class_max == 3",
    },
    {
        "q": "class:0",
        "k": 10,
        "relevant_image_ids": {"img-005", "img-012"},
        "resolver_mix": "filter",
        "reason": "fixture items with class_max == 0",
    },
    {
        "q": "key:true",
        "k": 10,
        "relevant_image_ids": {"img-001", "img-003", "img-007", "img-008", "img-009", "img-011", "img-013"},
        "resolver_mix": "filter",
        "reason": "fixture items with key_evidence == True",
    },
    {
        "q": "needs_geo:true",
        "k": 5,
        "relevant_image_ids": {"img-009"},
        "resolver_mix": "filter",
        "reason": "the only fixture item with needs_geo == True (centroid null)",
    },
    # -- location only (4) --
    {
        "q": "35th Ave SW",
        "k": 10,
        "relevant_image_ids": {"img-001", "img-002", "img-003", "img-008", "img-010", "img-013"},
        "resolver_mix": "location",
        "reason": "all fixture items within the default 400m bbox of 35th Ave SW / Oak Ave (adjacent clusters)",
    },
    {
        "q": "Riverside Dialysis Center",
        "k": 5,
        "relevant_image_ids": {"img-007"},
        "resolver_mix": "location",
        "reason": "the only item located at the dialysis center",
    },
    {
        "q": "Providence Mount St. Vincent",
        "k": 5,
        "relevant_image_ids": {"img-004"},
        "resolver_mix": "location",
        "reason": "the only item located at the nursing home",
    },
    {
        "q": "Pine St",
        "k": 5,
        "relevant_image_ids": {"img-011"},
        "resolver_mix": "location",
        "reason": "the only item located on Pine St",
    },
    # -- filter + semantic (3) --
    {
        "q": "buildings on fire class:3",
        "k": 10,
        "relevant_image_ids": {"img-001", "img-008", "img-013"},
        "resolver_mix": "filter+semantic",
        "reason": "fire captions that are also class_max == 3 (all three fire items happen to be class 3)",
    },
    {
        "q": "collapsed roof class:3",
        "k": 10,
        "relevant_image_ids": {"img-003", "img-011"},
        "resolver_mix": "filter+semantic",
        "reason": "collapse captions that are also class_max == 3",
    },
    {
        "q": "undamaged structure class:0",
        "k": 10,
        "relevant_image_ids": {"img-005", "img-012"},
        "resolver_mix": "filter+semantic",
        "reason": "undamaged captions that are also class_max == 0",
    },
    # -- location + semantic (3) --
    {
        "q": "buildings on fire near 35th Ave SW",
        "k": 10,
        "relevant_image_ids": {"img-001", "img-008", "img-013"},
        "resolver_mix": "location+semantic",
        "reason": "fire captions within the 35th Ave SW / Oak Ave cluster",
    },
    {
        "q": "flooding near 35th Ave SW",
        "k": 10,
        "relevant_image_ids": {"img-002", "img-010"},
        "resolver_mix": "location+semantic",
        "reason": "flood captions within the 35th Ave SW / Oak Ave cluster",
    },
    {
        "q": "structural damage near Riverside Dialysis Center",
        "k": 5,
        "relevant_image_ids": {"img-007"},
        "resolver_mix": "location+semantic",
        "reason": "the only item in the dialysis-center bbox",
    },
    # -- location + filter + semantic (2) --
    {
        "q": "buildings on fire near 35th Ave SW class:3",
        "k": 10,
        "relevant_image_ids": {"img-001", "img-008", "img-013"},
        "resolver_mix": "location+filter+semantic",
        "reason": "fire captions, class 3, within the 35th Ave SW / Oak Ave cluster",
    },
    {
        "q": "roof collapse near Pine St class:3",
        "k": 5,
        "relevant_image_ids": {"img-011"},
        "resolver_mix": "location+filter+semantic",
        "reason": "the only class-3 item in the Pine St bbox",
    },
]

assert len(SEARCH_QUERY_FIXTURE) == 20


def _precision_recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> tuple:
    """precision@k convention: divided by the number actually returned
    (never artificially penalized when fewer than k candidates exist at
    all) -- see module docstring."""
    top_k = retrieved_ids[:k]
    correct = len(set(top_k) & relevant_ids)
    precision = correct / len(top_k) if top_k else (1.0 if not relevant_ids else 0.0)
    recall = correct / len(relevant_ids) if relevant_ids else (1.0 if not top_k else 0.0)
    return precision, recall


def _run_queries(conn, embedder, location_resolver) -> list:
    per_query = []
    for case in SEARCH_QUERY_FIXTURE:
        result = search_archive(
            {"q": case["q"], "limit": case["k"]}, conn, embedder=embedder, location_resolver=location_resolver
        )
        retrieved_ids = [item["image_id"] for item in result["items"]]
        precision, recall = _precision_recall_at_k(retrieved_ids, case["relevant_image_ids"], case["k"])
        per_query.append(
            {
                "q": case["q"],
                "k": case["k"],
                "resolver_mix_intended": case["resolver_mix"],
                "resolved_by": result["resolved_by"],
                "took_ms": result["took_ms"],
                "retrieved_ids": retrieved_ids,
                "relevant_ids": sorted(case["relevant_image_ids"]),
                "precision_at_k": precision,
                "recall_at_k": recall,
            }
        )
    return per_query


def evaluate_search_recall_precision_offline() -> dict:
    embedder = DeterministicStubEmbedder()
    ctx = build_fixture_db(embedder=embedder)
    per_query = _run_queries(ctx["conn"], embedder, ctx["location_resolver"])

    mean_precision = sum(q["precision_at_k"] for q in per_query) / len(per_query)
    mean_recall = sum(q["recall_at_k"] for q in per_query) / len(per_query)

    return make_metric(
        name="search_precision_recall_at_k",
        status=STATUS_MEASURED,
        value={"mean_precision_at_k": mean_precision, "mean_recall_at_k": mean_recall},
        threshold=None,
        sample_count=len(per_query),
        details={
            "mode": (
                "OFFLINE: DeterministicStubEmbedder for the semantic-composed queries -- "
                "verifies metric machinery and filter-then-rank composition, NOT real BGE "
                "retrieval quality. The 8 structured-only/location-only queries in this set "
                "are genuine even in this mode (no embedder involved)."
            ),
            "per_query": per_query,
        },
    )


def evaluate_search_recall_precision_live() -> dict:
    try:
        embedder = BgeSmallEmbedder()
    except RuntimeError as exc:
        return deferred_metric(
            "search_precision_recall_at_k_live",
            f"BAAI/bge-small-en-v1.5 is not available locally -- never downloaded automatically: {exc}",
        )

    started_at = time.perf_counter()
    ctx = build_fixture_db(embedder=embedder)
    per_query = _run_queries(ctx["conn"], embedder, ctx["location_resolver"])
    elapsed_s = time.perf_counter() - started_at

    mean_precision = sum(q["precision_at_k"] for q in per_query) / len(per_query)
    mean_recall = sum(q["recall_at_k"] for q in per_query) / len(per_query)

    return make_metric(
        name="search_precision_recall_at_k_live",
        status=STATUS_MEASURED,
        value={"mean_precision_at_k": mean_precision, "mean_recall_at_k": mean_recall},
        threshold=None,
        sample_count=len(per_query),
        details={
            "mode": f"LIVE: real {BgeSmallEmbedder.MODEL_NAME} embedder",
            "elapsed_s": elapsed_s,
            "per_query": per_query,
        },
    )
