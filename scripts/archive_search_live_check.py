#!/usr/bin/env python3
"""B7 opt-in live check for archive search (location + semantic + structured
filter), against the local fixture archive.

Not part of the pytest suite, makes NO network calls. Uses the REAL
production search_archive() and the SAME fixture data
(tests/decision/archive_fixture_data.py) pytest uses, built through the
real archive_write.index_cleared_archive_item writer -- nothing here
reimplements search or indexing.

Embedder selection: this script tries to construct the real
BgeSmallEmbedder() first. If BAAI/bge-small-en-v1.5 is not already cached
locally, construction raises RuntimeError (BgeSmallEmbedder never
downloads a model) and this script falls back to DeterministicStubEmbedder,
printing an explicit message -- it never attempts to download anything.
Whichever embedder is selected is used for BOTH indexing the fixture
captions AND embedding each query, since comparing vectors from two
different embedding spaces would be meaningless.

Lightning is NOT required for this script -- see
scripts/archive_tagging_live_check.py for the batch-tagging live check.

Usage:
    python scripts/archive_search_live_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.archive_embedder import BgeSmallEmbedder, DeterministicStubEmbedder  # noqa: E402
from backend.decision.archive_search import search_archive  # noqa: E402
from tests.decision.archive_fixture_data import build_fixture_db  # noqa: E402

QUERIES = [
    "buildings on fire",
    "collapsed roof",
    "class:3",
    "buildings on fire near 35th Ave SW",
    "class:3 near 35th Ave SW",
    "47.558, -122.377",
]


def _resolve_embedder():
    try:
        embedder = BgeSmallEmbedder()
        print(f"Embedder: {BgeSmallEmbedder.MODEL_NAME} (real, local, cached) -- using it for this live check.\n")
        return embedder
    except RuntimeError as exc:
        print(
            "BGE model unavailable locally -- this script never downloads a model. "
            f"Reason: {exc}\n"
            "Falling back to DeterministicStubEmbedder (offline, deterministic, "
            "vocabulary-sensitive, but NOT real semantic quality) for this live check.\n"
        )
        return DeterministicStubEmbedder()


def main():
    embedder = _resolve_embedder()
    ctx = build_fixture_db(embedder=embedder)
    conn = ctx["conn"]
    location_resolver = ctx["location_resolver"]

    print(f"Fixture archive: {len(ctx['eligible_ids'])} indexed items.\n")
    print("=" * 70)

    for query in QUERIES:
        result = search_archive({"q": query, "limit": 10}, conn, embedder=embedder, location_resolver=location_resolver)

        print(f"query: {query!r}")
        print(f"resolved_by: {result['resolved_by']}")
        print(f"took_ms: {result['took_ms']}")
        if not result["items"]:
            print("  (no results)")
        for item in result["items"]:
            print(f"  {item['image_id']}  class={item['class_max']}  centroid={item['centroid']}")
            print(f"      caption: {item['caption']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
