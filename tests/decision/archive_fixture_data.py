"""Shared B7 fixture data -- NOT a test file (no test_* functions), a
helper module imported by tests/decision/test_archive_*.py AND by
scripts/archive_search_live_check.py / scripts/archive_tagging_live_check.py,
so pytest and the live-check scripts search over exactly the same
deterministic local archive.

Everything here is invented fixture content standing in for Member A's
real cleared-archive output and real local OSM road/facility export (see
backend/decision/archive_location.py and backend/decision/archive_store.py
docstrings for that pending integration seam) -- no network, no real
imagery, thumb_path values are placeholder strings only.

FIXTURE_ELIGIBLE_ITEMS: 13 already-cleared ArchiveItem-shaped records
(what Member A's privacy gate would have already passed), spanning
multiple locations, fire/collapse/flood/undamaged captions, class_max
0-3, varying captured_at across one demo day (2026-08-15, matching
archive_filter_parser._BARE_TIME_REFERENCE_DATE), key_evidence true/false,
and one needs_geo=true item with centroid=null.

FIXTURE_WITHHELD_SOURCE: ONE record representing what a real privacy-gate
withhold looks like from B7's side -- this is deliberately NEVER passed to
index_cleared_archive_item with eligible=True anywhere in this module or
in build_fixture_db(); tests import it specifically to prove it cannot be
indexed and never appears in search results (Part 9/13's required test).

FIXTURE_PLACES: local road/facility name -> centroid rows for
archive_location.FixtureLocalPlaceIndex.
"""

from datetime import datetime, timezone

from backend.decision import archive_store, archive_write
from backend.decision.archive_embedder import DeterministicStubEmbedder
from backend.decision.archive_location import FixtureLocalPlaceIndex, LocalLocationResolver
from backend.decision.archive_tag_extractor import DeterministicStubTagExtractor


def _ts(iso_text: str) -> float:
    return datetime.fromisoformat(iso_text).replace(tzinfo=timezone.utc).timestamp()


FIXTURE_PLACES = [
    {"name": "35th Ave SW", "centroid": [-122.3902, 47.5981]},
    {"name": "Oak Ave", "centroid": [-122.3951, 47.5990]},
    {"name": "Elm St", "centroid": [-122.4001, 47.6002]},
    {"name": "Pine St", "centroid": [-122.3820, 47.6105]},
    {"name": "Providence Mount St. Vincent", "centroid": [-122.3861, 47.5670]},
    {"name": "Riverside Dialysis Center", "centroid": [-122.3850, 47.6050]},
]

FIXTURE_ELIGIBLE_ITEMS = [
    {
        "image_id": "img-001",
        "thumb_path": "thumbs/img-001.jpg",
        "captured_at": _ts("2026-08-15T05:30:00"),
        "centroid": [-122.3903, 47.5982],
        "needs_geo": False,
        "caption": "Two-storey wood structure on 35th Ave SW fully engulfed in visible flames.",
        "class_max": 3,
        "key_evidence": True,
        "sector": "A",
    },
    {
        "image_id": "img-002",
        "thumb_path": "thumbs/img-002.jpg",
        "captured_at": _ts("2026-08-15T07:15:00"),
        "centroid": [-122.3899, 47.5979],
        "needs_geo": False,
        "caption": "Flooded intersection on 35th Ave SW with standing water covering the roadway.",
        "class_max": 2,
        "key_evidence": False,
        "sector": "A",
    },
    {
        "image_id": "img-003",
        "thumb_path": "thumbs/img-003.jpg",
        "captured_at": _ts("2026-08-15T04:00:00"),
        "centroid": [-122.3952, 47.5991],
        "needs_geo": False,
        "caption": "Roof has fully collapsed onto the structure below on Oak Ave.",
        "class_max": 3,
        "key_evidence": True,
        "sector": "B",
    },
    {
        "image_id": "img-004",
        "thumb_path": "thumbs/img-004.jpg",
        "captured_at": _ts("2026-08-15T06:45:00"),
        "centroid": [-122.3862, 47.5671],
        "needs_geo": False,
        "caption": "Building entrance near Providence Mount St. Vincent is damaged and partially blocked by debris.",
        "class_max": 2,
        "key_evidence": False,
        "sector": "B",
    },
    {
        "image_id": "img-005",
        "thumb_path": "thumbs/img-005.jpg",
        "captured_at": _ts("2026-08-15T03:00:00"),
        "centroid": [-122.4002, 47.6003],
        "needs_geo": False,
        "caption": "Single-family home on Elm St shows no visible damage.",
        "class_max": 0,
        "key_evidence": False,
        "sector": "C",
    },
    {
        "image_id": "img-006",
        "thumb_path": "thumbs/img-006.jpg",
        "captured_at": _ts("2026-08-15T08:00:00"),
        "centroid": [-122.4003, 47.6001],
        "needs_geo": False,
        "caption": "Minor cracking visible on the exterior wall of a building on Elm St.",
        "class_max": 1,
        "key_evidence": False,
        "sector": "C",
    },
    {
        "image_id": "img-007",
        "thumb_path": "thumbs/img-007.jpg",
        "captured_at": _ts("2026-08-15T09:00:00"),
        "centroid": [-122.3851, 47.6051],
        "needs_geo": False,
        "caption": "Riverside Dialysis Center exterior shows significant structural damage.",
        "class_max": 2,
        "key_evidence": True,
        "sector": "D",
    },
    {
        "image_id": "img-008",
        "thumb_path": "thumbs/img-008.jpg",
        "captured_at": _ts("2026-08-15T05:45:00"),
        "centroid": [-122.3905, 47.5983],
        "needs_geo": False,
        "caption": "Large fire visible with heavy smoke near the residential block on 35th Ave SW.",
        "class_max": 3,
        "key_evidence": True,
        "sector": "A",
    },
    {
        "image_id": "img-009",
        "thumb_path": "thumbs/img-009.jpg",
        "captured_at": _ts("2026-08-15T10:00:00"),
        "centroid": None,
        "needs_geo": True,
        "caption": "Structure appears destroyed; awaiting geolocation.",
        "class_max": 3,
        "key_evidence": True,
        "sector": None,
    },
    {
        "image_id": "img-010",
        "thumb_path": "thumbs/img-010.jpg",
        "captured_at": _ts("2026-08-15T07:30:00"),
        "centroid": [-122.3950, 47.5989],
        "needs_geo": False,
        "caption": "Standing water pools around the foundation on Oak Ave after flooding.",
        "class_max": 1,
        "key_evidence": False,
        "sector": "B",
    },
    {
        "image_id": "img-011",
        "thumb_path": "thumbs/img-011.jpg",
        "captured_at": _ts("2026-08-15T06:00:00"),
        "centroid": [-122.3821, 47.6104],
        "needs_geo": False,
        "caption": "Commercial building on Pine St with roof collapse and scattered debris.",
        "class_max": 3,
        "key_evidence": True,
        "sector": "E",
    },
    {
        "image_id": "img-012",
        "thumb_path": "thumbs/img-012.jpg",
        "captured_at": _ts("2026-08-15T02:00:00"),
        "centroid": [-122.4000, 47.6004],
        "needs_geo": False,
        "caption": "Undamaged park pavilion on Elm St, no structural concerns observed.",
        "class_max": 0,
        "key_evidence": False,
        "sector": "C",
    },
    {
        "image_id": "img-013",
        "thumb_path": "thumbs/img-013.jpg",
        "captured_at": _ts("2026-08-15T06:30:00"),
        "centroid": [-122.3901, 47.5980],
        "needs_geo": False,
        "caption": "Vehicle fire spreading toward an adjacent structure on 35th Ave SW.",
        "class_max": 3,
        "key_evidence": True,
        "sector": "C",
    },
]

# A source record a real privacy gate would withhold (person visible). Never
# indexed by build_fixture_db(); imported directly by tests that must prove
# it stays unreachable.
FIXTURE_WITHHELD_SOURCE = {
    "image_id": "img-withheld-001",
    "thumb_path": "thumbs/img-withheld-001.jpg",
    "captured_at": _ts("2026-08-15T05:35:00"),
    "centroid": [-122.3904, 47.5982],
    "needs_geo": False,
    "caption": "Person visible near a collapsed structure on 35th Ave SW requiring rescue.",
    "class_max": 3,
    "key_evidence": True,
    "sector": "A",
}


def build_fixture_db(conn=None, embedder=None, tag_extractor=None) -> dict:
    """Build (or populate) a local archive_store SQLite connection from
    FIXTURE_ELIGIBLE_ITEMS, running each caption through `tag_extractor`
    (Part 8) for its tags and `embedder` (Part 2) for its embedding, then
    writing it through archive_write.index_cleared_archive_item (Part 9) --
    exactly the real production path, just with fixture data and stub
    model clients. FIXTURE_WITHHELD_SOURCE is intentionally never passed to
    the writer here.

    conn defaults to a fresh in-memory archive_store connection.
    embedder/tag_extractor default to the deterministic offline stubs, so
    calling this from pytest never touches BGE, Lightning, or the network.

    Returns {"conn": ..., "place_index": FixtureLocalPlaceIndex,
    "location_resolver": LocalLocationResolver, "eligible_ids": [...]}.
    """
    active_conn = conn if conn is not None else archive_store.get_connection(":memory:")
    active_embedder = embedder if embedder is not None else DeterministicStubEmbedder()
    active_tag_extractor = tag_extractor if tag_extractor is not None else DeterministicStubTagExtractor()

    captions = [item["caption"] for item in FIXTURE_ELIGIBLE_ITEMS]
    tags_by_index = active_tag_extractor.extract_tags_batch(captions)

    for item, tags in zip(FIXTURE_ELIGIBLE_ITEMS, tags_by_index):
        archive_item = {
            "image_id": item["image_id"],
            "thumb_path": item["thumb_path"],
            "captured_at": item["captured_at"],
            "centroid": item["centroid"],
            "needs_geo": item["needs_geo"],
            "caption": item["caption"],
            "tags": tags,
            "class_max": item["class_max"],
            "key_evidence": item["key_evidence"],
        }
        archive_write.index_cleared_archive_item(
            active_conn, archive_item, active_embedder, eligible=True, sector=item["sector"]
        )

    place_index = FixtureLocalPlaceIndex(FIXTURE_PLACES)
    location_resolver = LocalLocationResolver(place_index)

    return {
        "conn": active_conn,
        "embedder": active_embedder,
        "place_index": place_index,
        "location_resolver": location_resolver,
        "eligible_ids": [item["image_id"] for item in FIXTURE_ELIGIBLE_ITEMS],
    }
