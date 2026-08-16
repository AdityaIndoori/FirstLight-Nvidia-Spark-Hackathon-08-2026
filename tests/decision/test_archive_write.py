import pytest

from backend.decision import archive_store
from backend.decision.archive_embedder import DeterministicStubEmbedder
from backend.decision.archive_write import ArchiveWriteError, index_cleared_archive_item, reindex_archive_item
from tests.decision.archive_fixture_data import FIXTURE_WITHHELD_SOURCE

_VALID_ITEM = {
    "image_id": "img-w1",
    "thumb_path": "thumbs/w1.jpg",
    "captured_at": 1_755_234_000.0,
    "centroid": [-122.39, 47.60],
    "needs_geo": False,
    "caption": "Two-storey structure with visible flames.",
    "tags": ["fire"],
    "class_max": 3,
    "key_evidence": True,
}


@pytest.fixture
def conn():
    connection = archive_store.get_connection(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def embedder():
    return DeterministicStubEmbedder()


# 37: cleared eligible archive item can be indexed
def test_eligible_true_indexes_item(conn, embedder):
    index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=True)
    item = archive_store.get_item(conn, "img-w1")
    assert item is not None
    assert item["caption"] == _VALID_ITEM["caption"]


def test_index_computes_embedding_via_the_embedder_abstraction(conn, embedder):
    index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=True)
    rows = archive_store.fetch_candidate_rows(conn)
    assert rows[0]["_embedding"] is not None


# 38: withheld/ineligible item cannot be indexed
def test_eligible_false_refuses_to_index(conn, embedder):
    withheld_item = {
        "image_id": FIXTURE_WITHHELD_SOURCE["image_id"],
        "thumb_path": FIXTURE_WITHHELD_SOURCE["thumb_path"],
        "captured_at": FIXTURE_WITHHELD_SOURCE["captured_at"],
        "centroid": FIXTURE_WITHHELD_SOURCE["centroid"],
        "needs_geo": FIXTURE_WITHHELD_SOURCE["needs_geo"],
        "caption": FIXTURE_WITHHELD_SOURCE["caption"],
        "tags": [],
        "class_max": FIXTURE_WITHHELD_SOURCE["class_max"],
        "key_evidence": FIXTURE_WITHHELD_SOURCE["key_evidence"],
    }
    with pytest.raises(ArchiveWriteError):
        index_cleared_archive_item(conn, withheld_item, embedder, eligible=False)

    assert archive_store.get_item(conn, FIXTURE_WITHHELD_SOURCE["image_id"]) is None
    assert archive_store.count_items(conn) == 0


def test_eligible_none_refuses_to_index(conn, embedder):
    with pytest.raises(ArchiveWriteError):
        index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=None)
    assert archive_store.count_items(conn) == 0


# deliberately strict: only the literal True is accepted, not a truthy stand-in
def test_eligible_truthy_non_bool_still_refused(conn, embedder):
    with pytest.raises(ArchiveWriteError):
        index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible="true")
    assert archive_store.count_items(conn) == 0


def test_eligible_missing_raises_type_error(conn, embedder):
    with pytest.raises(TypeError):
        index_cleared_archive_item(conn, _VALID_ITEM, embedder)


def test_invalid_archive_item_shape_refused(conn, embedder):
    bad_item = dict(_VALID_ITEM, caption="")
    with pytest.raises(ArchiveWriteError):
        index_cleared_archive_item(conn, bad_item, embedder, eligible=True)
    assert archive_store.count_items(conn) == 0


def test_sector_stored_internally_only(conn, embedder):
    index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=True, sector="C")
    item = archive_store.get_item(conn, "img-w1")
    assert "sector" not in item  # never a public ArchiveItem field


# --------------------------------------------------------------------------
# Part 10: edit/reindex
# --------------------------------------------------------------------------


def test_reindex_updates_caption_and_regenerates_embedding(conn, embedder):
    index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=True)
    rows_before = archive_store.fetch_candidate_rows(conn)
    embedding_before = rows_before[0]["_embedding"]

    updated = reindex_archive_item(conn, "img-w1", embedder, caption="Flooded intersection with standing water.")

    assert updated["caption"] == "Flooded intersection with standing water."
    rows_after = archive_store.fetch_candidate_rows(conn)
    embedding_after = rows_after[0]["_embedding"]
    assert not (embedding_before == embedding_after).all()


def test_reindex_without_caption_leaves_embedding_untouched(conn, embedder):
    index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=True)
    embedding_before = archive_store.fetch_candidate_rows(conn)[0]["_embedding"]

    reindex_archive_item(conn, "img-w1", embedder, key_evidence=False)

    embedding_after = archive_store.fetch_candidate_rows(conn)[0]["_embedding"]
    import numpy as np

    np.testing.assert_array_equal(embedding_before, embedding_after)


def test_reindex_updates_tags(conn, embedder):
    index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=True)
    updated = reindex_archive_item(conn, "img-w1", embedder, tags=["fire", "roof damage"])
    assert updated["tags"] == ["fire", "roof damage"]


def test_reindex_updates_key_evidence(conn, embedder):
    index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=True)
    updated = reindex_archive_item(conn, "img-w1", embedder, key_evidence=False)
    assert updated["key_evidence"] is False


# edit is never a second ingest path -- cannot create a row for something
# never indexed (i.e. never cleared)
def test_reindex_on_never_indexed_item_raises_and_creates_nothing(conn, embedder):
    with pytest.raises(ArchiveWriteError):
        reindex_archive_item(conn, "does-not-exist", embedder, caption="New caption.")
    assert archive_store.count_items(conn) == 0


def test_reindex_on_withheld_image_id_raises(conn, embedder):
    with pytest.raises(ArchiveWriteError):
        reindex_archive_item(
            conn, FIXTURE_WITHHELD_SOURCE["image_id"], embedder, caption="Attempted sneak edit."
        )
    assert archive_store.get_item(conn, FIXTURE_WITHHELD_SOURCE["image_id"]) is None


def test_reindex_invalid_tags_type_raises(conn, embedder):
    index_cleared_archive_item(conn, _VALID_ITEM, embedder, eligible=True)
    with pytest.raises(ArchiveWriteError):
        reindex_archive_item(conn, "img-w1", embedder, tags="not-a-list")
