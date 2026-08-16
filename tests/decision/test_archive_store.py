import numpy as np
import pytest

from backend.decision import archive_store
from backend.decision.archive_store import EMBEDDING_DIM

_ITEM_A = {
    "image_id": "img-a",
    "thumb_path": "thumbs/a.jpg",
    "captured_at": 1_755_234_000.0,
    "centroid": [-122.39, 47.60],
    "needs_geo": False,
    "caption": "Two-storey structure with visible flames.",
    "tags": ["fire"],
    "class_max": 3,
    "key_evidence": True,
}

_ITEM_B_NO_GEO = {
    "image_id": "img-b",
    "thumb_path": "thumbs/b.jpg",
    "captured_at": 1_755_237_600.0,
    "centroid": None,
    "needs_geo": True,
    "caption": "Structure appears destroyed; awaiting geolocation.",
    "tags": [],
    "class_max": 3,
    "key_evidence": True,
}


@pytest.fixture
def conn():
    connection = archive_store.get_connection(":memory:")
    yield connection
    connection.close()


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def test_fresh_connection_has_zero_items(conn):
    assert archive_store.count_items(conn) == 0


def test_upsert_and_get_item_round_trip(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(1))
    item = archive_store.get_item(conn, "img-a")

    assert item == {
        "image_id": "img-a",
        "thumb_path": "thumbs/a.jpg",
        "captured_at": 1_755_234_000.0,
        "centroid": [-122.39, 47.60],
        "needs_geo": False,
        "caption": "Two-storey structure with visible flames.",
        "tags": ["fire"],
        "class_max": 3,
        "key_evidence": True,
    }


def test_upsert_with_null_centroid_round_trips_as_null(conn):
    archive_store.upsert_item(conn, _ITEM_B_NO_GEO, None)
    item = archive_store.get_item(conn, "img-b")
    assert item["centroid"] is None
    assert item["needs_geo"] is True


def test_get_item_missing_returns_none(conn):
    assert archive_store.get_item(conn, "does-not-exist") is None


def test_upsert_replaces_existing_row(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(1))
    updated = dict(_ITEM_A, caption="Updated caption.", class_max=1)
    archive_store.upsert_item(conn, updated, _vector(2))

    assert archive_store.count_items(conn) == 1
    item = archive_store.get_item(conn, "img-a")
    assert item["caption"] == "Updated caption."
    assert item["class_max"] == 1


# embedding dimension validation on write
def test_upsert_rejects_wrong_embedding_dimension(conn):
    bad_vector = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        archive_store.upsert_item(conn, _ITEM_A, bad_vector)


def test_fetch_candidate_rows_returns_decoded_embedding(conn):
    vector = _vector(3)
    archive_store.upsert_item(conn, _ITEM_A, vector)

    rows = archive_store.fetch_candidate_rows(conn)
    assert len(rows) == 1
    embedding = rows[0]["_embedding"]
    assert embedding.shape == (EMBEDDING_DIM,)
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(embedding, vector, rtol=1e-5)


def test_fetch_candidate_rows_missing_embedding_is_none_not_a_crash(conn):
    archive_store.upsert_item(conn, _ITEM_A, None)
    rows = archive_store.fetch_candidate_rows(conn)
    assert rows[0]["_embedding"] is None


# corrupt embedding_dim on read -> excluded, never crashes (validate on read)
def test_fetch_candidate_rows_corrupt_embedding_dim_excluded(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(4))
    conn.execute("UPDATE archive_items SET embedding_dim = ? WHERE image_id = ?", (10, "img-a"))
    conn.commit()

    rows = archive_store.fetch_candidate_rows(conn)
    assert rows[0]["_embedding"] is None


def test_update_item_fields_only_changes_supplied_fields(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(5))
    archive_store.update_item_fields(conn, "img-a", caption="New caption only.")

    item = archive_store.get_item(conn, "img-a")
    assert item["caption"] == "New caption only."
    assert item["tags"] == ["fire"]
    assert item["class_max"] == 3


def test_update_item_fields_can_replace_embedding(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(6))
    new_vector = _vector(7)
    archive_store.update_item_fields(conn, "img-a", embedding=new_vector)

    rows = archive_store.fetch_candidate_rows(conn)
    np.testing.assert_allclose(rows[0]["_embedding"], new_vector, rtol=1e-5)


def test_update_item_fields_no_args_is_a_noop(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(8))
    archive_store.update_item_fields(conn, "img-a")
    item = archive_store.get_item(conn, "img-a")
    assert item["caption"] == _ITEM_A["caption"]


# structured filter: parameterized SQL (test 10) -- a hostile-looking value
# in a filter parameter is bound, never string-interpolated, so it can
# never break out of the query or corrupt the table.
def test_fetch_candidate_rows_filter_values_are_parameterized_not_interpolated(conn):
    hostile_item = dict(_ITEM_A, image_id="img-hostile")
    archive_store.upsert_item(conn, hostile_item, _vector(9), sector="C'; DROP TABLE archive_items;--")

    rows = archive_store.fetch_candidate_rows(conn, filters=[("sector = ?", "C'; DROP TABLE archive_items;--")])
    assert len(rows) == 1
    assert rows[0]["image_id"] == "img-hostile"
    assert archive_store.count_items(conn) == 1  # table still exists and intact


def test_fetch_candidate_rows_class_max_filter_narrows(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(10))
    other = dict(_ITEM_A, image_id="img-c", class_max=0)
    archive_store.upsert_item(conn, other, _vector(11))

    rows = archive_store.fetch_candidate_rows(conn, filters=[("class_max = ?", 3)])
    assert {r["image_id"] for r in rows} == {"img-a"}


# location bbox: centroid=null never matches (Part 5 invariant)
def test_fetch_candidate_rows_bbox_excludes_null_centroid(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(12))
    archive_store.upsert_item(conn, _ITEM_B_NO_GEO, None)

    bbox = (-123.0, 47.0, -122.0, 48.0)  # wide box that would include img-a's centroid
    rows = archive_store.fetch_candidate_rows(conn, bbox=bbox)
    assert {r["image_id"] for r in rows} == {"img-a"}


def test_fetch_candidate_rows_bbox_excludes_out_of_range_points(conn):
    archive_store.upsert_item(conn, _ITEM_A, _vector(13))  # centroid [-122.39, 47.60]
    far_away = dict(_ITEM_A, image_id="img-far", centroid=[10.0, 10.0])
    archive_store.upsert_item(conn, far_away, _vector(14))

    bbox = (-123.0, 47.0, -122.0, 48.0)
    rows = archive_store.fetch_candidate_rows(conn, bbox=bbox)
    assert {r["image_id"] for r in rows} == {"img-a"}
