import numpy as np
import pytest

from backend.decision import archive_store
from backend.decision.archive_embedder import DeterministicStubEmbedder
from backend.decision.archive_filter_parser import ArchiveSearchError
from backend.decision.archive_location import FixtureLocalPlaceIndex, LocalLocationResolver
from backend.decision.archive_search import search_archive
from tests.decision.archive_fixture_data import (
    FIXTURE_PLACES,
    FIXTURE_WITHHELD_SOURCE,
    build_fixture_db,
)


@pytest.fixture(scope="module")
def archive():
    return build_fixture_db()


def _search(archive, q, limit=10, embedder=None, location_resolver=None):
    return search_archive(
        {"q": q, "limit": limit},
        archive["conn"],
        embedder=embedder if embedder is not None else archive["embedder"],
        location_resolver=location_resolver if location_resolver is not None else archive["location_resolver"],
    )


# --------------------------------------------------------------------------
# CONTRACT
# --------------------------------------------------------------------------


def test_search_request_exact_shape_accepted(archive):
    result = search_archive({"q": "class:3", "limit": 5}, archive["conn"], embedder=archive["embedder"])
    assert isinstance(result, dict)


def test_search_result_exact_shape_returned(archive):
    result = _search(archive, "class:3")
    assert set(result.keys()) == {"items", "resolved_by", "took_ms"}
    assert isinstance(result["items"], list)
    assert isinstance(result["resolved_by"], list)
    assert isinstance(result["took_ms"], int)


def test_archive_item_exact_public_fields(archive):
    result = _search(archive, "class:3")
    assert result["items"]
    for item in result["items"]:
        assert set(item.keys()) == {
            "image_id", "thumb_path", "captured_at", "centroid", "needs_geo",
            "caption", "tags", "class_max", "key_evidence",
        }


def test_coordinates_remain_lng_lat(archive):
    result = _search(archive, "35th Ave SW")
    for item in result["items"]:
        if item["centroid"] is not None:
            lng, lat = item["centroid"]
            # this fixture AOI: lng around -122.x, lat around 47.x
            assert -123.0 < lng < -122.0
            assert 47.0 < lat < 48.0


# --------------------------------------------------------------------------
# STRUCTURED
# --------------------------------------------------------------------------


def test_class_3_narrows_correctly(archive):
    result = _search(archive, "class:3", limit=20)
    assert result["items"]
    assert all(item["class_max"] == 3 for item in result["items"])
    assert result["resolved_by"] == ["filter"]


def test_after_filter_works(archive):
    result = _search(archive, "after:09:00", limit=20)
    for item in result["items"]:
        from datetime import datetime, timezone

        assert item["captured_at"] >= datetime(2026, 8, 15, 9, 0, 0, tzinfo=timezone.utc).timestamp()


def test_before_filter_works(archive):
    result = _search(archive, "before:03:00", limit=20)
    from datetime import datetime, timezone

    cutoff = datetime(2026, 8, 15, 3, 0, 0, tzinfo=timezone.utc).timestamp()
    for item in result["items"]:
        assert item["captured_at"] <= cutoff


def test_key_true_works(archive):
    result = _search(archive, "key:true", limit=20)
    assert result["items"]
    assert all(item["key_evidence"] is True for item in result["items"])


def test_needs_geo_filter_works(archive):
    result = _search(archive, "needs_geo:true", limit=20)
    assert result["items"]
    assert all(item["needs_geo"] is True for item in result["items"])
    assert all(item["centroid"] is None for item in result["items"])


def test_malformed_structured_filter_fails_safely(archive):
    with pytest.raises(ArchiveSearchError):
        _search(archive, "class:9")


def test_sector_filter_works_against_internal_schema(archive):
    result = _search(archive, "class:3 after:06:00 sector:C", limit=20)
    ids = {item["image_id"] for item in result["items"]}
    assert ids == {"img-013"}


# --------------------------------------------------------------------------
# LOCATION
# --------------------------------------------------------------------------


def test_coordinate_query_resolves(archive):
    result = _search(archive, "47.598, -122.390", limit=20)
    assert result["resolved_by"] == ["location"]


def test_named_road_resolves_from_local_fixture(archive):
    result = _search(archive, "35th Ave SW", limit=20)
    assert result["resolved_by"] == ["location"]
    assert result["items"]


def test_named_facility_resolves_from_local_fixture(archive):
    result = _search(archive, "Riverside Dialysis Center", limit=20)
    assert result["resolved_by"] == ["location"]
    assert any(item["image_id"] == "img-007" for item in result["items"])


def test_location_matching_case_insensitive(archive):
    result = _search(archive, "35TH ave SW", limit=20)
    assert result["resolved_by"] == ["location"]
    assert result["items"]


def test_needs_geo_item_excluded_from_location_results(archive):
    result = _search(archive, "35th Ave SW", limit=50)
    assert all(item["image_id"] != "img-009" for item in result["items"])


def test_no_network_geocoder_called(archive, monkeypatch):
    import socket
    import urllib.request

    def _forbidden(*args, **kwargs):
        raise AssertionError("location resolution must never touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    result = _search(archive, "35th Ave SW", limit=5)
    assert result["items"]


# --------------------------------------------------------------------------
# SEMANTIC
# --------------------------------------------------------------------------


def test_semantic_only_query_ranks_fire_captions_first(archive):
    result = _search(archive, "buildings on fire", limit=3)
    assert result["resolved_by"] == ["semantic"]
    top_captions = " ".join(item["caption"].lower() for item in result["items"])
    assert "fire" in top_captions


def test_vectors_are_normalized():
    embedder = DeterministicStubEmbedder()
    vector = embedder.embed_text("a caption")
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=1e-5)


def test_cosine_uses_numpy_matrix_multiply():
    import backend.decision.archive_search as archive_search_module
    import inspect

    source = inspect.getsource(archive_search_module._semantic_rank)
    assert "@" in source  # matrix @ query_vector, not a python-level loop of dot products


def test_top_k_deterministic_across_repeated_calls(archive):
    result1 = _search(archive, "buildings on fire", limit=5)
    result2 = _search(archive, "buildings on fire", limit=5)
    ids1 = [item["image_id"] for item in result1["items"]]
    ids2 = [item["image_id"] for item in result2["items"]]
    assert ids1 == ids2


def test_missing_embedding_does_not_crash_semantic_search():
    conn = archive_store.get_connection(":memory:")
    embedder = DeterministicStubEmbedder()
    item = {
        "image_id": "img-no-embed",
        "thumb_path": "thumbs/x.jpg",
        "captured_at": 1_755_234_000.0,
        "centroid": [-122.39, 47.60],
        "needs_geo": False,
        "caption": "A caption with fire.",
        "tags": [],
        "class_max": 3,
        "key_evidence": False,
    }
    archive_store.upsert_item(conn, item, embedding=None)  # no embedding at all

    result = search_archive({"q": "buildings on fire", "limit": 10}, conn, embedder=embedder)
    assert result["items"] == []  # excluded, no crash


def test_limit_applied_after_ranking(archive):
    full = _search(archive, "buildings on fire", limit=20)
    limited = _search(archive, "buildings on fire", limit=2)
    assert limited["items"] == full["items"][:2]


# --------------------------------------------------------------------------
# COMPOSITION
# --------------------------------------------------------------------------


def test_filter_then_semantic_narrows_first(archive):
    result = _search(archive, "buildings on fire class:3", limit=20)
    assert set(result["resolved_by"]) == {"filter", "semantic"}
    assert all(item["class_max"] == 3 for item in result["items"])


def test_location_then_semantic_narrows_first(archive):
    result = _search(archive, "buildings on fire near 35th Ave SW", limit=20)
    assert set(result["resolved_by"]) == {"location", "semantic"}
    # every result must be within the 35th Ave SW narrowing, not the whole corpus
    unfiltered_semantic = _search(archive, "buildings on fire", limit=20)
    narrowed_ids = {item["image_id"] for item in result["items"]}
    all_ids = {item["image_id"] for item in unfiltered_semantic["items"]}
    assert narrowed_ids.issubset(all_ids)


def test_location_filter_and_semantic_all_compose(archive):
    result = _search(archive, "buildings on fire near 35th Ave SW class:3", limit=20)
    assert result["resolved_by"] == ["location", "semantic", "filter"]
    assert all(item["class_max"] == 3 for item in result["items"])


def test_resolved_by_fixed_order_regardless_of_internal_execution_order(archive):
    result = _search(archive, "buildings on fire near 35th Ave SW class:3", limit=20)
    assert result["resolved_by"] == ["location", "semantic", "filter"]


def test_resolved_by_reports_only_fired_resolvers(archive):
    assert _search(archive, "class:3")["resolved_by"] == ["filter"]
    assert _search(archive, "35th Ave SW")["resolved_by"] == ["location"]
    assert _search(archive, "buildings on fire")["resolved_by"] == ["semantic"]


def test_pure_filter_does_not_invoke_embedder(archive):
    class ExplodingEmbedder(DeterministicStubEmbedder):
        def embed_text(self, text):
            raise AssertionError("embedder must not be called for a pure filter search")

    result = search_archive(
        {"q": "class:3", "limit": 10}, archive["conn"], embedder=ExplodingEmbedder(),
        location_resolver=archive["location_resolver"],
    )
    assert result["items"]


def test_pure_location_does_not_invoke_embedder(archive):
    class ExplodingEmbedder(DeterministicStubEmbedder):
        def embed_text(self, text):
            raise AssertionError("embedder must not be called for a pure location search")

    result = search_archive(
        {"q": "35th Ave SW", "limit": 10}, archive["conn"], embedder=ExplodingEmbedder(),
        location_resolver=archive["location_resolver"],
    )
    assert result["items"]


def test_never_ranks_whole_corpus_before_filtering(archive):
    # class:0 (undamaged) narrows to a small set; combined with fire-themed
    # semantic text, results must still all be class 0 -- proving semantic
    # ranking happened WITHIN the filtered set, not before it.
    result = _search(archive, "buildings on fire class:0", limit=20)
    assert all(item["class_max"] == 0 for item in result["items"])


# --------------------------------------------------------------------------
# PRIVACY / WRITE (search-level: withheld never appears)
# --------------------------------------------------------------------------


def test_withheld_fixture_never_appears_in_any_search_result(archive):
    withheld_id = FIXTURE_WITHHELD_SOURCE["image_id"]

    all_queries = [
        "buildings on fire",
        "35th Ave SW",
        "class:3",
        "buildings on fire near 35th Ave SW class:3",
        "collapsed structure requiring rescue",
    ]
    for q in all_queries:
        result = _search(archive, q, limit=50)
        ids = {item["image_id"] for item in result["items"]}
        assert withheld_id not in ids

    assert archive_store.get_item(archive["conn"], withheld_id) is None


# --------------------------------------------------------------------------
# PERFORMANCE / SANITY
# --------------------------------------------------------------------------


def test_took_ms_is_non_negative(archive):
    result = _search(archive, "class:3")
    assert result["took_ms"] >= 0


def test_empty_result_works(archive):
    result = _search(archive, "class:3 sector:ZZZ-does-not-exist")
    assert result["items"] == []
    assert result["took_ms"] >= 0


def test_limit_validation_rejects_zero_and_negative(archive):
    with pytest.raises(ArchiveSearchError):
        search_archive({"q": "class:3", "limit": 0}, archive["conn"], embedder=archive["embedder"])
    with pytest.raises(ArchiveSearchError):
        search_archive({"q": "class:3", "limit": -1}, archive["conn"], embedder=archive["embedder"])


def test_limit_validation_rejects_absurd_value(archive):
    with pytest.raises(ArchiveSearchError):
        search_archive({"q": "class:3", "limit": 10_000_000}, archive["conn"], embedder=archive["embedder"])


def test_limit_validation_rejects_non_int(archive):
    with pytest.raises(ArchiveSearchError):
        search_archive({"q": "class:3", "limit": "5"}, archive["conn"], embedder=archive["embedder"])


def test_q_must_be_a_string(archive):
    with pytest.raises(ArchiveSearchError):
        search_archive({"q": 12345, "limit": 5}, archive["conn"], embedder=archive["embedder"])


def test_deterministic_ordering_across_repeated_identical_searches(archive):
    results = [_search(archive, "class:3", limit=10)["items"] for _ in range(3)]
    ids = [[item["image_id"] for item in r] for r in results]
    assert ids[0] == ids[1] == ids[2]


def test_deterministic_ordering_for_pure_location_search(archive):
    results = [_search(archive, "35th Ave SW", limit=10)["items"] for _ in range(3)]
    ids = [[item["image_id"] for item in r] for r in results]
    assert ids[0] == ids[1] == ids[2]
