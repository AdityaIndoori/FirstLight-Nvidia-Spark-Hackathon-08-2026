import numpy as np
import pytest

from backend.decision.archive_embedder import EMBEDDING_DIM, BgeSmallEmbedder, DeterministicStubEmbedder


def test_embedding_dim_is_384():
    assert EMBEDDING_DIM == 384


def test_stub_embedder_output_shape_and_dtype():
    vector = DeterministicStubEmbedder().embed_text("a caption about a fire")
    assert vector.shape == (EMBEDDING_DIM,)
    assert vector.dtype == np.float32


def test_stub_embedder_output_is_l2_normalized():
    vector = DeterministicStubEmbedder().embed_text("two-storey structure with visible flames")
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=1e-5)


def test_stub_embedder_empty_text_is_still_normalized_and_does_not_crash():
    vector = DeterministicStubEmbedder().embed_text("")
    assert vector.shape == (EMBEDDING_DIM,)
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=1e-5)


def test_stub_embedder_is_deterministic():
    embedder = DeterministicStubEmbedder()
    v1 = embedder.embed_text("buildings on fire")
    v2 = embedder.embed_text("buildings on fire")
    np.testing.assert_array_equal(v1, v2)


def test_stub_embedder_different_text_gives_different_vector():
    embedder = DeterministicStubEmbedder()
    v1 = embedder.embed_text("buildings on fire")
    v2 = embedder.embed_text("flooded intersection")
    assert not np.allclose(v1, v2)


# content-sensitivity: shared vocabulary should cosine-score higher than
# disjoint vocabulary -- the property archive_search's ranking depends on.
def test_stub_embedder_shared_vocabulary_scores_higher_than_disjoint():
    embedder = DeterministicStubEmbedder()
    query = embedder.embed_text("buildings on fire")
    fire_caption = embedder.embed_text("large fire visible with heavy smoke")
    flood_caption = embedder.embed_text("flooded intersection with standing water")

    fire_score = float(query @ fire_caption)
    flood_score = float(query @ flood_caption)
    assert fire_score > flood_score


# BGE: never loaded just by importing the module; construction fails
# loudly (never downloads) when the model isn't cached locally, which it
# is not on this machine (no sentence_transformers installed, no local
# HuggingFace cache for BAAI/bge-small-en-v1.5 -- see the live-check
# script for the real-model path).
def test_bge_embedder_import_alone_does_not_require_sentence_transformers():
    import backend.decision.archive_embedder  # noqa: F401 -- import must not raise


def test_bge_embedder_raises_runtime_error_when_model_unavailable():
    with pytest.raises(RuntimeError):
        BgeSmallEmbedder()
