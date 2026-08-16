"""B7 Part 2: embedding abstraction for archive captions and search queries.

Production embedder: BAAI/bge-small-en-v1.5, 384 dimensions, local CPU
inference, L2-normalized float32 output (BgeSmallEmbedder). Tests never
load it -- DeterministicStubEmbedder is offline, deterministic, and
content-sensitive enough to exercise real ranking behavior without any
model, network call, or download. Both implement the same Embedder
interface, so archive_search.py/archive_write.py never change based on
which is active.

Nothing here imports sentence_transformers/torch at module load time --
only BgeSmallEmbedder.__init__ does, and only when that class is actually
constructed, so importing this module (and running normal pytest, which
only ever constructs DeterministicStubEmbedder) never touches those
optional dependencies or the network.
"""

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

EMBEDDING_DIM = 384

_BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "near",
        "to", "from", "is", "are", "was", "were", "be", "been", "no", "not",
        "this", "that", "visible", "shows", "showing", "appears",
    }
)
"""Excluded from DeterministicStubEmbedder's bag-of-tokens -- without this,
common short function words (e.g. "on", as in most fixture captions'
"... on 35th Ave SW" phrasing) dominate the summed vector and drown out
the actual content words a query is looking for. Real BGE embeddings have
no such problem (attention already weighs content over function words);
this list exists only because the stub's hashing trick has no equivalent
notion of term importance."""


class Embedder(ABC):
    """Boundary: text -> a single (EMBEDDING_DIM,) float32 L2-normalized
    vector. Implementations must never mutate their input.
    """

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError


def _tokenize(text: str) -> list:
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS]


def _token_vector(token: str) -> np.ndarray:
    """Deterministic pseudo-random unit-ish vector for one token, seeded
    from a stable hash of the token text -- same token always produces the
    same vector, on any machine, any run, forever.
    """
    seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.normal(size=EMBEDDING_DIM)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        result = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        result[0] = 1.0
        return result
    return vector / norm


class DeterministicStubEmbedder(Embedder):
    """Offline, deterministic, VOCABULARY-SENSITIVE stand-in for a real
    semantic embedder -- NOT a substitute for BGE's actual quality, just
    good enough to make ranking tests meaningful without ML.

    Bag-of-hashed-tokens: each distinct lowercase alphanumeric token
    deterministically seeds a fixed pseudo-random direction in
    EMBEDDING_DIM-space (see _token_vector); the text's embedding is the
    L2-normalized sum of its tokens' vectors. Two captions sharing
    vocabulary get a meaningfully higher cosine similarity than two
    captions that share none -- e.g. "fire" appearing in both a query and
    a caption pulls their vectors together -- which is exactly the property
    archive_search.py's ranking tests exercise. Never random across runs
    (no unseeded RNG), never networked, never loads a model.
    """

    is_stub = True

    def embed_text(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        if not tokens:
            vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
            vector[0] = 1.0
        else:
            vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
            for token in tokens:
                vector += _token_vector(token)
            vector = _l2_normalize(vector)
        return vector.astype(np.float32)


class BgeSmallEmbedder(Embedder):
    """Real BAAI/bge-small-en-v1.5 embedder: local CPU inference only,
    384-dim, L2-normalized float32 output.

    Lazy import: sentence_transformers/torch are imported INSIDE __init__,
    never at module scope, so importing archive_embedder.py (which every
    B7 module does) never requires those packages to be installed. This
    class never downloads a model -- SentenceTransformer is constructed
    with local_files_only=True, so if BAAI/bge-small-en-v1.5 is not already
    in the local HuggingFace cache, construction raises RuntimeError with a
    clear message instead of reaching the network. Use
    scripts/archive_search_live_check.py to check local availability and
    exercise the real model; normal pytest never constructs this class.
    """

    is_stub = False
    MODEL_NAME = _BGE_MODEL_NAME

    def __init__(self, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence_transformers is not installed -- BgeSmallEmbedder requires it "
                "(and the model already cached locally). Use DeterministicStubEmbedder "
                "for offline work, or install sentence_transformers and cache the model "
                "once with network access before running scripts/archive_search_live_check.py."
            ) from exc

        try:
            self._model = SentenceTransformer(self.MODEL_NAME, device=device, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"{self.MODEL_NAME} is not cached locally (local_files_only=True was "
                "enforced -- this class never downloads a model at search time). Cache it "
                "once with network access first, or use DeterministicStubEmbedder."
            ) from exc

    def embed_text(self, text: str) -> np.ndarray:
        vector = self._model.encode(text, normalize_embeddings=True)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIM,):
            raise RuntimeError(
                f"{self.MODEL_NAME} produced dimension {vector.shape}, expected ({EMBEDDING_DIM},)"
            )
        return vector
