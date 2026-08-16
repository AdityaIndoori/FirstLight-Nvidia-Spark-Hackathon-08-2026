"""BGE-small text embedder for caption and query vectors, pinned to CPU.

WHY CPU, not "we did not get around to it": with the three vLLM pools resident
(Nano 0.25 + Lightning 0.35 + VL 0.22, about 98 GB observed) the GPU allocator
is full and loading the embedder on CUDA OOMs. Verified on the box. Eight
captions embed in milliseconds on CPU at this corpus size, so the pin costs
nothing and removes a class of demo-day failure.

WHY a hash fallback exists: archive search is a gate-7 deliverable and must
still return something ordered when sentence-transformers or its weights are
absent. The fallback is a signed hashing-trick bag of words, deterministic and
labelled "stub-hash-v1", so the status strip can say "stub engaged" instead of
the UI silently pretending a semantic model ran.

Vectors are L2-normalized float32, so cosine similarity is a plain dot product
and the archive can rank a few thousand rows with one numpy matmul.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading

import numpy as np

from . import config

# PUBLIC API
# encode(texts: list[str]) -> np.ndarray        (n, dim) float32, L2-normalized
# dim() -> int
# model_version() -> str                        "bge-small-en-v1.5" or "stub-hash-v1"
# available() -> bool                           True when the real model loaded
# force_stub(on: bool = True) -> None           deterministic path for tests and reseeds
# status() -> dict                              for the HUD status strip

STUB_VERSION = "stub-hash-v1"

_TOKEN = re.compile(r"[a-z0-9]+")
# Dropped so a query like "buildings on fire" ranks on the words that carry the
# signal. Tiny on purpose: an aggressive stoplist throws away caption nouns.
_STOP = frozenset(
    "a an and are as at be by for from in is it its of on or that the this to with near"
    .split()
)

_lock = threading.Lock()
_model = None
_version = ""
_dim = int(config.EMBED_DIM)
_load_error: str | None = None
_forced_stub = bool(os.environ.get("FIRSTLIGHT_EMBED_STUB"))


def force_stub(on: bool = True) -> None:
    """Pin the deterministic hash path. Tests and the demo reseed use this so
    search ordering does not depend on whether the weights are on the box."""
    global _forced_stub, _model, _version, _dim, _load_error
    with _lock:
        _forced_stub = bool(on)
        _model = None
        _version = ""
        _dim = int(config.EMBED_DIM)
        _load_error = None


def _load():
    """Lazy singleton. Loading is the slow part, so it happens once, under a
    lock, and never at import time: importing this module must not cost 0.4 GB."""
    global _model, _version, _dim, _load_error
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        if not _forced_stub:
            try:
                from sentence_transformers import SentenceTransformer

                real = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
                got = real.get_sentence_embedding_dimension()
                _dim = int(got or config.EMBED_DIM)
                _version = config.EMBED_MODEL.rsplit("/", 1)[-1]
                _model = real
                return _model
            except Exception as exc:  # missing package, missing weights, no torch
                _load_error = f"{type(exc).__name__}: {exc}"
        _dim = int(config.EMBED_DIM)
        _version = STUB_VERSION
        _model = _HashEmbedder(_dim)
        return _model


class _HashEmbedder:
    """Signed hashing trick over word tokens.

    Not a semantic model and never claims to be. It does give a stable, ordered
    similarity: captions sharing words with the query score above those that do
    not, which keeps the search panel functional with zero dependencies.
    """

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _TOKEN.findall((text or "").lower()):
                if tok in _STOP:
                    continue
                n = int.from_bytes(
                    hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big"
                )
                sign = 1.0 if (n >> 17) & 1 else -1.0
                out[row, n % self.dim] += sign
                # A second bucket per token spreads collisions without needing
                # a bigger vector, since the dim is fixed by the real model.
                out[row, (n >> 32) % self.dim] += sign * 0.5
        return out


def encode(texts: list[str]) -> np.ndarray:
    """Return (len(texts), dim) float32 rows, each L2-normalized.

    An all-zero row (empty text, or every token a stopword) stays zero rather
    than becoming a division by zero: it then scores 0.0 against every query,
    which is the honest answer for a caption with no content.
    """
    items = [t if isinstance(t, str) else "" for t in (texts or [])]
    if not items:
        return np.zeros((0, dim()), dtype=np.float32)
    model = _load()
    vecs = np.asarray(model.encode(items), dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    np.divide(vecs, norms, out=vecs, where=norms > 0)
    return vecs


def dim() -> int:
    _load()
    return int(_dim)


def model_version() -> str:
    _load()
    return _version


def available() -> bool:
    """True when the real embedder is the one answering."""
    _load()
    return _version != STUB_VERSION


def status() -> dict:
    _load()
    return {
        "embedder": _version,
        "dim": int(_dim),
        "device": config.EMBED_DEVICE,
        "stub": _version == STUB_VERSION,
        "load_error": _load_error,
    }


__all__ = ["STUB_VERSION", "available", "dim", "encode", "force_stub", "model_version", "status"]
