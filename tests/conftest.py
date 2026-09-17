"""Shared fixtures. No network and no Ollama: every vector here is seeded numpy.

The one exception is the golden-migration retrieval check, which needs a real
embedding model to reproduce the recorded answers; it skips itself when no
embedding daemon answers (see :func:`live_embedder`).
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN_DIR = os.environ.get(
    "NANOMEM_GOLDEN_DIR",
    os.path.join(os.path.dirname(ROOT), "scratch", "refound", "golden"))

D = 768


def unit_rows(n, dim=D, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, dim)).astype(np.float32)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


@pytest.fixture
def vault_path(tmp_path):
    return str(tmp_path / "v.dat")


@pytest.fixture
def offline_embedder(monkeypatch):
    """Force the deterministic offline encoder so no test ever touches the network."""
    from nanomem.embed import EmbeddingProvider
    monkeypatch.setattr(EmbeddingProvider, "embed_batch",
                        EmbeddingProvider._offline_encode_batch, raising=True)
    return EmbeddingProvider


def live_embedder():
    """An EmbeddingProvider backed by a real model, or ``None`` if none answers."""
    from nanomem.embed import EmbeddingProvider
    ep = EmbeddingProvider()
    probe = ep.embed_batch(["nanomem embedding probe"])
    fallback = ep._offline_encode_batch(["nanomem embedding probe"])
    if float(np.dot(probe[0], fallback[0])) > 0.999:
        return None
    return ep


def golden(name):
    p = os.path.join(GOLDEN_DIR, name)
    if not os.path.exists(p):
        pytest.skip(f"golden fixture not available: {p}")
    return p
