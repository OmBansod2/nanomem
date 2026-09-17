"""The embedding model is chosen by the caller, and so is its width.

Through 0.6.1 `EmbeddingProvider.dim` was the constant 768. `Vault` sizes its
engine from that number, so `embed_model=` was a parameter you could pass and
not a model you could use:

    Vault(embed_model="all-minilm")     # a real 384-d model
    -> ValueError: embedding has 384 dims, vault has 768   on the first add()

The container was never the limitation -- it stores whatever width it is given,
and an existing file's header width always wins. Only the constant was.
"""

import numpy as np
import pytest

from nanomem.embed import OFFLINE_DIM, EmbeddingProvider
from nanomem.engine import VaultEngine
from nanomem.vault import Vault


def test_an_explicit_dim_is_used_and_skips_the_probe():
    """Also the air-gapped path: no network call is made at all."""
    ep = EmbeddingProvider(model="anything", dim=1536)
    assert ep.dim == 1536
    ep2 = EmbeddingProvider(model="anything", dim=384)
    assert ep2.dim == 384


def test_the_probe_falls_back_to_the_offline_width(monkeypatch):
    """No daemon reachable -> the deterministic encoder answers, and it is 768."""
    monkeypatch.setattr(EmbeddingProvider, "_remote_embed_batch",
                        lambda self, texts: None)
    ep = EmbeddingProvider(model="unreachable")
    assert ep.dim == OFFLINE_DIM
    assert ep.embed_batch(["a", "b"]).shape == (2, OFFLINE_DIM)


def test_a_server_width_is_adopted(monkeypatch):
    """Whatever the model returns is the width, 768 or not."""
    def fake(self, texts):
        v = np.ones((len(texts), 384), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    monkeypatch.setattr(EmbeddingProvider, "_remote_embed_batch", fake)
    assert EmbeddingProvider(model="all-minilm").dim == 384


def test_embed_batch_caches_the_width_it_observed(monkeypatch):
    """A first embed_batch must set the width, so nothing probes twice."""
    calls = []

    def fake(self, texts):
        calls.append(len(texts))
        v = np.ones((len(texts), 512), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    monkeypatch.setattr(EmbeddingProvider, "_remote_embed_batch", fake)
    ep = EmbeddingProvider(model="m")
    ep.embed_batch(["one", "two"])
    assert ep._dim == 512
    before = len(calls)
    assert ep.dim == 512
    assert len(calls) == before, "reading .dim after a batch must not re-probe"


def test_the_offline_encoder_keeps_its_own_width(monkeypatch):
    """Its width is a property of the encoder, not of whatever was probed."""
    monkeypatch.setattr(EmbeddingProvider, "_remote_embed_batch",
                        lambda self, texts: None)
    ep = EmbeddingProvider(model="m", dim=1536)
    assert ep.dim == 1536
    assert ep._offline_encode_batch(["x"]).shape == (1, OFFLINE_DIM)


def test_a_vault_is_sized_from_the_chosen_model(vault_path, monkeypatch):
    """A 384-d model end to end: sized, written, searched, reopened.

    The stub hashes with hashlib, not builtins.hash: string hashing is salted
    per process, so an hash()-seeded stub gives the query and the document
    unrelated vectors, and a negative cosine is then dropped by the default
    min_score=0.0. That made the first version of this test pass alone and fail
    in the suite, which is a property of the test and not of the engine.
    """
    import hashlib

    def fake(self, texts):
        out = np.empty((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            out[i] = np.random.default_rng(seed).normal(size=384)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    monkeypatch.setattr(EmbeddingProvider, "_remote_embed_batch", fake)
    text = "The staging database listens on port 5433."
    v = Vault(vault_path, embed_model="all-minilm")
    assert v.engine.embed_dim == 384
    v.add(text)
    v.flush()
    hits = v.search(text, top_k=1)                 # identical text -> cosine 1
    assert hits and hits[0]["text"] == text
    assert hits[0]["cosine"] == pytest.approx(1.0, abs=1e-3)
    v.close()

    v2 = Vault(vault_path, embed_model="all-minilm")
    assert v2.engine.embed_dim == 384 and v2.engine.count() == 1
    v2.close()


def test_an_existing_vaults_width_wins_over_the_request(vault_path):
    """Reopening with a mismatched width warns and keeps the file's own."""
    VaultEngine(vault_path, embed_dim=384).close()
    with pytest.warns(UserWarning):
        e = VaultEngine(vault_path, embed_dim=768)
    assert e.embed_dim == 384
    e.close()
