"""Vault-level contract: the public API every caller in the repo uses."""

import json
import os
import sys

import numpy as np
import pytest

from conftest import D, unit_rows
from nanomem.vault import Vault, TextHopBridge


@pytest.fixture
def vault(tmp_path, offline_embedder):
    v = Vault(str(tmp_path / "v.dat"))
    yield v
    v.close()


def test_add_returns_id(vault):
    doc_id = vault.add("a fact about gardening tools")
    assert isinstance(doc_id, str) and doc_id.startswith("doc_")
    assert vault.last_id == doc_id
    assert vault.add("") == ""
    big = " ".join(f"word{i}" for i in range(2000))
    parent = vault.add(big)
    assert isinstance(parent, str)
    vault.flush()
    ids = {r["id"] for r in vault.get_all_records()}
    assert f"{parent}_chunk_1" in ids


def test_eager_creation_and_stats(tmp_path, offline_embedder):
    p = str(tmp_path / "u.dat")
    with Vault(p):
        pass
    assert os.path.exists(p)
    with Vault(p) as v:
        s = v.stats()
    assert s["total_documents"] == 0
    json.dumps(s)


def test_crud_cycle(vault):
    ids = [vault.add(f"fact number {i}", metadata={"k": i % 3}) for i in range(10)]
    vault.flush()
    assert len(vault) == 10
    assert vault.get(ids[3])["text"] == "fact number 3"
    assert ids[3] in vault
    assert vault.update(ids[3], text="fact number three, revised")
    got = vault.get(ids[3])
    assert got["text"] == "fact number three, revised" and got["revision"] == 2
    assert len(vault.get_all_records()) == 10
    assert vault.delete(id=ids[0]) == 1
    assert len(vault.get_all_records()) == 9
    assert vault.delete(where={"k": 1}) == 3
    assert len(vault.get_all_records()) == 6
    assert vault.delete(text_contains="nothing matches this") == 0


def test_get_all_records_order_is_insertion_order(vault):
    texts = [f"ordered record {i}" for i in range(30)]
    for t in texts:
        vault.add(t)
    assert [r["text"] for r in vault.get_all_records()] == texts


def test_records_survive_rebuild(vault):
    for i in range(20):
        vault.add(f"durable {i}", metadata={"keep": i < 10})
    vault.flush()
    before = vault.get_all_records(include_embeddings=True)
    vault.delete(where={"keep": False})
    after = vault.get_all_records(include_embeddings=True)
    assert len(after) == 10
    for b, a in zip(before[:10], after):
        assert b["id"] == a["id"] and b["text"] == a["text"]
        assert float(np.dot(b["embedding"], a["embedding"])) > 0.9999


def test_merge_without_dedup_loses_nothing(tmp_path, offline_embedder):
    a, b = str(tmp_path / "a.dat"), str(tmp_path / "b.dat")
    with Vault(a) as va:
        for i in range(5):
            va.add(f"shared text {i}")
    with Vault(b) as vb:
        for i in range(5):
            vb.add(f"shared text {i}")
        vb.add("unique to b")
    with Vault(a) as va:
        res = va.merge(b, deduplicate=False)
        assert res["added"] == 6 and res["duplicates_skipped"] == 0
        assert len(va.get_all_records()) == 11
        res2 = va.merge(b, deduplicate=True)
        assert res2["duplicates_skipped"] == 6


def test_export_split_and_unmerge(tmp_path, offline_embedder):
    src = str(tmp_path / "src.dat")
    with Vault(src) as v:
        for i in range(12):
            v.add(f"row {i}", metadata={"bucket": "x" if i < 6 else "y"})
        n = v.export(str(tmp_path / "x.dat"), where={"bucket": "x"}, purge=True)
        assert n == 6 and len(v.get_all_records()) == 6
        groups = v.split_by_key("bucket", str(tmp_path / "parts"))
        assert groups == {"y": 6}
    with Vault(str(tmp_path / "x.dat")) as t:
        assert len(t.get_all_records()) == 6


def test_prune_by_age(tmp_path, offline_embedder):
    p = str(tmp_path / "p.dat")
    with Vault(p) as v:
        import time
        now = time.time()
        for i in range(10):
            v.add(f"aged {i}", timestamp=now - i * 86400 * 10)
        v.flush()
        freed = v.prune(older_than_days=25)
        assert freed >= 0
        assert len(v.get_all_records()) == 3


def test_forget_threshold_is_on_the_cosine_scale(vault):
    import inspect
    sig = inspect.signature(Vault.forget)
    assert sig.parameters["min_score"].default == 0.42      # legacy 0.25 re-tuned


def test_detect_entity_has_no_inline_regex():
    import inspect
    src = inspect.getsource(Vault._detect_entity)
    assert "re.search" not in src and "re.compile" not in src
    assert Vault._detect_entity("My phone number is 020-4455-7788.") == "phone_number"
    assert Vault._detect_entity("anything", {"entity": "explicit"}) == "explicit"


def test_password_reaches_the_engine(tmp_path, offline_embedder):
    p = str(tmp_path / "enc.dat")
    with Vault(p, password="a long enough passphrase") as v:
        v.add("a protected note")
        v.flush()
        assert v.stats()["encrypted_at_rest"] is True
    from nanomem.errors import PasswordRequiredError
    with pytest.raises(PasswordRequiredError):
        Vault(p)


def test_search_multihop_uses_a_text_bridge(vault, monkeypatch):
    for i in range(20):
        vault.add(f"bridge document {i} about topic {i % 4}")
    vault.flush()
    calls = []

    class Recording(TextHopBridge):
        def bridge(self, q_vec, d1_vec, question, d1_text):
            calls.append((question, d1_text))
            return super().bridge(q_vec, d1_vec, question, d1_text)

    hits = vault.search_multihop("bridge document 3", top_k=4,
                                 bridge=Recording(vault.embedder))
    assert 0 < len(hits) <= 4
    assert len({h["id"] for h in hits}) == len(hits)          # deduped by id
    assert calls and calls[0][0] == "bridge document 3"
    assert {h.get("hop") for h in hits} <= {1, 2}


def test_search_multihop_falls_back_without_a_bridge(vault):
    class Broken:
        def bridge(self, *a, **k):
            return None
    for i in range(10):
        vault.add(f"plain document {i}")
    vault.flush()
    hits = vault.search_multihop("plain document 1", top_k=3, bridge=Broken())
    assert len(hits) == 3


def test_compact_is_exposed(vault):
    for i in range(120):
        vault.add(f"compactable {i}")
    vault.flush()
    info = vault.compact(recluster=True)
    assert info["records"] == 120 and info["reclustered"] is True
    assert len(vault.get_all_records()) == 120


# ---------------------------------------------------------------------------
# bulk ingest: the row-count hint, end to end
# ---------------------------------------------------------------------------
def _copies(v):
    """How many times this vault's arena has been copied to make room."""
    return v.engine.arena._store.copies


def test_add_batch_sizes_the_arena_for_the_batch_it_was_given(tmp_path, offline_embedder):
    """A bulk caller knows its length; the engine only knows it if told.

    Nothing in the library used to say so, and the arena grew into the count
    instead of being sized for it: 820.5 MB of peak ru_maxrss to build and then
    serve 71,433 documents on the committed engine, against 313.7 MB through
    this method today -- medians of four runs each
    (scratch/refound/ingest_ram_results.json, summary and replicate).
    """
    with Vault(str(tmp_path / "b.dat")) as v:
        V = unit_rows(500, D, seed=7)
        recs = [{"id": f"r{i}", "text": f"batch record {i}",
                 "embedding": V[i], "source": "batch",
                 "metadata": {"idx": i}} for i in range(500)]
        assert v.add_batch(recs) == 500
        assert v.engine.arena.n_rows == 500
        assert v.engine.arena.reserved_rows >= 500
        assert _copies(v) == 0
        assert len(v) == 500
        # The vectors here are seeded noise, so identity is checked by id, not
        # by asking the offline encoder to rediscover them.
        assert v.get("r250")["text"] == "batch record 250"
        assert v.get("r499")["metadata"]["idx"] == 499


def test_add_batch_hint_is_cumulative_across_batches(tmp_path, offline_embedder):
    """Ten batches in a row must not mean ten reallocations."""
    with Vault(str(tmp_path / "c.dat")) as v:
        V = unit_rows(1000, D, seed=8)
        for b in range(10):
            recs = [{"id": f"r{b}_{i}", "text": f"record {b} {i}",
                     "embedding": V[b * 100 + i]} for i in range(100)]
            v.add_batch(recs)
        assert v.engine.arena.n_rows == 1000
        assert v.engine.arena.reserved_rows >= 1000
        assert _copies(v) <= 1


def test_ingest_file_sizes_the_arena_for_its_own_chunks(tmp_path, offline_embedder):
    src = tmp_path / "long.txt"
    src.write_text("\n".join(f"line {i} of a plain text document" for i in range(4000)))
    with Vault(str(tmp_path / "f.dat")) as v:
        n = v.ingest_file(str(src))
        assert n > 50
        assert v.engine.arena.reserved_rows >= n
        assert _copies(v) == 0


def test_ingest_directory_hints_from_bytes_before_reading_anything(tmp_path, offline_embedder):
    """The chunk count is only known after the work; the byte count is free.

    ``ingest_directory`` therefore hints ``total bytes //
    BYTES_PER_CHUNK_ESTIMATE`` before it opens a file. The constant is measured,
    not asserted: scratch/refound/ingest_ram_results.json -> directory_estimate
    carries the bytes-per-chunk of four real trees at the shipped split.
    """
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    for f in range(6):
        d = tree if f % 2 else tree / "sub"
        (d / f"file{f}.txt").write_text(
            "\n".join(f"file {f} line {i} with a little text on it" for i in range(600)))

    hints = []
    with Vault(str(tmp_path / "d.dat")) as v:
        real = v.engine.reserve_additional_rows

        def spy(n):
            hints.append((int(n), int(v.engine.arena.n_rows)))
            return real(n)

        v.engine.reserve_additional_rows = spy
        out = v.ingest_directory(str(tree))

    assert out["files_indexed"] == 6
    first_hint, rows_at_first_hint = hints[0]
    assert rows_at_first_hint == 0            # hinted before anything was read
    assert first_hint > 0
    # A hint is allowed to be wrong; it is not allowed to be absurd.
    assert 0.2 <= first_hint / out["chunks_indexed"] <= 5.0
    assert Vault.BYTES_PER_CHUNK_ESTIMATE == 2048


def test_cli_ingest_accepts_an_explicit_row_count(tmp_path, offline_embedder, monkeypatch, capsys):
    """``nanomem ingest --expect-docs N`` is the hint a loader can type."""
    from nanomem import cli
    from nanomem.engine import VaultEngine
    src = tmp_path / "doc.txt"
    src.write_text("\n".join(f"line {i}" for i in range(300)))
    vault_path = str(tmp_path / "cli.dat")
    seen = []
    real = VaultEngine.reserve_additional_rows

    def spy(self, n):
        seen.append(int(n))
        return real(self, n)

    monkeypatch.setattr(VaultEngine, "reserve_additional_rows", spy)
    monkeypatch.setattr(sys, "argv",
                        ["nanomem", "ingest", str(src), "--vault", vault_path,
                         "--expect-docs", "5000"])
    cli.main()
    out = capsys.readouterr().out
    assert "Ingested" in out
    assert 5000 in seen                        # the typed count reached the arena
    with Vault(vault_path) as v:
        assert len(v) > 0
