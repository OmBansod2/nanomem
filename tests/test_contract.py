"""Every item in design/compat_contract.json, asserted against the v3 engine."""

import json
import os
import threading
import time

import numpy as np
import pytest

from conftest import D, unit_rows
from nanomem.engine import (VaultEngine, matches_filter, legacy_score,
                            legacy_score_to_cosine)

# The EXACT key set of a search hit. This is a contract, and the equality below
# is deliberate: adding a key is a decision, not a convenience. `superseded` and
# `superseded_at` were added in 0.8.1 -- a hit now says whether a LATER record in
# its own group replaced it, which a timestamp cannot express.
RESULT_KEYS = {"id", "doc_id", "text", "source", "metadata", "score", "cosine",
               "timestamp", "revision", "superseded", "superseded_at"}


# --- constructor -----------------------------------------------------------
def test_eager_file_creation(vault_path):
    assert not os.path.exists(vault_path)
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    assert os.path.exists(vault_path)
    assert os.path.getsize(vault_path) == 256
    assert os.listdir(os.path.dirname(vault_path)) == ["v.dat"]      # no sidecars
    e.close()


def test_second_instance_on_open_file(vault_path):
    a = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(3)
    for i in range(3):
        a.add_fact(f"fact {i}", V[i])
    a.flush()
    b = VaultEngine(filepath=vault_path, embed_dim=D)         # users.list_users does this
    assert b.stats()["total_documents"] == 3
    a.close(); b.close()


def test_close_remove_recreate(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    e.add_fact("x", unit_rows(1)[0]); e.flush()
    e.container.close()
    os.remove(vault_path)                                     # the v2 rebuild dance
    e2 = VaultEngine(filepath=vault_path, embed_dim=D)
    assert e2.stats()["total_documents"] == 0
    assert os.path.exists(vault_path)
    e2.close()


def test_embed_dim_mismatch(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    with pytest.raises(ValueError) as exc:
        e.add_fact("bad", np.ones(384, dtype=np.float32))
    assert "384" in str(exc.value) and str(D) in str(exc.value)
    e.close()
    with pytest.warns(UserWarning):
        e2 = VaultEngine(filepath=vault_path, embed_dim=384)
    assert e2.embed_dim == D
    e2.close()


# --- add_fact --------------------------------------------------------------
def test_add_fact_returns_id(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    got = e.add_fact("hello", unit_rows(1)[0])
    assert isinstance(got, str) and got.startswith("doc_")
    assert e.add_fact("hello", unit_rows(1)[0], id="explicit-1") == "explicit-1"
    e.close()


def test_metadata_id_always_set(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    doc_id = e.add_fact("t", unit_rows(1)[0], metadata={"user_id": "u"})
    e.flush()
    rec = e.get(doc_id)
    assert rec["metadata"]["id"] == doc_id
    e.close()


def test_roundtrip_idempotence(vault_path):
    """Explicit id/timestamp/revision/embedding survive a rebuild verbatim."""
    e = VaultEngine(filepath=vault_path, embed_dim=D, vector_dtype="float32")
    V = unit_rows(120, seed=3)
    src = []
    for i in range(120):
        r = {"id": f"id-{i}", "text": f"text {i} é中", "source": "s",
             "metadata": {"k": i % 5, "nested": {"a": [1, 2]}, "entity": "phone_number"},
             "timestamp": 1_700_000_000.0 + i, "revision": (i % 7) + 1}
        e.add_fact(r["text"], V[i], source=r["source"], metadata=dict(r["metadata"]),
                   timestamp=r["timestamp"], revision=r["revision"], id=r["id"])
        src.append(r)
    e.flush()
    for _ in range(3):
        recs = list(e.iter_records())
        assert len(recs) == 120
        for got, want in zip(recs, src):
            assert got["id"] == want["id"]
            assert got["text"] == want["text"]
            assert got["revision"] == want["revision"]
            assert abs(got["timestamp"] - want["timestamp"]) < 1e-9
            assert got["metadata"]["nested"] == {"a": [1, 2]}
        assert e.replace_all(recs) == 120
    V2 = np.vstack([r["embedding"] for r in e.iter_records()])
    assert float(np.min(np.sum(V2 * V, axis=1))) > 0.999999
    e.close()


def test_no_dedup_and_no_vector_rotation(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    v = unit_rows(1, seed=9)[0]
    for i in range(10):
        e.add_fact("identical text", v, source="chat_session",
                   metadata={"entity": "phone_number"}, revision=1)
    e.flush()
    assert e.count() == 10
    stored = np.vstack([r["embedding"] for r in e.iter_records()])
    assert float(np.min(stored @ v)) > 0.999999          # no anaphora re-rotation
    e.close()


def test_auto_revision_scoped_by_user_project_entity(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(6, seed=11)
    for i in range(3):
        e.add_fact(f"my phone number is 020-4455-778{i}", V[i], source="chat_session",
                   metadata={"user_id": "a", "project": "p"})
    for i in range(3):
        e.add_fact(f"my phone number is 020-1111-222{i}", V[3 + i], source="chat_session",
                   metadata={"user_id": "b", "project": "p"})
    e.flush()
    revs = [(r["metadata"].get("user_id"), r["revision"]) for r in e.iter_records()]
    assert revs == [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("b", 3)]
    e.close()


# --- search ----------------------------------------------------------------
def _seed(e, n=40, seed=5):
    V = unit_rows(n, seed=seed)
    for i in range(n):
        e.add_fact(f"document number {i}", V[i], metadata={"k": i % 4, "rare": i})
    return V


def test_search_result_shape(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = _seed(e)
    e.flush()
    hits = e.search("document number 3", V[3], top_k=5)
    assert 0 < len(hits) <= 5
    for h in hits:
        assert set(h) == RESULT_KEYS
        assert isinstance(h["id"], str) and isinstance(h["text"], str)
        assert isinstance(h["source"], str) and isinstance(h["metadata"], dict)
        assert isinstance(h["score"], float) and isinstance(h["cosine"], float)
        assert isinstance(h["revision"], int) and isinstance(h["timestamp"], float)
        # None is a real value here: it means the record is in no revision group,
        # so whether anything replaced it is UNKNOWN rather than False.
        assert h["superseded"] in (True, False, None)
        assert h["superseded_at"] is None or isinstance(h["superseded_at"], float)
        if h["superseded"] is not True:
            assert h["superseded_at"] is None
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    json.dumps(hits)                                          # proxy/server do this
    again = e.search("document number 3", V[3], top_k=5)
    assert again[0] is not hits[0] and again[0]["metadata"] is not hits[0]["metadata"]
    hits[0]["metadata"]["injected"] = True
    assert "injected" not in e.search("document number 3", V[3], top_k=1)[0]["metadata"]
    e.close()


def test_search_sees_unflushed_records(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(4, seed=7)
    for i in range(4):
        e.add_fact(f"pending {i}", V[i])
    assert e.stats()["memtable_pending"] == 4
    hits = e.search("pending 2", V[2], top_k=1)
    assert hits and hits[0]["text"] == "pending 2"
    e.close()


def test_min_score_is_pre_boost(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(2, seed=13)
    e.add_fact("my phone number is 020-4455-7788", V[0], source="chat_session",
               metadata={"user_id": "u"})
    e.add_fact("my phone number is 020-9911-2233", V[1], source="chat_session",
               metadata={"user_id": "u"})
    e.flush()
    q = V[0]
    assert e.search("what is my current phone number", q, top_k=5, min_score=0.0)
    assert e.search("what is my current phone number", q, top_k=5, min_score=1.5) == []
    e.close()


def test_metadata_filter_is_exact(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = _seed(e, n=60, seed=17)
    e.flush()
    # min_score defaults to 0.0, which (as in v2) also drops negative cosines,
    # so an exactness check has to lower it explicitly.
    hits = e.search("document", V[0], top_k=10, min_score=-1.0, metadata_filter={"k": 2})
    assert hits and all(h["metadata"]["k"] == 2 for h in hits)
    assert len(hits) == 10                       # 15 records carry k == 2
    rare = e.search("document", V[0], top_k=3, min_score=-1.0, metadata_filter={"rare": 57})
    assert len(rare) == 1 and rare[0]["metadata"]["rare"] == 57
    assert e.search("document", V[0], top_k=3, min_score=-1.0,
                    metadata_filter={"missing": 1}) == []
    assert all(h["metadata"]["k"] == 2
               for h in e.search("document", V[0], top_k=50, min_score=-1.0,
                                 metadata_filter={"k": 2}))
    e.close()


def test_temporal_direction(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(3, seed=19)
    base = V[0]
    for i, txt in enumerate(["my phone number is 9811111111",
                             "my phone number is 9855555555",
                             "my phone number is 9899999999"]):
        v = base + 0.02 * V[min(i, 2)]
        e.add_fact(txt, v / np.linalg.norm(v), source="chat_session",
                   metadata={"user_id": "demo"})
    e.flush()
    cur = e.search("what is my current phone number", base, top_k=3)
    old = e.search("what was my original phone number", base, top_k=3,
                   temporal_direction="historical")
    assert "9899999999" in cur[0]["text"]
    assert "9811111111" in old[0]["text"]
    e.close()


def test_zero_and_bad_queries(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    _seed(e, n=5)
    e.flush()
    assert e.search("q", np.zeros(D, dtype=np.float32), top_k=3) == []
    assert e.search("q", unit_rows(1)[0], top_k=0) == []
    e.close()


# --- flush / stats ---------------------------------------------------------
def test_flush_durability_across_instances(vault_path):
    a = VaultEngine(filepath=vault_path, embed_dim=D)
    b = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(10, seed=23)
    for i in range(10):
        a.add_fact(f"f{i}", V[i])
    assert b.stats()["total_documents"] == 0
    a.flush()
    assert a.stats()["memtable_pending"] == 0
    assert b.stats()["total_documents"] == 10                 # sees the other instance
    assert len(list(b.iter_records())) == 10
    a.flush()                                                  # idempotent
    assert a.stats()["compacted_blocks"] == b.stats()["compacted_blocks"]
    a.close(); b.close()


LEGACY_STATS = {"file_path": str, "file_size_mb": float, "total_documents": int,
                "compacted_blocks": int, "memtable_pending": int,
                "active_heap_ram_kb": float, "encrypted_at_rest": bool, "cipher": str}


def test_stats_keys_and_types(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    s0 = e.stats()
    for k, t in LEGACY_STATS.items():
        assert k in s0 and isinstance(s0[k], t), (k, type(s0[k]))
    json.dumps(s0)
    V = unit_rows(200, seed=29)
    for i in range(200):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    s1 = e.stats()
    assert s1["total_documents"] == 200
    assert s1["active_heap_ram_kb"] > s0["active_heap_ram_kb"]     # measured, not constant
    assert s1["encrypted_at_rest"] is False and s1["cipher"] == "none (plaintext)"
    from nanomem.engine import ENGINE_VERSION
    assert s1["engine_version"] == ENGINE_VERSION and s1["format_version"] == 3
    assert s1["max_boost"] == pytest.approx(1.10)     # documented cap on the boosts
    assert s1["max_boost"] == pytest.approx(s1["intent_boost"] + s1["group_hoist"]
                                            + s1["revision_lead"])
    # `peak_rss_kb()` reads `resource.getrusage`, which does not exist on
    # Windows, so the documented value there is None. Asserting a number
    # unconditionally made the CONTRACT test the one that did not know its
    # own contract.
    if s1["process_peak_rss_kb"] is None:
        assert os.name == "nt", "peak RSS is only unavailable where "\
                                "resource.getrusage is missing"
    else:
        assert s1["process_peak_rss_kb"] >= s1["active_heap_ram_kb"]
    e.close()


def test_active_heap_matches_measured_bytes(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(1000, seed=31)
    for i in range(1000):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    s = e.stats()
    total = (s["arena_bytes"] + s["landmark_bytes"] + s["column_bytes"]
             + s["record_bytes"] + s["memtable_bytes"] + s["index_bytes_estimated"])
    assert abs(s["active_heap_ram_kb"] - total / 1024.0) < 0.05 * (total / 1024.0)
    # arena_bytes is what is ALLOCATED (capacity-doubling), arena_used_bytes the
    # used slice. Reporting the slice under-counted the biggest allocation in the
    # process by up to 1.9x, which is what DECISIONS #15 forbids.
    assert s["arena_used_bytes"] == 1000 * D * 4
    assert s["arena_bytes"] == e.arena.vec.nbytes >= s["arena_used_bytes"]
    assert s["arena_bytes"] < 2 * s["arena_used_bytes"] + D * 4 * 64
    print(f"\nallocated {s['arena_bytes'] / 1e6:.2f} MB vs used "
          f"{s['arena_used_bytes'] / 1e6:.2f} MB at 1,000 docs")
    e.close()


# --- matches_filter --------------------------------------------------------
def _v2_matches_filter(meta, meta_filter):
    """The v2 implementation, copied verbatim as the oracle."""
    for k, v in meta_filter.items():
        if k not in meta:
            return False
        val = meta[k]
        if isinstance(v, (list, tuple, set)):
            if val not in v and str(val) not in [str(x) for x in v]:
                return False
        elif isinstance(val, (list, tuple, set)):
            if v not in val and str(v) not in [str(x) for x in val]:
                return False
        else:
            if val != v and str(val) != str(v):
                return False
    return True


def test_matches_filter_parity():
    rng = np.random.default_rng(0)
    values = [1, "1", 2, "two", [1, 2], ("a", "b"), {"x"}, 3.0, None, True]
    for _ in range(200):
        meta = {f"k{i}": values[int(rng.integers(len(values)))] for i in range(4)}
        flt = {f"k{int(rng.integers(6))}": values[int(rng.integers(len(values)))]}
        assert matches_filter(meta, flt) == _v2_matches_filter(meta, flt)
    assert VaultEngine._matches_filter({"a": 1}, {"a": "1"}) is True


def test_legacy_score_conversion():
    assert abs(legacy_score_to_cosine(0.25) - 0.417) < 1e-3
    assert abs(legacy_score_to_cosine(0.32) - 0.533) < 1e-3
    assert round(legacy_score_to_cosine(0.35), 2) == 0.58
    xs = np.linspace(-1, 1, 101)
    assert np.all(np.diff(legacy_score(xs)) > 0)               # strictly monotone


# --- container view --------------------------------------------------------
def test_container_view(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = _seed(e, n=120, seed=37)
    e.flush()
    toc = e.container.toc
    assert len(toc) == 3
    seen = 0
    for entry in toc:
        assert {"id", "doc_count", "offset", "length"} <= set(entry)
        blk = e.container.read_payload(entry["offset"], entry["length"],
                                       entry["doc_count"], entry["id"])
        assert set(blk) == {"texts", "sources", "metadatas", "timestamps",
                            "revisions", "values"}
        assert len(blk["texts"]) == entry["doc_count"]
        assert all("id" in m for m in blk["metadatas"])
        assert not np.shares_memory(blk["values"], e.arena.vec)      # owned copies
        blk["metadatas"][0]["mutated"] = True
        blk["values"][0] = 0.0
        seen += entry["doc_count"]
    assert seen == 120
    again = e.container.read_payload(toc[0]["offset"], toc[0]["length"],
                                      toc[0]["doc_count"], toc[0]["id"])
    assert "mutated" not in again["metadatas"][0]
    assert float(np.linalg.norm(again["values"][0])) > 0.9
    e.container.close()
    e.container.close()                                              # idempotent
    e.close()


# --- concurrency -----------------------------------------------------------
def test_thread_safety_smoke(vault_path):
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(400, seed=41)
    errors = []
    stop = threading.Event()

    def writer(lo, hi):
        try:
            for i in range(lo, hi):
                e.add_fact(f"w{i}", V[i], metadata={"w": i})
                if i % 25 == 0:
                    e.flush()
        except Exception as exc:                                   # pragma: no cover
            errors.append(exc)

    def reader():
        try:
            while not stop.is_set():
                hits = e.search("w", V[0], top_k=3)
                assert all(set(h) == RESULT_KEYS for h in hits)
                e.stats()
        except Exception as exc:                                   # pragma: no cover
            errors.append(exc)

    ws = [threading.Thread(target=writer, args=(i * 100, (i + 1) * 100)) for i in range(4)]
    rs = [threading.Thread(target=reader) for _ in range(3)]
    for t in ws + rs:
        t.start()
    for t in ws:
        t.join()
    stop.set()
    for t in rs:
        t.join()
    assert not errors
    e.flush()
    assert e.stats()["total_documents"] == 400
    e.close()
