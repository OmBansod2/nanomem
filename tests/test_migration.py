"""v2 -> v3 migration against the recorded golden vaults."""

import glob
import json
import os
import shutil
import stat

import numpy as np
import pytest

from conftest import D, GOLDEN_DIR, golden, live_embedder
from nanomem.engine import VaultEngine
from nanomem.errors import ReadOnlyVaultError
from nanomem.legacy_v2 import iter_v2_records

GOLDEN_JSON = os.path.join(GOLDEN_DIR, "chat_v2_golden.json")
BOOK_JSON = os.path.join(GOLDEN_DIR, "book_v2_golden.json")


def _copy(name, tmp_path):
    src = golden(name)
    dst = str(tmp_path / "v.dat")
    shutil.copyfile(src, dst)
    return src, dst


def test_book_migration_preserves_every_record(tmp_path):
    src, dst = _copy("book_v2.dat", tmp_path)
    want = list(iter_v2_records(src))
    e = VaultEngine(filepath=dst, embed_dim=D)
    assert e.stats()["format_version"] == 3
    assert e.count() == 845 == len(want)
    got = list(e.iter_records())
    # fp16 on disk is the default; measured worst-case cosine over all 845
    # migrated vectors on this corpus is 0.999957.
    for g, w in zip(got, want):
        assert g["id"] == w["id"]
        assert g["text"] == w["text"]
        assert g["source"] == w["source"]
        assert abs(g["timestamp"] - w["timestamp"]) < 1e-9
        assert g["revision"] == w["revision"]
        assert float(np.dot(g["embedding"], w["embedding"])) > 0.9999
    bak = dst + ".v2.bak"
    assert os.path.exists(bak)
    assert open(bak, "rb").read() == open(src, "rb").read()
    e.close()
    e2 = VaultEngine(filepath=dst, embed_dim=D)          # second open is a plain v3 open
    assert e2.stats()["format_version"] == 3 and e2.count() == 845
    e2.close()


def test_book_top4_equals_exhaustive(tmp_path):
    """The authoritative check: ranking over the migrated vectors is exact."""
    _, dst = _copy("book_v2.dat", tmp_path)
    e = VaultEngine(filepath=dst, embed_dim=D)
    V = e.arena.vec[:e.arena.n_rows]
    ids = e.arena.ids
    rng = np.random.default_rng(0)
    for _ in range(50):                                   # embedder-free: random queries
        q = rng.normal(size=D).astype(np.float32)
        q /= np.linalg.norm(q)
        got = [h["id"] for h in e.search("", q, top_k=4, min_score=-1.0)]
        want = [ids[i] for i in np.argsort(-(V @ q))[:4]]
        assert got == want
    ep = live_embedder()
    if ep is None:
        pytest.skip("no embedding daemon: skipping the 20 recorded book queries")
    for q in json.load(open(BOOK_JSON))["queries"]:
        qv = ep.embed(q); qv = qv / np.linalg.norm(qv)
        got = [h["id"] for h in e.search(q, qv, top_k=4)]
        want = [ids[i] for i in np.argsort(-(V @ qv))[:4]]
        assert got == want, q
    e.close()


def test_chat_migration_repairs_junk_entities(tmp_path):
    src, dst = _copy("chat_v2.dat", tmp_path)
    e = VaultEngine(filepath=dst, embed_dim=D)
    assert e.count() == 14
    recs = list(e.iter_records())
    for r in recs:
        ent = (r["metadata"] or {}).get("entity")
        assert ent is None or ent == ent.lower().replace(" ", "_")
        if "entity_v2" in r["metadata"]:
            assert r["metadata"]["entity_v2"] != r["metadata"].get("entity")
    ents = {r["metadata"].get("entity") for r in recs}
    assert "Update" not in ents and "phone_number" in ents
    e.close()


def test_chat_golden_answers(tmp_path):
    ep = live_embedder()
    if ep is None:
        pytest.skip("no embedding daemon: the recorded chat answers need a real model")
    _, dst = _copy("chat_v2.dat", tmp_path)
    G = json.load(open(GOLDEN_JSON))
    e = VaultEngine(filepath=dst, embed_dim=D)
    wrong = []
    for query, direction in G["queries"]:
        want = G["expected_top1_substring"][f"{query}|{direction}"]
        hits = e.search(query, ep.embed(query), top_k=3, temporal_direction=direction)
        if not hits or want not in hits[0]["text"]:
            wrong.append((query, direction, want))
    assert not wrong, f"{12 - len(wrong)}/12 correct; missed {wrong}"
    e.close()


def test_migration_backup_can_be_disabled(tmp_path, monkeypatch):
    _, dst = _copy("chat_v2.dat", tmp_path)
    monkeypatch.setenv("NANOMEM_KEEP_V2_BACKUP", "0")
    e = VaultEngine(filepath=dst, embed_dim=D)
    assert e.count() == 14
    assert not os.path.exists(dst + ".v2.bak")
    e.close()


def test_header_only_v2_file(tmp_path):
    import struct
    p = str(tmp_path / "empty.dat")
    open(p, "wb").write(struct.pack("<8sIIII40s", b"NANOMEM\x00", 2, D, 64, 50, b"\x00" * 40))
    e = VaultEngine(filepath=p, embed_dim=D)
    assert e.count() == 0 and e.stats()["format_version"] == 3
    e.add_fact("new life", np.ones(D, dtype=np.float32)); e.flush()
    assert e.count() == 1
    e.close()


def test_readonly_v2_fallback(tmp_path):
    _, dst = _copy("chat_v2.dat", tmp_path)
    e = VaultEngine(filepath=dst, embed_dim=D, migrate=False)
    assert e.stats()["format_version"] == 2 and e.stats()["read_only"] is True
    assert e.count() == 14
    hits = e.search("", np.ones(D, dtype=np.float32), top_k=3, min_score=-1.0)
    assert len(hits) == 3
    with pytest.raises(ReadOnlyVaultError):
        e.add_fact("nope", np.ones(D, dtype=np.float32))
    assert open(dst, "rb").read(8) == b"NANOMEM\x00"      # untouched
    e.close()


def test_encrypted_migration_keeps_no_plaintext_backup(tmp_path):
    """Migrating a v2 file into a PASSWORD vault must not leave the corpus readable.

    3.0.0 wrote ``<path>.v2.bak`` unconditionally, mode 0644. A v2 file's
    "cipher" is keyed only by the block id and a compiled-in constant, so the
    backup handed back every record the passphrase was meant to protect --
    measured on the 845-paragraph golden book vault.
    """
    from nanomem.legacy_v2 import iter_v2_records
    src = golden("chat_v2.dat")
    target = str(tmp_path / "mig.dat")
    shutil.copyfile(src, target)
    e = VaultEngine(filepath=target, embed_dim=768, password="a strong passphrase")
    assert e.stats()["encrypted_at_rest"] is True
    n = e.count()
    e.close()
    assert n > 0
    siblings = sorted(os.path.basename(p) for p in glob.glob(target + "*"))
    assert siblings == ["mig.dat"], siblings

    # ... and the opt-in still works, but the copy is owner-only and warned about
    target2 = str(tmp_path / "mig2.dat")
    shutil.copyfile(src, target2)
    with pytest.warns(RuntimeWarning, match="PLAINTEXT"):
        e2 = VaultEngine(filepath=target2, embed_dim=768,
                         password="a strong passphrase", migrate_backup=True)
    e2.close()
    bak = target2 + ".v2.bak"
    assert os.path.exists(bak)
    assert oct(os.stat(bak).st_mode & 0o777) == "0o600"
    assert len(list(iter_v2_records(bak))) == n        # honestly: still readable


def test_plaintext_migration_still_keeps_the_backup(tmp_path):
    src = golden("chat_v2.dat")
    target = str(tmp_path / "mig.dat")
    shutil.copyfile(src, target)
    e = VaultEngine(filepath=target, embed_dim=768)
    e.close()
    assert os.path.exists(target + ".v2.bak")
    assert open(target + ".v2.bak", "rb").read() == open(src, "rb").read()
