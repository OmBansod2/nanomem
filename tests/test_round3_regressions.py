"""Round-3 regressions: every one of these fails on nanomem 3.0.1.

Grouped by the defect they pin down. The ranking ones live in
``test_adjacent_attributes.py``; everything here is container, lifecycle,
concurrency and vault-level.
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from conftest import D, unit_rows

from nanomem import container as C
from nanomem import users as U
from nanomem.engine import VaultEngine
from nanomem.errors import (ClosedVaultError, IntegrityError, NanomemError,
                            NotEncryptedError, PasswordRequiredError)
from nanomem.vault import Vault

PW = "correct horse battery staple"


def _fill(path, n, dim=D, **kw):
    e = VaultEngine(filepath=path, embed_dim=dim, **kw)
    V = unit_rows(n, dim, seed=5)
    for i in range(n):
        e.add_fact(f"text number {i}", V[i], id=f"id{i}")
    e.flush()
    return e, V


# --------------------------------------------------------------------------
# BLOCKER: on_integrity_error="skip" attached the wrong text to later rows
# --------------------------------------------------------------------------
def test_skipped_block_does_not_shift_later_records(tmp_path):
    """``Arena.row_block`` indexes the arena's own buffer list, not the file's.

    3.0.1 stored ``meta.index`` -- the CONTAINER block position, which counts
    skipped blocks -- so with one block skipped every row after it read a
    different block's records (id12 came back as "text number 16") and the last
    rows raised IndexError out of ``arena.record``.
    """
    p = str(tmp_path / "v.dat")
    e, V = _fill(p, 20, block_capacity=4)
    off = e._cont.blocks[2].offset
    e.close()
    with open(p, "r+b") as f:                      # break block 2's header CRC
        f.seek(off + 64)
        b = f.read(4)
        f.seek(off + 64)
        f.write(bytes([b[0] ^ 0xFF]) + b[1:])

    e = VaultEngine(filepath=p, embed_dim=D, block_capacity=4,
                    on_integrity_error="skip")
    try:
        assert e._cont.integrity_errors == [2]
        assert e.arena.n_rows == 16                # 20 minus the skipped block
        for r in range(e.arena.n_rows):
            rec = e.arena.record(r)                # 3.0.1: IndexError from r=12
            assert rec["text"] == f"text number {e.arena.ids[r][2:]}", (r, rec)
        hit = e.search("q", V[12], top_k=1)[0]     # and through the public API
        assert hit["id"] == "id12" and hit["text"] == "text number 12"
    finally:
        e.close()


# --------------------------------------------------------------------------
# BLOCKER: a failed scan left the container half-committed
# --------------------------------------------------------------------------
def test_integrity_error_does_not_duplicate_rows_on_retry(tmp_path):
    """Blocks validated before the failure are committed; a retry re-raises.

    3.0.1 pushed them into the sink but left ``valid_end`` at the pre-scan
    offset, so three ``count()`` calls after one corrupt block turned 4 rows
    into 16 with only 8 distinct ids, and memory grew without bound.
    """
    p = str(tmp_path / "v.dat")
    w = VaultEngine(filepath=p, embed_dim=D, block_capacity=4)
    V = unit_rows(16, seed=7)
    for i in range(4):
        w.add_fact(f"a{i}", V[i], id=f"a{i}")
    w.flush()
    r = VaultEngine(filepath=p, embed_dim=D, block_capacity=4)     # reader sees 4
    assert r.count() == 4
    for i in range(4, 16):
        w.add_fact(f"a{i}", V[i], id=f"a{i}")
    w.flush()
    off, plen = w._cont.blocks[2].offset, w._cont.blocks[2].payload_len
    w.close()
    with open(p, "r+b") as f:                       # break block 2's trailer
        f.seek(off + C.BLOCK_HEADER_SIZE + plen)
        t = f.read(4)
        f.seek(off + C.BLOCK_HEADER_SIZE + plen)
        f.write(bytes([t[0] ^ 0xFF]) + t[1:])

    seen = []
    for _ in range(3):
        with pytest.raises(IntegrityError):
            r.count()
        seen.append((r.arena.n_rows, len(set(r.arena.ids))))
    assert seen == [(8, 8)] * 3, seen              # 3.0.1: (8,8) (12,8) (16,8)
    r.close()


# --------------------------------------------------------------------------
# BLOCKER: two processes creating the same vault
# --------------------------------------------------------------------------
_CREATE_WORKER = r"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from nanomem.engine import VaultEngine
path, tag, n, dim = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rng = np.random.default_rng(abs(hash(tag)) % 10000)
e = VaultEngine(filepath=path, embed_dim=dim, block_capacity=5)
for i in range(n):
    v = rng.normal(size=dim).astype(np.float32)
    e.add_fact(f"{tag}-{i}", v / np.linalg.norm(v), id=f"{tag}-{i}")
e.flush()
e.close()
print("ok")
"""


@pytest.mark.skipif(os.name == "nt", reason=(
    "advisory whole-file locking is weaker on Windows -- msvcrt has no shared lock and file_lock says so -- so simultaneous creation is not serialised the way fcntl serialises it"))
@pytest.mark.parametrize("procs,each", [(4, 30)])
def test_concurrent_creation_of_one_vault(tmp_path, procs, each):
    """The vault does NOT exist yet -- the first thing a deployment does.

    3.0.1 probed with ``os.path.exists`` and then opened
    ``O_CREAT | O_TRUNC`` outside any lock, so a second process truncated the
    first one's committed blocks and wrote a new ``vault_uuid``; since the uuid
    is inside every block tag the file was then permanently unopenable
    (measured: 2 of 4 trials dead, 1 of 10 silently half-empty).
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = str(tmp_path / "fresh.dat")
    assert not os.path.exists(path)
    worker = str(tmp_path / "cw.py")
    with open(worker, "w") as f:
        f.write(_CREATE_WORKER)
    running = [subprocess.Popen([sys.executable, worker, root, path, f"w{i}",
                                 str(each), "32"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for i in range(procs)]
    outs = [p.communicate() for p in running]
    for (out, err), p in zip(outs, running):
        assert p.returncode == 0, err
    e = VaultEngine(filepath=path, embed_dim=32, block_capacity=5)
    try:
        assert e._cont.integrity_errors == []
        assert e._cont.truncated_tail_bytes == 0
        assert e.count() == procs * each
        assert len({e.arena.ids[i] for i in range(e.arena.n_rows)}) == procs * each
    finally:
        e.close()


# --------------------------------------------------------------------------
# BLOCKER: writing after close() on a password-protected vault
# --------------------------------------------------------------------------
def test_write_after_close_is_refused(tmp_path):
    """3.0.1 wrote the record in CLEARTEXT and bricked the vault."""
    p = str(tmp_path / "v.dat")
    with Vault(p, password=PW) as v:
        for i in range(3):
            v.add(f"harmless note {i}")
    secret = "my bank pin is 4821"
    with pytest.raises(ClosedVaultError):
        v.add(secret)
        v.flush()
    assert secret.encode() not in open(p, "rb").read()
    with Vault(p, password=PW) as again:            # ... and it still opens
        assert len(again) == 3


def test_closed_engine_refuses_every_write(tmp_path):
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    e.add_fact("one", unit_rows(1, seed=1)[0])
    e.flush()
    e.close()
    with pytest.raises(ClosedVaultError):
        e.add_fact("two", unit_rows(1, seed=2)[0])
    e._mt.items.append({"id": "x"})                 # force flush to have work
    with pytest.raises(ClosedVaultError):
        e.flush()
    e._mt.items.clear()
    with pytest.raises(ClosedVaultError):
        e.replace_all([])


# --------------------------------------------------------------------------
# MAJOR: replace_all discarded unflushed records it had already given ids to
# --------------------------------------------------------------------------
def test_replace_all_keeps_pending_records(tmp_path):
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(65, seed=9)
    for i in range(60):
        e.add_fact(f"f{i}", V[i], id=f"f{i}")
    e.flush()
    snapshot = list(e.iter_records(include_embeddings=True))
    pending = [e.add_fact(f"p{i}", V[60 + i], id=f"p{i}") for i in range(5)]
    assert e.count() == 65
    assert e.replace_all(snapshot) == 65            # 3.0.1 returned 60
    assert e.count() == 65
    for pid in pending:
        assert e.get(pid) is not None, pid          # 3.0.1: None
    e.close()


# --------------------------------------------------------------------------
# MAJOR: export/split wrote a PLAINTEXT copy of an encrypted vault
# --------------------------------------------------------------------------
def test_export_inherits_the_source_password(tmp_path):
    src, dst = str(tmp_path / "a.dat"), str(tmp_path / "b.dat")
    v = Vault(src, password=PW)
    for i in range(6):
        v.add(f"secret fact {i}", metadata={"grp": "s" if i < 3 else "t"})
    v.flush()
    assert v.stats()["encrypted_at_rest"] is True
    assert v.export(dst, where={"grp": "s"}) == 3
    v.close()
    assert b"secret fact 0" not in open(dst, "rb").read()       # 3.0.1: present
    with pytest.raises(PasswordRequiredError):
        Vault(dst)
    with Vault(dst, password=PW) as t:
        assert t.stats()["encrypted_at_rest"] is True and len(t) == 3


def test_export_can_opt_out_of_the_password(tmp_path):
    src, dst = str(tmp_path / "a.dat"), str(tmp_path / "b.dat")
    with Vault(src, password=PW) as v:
        v.add("a note", metadata={"grp": "s"})
        v.flush()
        assert v.export(dst, where={"grp": "s"}, target_password=None) == 1
    with Vault(dst) as t:
        assert t.stats()["encrypted_at_rest"] is False


def test_merge_still_reads_a_plaintext_source(tmp_path):
    enc, plain = str(tmp_path / "e.dat"), str(tmp_path / "p.dat")
    with Vault(plain) as q:
        q.add("plain note one")
        q.flush()
    with Vault(enc, password=PW) as v:
        v.add("secret note")
        v.flush()
        assert v.merge(plain)["added"] == 1
        assert len(v) == 2


# --------------------------------------------------------------------------
# errors are all NanomemError; a closed/plaintext mismatch does not traceback
# --------------------------------------------------------------------------
def test_password_on_plaintext_is_a_nanomem_error(tmp_path):
    p = str(tmp_path / "v.dat")
    VaultEngine(filepath=p, embed_dim=D).close()
    with pytest.raises(NotEncryptedError):
        VaultEngine(filepath=p, embed_dim=D, password=PW)
    with pytest.raises(NanomemError):
        VaultEngine(filepath=p, embed_dim=D, password=PW)


def test_revision_out_of_int32_range_is_rejected_at_the_add(tmp_path):
    """3.0.1 accepted it, returned an id, then OverflowError'd on every later
    flush and close -- the memtable could never be written again."""
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(2, seed=4)
    with pytest.raises(ValueError):
        e.add_fact("too big", V[0], id="r1", revision=2 ** 31 + 5)
    e.add_fact("fine", V[1], id="r2")
    e.flush()                                        # 3.0.1: OverflowError
    assert e.count() == 1
    e.close()


# --------------------------------------------------------------------------
# a torn / damaged tail is never silent
# --------------------------------------------------------------------------
def test_damaged_last_block_header_is_reported(tmp_path):
    """The ambiguity is inherent; the silence was not.

    3.0.1 dropped the records with ``integrity_errors == []`` and no warning,
    and the next append ``ftruncate``d them away just as quietly.
    """
    p = str(tmp_path / "v.dat")
    e, V = _fill(p, 50, block_capacity=10)
    off = e._cont.blocks[-1].offset
    e.close()
    with open(p, "r+b") as f:
        f.seek(off + 41)                             # inside the nonce (CRC-covered)
        b = f.read(1)
        f.seek(off + 41)
        f.write(bytes([b[0] ^ 0x01]))

    with pytest.warns(RuntimeWarning, match="crashed write"):
        e = VaultEngine(filepath=p, embed_dim=D, block_capacity=10)
    assert e.count() == 40
    assert e.stats()["truncated_tail_bytes"] > 0
    with pytest.warns(RuntimeWarning, match="gone for good"):
        e.add_fact("after the damage", V[0], id="later")
        e.flush()
    e.close()


def test_on_torn_tail_raise_refuses_to_open_or_append(tmp_path):
    p = str(tmp_path / "v.dat")
    e, V = _fill(p, 50, block_capacity=10)
    off = e._cont.blocks[-1].offset
    e.close()
    with open(p, "r+b") as f:
        f.seek(off + 41)
        b = f.read(1)
        f.seek(off + 41)
        f.write(bytes([b[0] ^ 0x01]))
    with pytest.raises(IntegrityError):
        VaultEngine(filepath=p, embed_dim=D, block_capacity=10, on_torn_tail="raise")


def test_stale_reader_is_not_fooled_by_an_equal_file_size(tmp_path):
    """Same size, same inode, different content: ``modified()`` must not say
    "same". 3.0.1 compared the size alone, so a crash remnant whose length
    happened to match the replacement left a reader permanently stale."""
    p = str(tmp_path / "v.dat")
    e, V = _fill(p, 4, block_capacity=4)
    e.close()
    size = os.path.getsize(p)
    r = VaultEngine(filepath=p, embed_dim=D, block_capacity=4)
    assert r.count() == 4
    assert r._cont.modified() == "same"
    w = VaultEngine(filepath=p, embed_dim=D, block_capacity=4)
    for i in range(4, 8):
        w.add_fact(f"text number {i}", V[i % 4], id=f"id{i}")
    w.flush()
    w.close()
    r._cont._mtime_ns = None                       # simulate a size coincidence
    r._cont.scanned_end = os.path.getsize(p)
    assert r._cont.modified() == "same"             # size-only: fooled
    r._cont._mtime_ns = 0                           # with the mtime guard: caught
    assert r._cont.modified() == "replaced"
    r.close()
    assert size < os.path.getsize(p)


# --------------------------------------------------------------------------
# users.py, DECISIONS #12
# --------------------------------------------------------------------------
def test_delete_user_removes_the_plaintext_backup_and_temp_siblings(tmp_path):
    d = str(tmp_path)
    target = U.create_user("someone", directory=d)
    open(target + ".v2.bak", "wb").write(b"the user's migrated plaintext vault")
    open(target + ".tmp-1234-abcd", "wb").write(b"a partial rewrite")
    assert len(U.user_vault_siblings("someone", directory=d)) == 3
    assert U.delete_user("someone", directory=d) is True
    assert U.user_vault_siblings("someone", directory=d) == []   # 3.0.1: 2 left
    assert U.delete_user("someone", directory=d) is False


def test_list_users_ignores_siblings(tmp_path):
    d = str(tmp_path)
    U.create_user("alpha", directory=d)
    open(os.path.join(d, "memory_alpha.dat.v2.bak"), "wb").write(b"x")
    assert sorted(u["username"] for u in U.list_users(d)) == ["alpha"]


# --------------------------------------------------------------------------
# the block tag is computed once per block, not twice
# --------------------------------------------------------------------------
def test_block_tag_is_not_computed_twice_per_block(tmp_path, monkeypatch):
    from nanomem import crypto
    p = str(tmp_path / "v.dat")
    e, _ = _fill(p, 40, block_capacity=10, password=PW)
    e.close()
    calls = {"n": 0}
    real = crypto.block_tag

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(crypto, "block_tag", counting)
    e = VaultEngine(filepath=p, embed_dim=D, block_capacity=10, password=PW)
    blocks = len(e._cont.blocks)
    e.close()
    assert blocks == 4
    assert calls["n"] == blocks, f"{calls['n']} tags for {blocks} blocks"


# --------------------------------------------------------------------------
# header version checks run AFTER the authenticator
# --------------------------------------------------------------------------
def test_tampered_header_says_tampered_not_wrong_version(tmp_path):
    import struct
    import zlib
    p = str(tmp_path / "v.dat")
    _fill(p, 10, password=PW)[0].close()
    raw = bytearray(open(p, "rb").read(C.FILE_HEADER_SIZE))
    raw[12:16] = struct.pack("<I", 9)               # min_reader = 9
    raw[120:124] = struct.pack("<I", zlib.crc32(bytes(raw[:120])) & 0xFFFFFFFF)
    with open(p, "r+b") as f:
        f.seek(0)
        f.write(bytes(raw))
    with pytest.raises(NanomemError) as exc:
        VaultEngine(filepath=p, embed_dim=D, password=PW)
    assert "reader version" not in str(exc.value), str(exc.value)
    assert "tamper" in str(exc.value).lower()


# --------------------------------------------------------------------------
# a forged records section is a NanomemError, not numpy's ValueError
# --------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    b"\xff\xff\xff\xff" + b"\x00" * 40,          # count = 2**32-1
    b"\xe8\x03\x00\x00" + b"\x00" * 40,          # count = 1000, no data
    b"\x03\x00\x00\x00",                          # count = 3, nothing else
    b"",                                             # empty
])
def test_forged_records_section_raises_nanomem_error(payload):
    """A plaintext vault's trailer is an unkeyed SHA-256 that anyone can
    recompute, so a forged record count is reachable. 3.0.1 surfaced it as
    numpy's `ValueError: buffer is smaller than requested size`, which the
    documented single `except NanomemError` guard does not catch."""
    from nanomem.container import decode_records
    with pytest.raises(NanomemError):
        decode_records(payload)
