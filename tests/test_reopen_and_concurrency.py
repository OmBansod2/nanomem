"""Round-2 regressions.

Every test here fails on nanomem 3.0.0. They exist because the 135-test suite
that shipped with 3.0.0 measured latency, write cost and chat accuracy on the
SAME engine instance that ingested the corpus -- and the defect only appears
once a vault has been closed and opened again, which is the first thing any
real caller does.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import D, unit_rows

from nanomem import container as C
from nanomem.engine import VaultEngine
from nanomem.errors import ContainerReplacedError, IntegrityError


def _ingest(path, n, dim=D, seed=3, **kw):
    e = VaultEngine(filepath=path, embed_dim=dim, **kw)
    V = unit_rows(n, dim, seed=seed)
    for i in range(n):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    e.close()
    return V


# --------------------------------------------------------------------------
# the offsets themselves
# --------------------------------------------------------------------------
def test_reopened_container_reports_same(vault_path):
    """``valid_end`` is the real end of the last block, so a re-opened vault is
    not "replaced".

    3.0.0's ``scan()`` stored ``align_up(end)`` here while ``append_block()``
    stored the raw end, so ``valid_end`` sat up to 63 bytes past EOF and
    ``modified()`` answered "replaced" forever.
    """
    _ingest(vault_path, 137, seed=11)
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    size = os.path.getsize(vault_path)
    assert e._cont.valid_end == size, (e._cont.valid_end, size)
    assert e._cont.scanned_end == size
    assert e._cont.modified() == "same"
    assert e._cont.valid_end % C.ALIGN != 0 or size % C.ALIGN == 0
    e.close()


def test_writer_and_reader_agree_on_valid_end(vault_path):
    """The two code paths that set ``valid_end`` must produce the same number."""
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(60, seed=5)
    for i in range(60):
        e.add_fact(f"r{i}", V[i])
    e.flush()
    writer_valid_end = e._cont.valid_end
    assert e._cont.modified() == "same"
    e.close()
    r = VaultEngine(filepath=vault_path, embed_dim=D)
    assert r._cont.valid_end == writer_valid_end
    r.close()


def test_reopen_then_search_does_not_reload(vault_path):
    """No search on an unchanged vault may re-read the file."""
    _ingest(vault_path, 400, seed=6)
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    calls = {"n": 0}
    real = VaultEngine._full_reload

    def counting(self):
        calls["n"] += 1
        return real(self)

    VaultEngine._full_reload = counting
    try:
        q = unit_rows(1, seed=99)[0]
        for _ in range(20):
            e.search("", q, top_k=4)
            e.get(e.arena.ids[0])
            e.stats()
            e.count()
    finally:
        VaultEngine._full_reload = real
    assert calls["n"] == 0, f"{calls['n']} full reloads on an unchanged vault"
    e.close()


def test_reopened_search_and_write_latency(tmp_path):
    """Latency on the shape a caller runs: open an existing vault, then work.

    Asserted relative to the raw matmul on this machine, not to an absolute
    millisecond budget. 3.0.0 measured 51-89x the floor here.
    """
    n = 6000
    path = str(tmp_path / "v.dat")
    V = _ingest(path, n, seed=21)
    e = VaultEngine(filepath=path, embed_dim=D)
    A = e.arena.vec[:n]
    q = unit_rows(1, seed=77)[0]

    def p50(fn, reps=40):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1e3

    floor = p50(lambda: A @ q)
    got = p50(lambda: e.search("", q, top_k=4))
    write = p50(lambda: e.add_fact("late arrival", q), reps=20)
    print(f"\nreopened vault at {n} docs: search p50 {got:.3f} ms "
          f"(raw matmul {floor:.3f} ms, {got / max(floor, 1e-9):.2f}x), "
          f"add_fact p50 {write * 1000:.1f} us")
    assert got < max(6.0 * floor, 0.5)
    assert write < 1.0
    e.close()


def test_torn_tail_is_reported_and_cut(vault_path):
    _ingest(vault_path, 60, seed=4)
    with open(vault_path, "ab") as f:
        f.write(b"\x00" * 37)                       # crash remnant
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    assert e.count() == 60
    assert e.stats()["truncated_tail_bytes"] == 37
    e.add_fact("after the crash", unit_rows(1, seed=1)[0])
    e.flush()
    e.close()
    r = VaultEngine(filepath=vault_path, embed_dim=D)
    assert r.count() == 61
    assert r.stats()["truncated_tail_bytes"] == 0
    r.close()


# --------------------------------------------------------------------------
# corruption vs end-of-file
# --------------------------------------------------------------------------
def _flip(path, offset, bit=0x01):
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([b[0] ^ bit]))


def test_corrupt_block_header_mid_file_is_not_silent_eof(vault_path):
    """A stale-CRC block header with valid blocks after it is corruption.

    3.0.0 returned ``None`` from ``unpack_block_header`` and ``break``ed, so one
    flipped byte silently deleted that block and every block after it, with no
    error, no ``integrity_errors`` entry and a green ``stats()``.
    """
    _ingest(vault_path, 200, seed=9)
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    blocks = list(e._cont.blocks)
    assert len(blocks) >= 3
    e.close()
    _flip(vault_path, blocks[1].offset + 12)          # inside the block header
    with pytest.raises(IntegrityError):
        VaultEngine(filepath=vault_path, embed_dim=D)
    skipping = VaultEngine(filepath=vault_path, embed_dim=D, on_integrity_error="skip")
    assert skipping.stats()["integrity_errors"], "a skipped block must be reported"
    assert skipping.count() > blocks[1].n, "blocks after the damage must survive"
    skipping.close()


def test_truncated_last_block_is_still_end_of_file(vault_path):
    """The torn-tail case must NOT become an error: nothing parses after it."""
    _ingest(vault_path, 200, seed=9)
    size = os.path.getsize(vault_path)
    with open(vault_path, "r+b") as f:
        f.truncate(size - 500)
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    assert 0 < e.count() < 200
    assert e.stats()["integrity_errors"] == []
    e.close()


def test_replayed_block_is_rejected(vault_path):
    """A byte-for-byte duplicate of an earlier block has the wrong ``seq``."""
    _ingest(vault_path, 100, seed=13)
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    b0 = e._cont.blocks[0]
    e.close()
    with open(vault_path, "rb") as f:
        f.seek(b0.offset)
        chunk = f.read(b0.total_len)
    with open(vault_path, "ab") as f:
        pad = C.align_up(os.path.getsize(vault_path)) - os.path.getsize(vault_path)
        f.write(b"\x00" * pad)
        f.write(chunk)
    with pytest.raises(IntegrityError):
        VaultEngine(filepath=vault_path, embed_dim=D)


def test_swapped_blocks_are_rejected(vault_path):
    """Two equal-sized blocks exchanged keep valid MACs but not their order."""
    e = VaultEngine(filepath=vault_path, embed_dim=D)
    V = unit_rows(100, seed=14)
    for i in range(100):
        e.add_fact(f"row {i:06d}", V[i], id=f"fixed_id_{i:06d}")   # equal-width blocks
    e.flush()
    b0, b1 = e._cont.blocks[0], e._cont.blocks[1]
    assert b0.total_len == b1.total_len
    e.close()
    with open(vault_path, "r+b") as f:
        f.seek(b0.offset)
        c0 = f.read(b0.total_len)
        f.seek(b1.offset)
        c1 = f.read(b1.total_len)
        f.seek(b0.offset)
        f.write(c1)
        f.seek(b1.offset)
        f.write(c0)
    with pytest.raises(IntegrityError):
        VaultEngine(filepath=vault_path, embed_dim=D)


def test_readonly_open_of_a_missing_vault_creates_nothing(tmp_path):
    from nanomem.errors import ReadOnlyVaultError
    p = str(tmp_path / "nope.dat")
    # The round-2 verifier was right that `(ReadOnlyVaultError, Exception)`
    # accepts anything at all; it is the exact type now.
    with pytest.raises(ReadOnlyVaultError):
        VaultEngine(filepath=p, embed_dim=16, readonly=True)
    assert not os.path.exists(p)


@pytest.mark.skipif(os.name == "nt", reason=(
    "chmod 0600 is a POSIX concept; Windows has no mode bit to assert"))
def test_vault_file_is_owner_only(vault_path):
    _ingest(vault_path, 10, dim=16, seed=2)
    assert oct(os.stat(vault_path).st_mode & 0o777) == "0o600"


# --------------------------------------------------------------------------
# cross-process
# --------------------------------------------------------------------------
_WORKER = r"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from nanomem.engine import VaultEngine
path, tag, n, dim = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rng = np.random.default_rng(abs(hash(tag)) % 10000)
e = VaultEngine(filepath=path, embed_dim=dim)
ok = 0
for i in range(n):
    v = rng.normal(size=dim).astype(np.float32)
    e.add_fact(f"{tag}-{i}", v)
    e.flush()
    ok += 1
e.close()
print(ok)
"""


@pytest.mark.parametrize("procs,each", [(2, 60), (4, 40)])
def test_processes_appending_lose_nothing(tmp_path, procs, each):
    """Concurrent appends from several processes: every record survives.

    3.0.0 lost 19% of them silently at two processes and left the file
    unopenable at four, because a second process re-scanned from a stale
    *unaligned* offset that landed inside the 64-byte pad, failed to parse,
    rewound ``valid_end`` and then ``ftruncate``d away committed blocks.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = str(tmp_path / "shared.dat")
    VaultEngine(filepath=path, embed_dim=32).close()
    worker = str(tmp_path / "w.py")
    with open(worker, "w") as f:
        f.write(_WORKER)
    running = [subprocess.Popen([sys.executable, worker, root, path, f"p{i}",
                                 str(each), "32"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for i in range(procs)]
    outs = [p.communicate() for p in running]
    for (out, err), p in zip(outs, running):
        assert p.returncode == 0, err
        assert out.strip() == str(each), (out, err)
    e = VaultEngine(filepath=path, embed_dim=32)
    assert e.stats()["integrity_errors"] == []
    assert e.count() == procs * each, f"{e.count()} of {procs * each} records survived"
    ids = {e.arena.ids[i] for i in range(e.arena.n_rows)}
    assert len(ids) == procs * each
    e.close()


def test_rebuild_absorbs_a_concurrent_append(tmp_path):
    """A rebuild and an append cannot interleave: the append is kept, not lost.

    3.0.0's ``replace_all`` took no lock at all, so a successful ``add_fact``
    (id returned, ``flush()`` returned, no exception) vanished when another
    instance rebuilt the file.
    """
    path = str(tmp_path / "v.dat")
    V = unit_rows(40, 32, seed=17)
    a = VaultEngine(filepath=path, embed_dim=32)
    for i in range(20):
        a.add_fact(f"old {i}", V[i])
    a.flush()
    b = VaultEngine(filepath=path, embed_dim=32)          # second instance
    snapshot = list(b.iter_records())
    a.add_fact("IMPORTANT NEW FACT", V[30])               # committed by A
    a.flush()
    b.replace_all(snapshot)                               # rebuild from a stale snapshot
    r = VaultEngine(filepath=path, embed_dim=32)
    texts = [rec["text"] for rec in r.iter_records()]
    assert "IMPORTANT NEW FACT" in texts
    assert len(texts) == 21
    r.close()
    a.close()
    b.close()


def test_rebuild_after_replacement_raises_instead_of_guessing(tmp_path):
    path = str(tmp_path / "v.dat")
    V = unit_rows(10, 32, seed=18)
    a = VaultEngine(filepath=path, embed_dim=32)
    for i in range(5):
        a.add_fact(f"x{i}", V[i])
    a.flush()
    b = VaultEngine(filepath=path, embed_dim=32)
    snapshot = list(b.iter_records())
    a.replace_all(snapshot[:2])                            # A rewrites the file
    with pytest.raises(ContainerReplacedError):
        b.replace_all(snapshot)                            # B's view is stale
    a.close()
    b.close()


def test_count_agrees_with_stats_across_instances(tmp_path):
    path = str(tmp_path / "v.dat")
    V = unit_rows(60, 32, seed=19)
    a = VaultEngine(filepath=path, embed_dim=32)
    b = VaultEngine(filepath=path, embed_dim=32)
    for i in range(50):
        a.add_fact(f"x{i}", V[i])
    a.flush()
    assert b.count() == b.stats()["total_documents"] == 50
    a.close()
    b.close()


# --------------------------------------------------------------------------
# the memtable across a reload
# --------------------------------------------------------------------------
def test_pending_records_survive_an_arena_rebuild(tmp_path):
    """Interned ids are arena-local; a reload must re-resolve them.

    3.0.0 cached ``entity_id``/``group_id`` ints on memtable items and rebuilt
    the intern tables from disk on every reload, so two pending chat records
    tagged ``phone_number`` and ``credential`` both ended up meaning entity 2 --
    and the auto-revision counter restarted at 1.
    """
    path = str(tmp_path / "v.dat")
    V = unit_rows(12, 32, seed=23)
    e = VaultEngine(filepath=path, embed_dim=32)
    e.add_fact("my phone number is 555 000 1111", V[0],
               source="chat_session", metadata={"user_id": "u"})
    e.add_fact("my api key is sk-aaaa", V[1],
               source="chat_session", metadata={"user_id": "u"})
    before = [(it["entity_key"], it["revision"]) for it in e._mt.items]
    e._full_reload()
    after = [(e.arena.entity_names[it["entity_id"]], it["revision"]) for it in e._mt.items]
    assert [k for k, _ in before] == [k for k, _ in after]
    assert len(set(it["entity_id"] for it in e._mt.items)) == 2

    # and the revision counter keeps counting across a reload
    e.add_fact("my phone number is 555 000 2222", V[2],
               source="chat_session", metadata={"user_id": "u"})
    e._full_reload()
    e.add_fact("my phone number is 555 000 3333", V[3],
               source="chat_session", metadata={"user_id": "u"})
    revs = [it["revision"] for it in e._mt.items
            if it["entity_key"] == "phone_number"]
    assert revs == [1, 2, 3], revs
    e.close()
