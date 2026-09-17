"""The arena cache: an open that maps instead of rebuilding, and every way it
is allowed to be wrong.

The cache is DERIVED data. The vault is the truth. So the interesting tests are
not "does it make the fast path fast" -- that is a benchmark, and the numbers
live in ``scratch/refound/reopen_results.json`` -- but "does every corrupt,
stale, foreign, half-written or vanished cache produce exactly the answer a
cache-less open would have produced". That is what most of this file checks, one
failure mode per test, each by construction rather than by timing.

What a cached open does NOT re-read is the vault's blocks and the cache's own
bytes, and neither is hidden here: ``test_map_mode_does_not_notice_a_tampered_cache``
states one as a passing test, ``test_verify_mode_notices_a_tampered_cache``
shows the mode that re-reads both, and
``test_map_mode_refuses_a_vault_that_was_rewritten_in_place`` covers the case a
one-stat binding closed after a review found the default serving it in silence.

VOCABULARY, corrected here because three docstrings in this workstream got it
wrong. These are CHECKSUMS, not authentication. A cache is written only beside a
PLAINTEXT vault (an encrypted vault is refused one), and a plaintext vault's
block trailer is an unkeyed SHA-256 -- so the vault is forgeable
(``test_a_plaintext_vault_is_forgeable_with_no_cache_in_sight``), the cache's
content digest is forgeable
(``test_verify_mode_is_hijacked_by_a_forged_content_digest``), and the mtime
binding is defeated by ``os.utime``
(``test_the_stat_binding_is_defeated_by_restoring_mtime``). All three are
PASSING tests, deliberately, so that no future edit can re-promote any of this
to a security property without deleting a test that says otherwise. What these
modes buy is corruption detection, which is what a plaintext vault has ever
offered.
"""

import hashlib
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from conftest import unit_rows

from nanomem import arena as A
from nanomem import container as C
from nanomem.engine import VaultEngine

DIM = 64
ROWS = 400            # > A.ARENA_CACHE_MIN_ROWS, so a cache is written

#: THE COPYING SIDECAR, pinned explicitly. From 3.1.1 the default sidecar keeps
#: OFFSETS into the vault rather than copies of the vectors and the record
#: sections (``nanomem.arena.ARENA_VECTOR_SOURCES``), so a test about the
#: contents of the ``vec`` section -- planting a row in it, watching it be
#: mapped, watching it be copied out -- has to say which layout it is about.
#: Every such test below passes ``**COPYING``; the ones that do not are about
#: behaviour the layout does not change and run on the default.
COPYING = dict(arena_cache_vectors="cache", arena_cache_records="cache")


def fill(path, n=ROWS, start=0, dim=DIM, seed=3, **kw):
    """Build (or extend) a vault and close it. Returns the vectors used."""
    V = unit_rows(start + n, dim=dim, seed=seed)
    e = VaultEngine(path, embed_dim=dim, durable="none", **kw)
    for i in range(start, start + n):
        e.add_fact(f"record number {i}", V[i], source="wiki", metadata={"idx": i})
    e.flush()
    e.close()
    return V


def queries(n=12, dim=DIM, seed=99):
    return unit_rows(n, dim=dim, seed=seed)


def scores_digest(engine, Q):
    """SHA-256 over the full fp32 score vector of every query against every row.

    The whole scan, not the top-k: a digest catches a changed row that never
    reaches an answer, which is the only way to show that mapping the vectors
    changed no number anywhere.
    """
    h = hashlib.sha256()
    for q in Q:
        h.update(np.ascontiguousarray(engine.arena.scores(q), dtype=np.float32).tobytes())
    return h.hexdigest()


def top_ids(engine, Q, k=10):
    return [tuple(x["metadata"]["idx"] for x in engine.search("", q, top_k=k)) for q in Q]


def warm(path, **kw):
    """Open once so that the cache exists, then close."""
    VaultEngine(path, embed_dim=DIM, durable="none", **kw).close()


def warm_copying(path, **kw):
    """The same, for a test that is about what is IN the sidecar."""
    kw.update(COPYING)
    VaultEngine(path, embed_dim=DIM, durable="none", **kw).close()


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------
def test_first_open_scans_and_writes_second_open_maps(vault_path):
    fill(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    first = e.arena_cache_info()
    e.close()
    assert first["source"] == "scan"
    assert first["reason"] == "no readable cache"
    assert first["write"]["bytes"] > 0
    assert os.path.exists(A.arena_cache_path(vault_path))

    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    second = e.arena_cache_info()
    assert second["source"] == "cache"
    assert second["rows_from_cache"] == ROWS
    assert second["rows_scanned"] == 0
    assert e.count() == ROWS
    assert e.arena.from_cache is True
    e.close()


def test_the_cache_changes_no_score_no_answer_and_no_record(vault_path):
    """Every mode returns the same scores, the same top-10, the same text."""
    fill(vault_path)
    warm(vault_path)
    Q = queries()
    ref = None
    for mode in ("off", "map", "copy", "verify"):
        e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache=mode)
        got = {
            "scores": scores_digest(e, Q),
            "top": top_ids(e, Q),
            "ids": [e.arena.ids[r] for r in (0, 1, ROWS // 2, ROWS - 1)],
            "texts": [e.arena.record(r)["text"] for r in (0, ROWS // 2, ROWS - 1)],
            "meta": [e.arena.metadata(r)["idx"] for r in (0, ROWS // 2, ROWS - 1)],
            "ts": e.arena.ts[:ROWS].tolist(),
            "vec0": e.arena.vector(0).tobytes(),
            "records": [r["id"] for r in e.iter_records(include_embeddings=False)],
        }
        e.close()
        if ref is None:
            ref = got
        else:
            assert got == ref, f"arena_cache={mode!r} changed an answer"


def test_a_cached_open_holds_its_arrays_in_mapped_pages(vault_path):
    fill(vault_path)
    warm(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    st = e.stats()
    assert st["arena_source"] == "cache"
    assert st["arena_vectors_mapped"] is True
    # The vectors, the columns, the record sections and the id table are all in
    # the mapping; nothing of the arena is anonymous heap.
    assert st["mapped_bytes"] >= ROWS * DIM * 4
    assert st["record_mapped_bytes"] > 0
    assert st["id_table_mapped_bytes"] > 0
    assert st["resident_arena_mb"] == 0.0
    assert st["record_bytes"] == 0
    e.close()

    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    st = e.stats()
    assert st["arena_source"] == "scan"
    assert st["mapped_bytes"] == 0
    assert st["resident_arena_mb"] > 0
    assert st["record_bytes"] > 0
    e.close()


def test_copy_mode_maps_the_file_but_not_the_vectors(vault_path):
    fill(vault_path)
    warm_copying(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="copy",
                    **COPYING)
    st = e.stats()
    assert st["arena_from_cache"] is True
    assert st["arena_vectors_mapped"] is False       # copied into anonymous RAM
    assert st["resident_arena_mb"] > 0
    assert st["record_mapped_bytes"] > 0             # the records stay mapped
    e.close()


def test_the_lazy_tables_are_not_built_by_an_open(vault_path):
    """A cached open touches none of the four Python tables, and every one of
    them is still correct when something finally asks for it."""
    fill(vault_path)
    warm(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    assert e.arena._id_index is None
    assert e.arena._entity_names is None
    assert e.arena._group_keys is None
    assert e.arena._group_max is None
    wanted = e.arena.ids[7]
    assert e.get(wanted)["metadata"]["idx"] == 7
    assert e.arena._id_index is not None
    assert len(e.arena.id_index) == ROWS
    assert e.arena.entity_names == []                # this corpus tags nothing
    e.close()


def test_ids_are_decoded_one_at_a_time_and_match_the_scan(vault_path):
    fill(vault_path)
    warm(vault_path)
    a = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    b = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    assert isinstance(b.arena.ids, A._MappedStrings)
    assert [a.arena.ids[i] for i in range(ROWS)] == [b.arena.ids[i] for i in range(ROWS)]
    assert list(a.arena.ids) == list(b.arena.ids)
    assert b.arena.ids[-1] == a.arena.ids[ROWS - 1]
    a.close()
    b.close()


# --------------------------------------------------------------------------
# appending
# --------------------------------------------------------------------------
def test_a_small_append_is_scanned_and_the_cache_is_kept(vault_path):
    fill(vault_path)
    warm(vault_path)
    fill(vault_path, n=50, start=ROWS)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    info = e.arena_cache_info()
    assert info["source"] == "cache+append"
    assert info["rows_from_cache"] == ROWS
    assert info["rows_scanned"] == 50
    assert "write" not in info                       # below the refresh threshold
    assert e.count() == ROWS + 50
    assert e.get(e.arena.ids[ROWS + 49])["metadata"]["idx"] == ROWS + 49
    e.close()


@pytest.mark.skipif(os.name == "nt", reason=(
    "Windows will not os.replace a file this process has mapped, and a live arena cache is mapped by definition -- the write is recorded as a write_error, the old cache stays and the next open re-scans, so nothing is lost but the optimisation"))
def test_a_large_append_rewrites_the_cache(vault_path):
    fill(vault_path)
    warm(vault_path)
    fill(vault_path, n=A.ARENA_CACHE_REFRESH_ROWS + 10, start=ROWS)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none",
                    arena_cache_refresh_rows=A.ARENA_CACHE_REFRESH_ROWS)
    info = e.arena_cache_info()
    assert info["source"] == "cache+append"
    assert info["write"]["bytes"] > 0
    e.close()
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    assert e.arena_cache_info()["rows_from_cache"] == ROWS + A.ARENA_CACHE_REFRESH_ROWS + 10
    e.close()


def test_writing_into_a_cached_arena_leaves_the_mapping_behind(vault_path):
    """The first append copies the mapped rows into anonymous memory ONCE."""
    V = fill(vault_path)
    warm_copying(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", **COPYING)
    assert e.arena.vec.flags.writeable is False      # mapped, read-only
    before = [e.arena.ids[i] for i in range(ROWS)]
    e.add_fact("a new record", unit_rows(1, dim=DIM, seed=77)[0], source="wiki",
               metadata={"idx": -1})
    e.flush()
    assert e.arena.vec.flags.writeable is True
    assert e.arena.n_rows == ROWS + 1
    assert [e.arena.ids[i] for i in range(ROWS)] == before
    assert np.allclose(e.arena.vector(3), V[3].astype(np.float16).astype(np.float32))
    assert e.get(e.arena.ids[ROWS])["metadata"]["idx"] == -1
    assert e.stats()["arena_growth_copies"] == 1
    e.close()


def test_the_columns_survive_the_copy_out_of_the_mapping(vault_path):
    fill(vault_path)
    warm(vault_path)
    a = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    ts_ref, rev_ref = a.arena.ts[:ROWS].copy(), a.arena.rev[:ROWS].copy()
    span_ref = a.arena.doc_span[:ROWS].copy()
    a.close()
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    e.add_fact("one more", unit_rows(1, dim=DIM, seed=5)[0], source="wiki",
               metadata={"idx": -2})
    e.flush()
    assert np.array_equal(e.arena.ts[:ROWS], ts_ref)
    assert np.array_equal(e.arena.rev[:ROWS], rev_ref)
    assert np.array_equal(e.arena.doc_span[:ROWS], span_ref)
    assert e.arena.record(ROWS - 1)["metadata"]["idx"] == ROWS - 1
    e.close()


# --------------------------------------------------------------------------
# every way the cache is allowed to be wrong
# --------------------------------------------------------------------------
def _expect_rebuild(path, reason_contains, rows):
    e = VaultEngine(path, embed_dim=DIM, durable="none")
    info = e.arena_cache_info()
    assert info["source"] == "scan", info
    assert reason_contains in (info["reason"] or "")
    assert e.count() == rows
    e.close()


def test_a_missing_cache_is_simply_rebuilt(vault_path):
    fill(vault_path)
    warm(vault_path)
    os.remove(A.arena_cache_path(vault_path))
    _expect_rebuild(vault_path, "no readable cache", ROWS)


def test_a_cache_from_another_vault_is_refused(vault_path, tmp_path):
    fill(vault_path)
    warm(vault_path)
    other = str(tmp_path / "other.dat")
    fill(other, n=ROWS, seed=11)
    warm(other)
    shutil.copyfile(A.arena_cache_path(other), A.arena_cache_path(vault_path))
    _expect_rebuild(vault_path, "does not bind", ROWS)


def test_a_corrupt_cache_header_is_refused(vault_path):
    fill(vault_path)
    warm(vault_path)
    with open(A.arena_cache_path(vault_path), "r+b") as f:
        f.seek(64)
        f.write(b"\x00\xff\x00\xff")
    _expect_rebuild(vault_path, "no readable cache", ROWS)


def test_a_cache_with_a_foreign_magic_is_refused(vault_path):
    fill(vault_path)
    warm(vault_path)
    with open(A.arena_cache_path(vault_path), "r+b") as f:
        f.seek(0)
        f.write(b"NOTARENA")
    _expect_rebuild(vault_path, "no readable cache", ROWS)


def test_a_truncated_cache_is_refused(vault_path):
    fill(vault_path)
    warm(vault_path)
    p = A.arena_cache_path(vault_path)
    with open(p, "r+b") as f:
        f.truncate(os.path.getsize(p) - A.ARENA_CACHE_ALIGN)
    _expect_rebuild(vault_path, "no readable cache", ROWS)


def test_a_cache_whose_vault_was_rewritten_is_refused(vault_path):
    """Same length, same uuid, different bytes: the last block's trailer moves."""
    fill(vault_path)
    warm(vault_path)
    size = os.path.getsize(vault_path)
    with open(vault_path, "r+b") as f:       # scribble on the last block's trailer
        f.seek(size - C.TRAILER_SIZE)
        f.write(bytes(C.TRAILER_SIZE))
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none",
                    on_integrity_error="skip")
    assert e.arena_cache_info()["source"] == "scan"
    assert "does not bind" in e.arena_cache_info()["reason"]
    e.close()


def test_a_cache_written_for_a_different_row_count_is_refused(vault_path):
    """The header is rewritten to claim rows the vault does not have."""
    fill(vault_path)
    warm(vault_path)
    p = A.arena_cache_path(vault_path)
    with open(p, "rb") as f:
        raw = f.read(A.ARENA_CACHE_HEADER)
    hdr = A._parse_cache_header(raw)
    hdr["n_rows"] = ROWS + 1
    with open(p, "r+b") as f:
        f.seek(0)
        f.write(A._pack_cache_header(hdr))
    _expect_rebuild(vault_path, "", ROWS)


@pytest.mark.skipif(os.name == "nt", reason=(
    "the setup deletes a cache file this process has mapped, which POSIX allows and Windows refuses"))
def test_deleting_the_cache_mid_run_does_not_disturb_a_live_vault(vault_path):
    """POSIX keeps the inode alive for the mapping; the name is only a name."""
    fill(vault_path)
    warm(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    Q = queries()
    before = top_ids(e, Q)
    os.remove(A.arena_cache_path(vault_path))
    assert top_ids(e, Q) == before
    assert e.arena.record(ROWS - 1)["metadata"]["idx"] == ROWS - 1
    e.close()


@pytest.mark.skipif(os.name == "nt", reason=(
    "Windows will not remove a file this process has mapped"))
def test_replace_all_removes_the_cache_it_invalidates(vault_path):
    fill(vault_path)
    warm(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    recs = list(e.iter_records())[:20]
    e.replace_all(recs)
    assert not os.path.exists(A.arena_cache_path(vault_path))
    assert e.count() == 20
    e.close()


# --------------------------------------------------------------------------
# what a cached open does NOT check
# --------------------------------------------------------------------------
def _tamper_with_a_record(path):
    """Flip a byte inside the first block's payload (not its header, not its
    trailer), which is exactly what a container scan authenticates and a bound
    cache does not re-read."""
    off = C.FILE_HEADER_SIZE + C.BLOCK_HEADER_SIZE + 16
    with open(path, "r+b") as f:
        f.seek(off)
        b = f.read(1)
        f.seek(off)
        f.write(bytes([b[0] ^ 0xFF]))


def test_map_mode_refuses_a_vault_that_was_rewritten_in_place(vault_path):
    """THE DEFAULT NO LONGER SERVES THIS, and the change is the point of the test.

    Editing a block's payload and leaving its trailer stale is what a scan
    exists to catch: ``off`` and ``verify`` both raise ``IntegrityError``. Until
    the (size, mtime) binding was added, ``map`` -- the DEFAULT -- turned that
    hard error into a successful open with no error, no warning and no
    ``integrity_errors`` entry, because a bound cache never re-reads the blocks.

    The rewrite is the same length, so no length check can see it; what sees it
    is the vault's mtime moving away from the one the cache recorded. The cache
    is refused, the open falls back to a full scan, and the scan raises -- the
    same answer the other two modes give.
    """
    from nanomem.errors import IntegrityError
    fill(vault_path)
    warm(vault_path)
    before = os.path.getsize(vault_path)
    _tamper_with_a_record(vault_path)
    assert os.path.getsize(vault_path) == before          # same length, no length check helps
    with pytest.raises(IntegrityError):
        VaultEngine(vault_path, embed_dim=DIM, durable="none")    # arena_cache="map"


def test_the_stat_binding_is_defeated_by_restoring_mtime(vault_path):
    """...and here is exactly what that binding is worth, as a PASSING test.

    ``os.utime`` is one line. Put the mtime back and ``map`` serves the tampered
    block again, silently, just as it did before. The binding is a CORRUPTION
    check -- it catches a tool or a crash that rewrote the file, not an attacker
    -- and this test exists so that nobody can promote it to a security property
    without deleting a test that says otherwise.
    """
    fill(vault_path)
    warm(vault_path)
    st = os.stat(vault_path)
    _tamper_with_a_record(vault_path)
    os.utime(vault_path, ns=(st.st_atime_ns, st.st_mtime_ns))      # the whole attack
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")     # map
    info = e.arena_cache_info()
    assert info["source"] == "cache"
    assert info["vault_changed_since_cache"] == "unchanged"
    assert info["vault_blocks_checked"] == "none"
    assert e.count() == ROWS                                       # served anyway
    e.close()


def test_a_cache_that_records_no_stat_does_not_bind(vault_path):
    """An absent binding is not a passed one, so it is refused like any other
    cache that cannot prove it belongs to this vault at this length."""
    fill(vault_path)
    warm(vault_path)
    cpath = A.arena_cache_path(vault_path)
    raw = bytearray(open(cpath, "rb").read())
    hdr = A._parse_cache_header(bytes(raw[:A.ARENA_CACHE_HEADER]))
    hdr["source_stat_size"] = -1
    hdr["source_mtime_ns"] = -1
    raw[:A.ARENA_CACHE_HEADER] = A._pack_cache_header(hdr)
    with open(cpath, "wb") as f:
        f.write(bytes(raw))
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    info = e.arena_cache_info()
    assert info["source"] == "scan"
    assert "size and mtime" in info["reason"]
    assert e.count() == ROWS
    e.close()


def test_an_append_says_the_vault_grew_and_the_tail_is_the_part_re_read(vault_path):
    """The residual hole, named in the field the caller can read: a vault that
    GREW keeps its cache, so only the appended tail's trailers are checked on
    this open. An edit to the prefix followed by an append is still not caught.
    """
    fill(vault_path)
    warm(vault_path)
    fill(vault_path, n=50, start=ROWS)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    info = e.arena_cache_info()
    assert info["source"] == "cache+append"
    assert info["vault_changed_since_cache"] == "grew"
    assert info["vault_blocks_checked"] == "appended tail only"
    e.close()


def test_a_scan_and_verify_both_say_every_block_was_re_read(vault_path):
    fill(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    assert e.arena_cache_info()["vault_blocks_checked"] == "all"
    e.close()
    warm(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="verify")
    info = e.arena_cache_info()
    assert info["source"] == "cache"
    assert info["vault_blocks_checked"] == "all"
    e.close()
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")     # map
    assert e.arena_cache_info()["vault_blocks_checked"] == "none"
    e.close()


def test_verify_mode_notices_a_tampered_block(vault_path):
    from nanomem.errors import IntegrityError
    fill(vault_path)
    warm(vault_path)
    _tamper_with_a_record(vault_path)
    with pytest.raises(IntegrityError):
        VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="verify")


def test_cache_off_notices_a_tampered_block(vault_path):
    from nanomem.errors import IntegrityError
    fill(vault_path)
    warm(vault_path)
    _tamper_with_a_record(vault_path)
    with pytest.raises(IntegrityError):
        VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")


def _plant_a_row_in_the_cache(path, row, vec):
    """Overwrite one row of the cache's ``vec`` section, leaving every length,
    the header and the vault itself alone. This is the tamper the VAULT's MACs
    cannot see, because it never touches the vault."""
    raw = open(path + A.ARENA_CACHE_SUFFIX, "rb").read()
    hdr = A._parse_cache_header(raw[:A.ARENA_CACHE_HEADER])
    spec = hdr["sections"]["vec"]
    rb = int(spec["shape"][1]) * np.dtype(spec["dtype"]).itemsize
    off = int(spec["off"]) + int(row) * rb
    buf = bytearray(raw)
    buf[off:off + rb] = np.ascontiguousarray(vec, dtype=np.dtype(spec["dtype"])).tobytes()
    assert len(buf) == len(raw)              # same size, so no length check helps
    with open(path + A.ARENA_CACHE_SUFFIX, "wb") as f:
        f.write(bytes(buf))


def test_map_mode_does_not_notice_a_tampered_cache(vault_path):
    """The SECOND thing a cached open does not re-read: the cache itself.

    Nothing in ``map`` looks at the cache's own bytes, so a planted row comes
    back at cosine 1.0 and takes top-1 while the vault sits there intact.

    WITHDRAWN, because this docstring used to say it: "tampering with the vault
    is caught by its per-block MACs, which makes the cache a strictly weaker
    path to the same answers." It is not weaker. A cache is only ever written
    beside a PLAINTEXT vault -- an encrypted one is refused a cache outright --
    and a plaintext vault's block trailer is an unkeyed SHA-256 that anyone can
    recompute (nanomem/crypto.py, THREAT_MODEL). The vault is forgeable too, and
    ``test_a_plaintext_vault_is_forgeable_with_no_cache_in_sight`` below does it
    with no cache on disk at all. The sidecar inherits the vault's threat model;
    it does not lower it.
    """
    fill(vault_path)
    warm_copying(vault_path)
    Q = queries()
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    honest = top_ids(e, Q)
    e.close()

    _plant_a_row_in_the_cache(vault_path, 0, Q[0])
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", **COPYING)   # map
    assert e.arena_cache_info()["source"] == "cache"
    got = top_ids(e, Q)
    assert float(np.asarray(e.arena.scores(Q[0]))[0]) == pytest.approx(1.0, abs=1e-6)
    assert got[0][0] != honest[0][0]                 # the planted row took top-1
    e.close()


def test_verify_mode_notices_a_tampered_cache(vault_path):
    """...and ``verify`` catches it, as long as nobody updates the digest.

    The vault is untouched here, so re-reading the vault's blocks passes and
    proves nothing about the cache; only the cache's own content digest sees
    this edit. The open must still SUCCEED -- the vault is fine -- and serve
    exactly the cache-less answers.

    READ THIS WITH THE TEST BELOW IT. The digest is unkeyed and lives in the
    header of the file it describes, so this test shows ``verify`` catching a
    cache that was DAMAGED, not one that was FORGED. Forge it and ``verify``
    serves the planted row:
    ``test_verify_mode_is_hijacked_by_a_forged_content_digest``.
    """
    fill(vault_path)
    warm_copying(vault_path)
    Q = queries()
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    honest, honest_digest = top_ids(e, Q), scores_digest(e, Q)
    e.close()

    _plant_a_row_in_the_cache(vault_path, 0, Q[0])
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="verify",
                    **COPYING)
    info = e.arena_cache_info()
    assert info["source"] == "scan"
    assert "digest" in (info["reason"] or "")
    assert top_ids(e, Q) == honest
    assert scores_digest(e, Q) == honest_digest
    e.close()


def test_verify_mode_is_hijacked_by_a_forged_content_digest(vault_path):
    """THE WITHDRAWAL, as a passing test: ``verify`` is a corruption check.

    It was reported as "a real security defect, found and fixed" and as
    "verify SAFE". That was wrong, and this is the counter-example, ~10 lines
    long: plant the row, recompute the SHA-256 over the body, write it into the
    header's ``content_sha256``, recompute the header CRC32. Every check
    ``verify`` makes now passes and it serves the planted row at cosine 1.0.

    The digest is unkeyed and stored INSIDE the file it authenticates, behind a
    CRC32 that is also just a checksum, so the attacker who can edit the cache
    can always update both. There is no key anywhere to fix this with: a
    passphrase would make the vault's trailers real HMACs, and a vault with a
    passphrase is refused a cache. ``verify`` detects CORRUPTION in the cache
    and in the vault. It does not detect TAMPERING.
    """
    fill(vault_path)
    warm_copying(vault_path)
    Q = queries()
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    honest = top_ids(e, Q)
    e.close()

    _plant_a_row_in_the_cache(vault_path, 0, Q[0])
    cpath = A.arena_cache_path(vault_path)
    raw = bytearray(open(cpath, "rb").read())
    hdr = A._parse_cache_header(bytes(raw[:A.ARENA_CACHE_HEADER]))
    hdr["content_sha256"] = hashlib.sha256(
        bytes(raw[A.ARENA_CACHE_HEADER:int(hdr["total_bytes"])])).hexdigest()
    raw[:A.ARENA_CACHE_HEADER] = A._pack_cache_header(hdr)      # CRC recomputed in here
    with open(cpath, "wb") as f:
        f.write(bytes(raw))

    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="verify",
                    **COPYING)
    info = e.arena_cache_info()
    assert info["source"] == "cache"                 # every check passed
    assert info["reason"] is None
    assert float(np.asarray(e.arena.scores(Q[0]))[0]) == pytest.approx(1.0, abs=1e-6)
    assert top_ids(e, Q)[0][0] != honest[0][0]       # the planted row took top-1
    e.close()


def test_the_default_sidecar_has_no_vectors_to_plant_a_row_in(vault_path):
    """The forgeries above all need a ``vec`` section, and the DEFAULT layout
    has none.

    This is not a security claim -- a plaintext vault is forgeable directly, as
    the test below does with no cache on disk at all, and the sidecar has always
    inherited that threat model rather than lowering it. It is the narrower and
    checkable fact: from 3.1.1 the bytes a query is answered from are the
    VAULT's, so the second file beside it is no longer somewhere a row can be
    planted, and ``verify`` -- which re-reads every vault block -- now covers the
    bytes it serves rather than a digest of a copy of them.
    """
    fill(vault_path)
    warm(vault_path)                                  # the default layout
    with pytest.raises(KeyError):
        _plant_a_row_in_the_cache(vault_path, 0, queries()[0])


def test_a_plaintext_vault_is_forgeable_with_no_cache_in_sight(vault_path):
    """The premise the whole "the sidecar is weaker" story rested on, refuted.

    No cache exists here and ``arena_cache="off"`` never looks for one, so this
    is the full scan that re-reads every block and recomputes every trailer --
    the path that was called safe. Rewrite one row's vector inside the block
    body, recompute the UNKEYED trailer with the library's own
    ``crypto.block_tag(None, ...)``, and the scan accepts it and returns the
    planted row at cosine 1.0.

    This is not a bug in the container; crypto.py's THREAT_MODEL says exactly
    this ("in the default plaintext mode there is NO protection at all"). It is
    here because a sidecar cannot be a WEAKER path than a path that is already
    open, and no document in this repository may say otherwise again.
    """
    from nanomem import crypto
    V = fill(vault_path)
    cpath = A.arena_cache_path(vault_path)
    if os.path.exists(cpath):
        os.remove(cpath)
    victim, q = 7, queries()[0]

    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    honest = int(e.search("", q, top_k=1)[0]["metadata"]["idx"])
    blk = int(e.arena.row_block[victim])
    meta = list(e._cont.blocks)[blk]
    off_in_blk = victim - int(e.arena.block_start[blk])
    uuid = bytes(e.header.vault_uuid)
    e.close()
    assert honest != victim

    with open(vault_path, "r+b") as f:
        f.seek(int(meta.offset))
        hdr_bytes = f.read(C.BLOCK_HEADER_SIZE)
        body = bytearray(f.read(int(meta.payload_len)))
        vecs = np.frombuffer(bytes(body), dtype=np.float16,
                             count=meta.n * DIM, offset=0).reshape(meta.n, DIM).copy()
        vecs[off_in_blk] = q.astype(np.float16)
        body[:meta.n * DIM * 2] = vecs.tobytes()
        f.seek(int(meta.offset) + C.BLOCK_HEADER_SIZE)
        f.write(bytes(body))
        f.write(crypto.block_tag(None, uuid, hdr_bytes[:C.BLOCK_AUTH_LEN], bytes(body)))

    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    info = e.arena_cache_info()
    assert info["source"] == "scan" and info["vault_blocks_checked"] == "all"
    assert int(e.search("", q, top_k=1)[0]["metadata"]["idx"]) == victim
    assert float(e.search("", q, top_k=1)[0]["cosine"]) == pytest.approx(1.0, abs=1e-3)
    e.close()
    assert V is not None


def test_verify_mode_refuses_a_cache_that_carries_no_content_digest(vault_path):
    """A cache written without a digest has nothing to check, so ``verify`` must
    not accept it: an absent proof is not a passed one."""
    fill(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    state = e._cont.snapshot_state()
    state["engine_version"] = None
    A.ArenaSnapshot.write(A.arena_cache_path(vault_path), e.arena, state,
                          durable=False, digest=False)
    e.close()
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="verify",
                    **COPYING)
    info = e.arena_cache_info()
    assert info["source"] == "scan"
    assert "digest" in (info["reason"] or "")
    assert e.count() == ROWS
    e.close()


def test_verify_mode_falls_back_to_the_scan_and_serves_a_clean_vault(vault_path):
    fill(vault_path)
    warm(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="verify")
    info = e.arena_cache_info()
    assert info["source"] == "cache"
    assert info["verify_s"] >= 0.0
    assert e.count() == ROWS
    e.close()


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------
def test_an_encrypted_vault_never_writes_a_cache(vault_path):
    pw = "a passphrase for this test only"
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", password=pw)
    V = unit_rows(ROWS, dim=DIM, seed=4)
    for i in range(ROWS):
        e.add_fact(f"record number {i}", V[i], source="wiki", metadata={"idx": i})
    e.flush()
    e.close()
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", password=pw)
    assert e.arena_cache_info()["reason"] == "vault is encrypted"
    assert not os.path.exists(A.arena_cache_path(vault_path))
    e.close()


def test_a_read_only_open_uses_a_cache_but_never_writes_one(vault_path):
    fill(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", readonly=True)
    assert e.arena_cache_info()["source"] == "scan"
    assert not os.path.exists(A.arena_cache_path(vault_path))
    e.close()
    warm(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", readonly=True)
    assert e.arena_cache_info()["source"] == "cache"
    e.close()


def test_a_vault_below_the_minimum_gets_no_cache(vault_path):
    fill(vault_path, n=A.ARENA_CACHE_MIN_ROWS - 1)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    e.close()
    assert not os.path.exists(A.arena_cache_path(vault_path))


def test_arena_cache_off_writes_nothing_and_reads_nothing(vault_path):
    fill(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    assert e.arena_cache_info()["reason"] == "mode=off"
    e.close()
    assert not os.path.exists(A.arena_cache_path(vault_path))
    warm(vault_path)
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    assert e.arena_cache_info()["source"] == "scan"
    e.close()


def test_an_unknown_mode_is_refused(vault_path):
    with pytest.raises(ValueError):
        VaultEngine(vault_path, embed_dim=DIM, arena_cache="mmap")


def test_the_cache_temp_name_is_one_the_container_already_sweeps(vault_path):
    """The cache writes ``<vault>.tmp-arena-*``, which
    :func:`nanomem.container.cleanup_stale_temps` and ``users.delete_user``
    already glob for, so a crashed write cannot leave litter nobody owns."""
    assert A._TMP_SUFFIX == C.TMP_SUFFIX
    fill(vault_path)
    warm(vault_path)
    stale = vault_path + C.TMP_SUFFIX + "arena-1-deadbeef"
    with open(stale, "wb") as f:
        f.write(b"x" * 10)
    os.utime(stale, (0, 0))
    VaultEngine(vault_path, embed_dim=DIM, durable="none").close()
    assert not os.path.exists(stale)


@pytest.mark.skipif(os.name == "nt", reason=(
    "chmod 0600 is a POSIX concept; on Windows the file inherits the directory ACL and there is no mode bit to assert"))
def test_the_cache_file_is_owner_only(vault_path):
    fill(vault_path)
    warm(vault_path)
    mode = os.stat(A.arena_cache_path(vault_path)).st_mode & 0o777
    assert oct(mode) == "0o600"


def test_the_content_digest_the_cache_records_is_the_content_it_holds(vault_path):
    """Recorded at write time, checkable at any time, not checked at open --
    verifying it is O(bytes) and the point of the cache is that an open is
    not."""
    fill(vault_path)
    warm(vault_path)
    snap = A.ArenaSnapshot.open(A.arena_cache_path(vault_path))
    assert snap is not None
    assert snap.content_sha256() == snap.header["content_sha256"]
    snap.close()


# --------------------------------------------------------------------------
# cross-process
# --------------------------------------------------------------------------
_OPENER = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from nanomem.engine import VaultEngine
e = VaultEngine(sys.argv[2], embed_dim=int(sys.argv[3]), durable="none")
print("%s %d" % (e.arena_cache_info()["source"], e.count()))
e.close()
"""

_APPENDER = r"""
import os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from nanomem.engine import VaultEngine
path, dim, n, start = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rng = np.random.default_rng(1234)
e = VaultEngine(path, embed_dim=dim, durable="none")
for i in range(start, start + n):
    v = rng.normal(size=dim).astype(np.float32)
    e.add_fact("appended %d" % i, v / np.linalg.norm(v), source="wiki",
               metadata={"idx": i})
e.flush()
e.close()
print(n)
"""


def _script(tmp_path, body, name):
    p = str(tmp_path / name)
    with open(p, "w") as f:
        f.write(body)
    return p


def test_two_processes_opening_at_once_both_get_a_correct_vault(tmp_path, vault_path):
    """Both may decide to write the cache; ``os.replace`` makes the loser
    harmless, and neither may see a half-written file."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fill(vault_path)
    opener = _script(tmp_path, _OPENER, "opener.py")
    procs = [subprocess.Popen([sys.executable, opener, root, vault_path, str(DIM)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(4)]
    outs = [p.communicate() for p in procs]
    for (out, err), p in zip(outs, procs):
        assert p.returncode == 0, err
        src, n = out.split()
        assert int(n) == ROWS, out
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    assert e.count() == ROWS
    assert e.arena_cache_info()["source"] == "cache"
    e.close()


def test_another_process_appending_after_the_cache_was_written(tmp_path, vault_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fill(vault_path)
    warm(vault_path)
    appender = _script(tmp_path, _APPENDER, "appender.py")
    out = subprocess.run([sys.executable, appender, root, vault_path, str(DIM),
                          "60", str(ROWS)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    e = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    info = e.arena_cache_info()
    assert info["source"] == "cache+append"
    assert info["rows_from_cache"] == ROWS
    assert info["rows_scanned"] == 60
    assert e.count() == ROWS + 60
    assert e.get(e.arena.ids[ROWS + 59])["metadata"]["idx"] == ROWS + 59
    e.close()


def test_a_cached_reader_is_unharmed_by_a_writer_replacing_the_cache(tmp_path, vault_path):
    """A live reader holds a mapping of one inode while another process replaces
    the cache file underneath it. ``os.replace`` unlinks rather than rewrites, so
    the reader's pages stay exactly what they were -- and the reader still tracks
    the VAULT, which is what it is supposed to track: it reloads the appended
    rows like any other open engine and its answers stay equal to a cache-less
    engine's, which is the only thing that matters."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fill(vault_path)
    warm(vault_path)
    reader = VaultEngine(vault_path, embed_dim=DIM, durable="none")
    Q = queries()
    ids_before = [reader.arena.ids[i] for i in range(ROWS)]
    appender = _script(tmp_path, _APPENDER, "appender.py")
    subprocess.run([sys.executable, appender, root, vault_path, str(DIM),
                    str(A.ARENA_CACHE_REFRESH_ROWS + 5), str(ROWS)],
                   capture_output=True, text=True, check=True)
    subprocess.run([sys.executable, _script(tmp_path, _OPENER, "o2.py"), root,
                    vault_path, str(DIM)], capture_output=True, text=True, check=True)
    fresh = VaultEngine(vault_path, embed_dim=DIM, durable="none", arena_cache="off")
    assert reader.count() == fresh.count() == ROWS + A.ARENA_CACHE_REFRESH_ROWS + 5
    assert top_ids(reader, Q) == top_ids(fresh, Q)
    assert scores_digest(reader, Q) == scores_digest(fresh, Q)
    assert [reader.arena.ids[i] for i in range(ROWS)] == ids_before
    fresh.close()
    reader.close()
