"""Resident-arena shape: pre-sizing, the four residency modes, and exactness.

The rule these tests exist to hold: nanomem's headline is that its recall EQUALS
an exhaustive scan. Making the arena smaller must not change a single returned
id. Three of the four modes are therefore held to BIT-IDENTICAL output against
``float32``; the fourth (``int8``) is the one that is allowed to differ, and its
contract is written down here instead of assumed.

Every measured figure quoted in a comment comes from
``scratch/refound/memory_results.json``.
"""

import os

import numpy as np
import pytest

from conftest import unit_rows

from nanomem.arena import Arena, RESIDENCY_MODES, _next_capacity
from nanomem.engine import VaultEngine

DIM = 64
MODES = list(RESIDENCY_MODES)
EXACT_MODES = ["float32", "float16", "float16_mmap"]

MEMORY_RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scratch", "refound", "memory_results.json")


def measured():
    """The residency benchmark's own output, or a skip. Never a hard-coded number."""
    if not os.path.exists(MEMORY_RESULTS):
        pytest.skip(f"residency measurements not available: {MEMORY_RESULTS}")
    import json
    with open(MEMORY_RESULTS) as fh:
        return json.load(fh)


def build(path, n=260, dim=DIM, seed=3, **kw):
    V = unit_rows(n, dim, seed)
    e = VaultEngine(path, embed_dim=dim, **kw)
    for i in range(n):
        e.add_fact(f"record number {i}", V[i], source="wiki", metadata={"idx": i})
    e.flush()
    e.close()
    return V


# ---------------------------------------------------------------------------
# exactness: the whole point
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", EXACT_MODES)
def test_narrow_residency_returns_bit_identical_results(tmp_path, mode):
    """fp16 resident == fp32 resident, id for id and BIT for bit on the score.

    This is not luck. The vault stores fp16 on disk, so an fp32 arena is an
    upcast of exactly the values an fp16 arena holds, and the chunked kernel
    accumulates in fp32. Measured on the real 71,433-document corpus: 0 of 500
    queries changed in the top 10 (memory_results.json, exactness.vs_shipped_3_0_3).
    """
    path = str(tmp_path / "v.dat")
    V = build(path, n=300)
    ref = VaultEngine(path, embed_dim=DIM, residency="float32")
    got = VaultEngine(path, embed_dim=DIM, residency=mode, arena_dir=str(tmp_path))
    try:
        for qi in (0, 5, 17, 99, 250):
            q = V[qi]
            a = ref.search("", q, top_k=10)
            b = got.search("", q, top_k=10)
            assert [h["id"] for h in a] == [h["id"] for h in b], mode
            for ha, hb in zip(a, b):
                assert ha["cosine"] == hb["cosine"], (mode, ha["id"])
                assert ha["score"] == hb["score"], (mode, ha["id"])
    finally:
        ref.close()
        got.close()


@pytest.mark.parametrize("mode", EXACT_MODES)
def test_narrow_residency_scores_match_the_whole_vector(tmp_path, mode):
    """The whole score VECTOR matches, not just the winners.

    BITWISE at the shipped ``scan_chunk``. That is a measurement, not a theorem:
    BLAS is free to block a matvec differently for different shapes, and it does
    -- see :func:`test_chunk_size_only_moves_scores_by_one_ulp`. What is
    guaranteed is fp32 accumulation over the same fp16 values, so the difference
    can never exceed fp32 rounding.
    """
    path = str(tmp_path / "v.dat")
    V = build(path, n=420)
    ref = VaultEngine(path, embed_dim=DIM, residency="float32")
    got = VaultEngine(path, embed_dim=DIM, residency=mode, arena_dir=str(tmp_path))
    try:
        for qi in (1, 44, 200):
            q = np.ascontiguousarray(V[qi])
            assert np.array_equal(ref.arena.scores(q), got.arena.scores(q)), mode
    finally:
        ref.close()
        got.close()


@pytest.mark.parametrize("chunk", [32, 64, 257, 4096])
def test_chunk_size_only_moves_scores_by_one_ulp(tmp_path, chunk):
    """``scan_chunk`` must never change an answer, only the last bit of a score.

    Measured on the real 71,433 x 768 corpus over all 500 questions
    (memory_results.json, chunk_sweep): at the shipped chunk of 4,096 the score
    vectors are bitwise identical to the fp32 arena's; at 256 they differ by at
    most 1.49e-08, which is one fp32 ULP near 0.5, and no query's top 10 moves.
    """
    path = str(tmp_path / "v.dat")
    V = build(path, n=900)
    ref = VaultEngine(path, embed_dim=DIM, residency="float32")
    got = VaultEngine(path, embed_dim=DIM, residency="float16", scan_chunk=chunk,
                      arena_dir=str(tmp_path))
    try:
        for qi in (1, 44, 700):
            q = np.ascontiguousarray(V[qi])
            a, b = ref.arena.scores(q), got.arena.scores(q)
            assert np.abs(a - b).max() < 1e-6, chunk
            assert ([h["id"] for h in ref.search("", q, top_k=10)]
                    == [h["id"] for h in got.search("", q, top_k=10)]), chunk
    finally:
        ref.close()
        got.close()


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_the_stored_vector_exactly(tmp_path, mode):
    """``get``/``iter_records`` hand back the fp16-on-disk value under every mode.

    ``int8`` included: it quantises only what it SCANS, and reads the true vector
    back through the sidecar.
    """
    path = str(tmp_path / "v.dat")
    V = build(path, n=120)
    e = VaultEngine(path, embed_dim=DIM, residency=mode, arena_dir=str(tmp_path))
    try:
        want = V.astype(np.float16).astype(np.float32)
        for r in e.iter_records(include_embeddings=True):
            i = int(r["metadata"]["idx"])
            assert np.array_equal(np.asarray(r["embedding"], dtype=np.float32), want[i]), mode
    finally:
        e.close()


@pytest.mark.parametrize("mode", EXACT_MODES)
def test_search_batch_matches_across_modes(tmp_path, mode):
    path = str(tmp_path / "v.dat")
    V = build(path, n=200)
    ref = VaultEngine(path, embed_dim=DIM, residency="float32")
    got = VaultEngine(path, embed_dim=DIM, residency=mode, arena_dir=str(tmp_path))
    try:
        Q = V[[3, 40, 150]]
        assert ref.search_batch(Q, top_k=5) == got.search_batch(Q, top_k=5), mode
    finally:
        ref.close()
        got.close()


@pytest.mark.parametrize("mode", MODES)
def test_appending_after_open_works_in_every_mode(tmp_path, mode):
    """A memtable flush must land in whatever form the arena holds."""
    path = str(tmp_path / "v.dat")
    build(path, n=60)
    e = VaultEngine(path, embed_dim=DIM, residency=mode, arena_dir=str(tmp_path))
    try:
        extra = unit_rows(80, DIM, seed=99)
        for i in range(80):
            e.add_fact(f"late {i}", extra[i], source="wiki", metadata={"idx": 1000 + i})
        e.flush()
        assert e.count() == 140
        hit = e.search("", extra[7], top_k=1)[0]
        assert hit["metadata"]["idx"] == 1007, mode
        assert hit["cosine"] > 0.99
    finally:
        e.close()


# ---------------------------------------------------------------------------
# int8: the one mode that is allowed to differ, and by how much
# ---------------------------------------------------------------------------
def test_int8_reranks_the_pool_exactly_and_buries_the_rest(tmp_path):
    """``int8``'s contract, stated as a test rather than a hope.

    Inside the re-rank pool the score is the exact fp32 cosine. Outside it the
    score is shifted DOWN so an approximate score can never outrank an exact one
    -- which is what makes the top-k safe and what makes a score below the pool
    not a cosine. On the real corpus the residual disagreements with the fp32
    engine are 1-5 queries in 500 at every pool from 32 to 8,192 and EVERY ONE of
    them has a cosine gap of exactly 0.0 (duplicate paragraphs with bitwise
    identical embeddings) -- memory_results.json, int8_pool_sweep.
    """
    path = str(tmp_path / "v.dat")
    V = build(path, n=400)
    ref = VaultEngine(path, embed_dim=DIM, residency="float32")
    e = VaultEngine(path, embed_dim=DIM, residency="int8", rerank_pool=32,
                    arena_dir=str(tmp_path))
    try:
        q = np.ascontiguousarray(V[11])
        exact = ref.arena.scores(q)
        approx = e.arena.scores(q)
        pool = np.argpartition(-approx, 31)[:32]
        rest = np.setdiff1d(np.arange(approx.size), pool)
        # The pool is re-scored in fp32 from the true fp16 vectors. It is a
        # (pool, D) matvec and the reference is an (n_rows, D) one, so BLAS may
        # block them differently; the difference is bounded by fp32 rounding,
        # not by the int8 quantisation the pool exists to undo.
        assert np.abs(approx[pool] - exact[pool]).max() < 1e-6
        assert approx[rest].max() < approx[pool].min()
        assert approx[rest].max() < exact[pool].min()
        # Quantisation error outside the pool, for the record: it is what decides
        # whether the true winners reach the pool at all.
        raw = e.arena.vec[:e.arena.n_rows].astype(np.float32) * e.arena.vscale[
            :e.arena.n_rows, None]
        assert np.abs(raw @ q - exact).max() < 0.02
    finally:
        ref.close()
        e.close()


def test_int8_top_k_matches_fp32_when_the_pool_holds_the_winners(tmp_path):
    path = str(tmp_path / "v.dat")
    V = build(path, n=400)
    ref = VaultEngine(path, embed_dim=DIM, residency="float32")
    e = VaultEngine(path, embed_dim=DIM, residency="int8", rerank_pool=128,
                    arena_dir=str(tmp_path))
    try:
        for qi in (0, 60, 199, 333):
            a = [h["id"] for h in ref.search("", V[qi], top_k=4)]
            b = [h["id"] for h in e.search("", V[qi], top_k=4)]
            assert a == b, qi
    finally:
        ref.close()
        e.close()


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------
def test_reopen_presizes_the_arena_exactly(tmp_path):
    """A reopen must allocate the arena ONCE, at the file's true row count.

    3.0.3 grew into it by doubling and allocated 131,072 rows for 71,433, and
    the allocator handed almost none of the discarded copies back: 643.6 MB of
    RSS to serve a 71,433-document vault, against 286.0 MB once the arena is
    pre-sized (memory_results.json, summary.reopen_only.n71433).
    """
    path = str(tmp_path / "v.dat")
    build(path, n=260)                          # not a power of two, and > 64
    e = VaultEngine(path, embed_dim=DIM)
    try:
        assert e.arena.n_rows == 260
        assert e.arena.vec.shape[0] == 260      # exact, not 512
        assert e.arena.reserved_rows == 260
        assert e.arena.ts.shape[0] == 260
        assert e.arena.doc_span.shape[0] == 260
        s = e.stats()
        assert s["arena_reserved_rows"] == 260
        assert s["resident_arena_mb"] == s["resident_arena_used_mb"]
    finally:
        e.close()


def test_reserve_is_only_a_hint(tmp_path):
    """A short or absurd hint must never lose a row or change an answer."""
    path = str(tmp_path / "v.dat")
    V = build(path, n=300)
    e = VaultEngine(path, embed_dim=DIM)
    try:
        ref = [h["id"] for h in e.search("", V[5], top_k=5)]
    finally:
        e.close()

    a = Arena(DIM)
    a.reserve(7)                                 # far too small
    from nanomem.container import Container
    c = Container(path, embed_dim=DIM)
    c.scan(a)
    assert a.n_rows == 300
    assert a.vec.shape[0] >= 300
    c.close()

    b = Arena(DIM)
    b.reserve(10_000)                            # far too large
    c2 = Container(path, embed_dim=DIM)
    c2.scan(b)
    assert b.n_rows == 300
    assert np.array_equal(a.scores(V[5]), b.scores(V[5])[:300])
    c2.close()
    del ref


def test_unhinted_growth_never_copies_and_never_overshoots(tmp_path):
    """The policy that replaced capacity doubling, and why.

    Doubling was never about the overshoot, it was about the copy left behind:
    on this platform a freed numpy buffer is not handed back, so peak RSS tracks
    (live + everything the growth ever discarded). Growing a 768-d fp32 arena to
    71,433 rows 50 rows at a time, one subprocess per policy, measured by
    ru_maxrss (ingest_ram_results.json, growth_policy -- and the doubling row
    reproduces memory_results.json's 592.2 MB to 0.1%):

        policy                    final capacity   copies   peak RSS delta
        capacity doubling           131,072 rows       12       591.8 MB
        exact fit, reallocating      71,433 rows    1,429    13,505.9 MB
        reserved view (shipped)      71,433 rows        2       208.9 MB

    So the arena is now cut to the rows that exist (no overshoot) AND grows by
    re-viewing a reservation (no copy). ``_next_capacity`` still governs the
    per-row columns, which are 40 bytes a row rather than 3,072.
    """
    assert _next_capacity(0, 1, 3072) == 64
    assert _next_capacity(64, 65, 3072) == 128
    assert _next_capacity(65536, 71433, 3072) == 131072
    assert _next_capacity(131072, 71433, 3072) == 131072

    path = str(tmp_path / "v.dat")
    e = VaultEngine(path, embed_dim=DIM)
    try:
        V = unit_rows(300, DIM, seed=1)
        for i in range(300):
            e.add_fact(f"r{i}", V[i], source="wiki")
        e.flush()
        assert e.arena.n_rows == 300
        assert e.arena.vec.shape[0] == 300         # exact fit, not 512
        assert e.arena._store.copies == 0          # nothing was ever copied
        assert e.arena._store.reservation_rows >= 300
        s = e.stats()
        assert s["arena_bytes"] == s["arena_used_bytes"]
        assert s["arena_growth_copies"] == 0
        # The reservation is address space, not memory, and is reported apart
        # from both so the two can never be added together and called RAM.
        assert s["arena_reservation_bytes"] >= s["arena_bytes"]
    finally:
        e.close()


def test_reservation_survives_many_growths_without_copying(tmp_path):
    """4,000 rows in 50-row blocks: the whole point is the copy count."""
    path = str(tmp_path / "v.dat")
    e = VaultEngine(path, embed_dim=DIM)
    try:
        V = unit_rows(4000, DIM, seed=11)
        for i in range(4000):
            e.add_fact(f"r{i}", V[i], source="wiki", metadata={"idx": i})
        e.flush()
        assert e.arena.n_rows == 4000
        assert e.arena.vec.shape[0] == 4000
        # 80 blocks arrived. Capacity doubling would have reallocated (and left
        # behind) every power of two from 64 to 4,096.
        assert e.arena._store.copies <= 2
        assert e.search("", V[1234], top_k=1)[0]["metadata"]["idx"] == 1234
    finally:
        e.close()


def test_hint_rows_takes_repeated_hints_without_a_copy_each_time(tmp_path):
    """The bulk path hints once per batch; that must not mean a copy per batch.

    ``reserve`` is exact because its caller COUNTED the rows. ``hint_rows`` is
    for a caller that is estimating, or that will hint again with a bigger
    number in a moment -- one hint per file of a directory ingest -- so it keeps
    the reservation's headroom and re-views inside it.
    """
    a = Arena(DIM)
    for n in range(100, 2100, 100):
        a.hint_rows(n)
    assert a.vec.shape[0] == 2000
    assert a._store.copies <= 1                    # 20 hints, at most one copy
    assert a.reserved_rows == 2000

    b = Arena(DIM)
    b.reserve(2000)                                # counted, so exactly 2,000
    assert b.vec.shape[0] == 2000
    assert b._store.reservation_rows == 2000


def test_a_refused_reservation_still_produces_a_working_arena(monkeypatch):
    """No headroom available is not a failure; it is just no headroom.

    A strict-overcommit host, an rlimit or a small address space can refuse the
    reservation. The arena must then be sized to the rows that exist and keep
    working, which is what the fallback inside ``_VectorStore._adopt`` is for.
    """
    from nanomem import arena as _arena
    real = _arena._VectorStore._reserve_map
    seen = {"refused": 0}

    def stingy(self, rows):
        # Refuse anything bigger than what was actually asked for.
        if rows > 300:
            seen["refused"] += 1
            raise MemoryError("simulated: no room for the headroom")
        return real(self, rows)

    monkeypatch.setattr(_arena._VectorStore, "_reserve_map", stingy)
    a = Arena(DIM)
    V = unit_rows(300, DIM, seed=5)
    a._store.ensure(300, 0)
    a._store.array[:300] = V
    assert seen["refused"] >= 1
    assert a.vec.shape[0] == 300
    assert np.array_equal(a.vec[:300], V.astype(np.float32))


def test_reserve_rows_pre_sizes_a_bulk_ingest(tmp_path):
    """The public escape hatch for a loader that knows its corpus size.

    Measured over a 71,433 x 768 fp32 ingest: 592.2 MB of ru_maxrss growing by
    doubling, 209.3 MB after ``reserve_rows`` -- and 312.8 MB against 844.9 MB
    for the whole build-then-serve process (memory_results.json, growth_policy
    and summary.n71433).
    """
    path = str(tmp_path / "v.dat")
    e = VaultEngine(path, embed_dim=DIM)
    try:
        e.reserve_rows(300)
        assert e.arena.vec.shape[0] == 300
        V = unit_rows(300, DIM, seed=2)
        for i in range(300):
            e.add_fact(f"r{i}", V[i], source="wiki", metadata={"idx": i})
        e.flush()
        assert e.arena.n_rows == 300
        assert e.arena.vec.shape[0] == 300       # never grew
        assert e.search("", V[42], top_k=1)[0]["metadata"]["idx"] == 42
    finally:
        e.close()


# ---------------------------------------------------------------------------
# reporting and plumbing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_stats_reports_the_residency_honestly(tmp_path, mode):
    path = str(tmp_path / "v.dat")
    build(path, n=200)
    e = VaultEngine(path, embed_dim=DIM, residency=mode, arena_dir=str(tmp_path))
    try:
        s = e.stats()
        assert s["arena_residency"] == mode
        assert s["arena_residency_exact"] is (mode != "int8")
        res = e.arena.resident_bytes()
        anon, mapped = res["arena_bytes"], res["arena_mapped_bytes"]
        assert res["arena_residency"] == mode
        if mode == "float32":
            assert anon == 200 * DIM * 4
            assert mapped == 0
        elif mode == "float16":
            assert anon == 200 * DIM * 2
            assert mapped == 0
        elif mode == "float16_mmap":
            assert anon == 0                     # nothing anonymous at all
            assert mapped == 200 * DIM * 2
        else:
            assert anon == 200 * (DIM + 4)       # int8 rows + one fp32 scale
            assert mapped == 200 * DIM * 2
        # Anonymous and mapped are reported SEPARATELY and must never be summed
        # and called RAM: mapped pages are clean and the OS can drop them.
        assert s["resident_arena_mb"] == round(anon / (1024 * 1024), 3)
        assert s["resident_arena_mapped_mb"] == round(mapped / (1024 * 1024), 3)
        assert s["arena_dtype"] in ("float32", "float16", "int8")
    finally:
        e.close()


def test_unknown_residency_is_refused(tmp_path):
    with pytest.raises(ValueError):
        VaultEngine(str(tmp_path / "v.dat"), embed_dim=DIM, residency="bfloat16")
    with pytest.raises(ValueError):
        Arena(DIM, residency="q4")


@pytest.mark.skipif(os.name == "nt", reason=(
    "the sidecar is deliberately NOT unlinked at creation on Windows, which cannot unlink an open file -- tests/test_windows_paths.py asserts the lifecycle it uses there instead"))
def test_sidecar_is_unlinked_and_released(tmp_path):
    """The fp16 sidecar is a private temp file: never visible, never left behind."""
    d = str(tmp_path / "side")
    os.makedirs(d)
    path = str(tmp_path / "v.dat")
    build(path, n=150)
    e = VaultEngine(path, embed_dim=DIM, residency="float16_mmap", arena_dir=d)
    try:
        assert e.arena._sidecar is not None
        assert not os.path.exists(e.arena._sidecar.path_hint)
        assert os.listdir(d) == []               # unlinked the moment it was made
        assert e.arena._sidecar.mapped_bytes() == 150 * DIM * 2
    finally:
        e.close()
    assert e.arena._sidecar is None
    assert os.listdir(d) == []


def test_row_count_hint_survives_a_torn_tail(tmp_path):
    """A hint read from unauthenticated headers must never cost a row.

    It is derived from block headers with no HMAC and no decryption, so a damaged
    file can make it wrong. The only thing it is allowed to get wrong is an
    allocation size.
    """
    path = str(tmp_path / "v.dat")
    V = build(path, n=200)
    with open(path, "ab") as fh:
        fh.write(b"\x00" * 137)                  # a torn tail
    e = VaultEngine(path, embed_dim=DIM, on_torn_tail="ignore")
    try:
        assert e.arena.n_rows == 200
        assert e.search("", V[3], top_k=1)[0]["metadata"]["idx"] == 3
    finally:
        e.close()


def test_row_count_hint_is_capped_by_the_bytes_on_disk(tmp_path):
    """A forged row count must not become a forged allocation.

    ``row_count_hint`` runs BEFORE any block is authenticated -- that is the
    whole point of it, the arena has to be sized before the first row is read --
    and it reads ``n`` out of a plaintext header protected only by a CRC32 that
    anyone can recompute, which is what this test does. So an attacker with
    write access to the file can set ``n`` to anything. The cap is the payload:
    a block stores its own vectors, fp16 is the narrowest vector this format
    writes, so no block can hold more than ``payload_len // (embed_dim * 2)``
    rows.
    """
    import struct
    import zlib
    from nanomem.container import (Container, BLOCK_HEADER_SIZE, BLOCK_HEADER_FMT,
                                   FILE_HEADER_SIZE, align_up, unpack_block_header)

    path = str(tmp_path / "v.dat")
    build(path, n=60)                            # blocks of 50 and 10

    off = align_up(FILE_HEADER_SIZE)
    with open(path, "r+b") as fh:
        fh.seek(off)
        raw = fh.read(BLOCK_HEADER_SIZE)
        assert unpack_block_header(raw).n == 50
        fields = list(struct.unpack(BLOCK_HEADER_FMT, raw))
        payload_len = int(fields[5])
        fields[3] = 2 ** 31 - 1                  # n := absurd
        fields[12] = zlib.crc32(struct.pack(BLOCK_HEADER_FMT, *fields)[:68]) & 0xFFFFFFFF
        forged = struct.pack(BLOCK_HEADER_FMT, *fields)
        fh.seek(off)
        fh.write(forged)

    # The forgery really is well-formed: the parser accepts it and hands back
    # the absurd count. Without the cap the hint below would BE that count.
    reparsed = unpack_block_header(forged)
    assert reparsed is not None and reparsed.n == 2 ** 31 - 1

    size = os.path.getsize(path)
    c = Container(path, embed_dim=DIM)
    try:
        with open(path, "rb") as fh:
            hint = c.row_count_hint(fh, off, size)
    finally:
        c.close()

    assert hint < 2 ** 31 - 1
    assert hint <= payload_len // (DIM * 2) + 10  # block 0's cap, plus block 1's real 10
    assert hint <= size // (DIM * 2)              # and never more than the file can hold


def test_page_matrices_are_fp32_in_every_mode(tmp_path):
    """The PageLandmarkRouter ablation reads fp32 whatever the arena holds."""
    path = str(tmp_path / "v.dat")
    build(path, n=150)
    for mode in MODES:
        e = VaultEngine(path, embed_dim=DIM, residency=mode, arena_dir=str(tmp_path))
        try:
            pages = e.arena.page_matrices()
            assert sum(p.shape[0] for p in pages) == 150
            assert all(p.dtype == np.float32 for p in pages), mode
        finally:
            e.close()


def test_block_payload_is_an_owned_copy_in_every_mode(tmp_path):
    """``read_payload`` must never hand out a view into the arena."""
    path = str(tmp_path / "v.dat")
    build(path, n=150)
    for mode in MODES:
        e = VaultEngine(path, embed_dim=DIM, residency=mode, arena_dir=str(tmp_path))
        try:
            blocks = e.container.toc
            blk = e.container.read_payload(blocks[0]["offset"], blocks[0]["length"],
                                           blocks[0]["doc_count"], blocks[0]["id"])
            assert not np.shares_memory(blk["values"], e.arena.vec), mode
            assert blk["values"].dtype == np.float32
        finally:
            e.close()


# ---------------------------------------------------------------------------
# the default, held to the measurement that chose it
# ---------------------------------------------------------------------------
def test_the_shipped_default_residency_matches_what_was_measured():
    """``residency="float32"`` is a measured decision, not a leftover.

    The rule, written down before the numbers were in: a mode may become the
    DEFAULT only if it changes no answer, and it should become the default only
    if it does not cost the axis nanomem actually wins. ``float16`` clears the
    first bar and fails the second -- it is exact (0 of 500 top-10 changed at
    71,433) but 2.28x ``float32``'s p50 on a quiet machine. So the default stays
    ``float32``, the RAM comes from the allocator instead (643.6 MB -> 286.0 MB
    resident on a reopened 71,433-document vault), and ``float16`` stays one
    keyword away.

    This test fails if the library's default and
    ``scratch/refound/memory_results.json`` ever drift apart, so changing the
    default means re-running ``bench_memory.py``, not editing a keyword.
    """
    import inspect

    res = measured()
    sig = inspect.signature(VaultEngine.__init__)
    shipped = sig.parameters["residency"].default
    assert shipped == "float32"

    big = res["recommendation"]["sizes"]["n71433"]["arms"]
    # the default must be an EXACT mode
    assert big["fp32_reserved"]["exact_vs_shipped"] is True
    # and it must be the fastest of the exact modes that were measured
    duel = res["latency_duel"]["n71433"]["p50_vs_float32"]
    for mode in ("float16", "float16_mmap"):
        assert duel[mode] > 1.0, (mode, duel)
    # the mode that is NOT exact must not be the default, whatever it costs
    assert big["int8_rerank_reserved"]["exact_vs_shipped"] is False


def test_the_allocator_not_a_narrower_dtype_is_what_took_the_ram_off_the_default():
    """The default's saving comes from the allocator, not from a narrower dtype.

    WHAT THIS TEST USED TO ASSERT, AND WHY THAT DIED. Until engine 3.0.5 it
    compared ``fp32_reserved`` against the harness's ``legacy_303_fp32_doubling``
    arm and required the pre-sized arm to be under 55% of it. That comparison is
    no longer meaningful, and the reason is the whole point of 3.0.5.
    ``bench_memory.py`` builds its "3.0.3" baseline by disabling
    :meth:`Arena.reserve`, which WAS the entire difference under 3.0.4. Under
    3.0.5 the vectors live in a reservation that sizes itself exactly whether or
    not anybody calls ``reserve``, so disabling ``reserve`` no longer
    reconstructs anything: that arm's server figure moved 643.6 MB -> 285.7 MB
    between the two builds while ``fp32_reserved`` stayed at 286.0 -> 285.2, and
    the harness cannot self-detect it because its own assertion is on
    ``_next_capacity``, which still doubles for the COLUMN arrays.

    So the claim is asserted directly instead of by that proxy: the shipped
    fp32 default serves the 71,433-document vault out of an arena that is sized
    to the rows that exist, allocated once, never copied -- and it changes no
    answer. A narrower dtype is not what buys it; ``float16`` is one keyword away
    and is measured separately.

    If somebody repairs the harness's legacy arm into a real doubling baseline,
    the ``pytest.approx`` below is what will fail, and this test should go back
    to the ratio assertion.
    """
    res = measured()
    rows = {r["arm"]: r for r in res["summary"]["reopen_only"]["n71433"]}
    presized = rows["fp32_reserved"]["live_rss_delta_mb"]
    legacy = rows["legacy_303_fp32_doubling"]["live_rss_delta_mb"]

    # 1. The saving is real and is in the allocator: on a reopened vault the
    #    arena is the rows, exactly, allocated once and never copied.
    #
    #    AMENDED AT 0.4.0, and the amendment is the interesting part. The
    #    shipped default is now `arena_cache="map"`, so a reopened vault gets
    #    its rows as a file-backed MAPPING rather than as anonymous memory --
    #    and `arena_bytes` counts anonymous memory ONLY, deliberately, because
    #    counting clean evictable pages as RAM is the exact over-reporting that
    #    3.0.5 removed. So on a cached open `arena_bytes` is 0 and the vectors
    #    are under `arena_cache_mapped_bytes`. The claim here is about the
    #    arena's SIZE, so it reads whichever key is carrying the vectors, and
    #    it now pins both open paths instead of only the scanning one.
    arena = res["reopen_only"]["arms"]["n71433"]["fp32_reserved"]["arena"]
    rb = arena["resident_bytes"]
    n_rows, dim = arena["vec_shape"]
    vectors = rb["arena_bytes"] + rb["arena_cache_mapped_bytes"]
    assert vectors == n_rows * dim * 4                    # fp32, no overshoot
    assert rb["arena_used_bytes"] == vectors
    assert rb["arena_growth_copies"] == 0
    if rb["arena_vectors_mapped"]:
        # Nothing was allocated for the vectors at all, which is the stronger
        # statement and the one the cache exists to make.
        assert rb["arena_bytes"] == 0
        assert rb["arena_reservation_bytes"] == 0
    else:
        assert rb["arena_reservation_bytes"] == vectors

    # 2. The resident cost of serving is the vectors plus a small remainder,
    #    NOT the 1.83x a doubling capacity would have held.
    vectors_mb = vectors / 2 ** 20
    assert presized < vectors_mb * 1.45, (presized, vectors_mb)

    # 3. It is still an fp32 arena. The RAM did not come from a narrower dtype.
    assert rb["arena_residency"] == "float32" and rb["arena_dtype"] == "float32"
    assert rows["fp16_reserved"]["live_rss_delta_mb"] < presized

    # 4. Zero changed answers, on both arms.
    assert rows["fp32_reserved"]["top10_changed_vs_shipped"] == 0
    assert (rows["fp32_reserved"]["evidence_recall_at_4"]
            == rows["legacy_303_fp32_doubling"]["evidence_recall_at_4"])

    # 5. The dead proxy, pinned so it cannot be re-read as a 3.0.4 baseline.
    #    Comparable only when both arms took the same OPEN path. Since 0.4.0
    #    they often do not: bench_memory.py does not pin `arena_cache=`, so
    #    whichever arm runs first for a given dtype scans the vault and writes
    #    the sidecar, and the next arm with that dtype maps it. Two arms that
    #    differ in open path differ in RSS for a reason that has nothing to do
    #    with the allocator, so the proxy is asserted only where it still means
    #    something -- and where it does not, the confound itself is pinned. If
    #    somebody teaches that harness to pin the mode, this branch stops being
    #    taken and the ratio assertion comes back on its own.
    legacy_rb = (res["reopen_only"]["arms"]["n71433"]
                 ["legacy_303_fp32_doubling"]["arena"]["resident_bytes"])
    if legacy_rb["arena_vectors_mapped"] == rb["arena_vectors_mapped"]:
        assert legacy == pytest.approx(presized, rel=0.05), (
            "bench_memory.py's legacy_303_fp32_doubling arm now measures the "
            "SAME allocator as fp32_reserved (%.1f vs %.1f MB); it is not a "
            "3.0.4 baseline any more." % (legacy, presized))
    else:
        assert not legacy_rb["arena_from_cache"] and rb["arena_from_cache"], (
            "the two arms differ in open path, but not in the direction the "
            "arena cache explains: %.1f vs %.1f MB" % (legacy, presized))
        assert legacy > presized, (
            "the arm that mapped its arena should not hold MORE than the arm "
            "that scanned it: %.1f vs %.1f MB" % (legacy, presized))


def test_int8_was_measured_and_lost():
    """A negative result, kept as a test so it cannot quietly be re-sold.

    ``int8`` needs the fp16 sidecar to re-rank, so it holds MORE than a plain
    ``float16`` arena; it converts every row to fp32 for BLAS anyway, so it is
    slower than both; and it is the only mode that changes an answer. All three
    are read from the measurements rather than asserted from memory.
    """
    res = measured()
    big = res["recommendation"]["sizes"]["n71433"]["arms"]
    assert (big["int8_rerank_reserved"]["server_live_rss_mb"]
            > big["fp16_reserved"]["server_live_rss_mb"])
    assert res["latency_duel"]["n71433"]["p50_vs_float32"]["int8"] > 3.0
    assert big["int8_rerank_reserved"]["exact_vs_shipped"] is False


def test_an_impossible_hint_is_dropped_not_raised(tmp_path):
    """A hint that cannot be allocated must not break the ingest it was for.

    ``ingest_directory`` estimates its row count from bytes on disk, so the
    number reaching the arena is somebody's arithmetic, not a fact. An estimate
    the allocator refuses leaves the arena exactly as it was, still growing on
    demand.
    """
    path = str(tmp_path / "v.dat")
    e = VaultEngine(path, embed_dim=DIM)
    try:
        V = unit_rows(120, DIM, seed=9)
        for i in range(60):
            e.add_fact(f"r{i}", V[i], source="wiki", metadata={"idx": i})
        e.flush()
        before = e.arena.vec.shape[0]
        e.reserve_additional_rows(10 ** 13)          # ~30 petabytes of vectors
        assert e.arena.vec.shape[0] == before
        for i in range(60, 120):
            e.add_fact(f"r{i}", V[i], source="wiki", metadata={"idx": i})
        e.flush()
        assert e.arena.n_rows == 120
        assert e.search("", V[99], top_k=1)[0]["metadata"]["idx"] == 99
    finally:
        e.close()
