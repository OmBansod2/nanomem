"""The arena cache as an OFFSET MAP: what it stops duplicating, and what that
must not change.

The sidecar's job is to make an open O(1). Its 3.0.6 implementation did that by
copying, into a file beside the vault, an fp32 upcast of vectors the vault
already held in fp16 and a verbatim copy of every record section -- 209.3 MiB
and 43.8 MiB of the 257.5 MiB it occupied at 71,433 rows. Neither copy carries
information: the block table the sidecar already stored says where both live in
the vault.

These tests pin the two halves of that change:

  * what it REMOVES -- the ``vec`` and ``rec_data`` sections, and the bytes,
  * and what it must not disturb -- the score bits, the top-k, the records, the
    ids, appends after a cached open, and every refusal the cache already made.

Every arm here is checked against ``arena_cache="off"``, which reads the vault
and nothing else, so "exact" means "identical to the answer with no sidecar in
sight", not "identical to another sidecar".
"""
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from nanomem import arena as _arena
from nanomem.engine import VaultEngine

DIM = 48
N = 700
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LAYOUTS = {
    "fp32_sidecar": dict(arena_cache_vectors="cache", arena_cache_records="cache"),
    "rec_offsets": dict(arena_cache_vectors="cache", arena_cache_records="vault"),
    "offsets": dict(arena_cache_vectors="offsets", arena_cache_records="vault"),
    "offsets_ram": dict(arena_cache_vectors="offsets_ram", arena_cache_records="vault"),
    "offsets_rec_cached": dict(arena_cache_vectors="offsets",
                               arena_cache_records="cache"),
}


def _corpus(n=N, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, dim)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Q = rng.standard_normal((12, dim)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return V, Q


def _build(path, V, **kw):
    e = VaultEngine(path, embed_dim=V.shape[1], durable="none", router="off",
                    arena_cache="off", **kw)
    for i in range(V.shape[0]):
        e.add_fact("row %d" % i, V[i], source="s", metadata={"idx": i})
    e.flush()
    e.close()
    return path


def _open(path, dim, **kw):
    kw.setdefault("arena_cache", "map")
    return VaultEngine(path, embed_dim=dim, durable="none", router="off",
                       n_exhaustive=10 ** 9, **kw)


def _score_sha(e, Q):
    h = hashlib.sha256()
    for q in Q:
        h.update(np.ascontiguousarray(e.arena.scores(np.ascontiguousarray(q)),
                                      dtype=np.float32).tobytes())
    return h.hexdigest()


def _topk(e, Q, k=5):
    return [tuple(int(h["metadata"]["idx"]) for h in e.search("", q, top_k=k))
            for q in Q]


def _sections(path):
    raw = open(path + ".arena", "rb").read(_arena.ARENA_CACHE_HEADER)
    hdr = _arena._parse_cache_header(raw)
    assert hdr is not None
    return hdr, {k: int(v["len"]) for k, v in hdr["sections"].items()}


# ---------------------------------------------------------------------------
# what it removes
# ---------------------------------------------------------------------------
def test_an_offset_sidecar_carries_no_vectors_and_no_records(tmp_path):
    """The two biggest sections are simply absent, and the file shrinks by them.

    This is the whole mechanism stated as a file-format fact: the sidecar that
    used to be dominated by ``vec`` has no ``vec``, and the one that copied
    ``rec_data`` has no ``rec_data``. What remains is the block table -- which
    was always there -- plus the columns and the id table.
    """
    V, _Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    sizes = {}
    for name, kw in LAYOUTS.items():
        p = str(tmp_path / (name + ".dat"))
        shutil.copyfile(src, p)
        _open(p, DIM, **kw).close()
        hdr, sec = _sections(p)
        sizes[name] = os.path.getsize(p + ".arena")
        if kw["arena_cache_vectors"].startswith("offsets"):
            assert hdr["vec_source"] == "vault"
            assert "vec" not in sec
        else:
            assert hdr["vec_source"] == "cache"
            assert sec["vec"] == V.shape[0] * DIM * 4
        if kw["arena_cache_records"] == "vault":
            assert hdr["rec_source"] == "vault"
            assert "rec_data" not in sec and "rec_off" not in sec
        else:
            assert sec["rec_data"] > 0

    assert sizes["offsets"] < sizes["rec_offsets"] < sizes["fp32_sidecar"]
    # the vectors are the bulk of it, so dropping them is the bulk of the saving
    assert sizes["offsets"] < 0.35 * sizes["fp32_sidecar"]


def test_the_offsets_a_sidecar_keeps_are_the_block_table_it_already_had(tmp_path):
    """No new section pays for the offsets: they are derived from ``blocks``.

    If this ever stops holding -- if a row-offset array appears -- the claim
    "the replacement costs zero new bytes" has to be restated, so it is pinned
    rather than asserted in prose.
    """
    V, _Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    a = str(tmp_path / "a.dat")
    b = str(tmp_path / "b.dat")
    shutil.copyfile(src, a)
    shutil.copyfile(src, b)
    _open(a, DIM, **LAYOUTS["fp32_sidecar"]).close()
    _open(b, DIM, **LAYOUTS["offsets"]).close()
    _hdr_a, sec_a = _sections(a)
    _hdr_b, sec_b = _sections(b)
    assert set(sec_b) - set(sec_a) == set()
    assert sec_b["blocks"] == sec_a["blocks"]


# ---------------------------------------------------------------------------
# what it must not change
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_every_sidecar_layout_scores_bit_for_bit_like_no_sidecar(tmp_path, layout):
    V, Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    ref = _open(src, DIM, arena_cache="off")
    want_sha, want_top = _score_sha(ref, Q), _topk(ref, Q)
    ref.close()

    p = str(tmp_path / (layout + ".dat"))
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS[layout]).close()          # write the sidecar
    e = _open(p, DIM, **LAYOUTS[layout])
    assert e.arena_cache_info()["source"] == "cache"
    assert _score_sha(e, Q) == want_sha
    assert _topk(e, Q) == want_top
    e.close()


def test_records_and_ids_come_back_from_the_mapped_vault(tmp_path):
    """A record read through an offset into the vault is the same record.

    ``record()`` slices one block's section out of whichever mapping holds it,
    so this is the test that the offsets point at the right bytes rather than at
    bytes that merely parse.
    """
    V, _Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["offsets"]).close()
    e = _open(p, DIM, **LAYOUTS["offsets"])
    for r in (0, 1, 49, 50, 51, 349, N - 1):
        rec = e.arena.record(r)
        assert rec["text"] == "row %d" % r
        assert rec["metadata"]["idx"] == r
    assert len(e.arena.id_index) == N
    assert e.arena.ids[N - 1] == e.arena.record(N - 1).get("id", e.arena.ids[N - 1])
    e.close()


def test_a_scattered_gather_matches_the_full_scan(tmp_path):
    """``vectors(rows)`` is the screen's path and is a different kernel.

    The full scan walks blocks in order; a gather jumps between them. Both have
    to produce the same numbers, so the gather is checked against the scan's own
    rows rather than against itself.
    """
    V, _Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    ref = _open(src, DIM, arena_cache="off")
    want = ref.arena.matrix(0, N).copy()
    ref.close()
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["offsets"]).close()
    e = _open(p, DIM, **LAYOUTS["offsets"])
    rng = np.random.default_rng(3)
    rows = np.sort(rng.choice(N, 137, replace=False))
    assert np.array_equal(e.arena.vectors(rows), want[rows])
    assert np.array_equal(e.arena.vectors(rows[::-1]), want[rows[::-1]])
    assert np.array_equal(e.arena.matrix(37, 421), want[37:421])
    assert np.array_equal(e.arena.exact16(), want.astype(np.float16))
    e.close()


def test_the_one_shot_upcast_reads_the_vault_instead_of_faulting_it_in(tmp_path):
    """``read_rows`` and ``fill`` must agree byte for byte.

    They are two ways to the same bytes -- one through the mapping, one through
    ``read`` -- and the second exists only so the upcast is not billed for
    faulting in 104.6 MiB it is about to throw away. If they ever disagree, the
    arm that upcasts and the arm that scans stop being the same arm.
    """
    V, _Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["offsets"]).close()
    e = _open(p, DIM, **LAYOUTS["offsets"])
    vault = e.arena._vault
    for lo, hi in ((0, N), (0, 1), (49, 51), (137, 421), (N - 1, N)):
        a = np.empty((hi - lo, DIM), dtype=np.float32)
        b = np.empty((hi - lo, DIM), dtype=np.float32)
        vault.fill(lo, hi, a)
        vault.read_rows(lo, hi, b)
        assert np.array_equal(a, b)
    e.close()


def test_offsets_ram_upcasts_once_on_first_touch_and_then_owns_its_rows(tmp_path):
    """The lazy variant: the OPEN reads nothing, the first query materialises.

    That ordering is the point -- an upcast at open would put O(rows) back into
    the open this whole file exists to keep O(1).
    """
    V, Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["offsets_ram"]).close()
    e = _open(p, DIM, **LAYOUTS["offsets_ram"])
    assert e.arena.vault_backed is True
    assert e.arena.vec.shape[0] == 0            # nothing has been read yet
    e.arena.scores(np.ascontiguousarray(Q[0]))
    assert e.arena.vault_backed is False
    assert e.arena.vec.shape == (N, DIM)
    ref = _open(src, DIM, arena_cache="off")
    assert _score_sha(e, Q) == _score_sha(ref, Q)
    ref.close()
    e.close()


def test_appending_to_a_vault_backed_arena_moves_it_into_memory_once(tmp_path):
    """Row ``n`` cannot be written into a read-only mapping of someone's vault.

    The append pays exactly one copy and the arena is ordinary afterwards -- and
    the rows it already served keep their values.
    """
    V, Q = _corpus(n=N + 60)
    src = _build(str(tmp_path / "src.dat"), V[:N])
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["offsets"]).close()
    e = _open(p, DIM, **LAYOUTS["offsets"])
    before = e.arena.matrix(0, N).copy()
    assert e.arena.vault_backed is True
    for i in range(N, N + 60):
        e.add_fact("row %d" % i, V[i], source="s", metadata={"idx": i})
    e.flush()
    assert e.arena.vault_backed is False
    assert e.arena.n_rows == N + 60
    assert np.array_equal(e.arena.matrix(0, N), before)
    assert e.arena.record(N + 59)["metadata"]["idx"] == N + 59
    assert e.arena.record(3)["metadata"]["idx"] == 3   # still reading the vault
    e.close()

    ref = _open(p, DIM, arena_cache="off")
    got = _open(p, DIM, **LAYOUTS["offsets"])
    assert _score_sha(got, Q) == _score_sha(ref, Q)
    assert _topk(got, Q) == _topk(ref, Q)
    got.close()
    ref.close()


def test_a_cached_open_that_has_to_scan_a_tail_still_binds_the_offsets(tmp_path):
    """``cache+append``: rows the cache covers come from the vault mapping, the
    rest from a scan, and the two halves have to line up."""
    V, Q = _corpus(n=N + 120)
    src = _build(str(tmp_path / "src.dat"), V[:N])
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["offsets"]).close()
    grow = _open(p, DIM, arena_cache="off")
    for i in range(N, N + 120):
        grow.add_fact("row %d" % i, V[i], source="s", metadata={"idx": i})
    grow.flush()
    grow.close()

    e = _open(p, DIM, **LAYOUTS["offsets"])
    info = e.arena_cache_info()
    assert info["source"] == "cache+append"
    assert info["rows_from_cache"] == N and info["rows_scanned"] == 120
    ref = _open(p, DIM, arena_cache="off")
    assert _score_sha(e, Q) == _score_sha(ref, Q)
    assert e.arena.record(N + 119)["metadata"]["idx"] == N + 119
    ref.close()
    e.close()


def test_float16_residency_writes_its_own_width_into_the_sidecar(tmp_path):
    """An fp16 sidecar is not a fourth knob: it is ``residency="float16"``.

    Worth pinning because it is the arm between "copy everything" and "copy
    nothing", and it is reached without any new code.
    """
    V, Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    kw = dict(residency="float16", arena_cache_vectors="cache",
              arena_cache_records="vault")
    _open(p, DIM, **kw).close()
    hdr, sec = _sections(p)
    assert hdr["dtype"] == np.dtype(np.float16).str
    assert sec["vec"] == N * DIM * 2
    e = _open(p, DIM, **kw)
    ref = _open(src, DIM, arena_cache="off")
    assert _score_sha(e, Q) == _score_sha(ref, Q)
    ref.close()
    e.close()


def test_a_float32_vault_is_offset_mapped_at_its_own_width(tmp_path):
    """The offsets do not assume fp16. A vault written ``vector_dtype="float32"``
    has wider vector runs and the same arithmetic must come out."""
    V, Q = _corpus()
    src = str(tmp_path / "src.dat")
    _build(src, V, vector_dtype="float32")
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, vector_dtype="float32", **LAYOUTS["offsets"]).close()
    e = _open(p, DIM, vector_dtype="float32", **LAYOUTS["offsets"])
    assert e.arena.vault_backed is True
    ref = _open(src, DIM, vector_dtype="float32", arena_cache="off")
    assert _score_sha(e, Q) == _score_sha(ref, Q)
    ref.close()
    e.close()


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------
def test_an_encrypted_vault_gets_no_sidecar_and_no_mapping(tmp_path):
    """An encrypted block's payload is ciphertext: there are no vectors in it to
    point at. The cache was already refused there; asking for offsets must not
    open a way round that."""
    V, _Q = _corpus(n=300)
    p = str(tmp_path / "enc.dat")
    e = VaultEngine(p, embed_dim=DIM, durable="none", router="off", password="pw",
                    arena_cache="map", arena_cache_vectors="offsets",
                    arena_cache_records="vault")
    for i in range(300):
        e.add_fact("row %d" % i, V[i], source="s", metadata={"idx": i})
    e.flush()
    e.close()
    assert not os.path.exists(p + ".arena")
    e = VaultEngine(p, embed_dim=DIM, durable="none", router="off", password="pw",
                    arena_cache="map", arena_cache_vectors="offsets",
                    arena_cache_records="vault")
    assert e.arena_cache_info()["reason"] == "vault is encrypted"
    assert e.arena.vault_backed is False
    e.close()
    assert not os.path.exists(p + ".arena")


def test_a_sidecar_written_for_another_layout_is_refused_and_rebuilt(tmp_path):
    """The knob decides what is on disk, so flipping it has to cost one rebuild.

    Using a cache that duplicates different things than the caller asked for
    would make the knob decide nothing, which is worse than the rebuild.
    """
    V, Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["fp32_sidecar"]).close()
    big = os.path.getsize(p + ".arena")
    e = _open(p, DIM, **LAYOUTS["offsets"])
    assert e.arena_cache_info()["reason"] == "cache was written for a different sidecar layout"
    assert e.arena_cache_info()["source"] == "scan"
    e.close()
    assert os.path.getsize(p + ".arena") < big
    e = _open(p, DIM, **LAYOUTS["offsets"])
    assert e.arena_cache_info()["source"] == "cache"
    ref = _open(src, DIM, arena_cache="off")
    assert _score_sha(e, Q) == _score_sha(ref, Q)
    ref.close()
    e.close()


def test_a_vault_truncated_after_its_sidecar_is_refused(tmp_path):
    """The binding is the vault's, not the sidecar's: an offset map into a file
    that has since been cut has to be refused before it is read."""
    V, _Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["offsets"]).close()
    with open(p, "r+b") as f:
        f.truncate(os.path.getsize(p) - 4096)
    with pytest.warns(RuntimeWarning):
        e = _open(p, DIM, **LAYOUTS["offsets"])
    assert e.arena_cache_info()["source"] == "scan"
    assert e.arena.n_rows < N
    e.close()


# ---------------------------------------------------------------------------
# damage UNDER a live mapping
#
# These run in a SUBPROCESS on purpose. The failure they guard against is
# SIGBUS, which is a signal and not an exception: a regression here does not
# fail an assertion, it kills the interpreter that runs it. In-process these
# tests would take the whole suite down with them and report nothing.
# ---------------------------------------------------------------------------
_TRUNC_PROBE = r"""
import os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from nanomem.engine import VaultEngine
from nanomem.errors import VaultShrankError

path, dim, layout = sys.argv[2], int(sys.argv[3]), sys.argv[4]
kw = dict(arena_cache="map", durable="none", router="off", n_exhaustive=10 ** 9)
if layout != "shipped_default":
    kw.update(LAYOUTS[layout])
e = VaultEngine(path, embed_dim=dim, **kw)
q = np.zeros(dim, dtype=np.float32); q[0] = 1.0
e.arena.scores(q)                      # warm whatever the layout warms
e.arena.record(0)
size = os.path.getsize(path)
with open(path, "r+b") as f:
    f.truncate(size // 2)              # somebody else cuts the vault in half
for name, call in (("scores", lambda: e.arena.scores(q)),
                   ("record", lambda: e.arena.record(e.arena.n_rows - 1))):
    try:
        out = call()
        print("%s=returned" % name)
    except VaultShrankError:
        print("%s=raised" % name)
e.close()
print("exit=clean")
"""


@pytest.mark.parametrize("layout", ["fp32_sidecar", "rec_offsets", "offsets",
                                    "offsets_ram", "shipped_default"])
def test_truncation_under_a_live_mapping_raises_instead_of_killing_the_process(
        tmp_path, layout):
    """Cut the vault in half while the arena is mapped to it, then read.

    Before the guard this was exit 138 -- SIGBUS, no traceback, no exception --
    for every layout that reads the vault instead of a copy of it, and the copy
    layouts survived the identical damage because they had a copy. The point of
    the check is not that the data comes back (it is gone) but that a caller
    gets an error it can catch.
    """
    V, _Q = _corpus(n=300)
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / ("v_%s.dat" % layout))
    shutil.copyfile(src, p)
    _open(p, DIM, **(LAYOUTS.get(layout) or {})).close()     # write the sidecar
    probe = ("LAYOUTS = %r\n" % (LAYOUTS,)) + _TRUNC_PROBE
    r = subprocess.run([sys.executable, "-c", probe, str(_ROOT), p, str(DIM),
                        layout], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (
        "layout %s died with returncode %s (-7/138 is SIGBUS)\n%s\n%s"
        % (layout, r.returncode, r.stdout, r.stderr))
    out = dict(line.split("=", 1) for line in r.stdout.split() if "=" in line)
    assert out.get("exit") == "clean", r.stdout
    reads_the_vault = LAYOUTS.get(layout, {}).get("arena_cache_records") == "vault"
    if layout == "shipped_default":
        # whatever the default is, both calls must be survivable
        assert out["scores"] in ("returned", "raised")
        assert out["record"] in ("returned", "raised")
    elif reads_the_vault:
        assert out["record"] == "raised", r.stdout
    else:
        assert out["record"] == "returned", r.stdout


def test_a_trimmed_torn_tail_under_a_live_mapping_does_not_raise(tmp_path):
    """The guard must not fire on nanomem's OWN recovery.

    A writer that finds unreadable trailing bytes truncates them
    (``container._append``) -- the file SHRINKS under any reader mapped to it,
    and every byte the reader's block table addresses is still there. The check
    is therefore against the last addressed byte, not against the mapped
    length, and this is the test that says so.
    """
    V, _Q = _corpus(n=300)
    p = _build(str(tmp_path / "v.dat"), V)
    _open(p, DIM, **LAYOUTS["offsets"]).close()          # write the sidecar
    good = os.path.getsize(p)
    with open(p, "ab") as f:                 # a torn tail: unreadable bytes
        f.write(b"\x7f" * 40_000)
    reader = _open(p, DIM, **LAYOUTS["offsets"])
    q = np.ascontiguousarray(V[0])
    before = np.asarray(reader.arena.scores(q)).copy()
    assert reader.arena._vault is not None
    assert reader.arena._vault.map_bytes > good          # mapped OVER the tail

    with pytest.warns(RuntimeWarning):                   # the writer trims it
        w = VaultEngine(p, embed_dim=DIM, durable="none", router="off",
                        arena_cache="off")
        w.add_fact("appended", V[1], source="s", metadata={"idx": -1})
        w.flush()
        w.close()
    assert os.path.getsize(p) < reader.arena._vault.map_bytes

    after = np.asarray(reader.arena.scores(q))           # must NOT raise
    assert np.array_equal(before, after)
    assert reader.arena.record(N_TRUNC_PROBE_ROW)["text"] == "row %d" % N_TRUNC_PROBE_ROW
    reader.close()


N_TRUNC_PROBE_ROW = 299


def test_a_block_table_that_points_past_the_vault_is_refused(tmp_path):
    """``_VaultBacking.open`` returns ``None`` rather than mapping nonsense.

    Checked directly, because the engine's fall-back to a full scan would hide
    whether the refusal happened here or one layer up.
    """
    from nanomem import container as _c
    V, _Q = _corpus(n=200)
    src = _build(str(tmp_path / "src.dat"), V)
    e = _open(src, DIM, arena_cache="off")
    table = _c.pack_block_table(list(e._cont.blocks))
    valid_end = int(e._cont.valid_end)
    e.close()
    ok = _arena._VaultBacking.open(src, table, DIM, valid_end, fp16=True)
    assert ok is not None and ok.n_rows == 200
    ok.close()
    assert _arena._VaultBacking.open(src, table, DIM, 128, fp16=True) is None
    assert _arena._VaultBacking.open(src, table, DIM, valid_end, fp16=False) is None
    assert _arena._VaultBacking.open(str(tmp_path / "nope.dat"), table, DIM,
                                     valid_end, fp16=True) is None
    bad = table.copy()
    bad[-1]["offset"] = 10 ** 12
    assert _arena._VaultBacking.open(src, bad, DIM, valid_end, fp16=True) is None


def test_a_block_with_a_landmark_table_shifts_its_vectors_and_is_followed(tmp_path):
    """``vec_off = offset + 96 + m*D*4`` -- the ``m`` term, exercised.

    No vault this engine writes today carries landmarks (``m`` is 0 in every
    block of every corpus measured), so the one term of the offset arithmetic
    that a real file never exercises is exercised here directly: blocks are
    appended through :func:`container.build_block_blob` with a landmark table,
    and the offsets have to skip it. Get this wrong and every row of such a
    vault comes back shifted by ``m * D * 4`` bytes.
    """
    from nanomem import container as _c
    V, Q = _corpus(n=400, dim=DIM)
    path = str(tmp_path / "lm.dat")
    cont = _c.Container(path, embed_dim=DIM, vector_dtype="float16")
    per, m = 50, 8
    for b in range(8):
        rows = V[b * per:(b + 1) * per]

        def make(seq, rows=rows, b=b):
            return _c.build_block_blob(
                cont.header, cont.keys, seq, rows, rows[:m],
                [float(i) for i in range(per)], [0] * per,
                ["id-%d-%d" % (b, i) for i in range(per)], [""] * per,
                [('{"text":"row %d","source":"s","metadata":{"idx":%d}}'
                  % (b * per + i, b * per + i)).encode() for i in range(per)])

        cont.append_block(make, None)
    cont.close()

    probe = VaultEngine(path, embed_dim=DIM, durable="none", router="off",
                        arena_cache="off", n_exhaustive=10 ** 9)
    assert probe.arena.n_rows == 400
    table = _c.pack_block_table(list(probe._cont.blocks))
    assert max(int(x["m"]) for x in table) == m
    want = probe.arena.matrix(0, 400).copy()
    probe.close()

    back = _arena._VaultBacking.open(path, table, DIM, os.path.getsize(path), fp16=True)
    assert back is not None
    got = np.empty((400, DIM), dtype=np.float32)
    back.fill(0, 400, got)
    assert np.array_equal(got, want)
    back.close()

    _open(path, DIM, **LAYOUTS["offsets"]).close()
    e = _open(path, DIM, **LAYOUTS["offsets"])
    assert e.arena.vault_backed is True
    assert np.array_equal(e.arena.matrix(0, 400), want)
    assert e.arena.record(399)["metadata"]["idx"] == 399
    ref = _open(path, DIM, arena_cache="off")
    assert _score_sha(e, Q) == _score_sha(ref, Q)
    ref.close()
    e.close()


def test_the_byte_report_calls_a_mapped_vault_mapped(tmp_path):
    """``stats()`` must not count the vault's clean pages as heap, and must not
    count them twice either."""
    V, Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    _open(p, DIM, **LAYOUTS["offsets"]).close()
    e = _open(p, DIM, **LAYOUTS["offsets"])
    e.arena.scores(np.ascontiguousarray(Q[0]))
    res = e.arena.resident_bytes()
    assert res["arena_vault_backed"] is True
    assert res["arena_bytes"] == 0
    assert res["arena_vault_vector_bytes"] == N * DIM * 2
    assert res["arena_mapped_bytes"] >= res["arena_vault_vector_bytes"]
    e.close()


def test_the_shipped_defaults_are_the_ones_the_measurement_chose(tmp_path):
    """The defaults are a decision, so they are pinned rather than assumed.

    ``scratch/refound/sidecar_size_results.json`` picked them: offsets_ram is
    the only arm that cleared every pre-registered gate AND stayed inside the
    pre-registered peak-RSS cap, and vault-sourced records cost nothing
    measurable. If either default moves, this test is where the results file has
    to be re-read.
    """
    V, Q = _corpus()
    src = _build(str(tmp_path / "src.dat"), V)
    p = str(tmp_path / "v.dat")
    shutil.copyfile(src, p)
    e = VaultEngine(p, embed_dim=DIM, durable="none", router="off")
    assert e.arena_cache_vectors == "offsets_ram"
    assert e.arena_cache_records == "vault"
    e.close()
    e = VaultEngine(p, embed_dim=DIM, durable="none", router="off",
                    n_exhaustive=10 ** 9)
    assert e.arena_cache_info()["source"] == "cache"
    hdr, sec = _sections(p)
    assert "vec" not in sec and "rec_data" not in sec
    ref = _open(src, DIM, arena_cache="off")
    assert _score_sha(e, Q) == _score_sha(ref, Q)
    ref.close()
    e.close()


def test_the_knobs_reject_a_value_they_do_not_implement(tmp_path):
    V, _Q = _corpus(n=120)
    p = _build(str(tmp_path / "v.dat"), V)
    with pytest.raises(ValueError):
        VaultEngine(p, embed_dim=DIM, durable="none", arena_cache_vectors="pointers")
    with pytest.raises(ValueError):
        VaultEngine(p, embed_dim=DIM, durable="none", arena_cache_records="elsewhere")
