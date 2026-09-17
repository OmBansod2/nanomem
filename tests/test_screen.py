"""The exactness-preserving PCA + Cauchy-Schwarz search screen.

The flag ships OFF. What these tests defend is the one thing the flag promises:
turning it ON cannot change an answer. So most of this file is not "does it go
faster" -- it is "can the bound ever be wrong", asked in the ways that would
actually catch it.

Two of them have teeth on purpose:

``test_bound_holds_for_a_DELIBERATELY_BAD_basis``
    feeds the screen bases it was never designed for -- non-orthonormal,
    rank-deficient, badly scaled, all-zero -- and demands admissibility anyway.
    That is not a hypothetical: it is exactly the situation an APPEND creates,
    where the basis was fitted from rows that no longer represent the corpus.

``test_the_orthonormality_term_is_what_makes_the_bad_basis_safe``
    removes the ``(I - B.T B)`` term from the bound and asserts the result IS
    violated. Without it the first test could pass for the wrong reason -- a
    bound so loose that nothing could ever break it -- and nobody would know the
    term was load-bearing.

No fixture vocabulary, no persona names, no corpus on disk: every vector here is
seeded numpy.
"""

import os

import numpy as np
import pytest

from conftest import unit_rows                                   # noqa: F401
from nanomem import screen as S
from nanomem.engine import VaultEngine

D = 128


def structured_rows(n, dim=D, rank=24, seed=0, noise=0.25):
    """Unit rows with a low intrinsic dimension, so a projection can be tight.

    Isotropic Gaussian rows have no subspace to find -- the screen correctly
    refuses to help there, which :func:`test_isotropic_corpus_falls_back`
    checks. Real embeddings are not isotropic, and neither are these.
    """
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(rank, dim)).astype(np.float32)
    V = rng.normal(size=(n, rank)).astype(np.float32) @ B
    V += noise * rng.normal(size=(n, dim)).astype(np.float32)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def fill(path, V, source="corpus.txt", **kw):
    e = VaultEngine(path, embed_dim=V.shape[1], **kw)
    e.reserve_additional_rows(V.shape[0])
    for i in range(V.shape[0]):
        e.add_fact(f"paragraph {i}", V[i], source=source)
    e.flush()
    return e


def keyed(hits):
    return [(h["id"], h["cosine"]) for h in hits]


# ---------------------------------------------------------------------------
# the bound itself
# ---------------------------------------------------------------------------
def test_bound_is_admissible_on_a_fitted_basis():
    V = structured_rows(2000)
    Q = structured_rows(200, seed=7)
    sc = S.PCAScreen(D, d_out=32).fit([V])
    sc.extend(V)
    worst = -1e9
    for q in Q:
        ub = sc.bounds(q)
        true = V @ q
        worst = max(worst, float((true - ub).max()))
    assert worst <= 0.0, f"bound violated by {worst}"


@pytest.mark.parametrize("d_out", [1, 8, 32, 127, 128])
def test_bound_is_admissible_at_every_width(d_out):
    V = structured_rows(600, seed=3)
    Q = structured_rows(60, seed=11)
    sc = S.PCAScreen(D, d_out=d_out).fit([V])
    sc.extend(V)
    for q in Q:
        assert float((V @ q - sc.bounds(q)).max()) <= 0.0


def _bad_bases(rng):
    """Bases the fit would never produce, which an append effectively simulates."""
    yield "gaussian", rng.normal(size=(D, 16)).astype(np.float32)
    yield "scaled_10x", 10.0 * rng.normal(size=(D, 16)).astype(np.float32)
    yield "scaled_tiny", 1e-3 * rng.normal(size=(D, 16)).astype(np.float32)
    dup = rng.normal(size=(D, 1)).astype(np.float32)
    yield "rank_1_repeated", np.repeat(dup, 16, axis=1)
    yield "all_zero", np.zeros((D, 16), dtype=np.float32)
    ones = np.ones((D, 16), dtype=np.float32)
    yield "all_ones", ones
    q, _ = np.linalg.qr(rng.normal(size=(D, 16)))
    yield "orthonormal_times_3", np.ascontiguousarray(3.0 * q, dtype=np.float32)


def test_bound_holds_for_a_DELIBERATELY_BAD_basis():
    """ADVERSARIAL. The bound must be admissible for ANY matrix B.

    This is the property the whole "appends cannot invalidate correctness"
    argument rests on: a basis fitted from rows 0..N still has to bound row
    N+1, and nothing about the fit is allowed to be a premise. If the
    implementation ever starts assuming ``B.T B == I`` -- the natural
    simplification, and the one the research prototype made -- this test fails.
    """
    rng = np.random.default_rng(0)
    V = structured_rows(800, seed=5)
    Q = structured_rows(80, seed=6)
    for name, B in _bad_bases(rng):
        sc = S.PCAScreen(D, d_out=B.shape[1])
        sc.fit([V[:1]])                       # get the plumbing, then overwrite
        sc.basis = np.ascontiguousarray(B, dtype=np.float32)
        B64 = B.astype(np.float64)
        sc._G = np.ascontiguousarray(B64.T @ B64)
        sc._nIG = float(np.linalg.norm(np.eye(B.shape[1]) - sc._G, 2))
        sc.n_rows = 0
        sc._cap = 0
        sc._xmax = 1.0
        sc._Y = np.zeros((0, B.shape[1]), dtype=np.float32)
        sc._res = np.zeros(0, dtype=np.float32)
        sc._yn = np.zeros(0, dtype=np.float32)
        sc.extend(V)
        worst = max(float((V @ q - sc.bounds(q)).max()) for q in Q)
        assert worst <= 0.0, f"basis {name!r} broke the bound by {worst}"


def test_the_orthonormality_term_is_what_makes_the_bad_basis_safe():
    """The previous test must not be passing because the bound is simply huge.

    Recompute it WITHOUT the ``||pq|| * ||B.T r||`` term -- i.e. the
    textbook Cauchy-Schwarz screen that assumes an orthonormal basis -- against
    a basis that is not orthonormal, and assert it IS violated. If this ever
    stops failing, the adversarial test above has stopped proving anything.
    """
    rng = np.random.default_rng(1)
    V = structured_rows(800, seed=5)
    Q = structured_rows(80, seed=6)
    B = np.ascontiguousarray(3.0 * np.linalg.qr(rng.normal(size=(D, 16)))[0],
                             dtype=np.float32)
    Y = V @ B
    yn2 = np.einsum("ij,ij->i", Y, Y)
    xn2 = np.einsum("ij,ij->i", V, V)
    res = np.sqrt(np.maximum(xn2 - yn2, 0.0))
    violations = 0
    for q in Q:
        pq = q @ B
        resq = float(np.sqrt(max(float(q @ q) - float(pq @ pq), 0.0)))
        naive = Y @ pq + res * resq              # the term left out on purpose
        violations += int(((V @ q) - naive > 0).sum())
    assert violations > 0, ("the naive bound was never violated, so the "
                            "adversarial test proves nothing")


def test_appended_rows_are_bounded_by_the_old_basis():
    """Fit on a quarter of the corpus, extend with the rest, bound all of it."""
    V = structured_rows(2000, seed=2)
    sc = S.PCAScreen(D, d_out=32).fit([V[:500]])
    assert sc.fit_rows == 500
    sc.extend(V[:500])
    sc.extend(V[500:])
    assert sc.n_rows == 2000
    assert sc.fit_rows == 500, "extend() must never refit the basis"
    Q = structured_rows(100, seed=9)
    for q in Q:
        assert float((V @ q - sc.bounds(q)).max()) <= 0.0


def test_bounds_of_an_empty_screen():
    sc = S.PCAScreen(D, d_out=8).fit([])
    assert sc.n_rows == 0
    assert sc.bounds(np.zeros(D, dtype=np.float32)).shape == (0,)


# ---------------------------------------------------------------------------
# the flag, through the engine
# ---------------------------------------------------------------------------
def test_flag_is_off_by_default(vault_path):
    e = fill(vault_path, structured_rows(300))
    assert e.screen_mode == "off"
    assert e.stats()["screen"] == "off"
    e.search("a question", structured_rows(1)[0], top_k=4)
    assert e.screen_info()["engaged"] == 0
    e.close()


def test_search_returns_the_exact_scan_result(tmp_path):
    V = structured_rows(4000, seed=4)
    Q = structured_rows(150, seed=12)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=100)
    off = VaultEngine(p, embed_dim=D, screen="off")
    assert on.build_screen()["built"] is True
    for k in (1, 4, 10):
        for q in Q:
            a = on.search("a question", q, top_k=k)
            b = off.search("a question", q, top_k=k)
            assert set(h["id"] for h in a) == set(h["id"] for h in b)
            # BITWISE, not np.isclose: the flag's promise is the same answer.
            assert [h["cosine"] for h in a] == [h["cosine"] for h in b]
            assert [h["score"] for h in a] == [h["score"] for h in b]
    assert on.screen_info()["engaged"] > 0
    on.close(); off.close()


def test_min_score_is_applied_identically(tmp_path):
    V = structured_rows(3000, seed=14)
    Q = structured_rows(60, seed=15)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=100)
    off = VaultEngine(p, embed_dim=D, screen="off")
    on.build_screen()
    for q in Q:
        for ms in (0.0, 0.3, 0.9, 1.5):
            assert keyed(on.search("a question", q, top_k=5, min_score=ms)) == \
                   keyed(off.search("a question", q, top_k=5, min_score=ms))
    on.close(); off.close()


def test_appending_through_the_engine_stays_exact(tmp_path):
    V = structured_rows(4000, seed=16)
    p = str(tmp_path / "v.dat")
    e = fill(p, V[:1500], screen="pca", screen_dims=32, screen_min_rows=100)
    built = e.build_screen()
    assert built["rows"] == 1500
    for i in range(1500, 4000):
        e.add_fact(f"paragraph {i}", V[i], source="corpus.txt")
    e.flush()
    # The screen reconciles lazily, on the next search or build_screen(): a
    # spill must not pay for a projection nobody has asked for yet.
    assert e.screen_info()["rows"] == 1500
    e.search("a question", V[0], top_k=4)
    after = e.screen_info()
    assert after["rows"] == 4000
    assert after["fit_rows"] == built["fit_rows"], "an append refitted the basis"
    Q = structured_rows(80, seed=17)
    for q in Q:
        e.screen_mode = "pca"
        a = e.search("a question", q, top_k=8)
        e.screen_mode = "off"
        b = e.search("a question", q, top_k=8)
        e.screen_mode = "pca"
        assert keyed(a) == keyed(b)
    e.close()


def test_two_engines_can_disagree_and_it_is_NOT_the_screen(tmp_path):
    """A hazard that looks exactly like a screen bug, pinned so it is not one.

    nanomem stores vectors as fp16 on disk and keeps them fp32 while resident.
    A writer that has appended and flushed but NOT reopened therefore still holds
    the fp32 originals, while a second engine opening the same file gets them
    back through the fp16 round trip. The two engines hold genuinely different
    numbers -- measured max vector difference 1.2e-04, max cosine difference
    5.9e-05 -- and their top-k can differ whatever the screen is doing.

    This test asserts that the disagreement is present with the flag OFF ON BOTH
    SIDES, which is the whole point: it belongs to fp16 storage, not to the
    screen. It exists because a comparison written the wrong way round (engine A
    with the flag on against engine B with it off) produced 88 mismatches in 100
    queries while the screen was in fact exact, and the next person to see that
    should find this test rather than go looking for a bound bug.

    The comparison that DOES isolate the flag is toggling ``screen_mode`` inside
    ONE engine, which is what every other test here and the bench's phase E do.
    """
    V = structured_rows(1200, seed=41)
    p = str(tmp_path / "v.dat")
    writer = fill(p, V[:900], screen="pca", screen_dims=32, screen_min_rows=100)
    writer.build_screen()
    for i in range(900, 1200):
        writer.add_fact(f"paragraph {i}", V[i], source="corpus.txt")
    writer.flush()
    reader = VaultEngine(p, embed_dim=D, screen="off")

    # Same rows, different numbers -- before any search happens.
    assert writer.arena.n_rows == reader.arena.n_rows == 1200
    delta = float(np.abs(writer.arena.matrix().astype(np.float64)
                         - reader.arena.matrix().astype(np.float64)).max())
    assert delta > 0.0, "fp16 round trip lost nothing; this hazard has gone away"

    # The screen is exact against the arena it was built on: flag toggled inside
    # ONE engine agrees bitwise, appended rows and all.
    Q = structured_rows(60, seed=42)
    for q in Q:
        writer.screen_mode = "pca"
        a = writer.search("a question", q, top_k=6)
        writer.screen_mode = "off"
        b = writer.search("a question", q, top_k=6)
        writer.screen_mode = "pca"
        assert keyed(a) == keyed(b)

    # And the cross-engine disagreement is there with the flag OFF on both
    # sides, so it was never the screen's.
    writer.screen_mode = "off"
    cross = sum(keyed(writer.search("a question", q, top_k=6))
                != keyed(reader.search("a question", q, top_k=6)) for q in Q)
    writer.screen_mode = "pca"
    assert cross > 0, ("the fp16 hazard did not reproduce, so this test no "
                       "longer proves what it claims")
    writer.close(); reader.close()


def test_replace_all_rebuilds_the_screen(tmp_path):
    V = structured_rows(2000, seed=18)
    p = str(tmp_path / "v.dat")
    e = fill(p, V, screen="pca", screen_dims=32, screen_min_rows=100)
    e.build_screen()
    q = structured_rows(1, seed=19)[0]
    e.search("a question", q, top_k=4)
    recs = list(e.iter_records())[::-1]          # same records, new row order
    e.replace_all(recs)
    hits = e.search("a question", q, top_k=4)
    e.screen_mode = "off"
    assert keyed(hits) == keyed(e.search("a question", q, top_k=4))
    e.close()


def test_gather_is_calibrated_and_bitwise(tmp_path):
    """The survivor gather must reproduce the full scan's arithmetic exactly.

    MEASURED (Apple Accelerate / numpy 2.5.3, 768 dims): a gathered sub-scan of
    1-8 rows differs from the full scan in the last bit; 16 rows and up is
    bitwise identical. The engine measures that threshold instead of hard-coding
    it, and stands the screen down if no size on its ladder reproduces the scan.
    """
    V = structured_rows(2000, seed=20)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    e = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=100)
    e.build_screen()
    info = e.screen_info()
    assert info["gather_bitwise_exact"] is True
    assert info["gather_pad"] in VaultEngine._GATHER_LADDER
    q = structured_rows(1, seed=21)[0]
    q = np.asarray(q, dtype=np.float32) / np.linalg.norm(q)
    full = e.arena.scores(q)
    rows = np.sort(np.random.default_rng(0).choice(
        e.arena.n_rows, info["gather_pad"], replace=False)).astype(np.int64)
    assert np.array_equal(e.arena.scores_rows(rows, q), full[rows])
    e.close()


# ---------------------------------------------------------------------------
# automatic fallback: every reason the screen must stand down
# ---------------------------------------------------------------------------
def test_below_the_engagement_floor_nothing_is_built(tmp_path):
    V = structured_rows(500, seed=22)
    p = str(tmp_path / "v.dat")
    e = fill(p, V, screen="pca", screen_dims=32, screen_min_rows=100000)
    assert e.build_screen()["built"] is False
    e.search("a question", V[0], top_k=4)
    assert e.screen_info()["engaged"] == 0
    e.close()


def test_int8_residency_refuses_the_screen(tmp_path):
    V = structured_rows(1500, seed=23)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    e = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=10,
                    residency="int8")
    assert e.build_screen()["built"] is False
    e.search("a question", V[0], top_k=4)
    assert e.screen_info()["engaged"] == 0
    e.close()


def test_a_broken_basis_falls_back_instead_of_raising(tmp_path):
    V = structured_rows(1500, seed=24)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=10)
    off = VaultEngine(p, embed_dim=D, screen="off")
    on.build_screen()
    q = structured_rows(1, seed=25)[0]
    on._screen.basis = None
    on._screen_arena = None
    assert keyed(on.search("a question", q, top_k=4)) == \
           keyed(off.search("a question", q, top_k=4))
    on.close(); off.close()


def test_a_metadata_filter_stands_the_screen_down(tmp_path):
    V = structured_rows(1500, seed=26)
    p = str(tmp_path / "v.dat")
    e = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32,
                    screen_min_rows=10)
    for i in range(V.shape[0]):
        e.add_fact(f"paragraph {i}", V[i], source="corpus.txt",
                   metadata={"bucket": i % 4})
    e.flush()
    e.build_screen()
    q = structured_rows(1, seed=27)[0]
    got = e.search("a question", q, top_k=4, metadata_filter={"bucket": 2})
    assert e.screen_info()["engaged"] == 0
    e.screen_mode = "off"
    assert keyed(got) == keyed(
        e.search("a question", q, top_k=4, metadata_filter={"bucket": 2}))
    e.close()


def test_a_personal_vault_stands_the_screen_down(tmp_path):
    """Entity boosts can lift a candidate by up to ``stats()['max_boost']``.

    A screen that prunes on COSINE cannot see that coming, so it must not run
    at all where a boost is possible -- which is exactly where the vault holds
    entity-tagged records.
    """
    V = structured_rows(600, seed=28)
    p = str(tmp_path / "v.dat")
    e = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32,
                    screen_min_rows=10)
    for i in range(V.shape[0]):
        e.add_fact(f"my phone number is 020-4455-{i:04d}", V[i],
                   source="chat_session")
    e.flush()
    assert e._has_personal_records() is True
    e.search("what is my phone number", V[0], top_k=4)
    assert e.screen_info()["engaged"] == 0
    e.close()


def test_an_explicit_historical_direction_stands_the_screen_down(tmp_path):
    V = structured_rows(1500, seed=29)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    e = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=10)
    e.build_screen()
    e.search("a question", V[0], top_k=4, temporal_direction="historical")
    assert e.screen_info()["engaged"] == 0
    e.search("a question", V[0], top_k=4)
    assert e.screen_info()["engaged"] == 1
    e.close()


def test_isotropic_corpus_falls_back_rather_than_slowing_down(tmp_path):
    """No subspace to find -> the bound admits too much -> fall back, stay exact.

    ``screen_max_frac`` is what makes the flag safe to leave on: a corpus the
    projection cannot compress does not get a slow search, it gets the ordinary
    one.
    """
    rng = np.random.default_rng(30)
    V = rng.normal(size=(3000, D)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=16, screen_min_rows=10)
    off = VaultEngine(p, embed_dim=D, screen="off")
    on.build_screen()
    Q = rng.normal(size=(40, D)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    for q in Q:
        assert keyed(on.search("a question", q, top_k=4)) == \
               keyed(off.search("a question", q, top_k=4))
    info = on.screen_info()
    assert info["fell_back"] > 0, "an isotropic corpus should trip screen_max_frac"
    on.close(); off.close()


# ---------------------------------------------------------------------------
# adversarial corpora
# ---------------------------------------------------------------------------
def test_a_corpus_packed_with_near_ties(tmp_path):
    """Hundreds of documents within 1e-6 of the k-th best score.

    This is where an off-by-an-epsilon threshold loses a document: every one of
    these sits inside the bound's own safety envelope.
    """
    rng = np.random.default_rng(31)
    base = rng.normal(size=D).astype(np.float32)
    base /= np.linalg.norm(base)
    V = np.repeat(base[None, :], 800, axis=0)
    V += 1e-6 * rng.normal(size=(800, D)).astype(np.float32)
    V = np.vstack([V, structured_rows(1200, seed=32)])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=10)
    off = VaultEngine(p, embed_dim=D, screen="off")
    on.build_screen()
    for q in [base] + list(structured_rows(30, seed=33)):
        a = on.search("a question", q, top_k=12)
        b = off.search("a question", q, top_k=12)
        assert keyed(a) == keyed(b)
    on.close(); off.close()


def test_exact_duplicate_rows(tmp_path):
    V = structured_rows(1000, seed=34)
    V = np.vstack([V, V[:200]])                  # 200 exact duplicates
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=10)
    off = VaultEngine(p, embed_dim=D, screen="off")
    on.build_screen()
    for q in structured_rows(40, seed=35):
        a = on.search("a question", q, top_k=6)
        b = off.search("a question", q, top_k=6)
        # The ORDERED id list, not just the scores. Where two documents carry
        # the IDENTICAL float32 cosine the exhaustive top-k is not unique, and
        # this assertion held only up to ties until `engine._select_top_k`
        # started breaking them on the row id: `np.argpartition` chooses among
        # equal elements differently depending on how long the array is, and
        # the screen hands it a shorter one. 9 of 3000 searches differed at
        # 71,433 docs before the fix -- 1 of them a different id SET, all 9
        # with a bitwise-identical score sequence, none a real miss
        # (pca_screen_results.json :: verdict.C1_exactness_BEFORE_the_fix).
        assert keyed(a) == keyed(b)
    on.close(); off.close()


def test_ties_are_broken_by_row_id_not_by_array_position():
    """The selection rule is a function of the SET of rows and nothing else.

    This is the unit-level statement of what makes the flag safe on a corpus
    that contains duplicates. ``_select_top_k`` is handed the same tied rows
    three ways -- the whole array, the whole array PERMUTED, and a subset that
    still contains every tied row, which is exactly what the screen hands it --
    and must choose the same row ids every time.

    An ``argpartition``-only selection FAILS this: its order among equal
    elements is unspecified and depends on the array's length. The assertion at
    the bottom shows that directly, so this test cannot pass merely because the
    tie never mattered.
    """
    from nanomem.engine import _select_top_k

    n, k = 400, 5
    final = np.full(n, 0.5)
    final[:40] = 0.9                        # 40 rows tied for 5 places
    final[40:80] = 0.7
    rows = np.arange(n, dtype=np.int64)

    whole = rows[_select_top_k(final, rows, k)]
    assert whole.tolist() == [0, 1, 2, 3, 4], whole

    rng = np.random.default_rng(11)
    perm = rng.permutation(n)
    shuffled = rows[perm][_select_top_k(final[perm], rows[perm], k)]
    assert shuffled.tolist() == whole.tolist()

    # The survivor set a screen would produce: every tied row plus some others,
    # ordered ascending, but only a fraction of the corpus.
    sub = np.sort(np.concatenate([np.arange(80), np.arange(300, 400)]))
    subset = rows[sub][_select_top_k(final[sub], rows[sub], k)]
    assert subset.tolist() == whole.tolist()

    # And the thing this replaced does not survive the same treatment.
    def positional(f, kk):
        idx = np.argpartition(-f, kk - 1)[:kk]
        return idx[np.argsort(-f[idx], kind="stable")]
    assert rows[sub][positional(final[sub], k)].tolist() != whole.tolist()


def test_a_query_that_matches_nothing(tmp_path):
    V = structured_rows(1500, seed=36)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=10)
    off = VaultEngine(p, embed_dim=D, screen="off")
    on.build_screen()
    q = np.zeros(D, dtype=np.float32); q[0] = 1.0
    assert keyed(on.search("a question", q, top_k=4)) == \
           keyed(off.search("a question", q, top_k=4))
    on.close(); off.close()


def test_top_k_larger_than_the_corpus(tmp_path):
    V = structured_rows(1200, seed=37)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=10)
    off = VaultEngine(p, embed_dim=D, screen="off")
    on.build_screen()
    q = structured_rows(1, seed=38)[0]
    assert keyed(on.search("a question", q, top_k=5000)) == \
           keyed(off.search("a question", q, top_k=5000))
    on.close(); off.close()


def test_bad_screen_argument_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        VaultEngine(str(tmp_path / "v.dat"), embed_dim=D, screen="pcaish")


def test_stats_reports_the_screen(tmp_path):
    V = structured_rows(1500, seed=39)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    e = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32, screen_min_rows=10)
    e.build_screen()
    st = e.stats()
    assert st["screen"] == "pca"
    assert st["screen_built"] is True
    assert st["screen_exact"] is True
    assert st["screen_resident_mb"] > 0
    e.close()


def test_resident_cost_matches_the_documented_formula():
    V = structured_rows(5000, seed=40)
    sc = S.PCAScreen(D, d_out=32).fit([V])
    sc.extend(V)
    assert sc.bytes_per_row() == 32 * 4 + 8
    expected = D * 32 * 4 + 32 * 32 * 8 + 5000 * (32 * 4 + 8)
    assert sc.used_bytes() == expected
    assert sc.resident_bytes() >= sc.used_bytes()


def test_screen_module_imports_nothing_but_numpy():
    """numpy is nanomem's only runtime dependency, and screen.py keeps it that way."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(S.__file__)),
                            "screen.py"), encoding="utf-8").read()
    for line in src.splitlines():
        t = line.strip()
        if t.startswith("import ") or t.startswith("from "):
            mod = t.split()[1].split(".")[0]
            assert mod in ("numpy", "typing", "time"), t


# ---------------------------------------------------------------------------
# quarantine: a suite-wide version of test_entities' single-module audit
# ---------------------------------------------------------------------------
# Deliberately narrow: "persona" alone matches "personal account", which is
# ordinary wording in a ranking test. These are the shapes a FILENAME takes.
_QUARANTINE_NEEDLES = ("chat_benchmark", "persona4", "_personas", "heldout")


def test_no_test_module_names_a_quarantined_fixture():
    """MECHANICAL, suite-wide. Every test file, not just the one that was fixed.

    ``test_entities.py`` audits ITSELF for wired-in fixture paths. That was not
    enough: ``test_round5_temporal.py`` globbed the same directory and read the
    text of every ``*chat_benchmark*.json`` and ``*personas*.json`` it found --
    which is ``clean_chat_benchmark_persona4.json``, scored-once material, and
    ``clean_chat_benchmark_heldout.json``, on every single ``pytest`` run. It
    was found by instrumenting ``builtins.open`` for a whole suite run and
    reading the paths back, because nothing in the suite's own output said so
    and an access timestamp would not have settled it (measured on this
    filesystem: reads do not reliably bump ``atime``).

    So the audit is now over every module in this directory. Docstrings and
    comments are exempt -- they are where the ban has to be discussed -- and so
    is a string that only names the offline extractor's own asset.
    """
    import ast
    here = os.path.dirname(os.path.abspath(__file__))
    me = os.path.basename(__file__)
    bad = []
    for fn in sorted(os.listdir(here)):
        if not (fn.startswith("test_") and fn.endswith(".py")) or fn == me:
            continue
        tree = ast.parse(open(os.path.join(here, fn), encoding="utf-8").read())
        doc_ids = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) \
                    and ast.get_docstring(node) is not None:
                doc_ids.add(id(node.body[0].value))
        exempt = [n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and "quarantined_fixture" in n.name]
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in doc_ids:
                continue
            if any(f.lineno <= node.lineno <= f.end_lineno for f in exempt):
                continue
            low = node.value.lower()
            if any(n in low for n in _QUARANTINE_NEEDLES):
                bad.append((fn, node.lineno, node.value[:60]))
    assert bad == [], bad


def test_this_study_names_no_corpus_path_at_all():
    """screen.py and this file open nothing. Proof by construction, not by atime."""
    import ast
    for path in (S.__file__, os.path.abspath(__file__)):
        tree = ast.parse(open(path, encoding="utf-8").read())
        doc_ids = {id(n.body[0].value) for n in ast.walk(tree)
                   if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
                   and ast.get_docstring(n) is not None}
        # This function's own body searches FOR those needles, so it is exempt
        # exactly as test_entities' single-module audit exempts itself.
        exempt = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "test_this_study_names_no_corpus_path_at_all"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in doc_ids \
                    and not any(f.lineno <= node.lineno <= f.end_lineno
                                for f in exempt):
                low = node.value.lower()
                assert "refound" not in low and "scratch" not in low, \
                    (os.path.basename(path), node.lineno, node.value[:60])


@pytest.mark.parametrize("residency", ["float32", "float16", "float16_mmap"])
def test_exact_under_every_residency_the_arena_calls_exact(tmp_path, residency):
    """The three modes whose ``Arena.scores`` is documented EXACT must all agree.

    ``int8`` is the fourth mode and is refused outright (see
    :func:`test_int8_residency_refuses_the_screen`): there ``Arena.scores`` is
    not a cosine below its re-rank pool, so there is no exact scan for the
    screen to be identical to.
    """
    V = structured_rows(3000, seed=41)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32,
                     screen_min_rows=100, residency=residency)
    off = VaultEngine(p, embed_dim=D, screen="off", residency=residency)
    assert on.build_screen()["built"] is True
    for q in structured_rows(60, seed=42):
        a = on.search("a question", q, top_k=6)
        b = off.search("a question", q, top_k=6)
        assert set(h["id"] for h in a) == set(h["id"] for h in b)
        assert [h["cosine"] for h in a] == [h["cosine"] for h in b]
    assert on.screen_info()["engaged"] > 0
    on.close(); off.close()


def test_search_batch_is_untouched_by_the_flag(tmp_path):
    """``search_batch`` is one matmul over every query; the screen skips it."""
    V = structured_rows(2000, seed=43)
    p = str(tmp_path / "v.dat")
    fill(p, V).close()
    on = VaultEngine(p, embed_dim=D, screen="pca", screen_dims=32,
                     screen_min_rows=100)
    off = VaultEngine(p, embed_dim=D, screen="off")
    on.build_screen()
    Q = structured_rows(12, seed=44)
    assert on.search_batch(Q, top_k=5) == off.search_batch(Q, top_k=5)
    on.close(); off.close()
