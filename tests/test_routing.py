"""nanomem.routing: pure numpy, deterministic, no I/O."""

import math

import numpy as np
import pytest

from nanomem import routing as R


def blobs(n=2000, dim=64, k=20, seed=0, spread=0.25):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    lab = rng.integers(k, size=n)
    V = centres[lab] + spread * rng.normal(size=(n, dim))
    return R.unit_rows(V), lab


def test_unit_rows():
    V = R.unit_rows(np.random.default_rng(0).normal(size=(10, 8)))
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-6)
    assert R.unit_rows(np.zeros((2, 4))).shape == (2, 4)      # no NaN on a zero row


def test_spherical_kmeans_is_deterministic_and_covers_blobs():
    V, lab = blobs()
    M1, a1 = R.spherical_kmeans(V, 20, seed=0)
    M2, a2 = R.spherical_kmeans(V, 20, seed=0)
    assert np.array_equal(a1, a2) and np.allclose(M1, M2)
    assert M1.shape == (20, V.shape[1])
    assert np.allclose(np.linalg.norm(M1, axis=1), 1.0, atol=1e-5)
    assert len(set(a1.tolist())) >= 15                        # no mass cluster collapse
    diff, _ = R.spherical_kmeans(V, 20, seed=1)
    assert not np.allclose(M1, diff)


def test_kmeans_plusplus_picks_distinct_seeds():
    V, _ = blobs(n=500, k=10)
    idx = R.kmeans_plusplus_init(V, 10, np.random.default_rng(0))
    assert len(set(idx.tolist())) == 10


def test_landmarks_degenerate_to_rows():
    V = R.unit_rows(np.random.default_rng(0).normal(size=(5, 16)))
    M = R.spherical_kmeans_landmarks(V, 8, seed=0)
    assert M.shape == (5, 16) and np.allclose(M, V, atol=1e-6)


def test_pool_max_and_power_mean():
    S = np.array([0.1, 0.9, 0.2, 0.3, 0.4], dtype=np.float32)
    starts = np.array([0, 2, 5], dtype=np.int64)
    assert np.allclose(R.pool(S, starts, "max"), [0.9, 0.4])
    pm = R.pool(S, starts, 2)
    assert pm[0] < 0.9 and pm[1] < 0.4                        # mean <= max


def test_beam_size():
    assert R.beam_size(100, 0.25, 4) == 25
    assert R.beam_size(10, 0.01, 4) == 4
    assert R.beam_size(3, 0.9, 4) == 3
    assert R.beam_size(0, 0.25, 4) == 0


def test_global_cell_router_partitions_every_row():
    V, _ = blobs(n=3000)
    r = R.GlobalCellRouter(cell_target=25, beam_frac=0.25).fit(V)
    assert r.n_cells == math.ceil(3000 / 25)
    assert sorted(r.cell_rows.tolist()) == list(range(3000))
    assert int(r.cell_start[-1]) == 3000
    rows = r.route(V[0])
    assert rows.size > 0 and np.all(np.diff(rows) > 0)         # sorted, unique
    assert 0.15 < r.scanned_fraction(V[0]) < 0.45


def test_global_cell_router_recall_on_clustered_data():
    V, _ = blobs(n=4000, k=40, seed=1)
    r = R.GlobalCellRouter(cell_target=25, beam_frac=0.25).fit(V)
    rng = np.random.default_rng(2)
    hit = 0
    trials = 100
    for _ in range(trials):
        q = R.unit_rows(rng.normal(size=(1, V.shape[1])))[0]
        exact = int(np.argmax(V @ q))
        hit += int(np.isin(exact, r.route(q)))
    assert hit / trials >= 0.80                                # measured ~0.9 on this data


def test_page_landmark_router_ablation():
    V, _ = blobs(n=1000)
    pages = [V[i:i + 50] for i in range(0, 1000, 50)]
    router = R.PageLandmarkRouter.from_pages(pages, L=8, beam_frac=0.25)
    assert router.n_blocks == 20
    beam = router.route(V[0])
    assert beam.size == 5 and len(set(beam.tolist())) == 5
    single = R.PageLandmarkRouter.from_pages(pages, L=1, beam_frac=0.25)
    assert single.landmarks.shape[0] == 20


def test_recluster_is_a_permutation_and_improves_locality():
    V, lab = blobs(n=2000, k=20, seed=3)
    rng = np.random.default_rng(4)
    perm0 = rng.permutation(2000)
    Vr = V[perm0]
    perm = R.recluster(Vr, cell_target=50)
    assert sorted(perm.tolist()) == list(range(2000))
    before = np.mean([lab[perm0[i]] == lab[perm0[i + 1]] for i in range(1999)])
    after = np.mean([lab[perm0[perm[i]]] == lab[perm0[perm[i + 1]]] for i in range(1999)])
    assert after > before + 0.3                                # locality restored


def test_nn_coverage_prefers_a_reclustered_layout():
    V, _ = blobs(n=1500, k=15, seed=5)
    r = R.GlobalCellRouter(cell_target=25, beam_frac=0.25).fit(V)
    cov = R.nn_coverage(V, r, sample=100)
    assert 0.0 <= cov <= 1.0 and cov > 0.5


def test_defaults_are_the_documented_ones():
    assert R.DEFAULTS["n_exhaustive"] == 50_000
    assert R.DEFAULTS["cell_target"] == 25
    assert R.DEFAULTS["beam_frac"] == 0.25
    assert R.DEFAULTS["pool"] == "max"


def test_routing_defaults_match_the_engine_signature():
    """routing.DEFAULTS must describe what a Vault actually does.

    3.0.0 advertised ``landmarks_per_block = 8`` here while ``VaultEngine``
    defaulted to 0, and the only guard checked ``R.DEFAULTS`` against itself.
    """
    import inspect
    from nanomem.engine import VaultEngine
    sig = inspect.signature(VaultEngine.__init__).parameters
    for key in ("n_exhaustive", "cell_target", "beam_frac", "beam_min_cells",
                "landmarks_per_block", "block_capacity"):
        assert sig[key].default == R.DEFAULTS[key], key
    # The DECISIONS #3 sweep HAS now been run -- 30 budgets on the reclustered
    # 71,433-paragraph corpus, scratch/refound/router_gate_results.json -- and it
    # cleared the recall gate at 13 of them. The default is still "off", because
    # recall was only the first of three conditions and the router fails the
    # other two: its speed-up over the exact scan changes sign between runs, and
    # the k-means fit it repeats at every open costs far more than the per-query
    # saving ever returns. See routing.py's module docstring for the numbers.
    # Flipping this line is a measurement, not an edit: re-run the sweep.
    assert sig["router"].default == "off"


def test_routing_imports_nothing_from_nanomem():
    src = open(R.__file__, encoding="utf-8").read()
    assert "from ." not in src and "import nanomem" not in src


def test_engine_router_auto_engages_after_recluster(tmp_path):
    """The opt-in cell router: one automatic re-cluster, then a routed scan.

    ``n_exhaustive`` is lowered here so the path is exercised without ingesting
    50,000 documents; the shipped default keeps every vault below that size on an
    exact scan.
    """
    import numpy as np

    from nanomem.engine import VaultEngine

    V, _ = blobs(n=3000, dim=768, k=60, seed=9, spread=0.12)
    rng = np.random.default_rng(1)
    order = rng.permutation(3000)                       # honest random insertion order
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=768, n_exhaustive=1000, router="auto",
                    vector_dtype="float32")
    for i in order:
        e.add_fact(f"row {int(i)}", V[i], metadata={"idx": int(i)})
    e.flush()
    assert e.stats()["routing_mode"] == "cells"
    assert e.count() == 3000
    gate = e.layout_gate()
    assert gate["coverage"] is not None and gate["coverage"] > 0.5

    arena = e.arena.vec[:e.arena.n_rows]
    hits = miss = 0
    for i in range(100):
        q = V[int(order[i])]
        exact = {e.arena.ids[j] for j in np.argsort(-(arena @ q))[:4]}
        got = {h["id"] for h in e.search("", q, top_k=4, min_score=-1.0)}
        hits += len(exact & got)
        miss += 4 - len(exact & got)
    recall = hits / (hits + miss)
    print(f"\nrouted recall@4 vs exhaustive on 3,000 reclustered docs: {100 * recall:.1f}% "
          f"at a {100 * e.beam_frac:.0f}% cell budget")
    # This is a UNIT check on synthetic, separable, 3,000 documents -- it shows
    # the wiring works, not that the router is worth using. The real measurement
    # of the shipped GlobalCellRouter is the DECISIONS #3 sweep over the
    # reclustered 71,433-paragraph corpus (scratch/refound/router_gate_results.json).
    assert recall > 0.85
    e.close()


# ---------------------------------------------------------------------------
# 3.0.4: the DECISIONS #3 sweep exists now, and the router it measured was
# first made as fast and as small as it reasonably could be -- a slow router is
# not a fair thing to judge. These guard the three optimisations that were made
# for that measurement (scratch/refound/router_gate_results.json).
# ---------------------------------------------------------------------------
def _expand_ranges_reference(starts, lengths):
    parts = [np.arange(s, s + l) for s, l in zip(starts, lengths) if l > 0]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)


def test_expand_ranges_matches_the_loop_it_replaced():
    rng = np.random.default_rng(0)
    for trial in range(300):
        m = int(rng.integers(0, 8))
        starts = np.sort(rng.integers(0, 60, m))
        lengths = rng.integers(0, 7, m)
        got = R._expand_ranges(starts, lengths)
        assert np.array_equal(got, _expand_ranges_reference(starts, lengths)), (starts, lengths)
        assert got.dtype == np.int64


def test_route_is_identical_to_the_per_cell_concatenation():
    """The vectorised beam expansion must not change WHICH rows are scanned.

    3.0.3 built the beam with a Python ``for cell in beam`` loop that ran 715
    times per query at 71,433 documents and the shipped ``beam_frac``.
    """
    V, _ = blobs(n=3000, dim=64, k=30, seed=7)
    rng = np.random.default_rng(8)
    for cell_target in (25, 50, 200):
        r = R.GlobalCellRouter(cell_target=cell_target, beam_frac=0.25).fit(V)
        for _ in range(25):
            q = R.unit_rows(rng.normal(size=(1, V.shape[1])))[0]
            for beam_frac in (0.02, 0.1, 0.25, 1.0):
                cells = r.route_cells(q, beam_frac=beam_frac)
                want = np.sort(np.concatenate(
                    [r.cell_rows[r.cell_start[c]:r.cell_start[c + 1]] for c in cells]))
                assert np.array_equal(r.route(q, beam_frac=beam_frac), want)


def test_contiguous_is_detected_and_scan_agrees_with_the_gather():
    """A cell-ordered arena turns the beam into slices; the answer must not move.

    Slices are the whole latency case for the router: a fancy-index gather of
    the beam costs more than the exact scan it is meant to replace.
    """
    V, _ = blobs(n=2000, dim=64, k=25, seed=11)
    r = R.GlobalCellRouter(cell_target=25, beam_frac=0.25).fit(V)
    assert r.contiguous is False                       # arbitrary insertion order

    W = np.ascontiguousarray(V[r.cell_rows])           # store rows in cell order
    # Re-fitting on W is NOT the way to get there: k-means++ over the new row
    # order finds a different partition, so its assignment is not sorted.
    assert R.GlobalCellRouter(cell_target=25, beam_frac=0.25).fit(W).contiguous is False
    c = R.GlobalCellRouter.adopt_layout(r.centroids, r.cell_start,
                                        cell_target=25, beam_frac=0.25)
    assert c.contiguous is True
    assert np.array_equal(c.cell_rows, np.arange(W.shape[0]))

    rng = np.random.default_rng(12)
    for _ in range(25):
        q = R.unit_rows(rng.normal(size=(1, V.shape[1])))[0]
        rows_gather = c.route(q)
        rows_scan, cos_scan = c.scan(W, q)
        assert np.array_equal(np.sort(rows_scan), rows_gather)
        order = np.argsort(rows_scan)
        assert np.allclose(cos_scan[order], W[rows_gather] @ q, atol=1e-6)
        ranges = c.route_ranges(q)
        assert ranges.shape[1] == 2
        assert int((ranges[:, 1] - ranges[:, 0]).sum()) == rows_gather.size


def test_scan_falls_back_to_the_gather_when_rows_are_not_in_cell_order():
    V, _ = blobs(n=1200, dim=32, k=12, seed=13)
    r = R.GlobalCellRouter(cell_target=25, beam_frac=0.3).fit(V)
    assert r.contiguous is False
    q = V[0]
    rows, cos = r.scan(V, q)
    assert np.array_equal(rows, r.route(q))
    assert np.allclose(cos, V[rows] @ q, atol=1e-6)


def test_chunked_assignment_matches_a_single_full_matmul():
    """``_nearest_centroid`` blocks the (n, k) similarity matrix it never builds.

    That matrix is 816 MB at n = 71,433 / k = 2,858 and was allocated once per
    Lloyd iteration; it is the largest single term in the 3,363 MB peak RSS the
    standard competitor run recorded for the routed arm.
    """
    rng = np.random.default_rng(14)
    A = R.unit_rows(rng.normal(size=(900, 48)))
    M = R.unit_rows(rng.normal(size=(37, 48)))
    S = A @ M.T
    want_assign = np.argmax(S, axis=1)
    want_best = S[np.arange(A.shape[0]), want_assign]
    for chunk in (1, 7, 128, 4096):
        assign, best = R._nearest_centroid(A, M, chunk=chunk)
        assert np.array_equal(assign, want_assign)
        assert np.allclose(best, want_best, atol=1e-6)


def test_cluster_sums_matches_the_boolean_mask_loop():
    """``_cluster_sums`` replaces k full-corpus boolean masks (O(n*k))."""
    rng = np.random.default_rng(15)
    A = R.unit_rows(rng.normal(size=(500, 24)))
    k = 20
    assign = rng.integers(0, k, 500).astype(np.int32)
    assign[assign == 3] = 4                                  # leave cluster 3 empty
    sums, counts = R._cluster_sums(A, assign, k)
    assert counts[3] == 0 and np.allclose(sums[3], 0.0)
    for j in range(k):
        sel = assign == j
        assert int(counts[j]) == int(sel.sum())
        if sel.any():
            assert np.allclose(sums[j], A[sel].sum(axis=0), atol=1e-4)


def test_kmeans_partition_survived_the_rewrite():
    """Same membership as the mask-loop implementation it replaced.

    ``reduceat`` accumulates sequentially where ``ndarray.sum`` pairwise-reduces,
    so centroids may differ in the last float32 ulp. Membership may not.
    """
    def reference(V, k, seed=0, iters=10):
        A = R.unit_rows(V)
        n = A.shape[0]
        rng = np.random.default_rng(seed)
        M = A[R.kmeans_plusplus_init(A, k, rng)].copy()
        for _ in range(iters):
            S = A @ M.T
            assign = np.argmax(S, axis=1).astype(np.int32)
            best = S[np.arange(n), assign]
            for j in range(k):
                sel = assign == j
                if sel.any():
                    M[j] = A[sel].sum(axis=0)
                else:
                    M[j] = A[int(np.argmin(best))]
                    best[int(np.argmin(best))] = np.inf
            nrm = np.linalg.norm(M, axis=1, keepdims=True)
            np.maximum(nrm, 1e-9, out=nrm)
            M /= nrm
        return np.argmax(A @ M.T, axis=1).astype(np.int32)

    for seed, n, k in ((0, 1500, 30), (2, 800, 64)):
        V, _ = blobs(n=n, dim=48, k=max(4, k // 3), seed=seed)
        assert np.array_equal(R.spherical_kmeans(V, k, seed=0)[1], reference(V, k, seed=0))


# ---------------------------------------------------------------------------
# DECISIONS #3, settled. The sweep ran (scratch/refound/sweep_router_gate.py ->
# router_gate_results.json): the RECALL gate passes and the router still ships
# OFF. These tests guard the two properties that decision rests on -- that
# `router="auto"` changes NOTHING below `n_exhaustive`, and that the routed path
# is the same scan as the exhaustive one when the beam is the whole corpus --
# plus the decision itself, so a future flip has to come with a re-run.
# ---------------------------------------------------------------------------
def _gate_results():
    """The sweep's results file, or a skip. It is measurement output, not code."""
    import json
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scratch", "refound", "router_gate_results.json")
    if not os.path.exists(p):
        pytest.skip("router gate results not present: %s" % p)
    return json.load(open(p, encoding="utf-8"))


def test_router_auto_changes_nothing_below_n_exhaustive(tmp_path):
    """Below the threshold, `router="auto"` must be the exact scan, id for id.

    This is the invariant that makes the option safe to offer at all: a caller
    who sets ``router="auto"`` on a vault that never grows past ``n_exhaustive``
    has asked for nothing and must get exactly what ``router="off"`` returns --
    same ids, same cosines, same order. Measured at 10,000 documents in the
    sweep (phase E, 200 questions, zero mismatches); this is the unit-scale
    version that runs in CI.
    """
    import numpy as np

    from nanomem.engine import VaultEngine

    V, _ = blobs(n=600, dim=64, k=12, seed=5, spread=0.20)
    Q, _ = blobs(n=40, dim=64, k=12, seed=6, spread=0.20)
    out = {}
    for mode in ("off", "auto"):
        e = VaultEngine(filepath=str(tmp_path / ("v_%s.dat" % mode)), embed_dim=64,
                        n_exhaustive=5000, router=mode, vector_dtype="float32")
        for i in range(V.shape[0]):
            e.add_fact("row %d" % i, V[i], metadata={"idx": i})
        e.flush()
        assert e.stats()["routing_mode"] == "exhaustive"     # not engaged: n < 5000
        # Compare on the caller's own key, not on ``id``: a doc id is minted per
        # vault and two separately-built vaults never agree on one.
        out[mode] = [[(h["metadata"]["idx"], round(float(h["cosine"]), 6))
                      for h in e.search("", Q[j], top_k=5, min_score=-1.0)]
                     for j in range(Q.shape[0])]
        e.close()
    assert out["off"] == out["auto"]


def test_routed_scan_with_a_full_beam_is_the_exhaustive_scan(tmp_path):
    """`beam_frac=1.0` scans every cell, so routing must be exact.

    If this ever fails, the loss is in the candidate gather or the re-cluster,
    not in the approximation -- which is the difference between "the router
    trades recall for speed" and "the router is broken".
    """
    import numpy as np

    from nanomem.engine import VaultEngine

    V, _ = blobs(n=2000, dim=64, k=40, seed=7, spread=0.15)
    order = np.random.default_rng(2).permutation(V.shape[0])
    args = dict(embed_dim=64, vector_dtype="float32")
    e_off = VaultEngine(filepath=str(tmp_path / "off.dat"), router="off",
                        n_exhaustive=10 ** 9, **args)
    e_on = VaultEngine(filepath=str(tmp_path / "on.dat"), router="auto",
                       n_exhaustive=500, beam_frac=1.0, cell_target=25, **args)
    for i in order:
        for e in (e_off, e_on):
            e.add_fact("row %d" % int(i), V[i], metadata={"idx": int(i)})
    for e in (e_off, e_on):
        e.flush()
    assert e_on.stats()["routing_mode"] == "cells"
    assert e_off.stats()["routing_mode"] == "exhaustive"
    for j in range(60):
        q = V[int(order[j])]
        a = [h["metadata"]["idx"] for h in e_off.search("", q, top_k=5, min_score=-1.0)]
        b = [h["metadata"]["idx"] for h in e_on.search("", q, top_k=5, min_score=-1.0)]
        assert a == b, (j, a, b)
    e_off.close()
    e_on.close()


def test_the_shipped_default_is_the_one_the_sweep_decided():
    """The results file and the library must not drift apart.

    ``verdict.decision.router_default_should_be`` is computed by the sweep from
    its own measurements under a rule written down before the numbers were in
    (recall within 1.0 pt, faster in every replicate, and paying back its open
    cost inside 10,000 queries). Whatever that field says, the engine's default
    has to match it. Changing the default therefore means re-running the sweep,
    not editing a keyword.
    """
    import inspect

    from nanomem.engine import VaultEngine

    d = _gate_results()["verdict"]["decision"]
    assert d["router_default_should_be"] in ("off", "auto")
    assert d["shipped_default"] == d["router_default_should_be"], d.get("one_sentence")
    sig = inspect.signature(VaultEngine.__init__).parameters
    assert sig["router"].default == d["shipped_default"]


def test_the_gate_file_can_be_audited_from_itself():
    """The CI that decides the default must be recomputable from the file.

    The first version of ``router_gate_results.json`` stored only summary
    statistics for each budget -- recall, the mean difference and the interval --
    and NO per-question data, so the paired bootstrap the whole DECISIONS #3 gate
    turns on could not be checked from the file. An auditor had to re-derive it
    from the cached embeddings, which is exactly the work the file exists to make
    unnecessary.

    Phase A now persists one digit per question per arm (0, 1 or 2 of the 2 gold
    paragraphs), and the sweep recomputes recall, the mean and the 95% interval
    from those digits and stores the comparison in ``phase_A_self_audit``. This
    asserts the raw data is still there and that the file still agrees with
    itself. Re-create it with ``python3 scratch/refound/sweep_router_gate.py
    --persist-phase-a`` (or ``--audit-phase-a`` to re-check only).
    """
    res = _gate_results()
    audit = res.get("phase_A_self_audit")
    if audit is None:
        pytest.skip("this results file predates the per-question audit block")
    assert audit["ran"] is True
    assert audit["arms_checked"] == len(res["phase_A_recall_gate"])
    assert audit["file_is_self_consistent"] is True, audit.get("disagreements")
    assert audit["max_abs_ci_delta_pt"] == 0.0, audit.get("disagreements")
    n = audit["questions"]
    for arm in res["phase_A_recall_gate"]:
        s = arm["per_question_gold_hits"]
        assert len(s) == n and set(s) <= set("012"), (arm["cell_target"], arm["beam_frac"])


def test_the_recall_gate_itself_passed_and_the_file_says_which_gate_failed():
    """Guard the SHAPE of the finding, so it cannot quietly become a different one.

    The honest summary of DECISIONS #3 is "recall passed, cost did not". If a
    re-run ever makes the recall gate fail, or makes every cost gate pass, that
    is a real change of result and this test should fail so somebody rewrites
    the prose instead of leaving it stale.
    """
    d = _gate_results()["verdict"]["decision"]
    assert d["recall_gate_passes"] is True, d["recall_gate_evidence"]
    if d["router_default_should_be"] == "off":
        assert d["gates_failed"], "default is off but no gate is recorded as failing"
        assert "recall" not in d["gates_failed"]
