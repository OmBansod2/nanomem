"""DECISIONS #3: the sweep that settles whether nanomem's IVF router ships ON.

DECISIONS.md, core decision 3:

    "Router above `n_exhaustive` = global spherical k-means cells (IVF-style),
     opt-in `router="auto"`. `beam_frac` chosen by the sweep with tolerance
     '95% CI lower bound >= -1.0 pt vs exhaustive'. Gate: the router engages by
     default above `n_exhaustive` only if the sweep shows it within 1.0 pt of
     exhaustive at its default budget on the 71k corpus in random insertion
     order after `compact(recluster=True)`; otherwise 3.0 ships exhaustive-only
     and prints the numbers."

WHAT THIS RUNS, and why each phase exists.

  prep  71,433 HotpotQA train paragraphs (cached nomic-embed-text vectors, never
        re-embedded), L2-normalised ONCE, shuffled into a random insertion order
        with `default_rng(0).permutation`, then put through the SHIPPED
        `routing.recluster(V, cell_target=block_capacity)` -- byte-for-byte the
        permutation `VaultEngine.compact(recluster=True)` applies. Every later
        phase loads those arrays; nothing re-normalises or re-shuffles.

  A     RECALL GATE. GlobalCellRouter fitted at each `cell_target`, evaluated at
        each `beam_frac` against the exact scan on the SAME rows, with a paired
        bootstrap (10,000 resamples, paired over questions) on the per-question
        recall@4 difference. `PageLandmarkRouter` is measured beside it as the
        ablation DECISIONS #3 asks for. This phase alone decides the gate.

  B     LATENCY, ARENA LEVEL, at 71,433. The gate is a necessary condition, not
        a sufficient one: a router that is exact enough and SLOWER than the scan
        it replaces is a regression, and flipping a default into a regression is
        the failure mode this whole re-founding exists to avoid. Two scan paths
        are timed because they are not the same cost: a fancy-index gather of
        the beam, which is what 3.0.3 does, and contiguous slices, which is what
        the beam becomes when the rows are stored in cell order.

  C     LATENCY AND RECALL, ENGINE LEVEL, at 71,433: a real vault, written with
        fp16 vectors, closed, RE-OPENED from disk, warmed, then timed. This is
        the number a caller actually gets, and it carries the costs phase B
        cannot see -- the one-time re-cluster, and the k-means fit that is paid
        again on every open because no centroid is persisted.

  D     SYNTHETIC 214,299 documents, so the "constant-factor win" claim is
        checked at a size no real corpus here reaches. The distractors are the
        DECISIONS #16 recipe (N(0, 0.35^2) noise copies, `default_rng(0)`), so
        the corpus is reproducible; it is SYNTHETIC and labelled as such.

  E     BELOW `n_exhaustive`, nothing may move. 10,000 documents, `router="auto"`
        against `router="off"`, id-for-id on 200 queries.

Every phase that reports memory or latency runs in its own subprocess, so one
arm's peak RSS and one arm's cache state cannot contaminate another's.

Usage:  python3 sweep_router_gate.py [--quick]
Writes: scratch/refound/router_gate_results.json
"""
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PKG = os.path.join(REPO, "nanomem_standalone")
OUT = os.path.join(HERE, "router_gate_results.json")
WORK = os.path.join(tempfile.gettempdir(), "nanomem_router_gate")
TOP_K = 4
N_QUESTIONS = 1000
N_TIMED = 500
BOOT_ITERS = 10_000
BOOT_SEED = 3
GATE_PT = -1.0                      # CI lower bound must be >= this

CELL_TARGETS = [12, 25, 50, 100, 250]
BEAM_FRACS = [0.02, 0.05, 0.10, 0.15, 0.25, 0.50]

sys.path.insert(0, PKG)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def timeit(fn, n, warm=20):
    for _ in range(warm):
        fn(0)
    ts = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        ts[i] = (time.perf_counter() - t0) * 1000.0
    return summarise(ts)


def summarise(ts):
    ts = np.asarray(ts, dtype=np.float64)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 4),
            "p95_ms": round(float(np.percentile(ts, 95)), 4),
            "mean_ms": round(float(ts.mean()), 4), "n": int(ts.size)}


def interleaved(fns, n, warm=20):
    """Time several arms against the SAME machine, query by query.

    This box is shared with whatever else is running (load average 3-21 during
    this session), and an arm measured at 16:59 cannot be compared with one
    measured at 17:04. Round-robin at single-query granularity is the only way
    the ratios survive that: each round charges every arm one query under the
    same conditions, so drift hits all arms equally.
    """
    names = list(fns)
    for name in names:
        for _ in range(warm):
            fns[name](0)
    ts = {name: np.empty(n, dtype=np.float64) for name in names}
    for i in range(n):
        for name in names:
            t0 = time.perf_counter()
            fns[name](i)
            ts[name][i] = (time.perf_counter() - t0) * 1000.0
    return {name: summarise(ts[name]) for name in names}


def load_avg():
    return [round(x, 2) for x in os.getloadavg()]


def paired_bootstrap(diff_pts, iters=BOOT_ITERS, seed=BOOT_SEED):
    """95% CI of the mean per-question difference, resampled over questions."""
    r = np.random.default_rng(seed)
    n = diff_pts.size
    means = np.empty(iters)
    step = 500
    for i in range(0, iters, step):
        m = min(step, iters - i)
        means[i:i + m] = diff_pts[r.integers(0, n, (m, n))].mean(axis=1)
    return (round(float(diff_pts.mean()), 4),
            round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4))


def hits_exact(V, Q, G, block=250):
    """Per-question recall@4 over the two gold paragraphs, exact scan."""
    out = np.zeros(Q.shape[0], dtype=np.float64)
    for i in range(0, Q.shape[0], block):
        S = V @ Q[i:i + block].T
        idx = np.argpartition(-S, TOP_K - 1, axis=0)[:TOP_K]
        for j in range(idx.shape[1]):
            top = set(idx[:, j].tolist())
            g = G[i + j]
            out[i + j] = (int(g[0]) in top) + (int(g[1]) in top)
    return out / 2.0


def hit_string(h):
    """Per-question gold hits as one digit per question: 0, 1 or 2 out of 2.

    The whole DECISIONS #3 gate turns on a paired bootstrap CI, and a file that
    stores only the summary statistic cannot be audited from itself -- neither
    the CI nor the recall can be recomputed from it. ``h`` holds the FRACTION of
    the two gold paragraphs each question retrieved, so ``2*h`` is an integer
    count and a whole arm fits in one 1,000-character string. From two such
    strings: recall@4 is ``mean(digits)/2``, and the paired per-question
    difference the bootstrap resamples is ``(router - exhaustive) * 50`` points.
    """
    return "".join(str(int(round(float(x) * 2))) for x in np.asarray(h).ravel())


def hits_from_string(s):
    """Inverse of :func:`hit_string` -- the per-question fractions."""
    return np.array([int(c) for c in s], dtype=np.float64) / 2.0


def hits_from_rows(rows, cos, g):
    if rows.size == 0:
        return 0.0
    k = min(TOP_K, rows.size)
    sel = np.argpartition(-cos, k - 1)[:k]
    top = set(rows[sel].tolist())
    return ((int(g[0]) in top) + (int(g[1]) in top)) / 2.0


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------
def prep():
    os.makedirs(WORK, exist_ok=True)
    stamp = os.path.join(WORK, "prep.json")
    if os.path.exists(stamp):
        return json.load(open(stamp))
    from nanomem import routing as R
    z = np.load(os.path.join(HERE, "hotpot_train_8k_embeds.npz"))
    C, Q, G = z["C"], z["Q"], z["G"]
    n = C.shape[0]
    C = (C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    Q = (Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    norms = {"doc_min": float(np.linalg.norm(C, axis=1).min()),
             "doc_max": float(np.linalg.norm(C, axis=1).max()),
             "query_min": float(np.linalg.norm(Q, axis=1).min()),
             "query_max": float(np.linalg.norm(Q, axis=1).max())}

    qrng = np.random.default_rng(11)
    qsel = np.sort(qrng.choice(Q.shape[0], size=min(N_QUESTIONS, Q.shape[0]), replace=False))

    perm = np.random.default_rng(0).permutation(n)          # random insertion order
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    Crand = np.ascontiguousarray(C[perm])
    Grand = inv[G]

    t0 = time.perf_counter()
    order = R.recluster(Crand, cell_target=50)              # == engine block_capacity
    recluster_s = time.perf_counter() - t0
    inv2 = np.empty(n, dtype=np.int64)
    inv2[order] = np.arange(n)
    Cre = np.ascontiguousarray(Crand[order])
    Gre = inv2[Grand]

    np.save(os.path.join(WORK, "Cre.npy"), Cre)
    np.save(os.path.join(WORK, "Gre.npy"), Gre[qsel])
    np.save(os.path.join(WORK, "Crand.npy"), Crand)
    np.save(os.path.join(WORK, "Grand.npy"), Grand[qsel])
    np.save(os.path.join(WORK, "Qs.npy"), Q[qsel])
    np.save(os.path.join(WORK, "qsel.npy"), qsel)
    info = {"documents": int(n), "questions": int(qsel.size), "embed_dim": int(C.shape[1]),
            "source_npz": "hotpot_train_8k_embeds.npz",
            "insertion_order": "np.random.default_rng(0).permutation(71433)",
            "question_selection": "np.random.default_rng(11).choice(8000, 1000, replace=False)",
            "recluster": "nanomem.routing.recluster(V, cell_target=50) -- identical to "
                         "VaultEngine.compact(recluster=True) at block_capacity=50",
            "recluster_seconds_numpy_only": round(recluster_s, 3),
            "vector_norms_after_prep": {k: round(v, 7) for k, v in norms.items()}}
    json.dump(info, open(stamp, "w"), indent=1)
    return info


def load(name):
    return np.load(os.path.join(WORK, name + ".npy"))


def vault_path(arm):
    return os.path.join(WORK, "vault_%s.dat" % arm)


def engine_kwargs(cfg, dim):
    kw = dict(filepath=vault_path(cfg["arm"]), embed_dim=dim, router=cfg["router"],
              n_exhaustive=cfg["n_exhaustive"])
    if cfg.get("beam_frac") is not None:
        kw["beam_frac"] = cfg["beam_frac"]
    if cfg.get("cell_target") is not None:
        kw["cell_target"] = cfg["cell_target"]
    return kw


# ---------------------------------------------------------------------------
# workers (each runs in its own process)
# ---------------------------------------------------------------------------
def worker_recall(cfg):
    from nanomem import routing as R
    Cre, Q, G = load("Cre"), load("Qs"), load("Gre")
    layout = cfg.get("layout", "reclustered")
    if layout == "random":
        Cre, G = load("Crand"), load("Grand")
    base = rss_mb()
    ex = hits_exact(Cre, Q, G)
    out = {"layout": layout, "exhaustive_recall_at_4_pct": round(100 * float(ex.mean()), 4),
           "exhaustive_per_question_gold_hits": hit_string(ex),
           "per_question_encoding": "one digit per question, 0/1/2 of the 2 gold "
                                    "paragraphs in the top 4; recall@4 = mean/2, "
                                    "paired difference = (router - exhaustive) * 50 pt",
           "arms": []}
    ct = cfg["cell_target"]
    t0 = time.perf_counter()
    rt = R.GlobalCellRouter(cell_target=ct).fit(Cre)
    fit_s = time.perf_counter() - t0
    fit_rss = rss_mb() - base
    for bf in cfg["beam_fracs"]:
        h = np.zeros(Q.shape[0])
        scanned = 0
        for j in range(Q.shape[0]):
            rows = rt.route(Q[j], beam_frac=bf)
            scanned += rows.size
            h[j] = hits_from_rows(rows, Cre[rows] @ Q[j], G[j])
        m, lo, hi = paired_bootstrap((h - ex) * 100.0)
        out["arms"].append({
            "router": "GlobalCellRouter", "cell_target": ct, "cells": int(rt.n_cells),
            "beam_frac": bf, "beam_cells": int(R.beam_size(rt.n_cells, bf, 4)),
            "recall_at_4_pct": round(100 * float(h.mean()), 4),
            "delta_vs_exhaustive_pt": m, "ci95_pt": [lo, hi],
            "passes_gate": bool(lo >= GATE_PT),
            "scan_fraction": round(scanned / float(Q.shape[0] * Cre.shape[0]), 5),
            "fit_seconds": round(fit_s, 3),
            "fit_peak_rss_delta_mb": round(fit_rss, 1),
            "per_question_gold_hits": hit_string(h)})
    return out


def worker_ablation(cfg):
    """PageLandmarkRouter -- the ablation DECISIONS #3 keeps, never the default."""
    from nanomem import routing as R
    Cre, Q, G = load("Cre"), load("Qs"), load("Gre")
    ex = hits_exact(Cre, Q, G)
    n, c = Cre.shape[0], 50                       # one page == one 50-row block
    nb = math.ceil(n / c)
    starts = np.arange(nb) * c
    stops = np.minimum(starts + c, n)
    out = {"exhaustive_recall_at_4_pct": round(100 * float(ex.mean()), 4),
           "exhaustive_per_question_gold_hits": hit_string(ex), "arms": []}
    for L in cfg["landmarks"]:
        t0 = time.perf_counter()
        pr = R.PageLandmarkRouter.from_pages((Cre[a:b] for a, b in zip(starts, stops)), L=L)
        fit_s = time.perf_counter() - t0
        for bf in cfg["beam_fracs"]:
            h = np.zeros(Q.shape[0])
            scanned = 0
            for j in range(Q.shape[0]):
                pages = np.sort(pr.route(Q[j], beam_frac=bf))
                rows = R._expand_ranges(starts[pages], stops[pages] - starts[pages])
                scanned += rows.size
                h[j] = hits_from_rows(rows, Cre[rows] @ Q[j], G[j])
            m, lo, hi = paired_bootstrap((h - ex) * 100.0)
            out["arms"].append({
                "router": "PageLandmarkRouter", "landmarks_per_block": L, "pages": nb,
                "beam_frac": bf, "recall_at_4_pct": round(100 * float(h.mean()), 4),
                "delta_vs_exhaustive_pt": m, "ci95_pt": [lo, hi],
                "passes_gate": bool(lo >= GATE_PT),
                "scan_fraction": round(scanned / float(Q.shape[0] * n), 5),
                "fit_seconds": round(fit_s, 3),
                "per_question_gold_hits": hit_string(h)})
    return out


def worker_latency(cfg):
    """Arena-level scan cost. Exhaustive and both router paths, INTERLEAVED.

    The exhaustive reference is timed twice: once before any router structure
    exists (the condition a `router="off"` process actually runs in) and once
    at the end, after the router's centroids and the cell-ordered copy are
    resident. Both are reported; the first is the one the ratio uses, because
    turning the router ON is what would pay for those bytes.
    """
    from nanomem import routing as R
    Cre, Q = load("Cre"), load("Qs")
    nq = min(N_TIMED, Q.shape[0])
    ct, bf = cfg["cell_target"], cfg["beam_frac"]
    out = {"cell_target": ct, "beam_frac": bf, "load_avg_start": load_avg()}
    base = rss_mb()

    def f_exact(i):
        cos = Cre @ Q[i % nq]
        return np.argpartition(-cos, TOP_K - 1)[:TOP_K]

    out["exhaustive_alone"] = timeit(f_exact, nq)
    out["exhaustive_alone_peak_rss_delta_mb"] = round(rss_mb() - base, 1)

    t0 = time.perf_counter()
    rt = R.GlobalCellRouter(cell_target=ct).fit(Cre)
    out["fit_seconds"] = round(time.perf_counter() - t0, 3)
    out["cells"] = int(rt.n_cells)
    out["fit_peak_rss_delta_mb"] = round(rss_mb() - base, 1)

    def f_gather(i):
        q = Q[i % nq]
        rows = rt.route(q, beam_frac=bf)
        cos = Cre[rows] @ q
        k = min(TOP_K, rows.size)
        return rows[np.argpartition(-cos, k - 1)[:k]]

    W = np.ascontiguousarray(Cre[rt.cell_rows])
    rt2 = R.GlobalCellRouter.adopt_layout(rt.centroids, rt.cell_start,
                                          cell_target=ct, beam_frac=bf)
    out["cell_ordered_copy_mb"] = round(W.nbytes / (1024.0 * 1024.0), 1)

    def f_contig(i):
        q = Q[i % nq]
        rows, cos = rt2.scan(W, q, beam_frac=bf)
        k = min(TOP_K, rows.size)
        return rows[np.argpartition(-cos, k - 1)[:k]]

    out["interleaved"] = interleaved(
        {"exhaustive": f_exact, "router_gather": f_gather, "router_contiguous": f_contig}, nq)
    out["exhaustive_after_router_resident"] = timeit(f_exact, nq)
    out["peak_rss_delta_mb"] = round(rss_mb() - base, 1)
    out["load_avg_end"] = load_avg()
    ex = out["interleaved"]["exhaustive"]["p50_ms"]
    out["speedup_p50"] = {k: round(ex / v["p50_ms"], 3)
                          for k, v in out["interleaved"].items()}
    return out


def worker_engine(cfg):
    """Build one real vault, close it, RE-OPEN it, and score it. Own process."""
    from nanomem.engine import VaultEngine
    Cre, Q, G = load("Crand"), load("Qs"), load("Grand")   # random order, as a user writes
    n = Cre.shape[0]
    path = vault_path(cfg["arm"])
    for suffix in ("", ".v2.bak"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    la_start = load_avg()
    base = rss_mb()
    kw = engine_kwargs(cfg, Cre.shape[1])
    t0 = time.perf_counter()
    e = VaultEngine(**kw)
    for i in range(n):
        e.add_fact("doc %d" % i, Cre[i], source="wiki", metadata={"row": i})
    add_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    e.flush()                      # the crossing point: recluster + k-means fit land here
    flush_s = time.perf_counter() - t0
    e.close()
    del e
    disk_mb = os.path.getsize(path) / (1024.0 * 1024.0)

    t0 = time.perf_counter()
    e = VaultEngine(**kw)
    open_s = time.perf_counter() - t0
    st = e.stats()
    nq = min(N_TIMED, Q.shape[0])
    hits = np.zeros(nq)
    for j in range(nq):
        got = {h["metadata"].get("row") for h in
               e.search("", Q[j], top_k=TOP_K, min_score=-1.0)}
        hits[j] = ((int(G[j][0]) in got) + (int(G[j][1]) in got)) / 2.0
    gate = e.layout_gate() if st["routing_mode"] == "cells" else {"coverage": None}
    e.close()
    return {"arm": cfg["arm"], "router": cfg["router"], "n_exhaustive": cfg["n_exhaustive"],
            "beam_frac": st.get("beam_fraction"), "cell_target": cfg.get("cell_target"),
            "documents": int(st["total_documents"]), "routing_mode": st["routing_mode"],
            "beam_cells": st.get("beam_cells"), "layout_gate_coverage": gate.get("coverage"),
            "add_fact_seconds": round(add_s, 3),
            "final_flush_seconds": round(flush_s, 3),
            "ingest_seconds_total": round(add_s + flush_s, 3),
            "reopen_seconds": round(open_s, 3),
            "disk_mb": round(disk_mb, 1), "peak_rss_delta_mb": round(rss_mb() - base, 1),
            "recall_at_4_pct": round(100 * float(hits.mean()), 4),
            "per_question_hits": hits.tolist(),
            "load_avg_start": la_start, "load_avg_end": load_avg()}


def worker_engine_latency(cfg):
    """Open every already-built vault at once and interleave their queries."""
    from nanomem.engine import VaultEngine
    Q = load("Qs")
    nq = min(N_TIMED, Q.shape[0])
    out = {"load_avg_start": load_avg(), "arms": [a["arm"] for a in cfg["arms"]]}
    engines, fns = {}, {}
    for a in cfg["arms"]:
        e = VaultEngine(**engine_kwargs(a, int(load("Crand").shape[1])))
        engines[a["arm"]] = e
        fns[a["arm"]] = (lambda eng: (lambda i: eng.search("", Q[i % nq], top_k=TOP_K,
                                                           min_score=-1.0)))(e)
        out.setdefault("routing_mode", {})[a["arm"]] = e.stats()["routing_mode"]
    out["interleaved"] = interleaved(fns, nq)
    for e in engines.values():
        e.close()
    ex = out["interleaved"]["off"]["p50_ms"]
    out["speedup_p50_vs_off"] = {k: round(ex / v["p50_ms"], 3)
                                 for k, v in out["interleaved"].items()}
    out["load_avg_end"] = load_avg()
    return out


def worker_synthetic(cfg):
    """~214k SYNTHETIC documents: exhaustive and both router paths, interleaved.

    The corpus is the real 71,433 reclustered vectors plus `extra_copies` noisy
    copies built with the DECISIONS #16 distractor recipe -- v + N(0, 0.35^2),
    re-normalised, `np.random.default_rng(0)`. It exists to check the
    constant-factor claim at a size no corpus here reaches. It is synthetic: the
    distractors are near-duplicates of the real rows, which makes retrieval
    HARDER than a genuine 214k corpus would, so treat the absolute recall as a
    stress figure and the LATENCY RATIO as the result.
    """
    from nanomem import routing as R
    Cre, Q, G = load("Cre"), load("Qs"), load("Gre")
    n0, D = Cre.shape
    rng = np.random.default_rng(0)
    parts = [Cre]
    for _ in range(int(cfg["extra_copies"])):
        X = Cre + 0.35 * rng.normal(size=(n0, D)).astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        parts.append(X.astype(np.float32))
    V = np.ascontiguousarray(np.vstack(parts))
    del parts
    n = V.shape[0]
    nq = min(N_TIMED, Q.shape[0])
    ct, bf = cfg["cell_target"], cfg["beam_frac"]
    base = rss_mb()
    out = {"documents": int(n), "extra_copies": int(cfg["extra_copies"]),
           "cell_target": ct, "beam_frac": bf, "load_avg_start": load_avg(),
           "distractor_recipe": "DECISIONS #16: v + N(0, 0.35^2), re-normalised, "
                                "np.random.default_rng(0); SYNTHETIC"}

    def f_exact(i):
        return np.argpartition(-(V @ Q[i % nq]), TOP_K - 1)[:TOP_K]

    out["exhaustive_alone"] = timeit(f_exact, nq, warm=5)
    ex_hits = hits_exact(V, Q, G, block=100)
    out["exhaustive_recall_at_4_pct"] = round(100 * float(ex_hits.mean()), 4)

    t0 = time.perf_counter()
    rt = R.GlobalCellRouter(cell_target=ct).fit(V)
    out["fit_seconds"] = round(time.perf_counter() - t0, 2)
    out["cells"] = int(rt.n_cells)

    def f_gather(i):
        q = Q[i % nq]
        rows = rt.route(q, beam_frac=bf)
        cos = V[rows] @ q
        k = min(TOP_K, rows.size)
        return rows[np.argpartition(-cos, k - 1)[:k]]

    hg = np.zeros(Q.shape[0])
    scanned = 0
    for j in range(Q.shape[0]):
        rows = rt.route(Q[j], beam_frac=bf)
        scanned += rows.size
        hg[j] = hits_from_rows(rows, V[rows] @ Q[j], G[j])
    m, lo, hi = paired_bootstrap((hg - ex_hits) * 100.0)
    out["router_recall_at_4_pct"] = round(100 * float(hg.mean()), 4)
    out["router_delta_vs_exhaustive_pt"] = m
    out["router_ci95_pt"] = [lo, hi]
    out["router_passes_gate"] = bool(lo >= GATE_PT)
    out["scan_fraction"] = round(scanned / float(Q.shape[0] * n), 5)

    W = np.ascontiguousarray(V[rt.cell_rows])
    rt2 = R.GlobalCellRouter.adopt_layout(rt.centroids, rt.cell_start,
                                          cell_target=ct, beam_frac=bf)

    def f_contig(i):
        q = Q[i % nq]
        rows, cos = rt2.scan(W, q, beam_frac=bf)
        k = min(TOP_K, rows.size)
        return rows[np.argpartition(-cos, k - 1)[:k]]

    out["interleaved"] = interleaved(
        {"exhaustive": f_exact, "router_gather": f_gather, "router_contiguous": f_contig},
        nq, warm=5)
    exp = out["interleaved"]["exhaustive"]["p50_ms"]
    out["speedup_p50"] = {k: round(exp / v["p50_ms"], 3) for k, v in out["interleaved"].items()}
    out["peak_rss_delta_mb"] = round(rss_mb() - base, 1)
    out["load_avg_end"] = load_avg()
    return out


def worker_below(cfg):
    """Below n_exhaustive, router='auto' must change absolutely nothing."""
    from nanomem.engine import VaultEngine
    C, Q = load("Crand")[:cfg["docs"]], load("Qs")[:200]
    res = {}
    for mode in ("off", "auto"):
        path = os.path.join(WORK, "below_%s.dat" % mode)
        if os.path.exists(path):
            os.remove(path)
        e = VaultEngine(filepath=path, embed_dim=C.shape[1], router=mode, n_exhaustive=50_000)
        ids = [e.add_fact("doc %d" % i, C[i], source="wiki", metadata={"row": i})
               for i in range(C.shape[0])]
        e.flush()
        e.close()
        e = VaultEngine(filepath=path, embed_dim=C.shape[1], router=mode, n_exhaustive=50_000)
        st = e.stats()
        outs = [[(h["metadata"].get("row"), round(float(h["cosine"]), 6)) for h in
                 e.search("", Q[j], top_k=TOP_K, min_score=-1.0)] for j in range(Q.shape[0])]
        e.close()
        res[mode] = {"routing_mode": st["routing_mode"], "router": st["router"],
                     "documents": int(st["total_documents"]), "results": outs}
    same = res["off"]["results"] == res["auto"]["results"]
    return {"docs": cfg["docs"], "n_exhaustive": 50_000, "questions": 200,
            "routing_mode_off": res["off"]["routing_mode"],
            "routing_mode_auto": res["auto"]["routing_mode"],
            "identical_id_and_cosine_for_every_question": bool(same),
            "first_mismatch": None if same else next(
                (j for j in range(len(res["off"]["results"]))
                 if res["off"]["results"][j] != res["auto"]["results"][j]), None)}


WORKERS = {"recall": worker_recall, "ablation": worker_ablation, "latency": worker_latency,
           "engine": worker_engine, "engine_latency": worker_engine_latency,
           "synthetic": worker_synthetic, "below": worker_below}


def run(kind, cfg, label):
    """Run one worker in a clean subprocess and return its JSON result."""
    payload = json.dumps({"kind": kind, "cfg": cfg})
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", payload],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print("  !! %s FAILED\n%s" % (label, p.stderr[-2500:]), flush=True)
        return {"failed": True, "label": label, "stderr": p.stderr[-2500:]}
    out = json.loads(p.stdout.strip().splitlines()[-1])
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    print("  %-46s %.1fs" % (label, out["wall_seconds"]), flush=True)
    return out


# ---------------------------------------------------------------------------
def main():
    quick = "--quick" in sys.argv
    import nanomem
    info = prep()
    print("prep: %d docs, %d questions, recluster %.2fs"
          % (info["documents"], info["questions"], info["recluster_seconds_numpy_only"]),
          flush=True)

    res = {
        "what": "DECISIONS #3 router gate: does nanomem's IVF cell router ship ON by default?",
        "engine_version": nanomem.ENGINE_VERSION,
        "package_version": getattr(nanomem, "__version__", "?"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": sys.version.split()[0], "numpy": np.__version__,
                    "cpu_count": os.cpu_count(),
                    "load_avg_start": [round(x, 2) for x in os.getloadavg()]},
        "protocol": {
            "gate": ("DECISIONS #3: engage the router by default above n_exhaustive only "
                     "if the 95%% CI lower bound of (router - exhaustive) recall@4 is "
                     ">= %.1f pt at the chosen default budget, on the 71,433-doc corpus "
                     "in random insertion order after compact(recluster=True)." % GATE_PT),
            "gate_is_necessary_not_sufficient": (
                "A router exists to be cheaper than the scan it replaces. Passing the "
                "recall gate earns the router the right to be considered; phases B-D "
                "decide whether turning it on is an improvement or a regression."),
            "top_k": TOP_K, "metric": "recall@4 over the 2 gold paragraphs per question",
            "bootstrap": "%d resamples, paired over questions, seed %d" % (BOOT_ITERS, BOOT_SEED),
            "timed_queries_per_arm": N_TIMED,
            "isolation": "one subprocess per arm; ru_maxrss delta measured inside it",
            "prep": info},
        "phase_A_recall_gate": [], "phase_A_ablation": None,
        "phase_B_latency_71433": [], "phase_C_engine_71433": [],
        "phase_D_synthetic_214299": [], "phase_E_below_n_exhaustive": None,
        "failures": []}

    cts = CELL_TARGETS if not quick else [25]
    print("\nPHASE A -- recall gate (%d cell_targets x %d beam_fracs)" % (len(cts), len(BEAM_FRACS)),
          flush=True)
    for ct in cts:
        r = run("recall", {"cell_target": ct, "beam_fracs": BEAM_FRACS}, "recall cell_target=%d" % ct)
        if r.get("failed"):
            res["failures"].append(r)
        else:
            res["exhaustive_recall_at_4_pct"] = r["exhaustive_recall_at_4_pct"]
            res["phase_A_recall_gate"].extend(r["arms"])
    r = run("recall", {"cell_target": 25, "beam_fracs": [0.10, 0.25], "layout": "random"},
            "recall cell_target=25 RANDOM layout")
    if not r.get("failed"):
        res["phase_A_random_layout_control"] = r
    r = run("ablation", {"landmarks": [8, 16], "beam_fracs": [0.10, 0.25]},
            "ablation PageLandmarkRouter")
    if r.get("failed"):
        res["failures"].append(r)
    else:
        res["phase_A_ablation"] = r

    passing = [a for a in res["phase_A_recall_gate"] if a["passes_gate"]]
    passing.sort(key=lambda a: (a["scan_fraction"], a["cell_target"]))
    cands = passing[:4]
    res["gate_passing_configs"] = [{k: a[k] for k in
                                    ("cell_target", "beam_frac", "recall_at_4_pct",
                                     "delta_vs_exhaustive_pt", "ci95_pt", "scan_fraction")}
                                   for a in passing]
    print("\n  %d of %d configs pass the recall gate; cheapest: %s"
          % (len(passing), len(res["phase_A_recall_gate"]),
             [(a["cell_target"], a["beam_frac"]) for a in cands]), flush=True)

    print("\nPHASE B -- arena-level latency at 71,433 (arms interleaved per query)", flush=True)
    for a in cands:
        res["phase_B_latency_71433"].append(run(
            "latency", {"cell_target": a["cell_target"], "beam_frac": a["beam_frac"]},
            "latency ct=%d bf=%.2f" % (a["cell_target"], a["beam_frac"])))

    print("\nPHASE C -- engine end to end at 71,433 (reopened vault)", flush=True)
    best = cands[0] if cands else {"cell_target": 25, "beam_frac": 0.25}
    engine_arms = [
        {"arm": "off", "router": "off", "n_exhaustive": 50_000},
        {"arm": "auto_shipped_budget", "router": "auto", "n_exhaustive": 1_000,
         "cell_target": 25, "beam_frac": 0.25},
        {"arm": "auto_gate_budget", "router": "auto", "n_exhaustive": 1_000,
         "cell_target": best["cell_target"], "beam_frac": best["beam_frac"]},
    ]
    res["engine_arm_configs"] = engine_arms
    for cfg in engine_arms:
        r = run("engine", cfg, "engine build+score %s" % cfg["arm"])
        if r.get("failed"):
            res["failures"].append(r)
        else:
            res["phase_C_engine_71433"].append(r)
    if len(res["phase_C_engine_71433"]) == len(engine_arms):
        res["phase_C_engine_latency_interleaved"] = run(
            "engine_latency", {"arms": engine_arms}, "engine latency (interleaved)")

    print("\nPHASE D -- synthetic 214,299 (DECISIONS #16 distractor recipe)", flush=True)
    r = run("synthetic", {"extra_copies": 2, "cell_target": best["cell_target"],
                          "beam_frac": best["beam_frac"]}, "synthetic 214,299 (interleaved)")
    if r.get("failed"):
        res["failures"].append(r)
    else:
        res["phase_D_synthetic_214299"].append(r)

    print("\nPHASE E -- below n_exhaustive", flush=True)
    r = run("below", {"docs": 10_000}, "below n_exhaustive (10,000 docs)")
    res["phase_E_below_n_exhaustive"] = r
    if r.get("failed"):
        res["failures"].append(r)

    res["verdict"] = verdict(res)
    res["machine"]["load_avg_end"] = [round(x, 2) for x in os.getloadavg()]
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT, flush=True)
    print(json.dumps(res["verdict"], indent=1), flush=True)


def verdict(res):
    A = res["phase_A_recall_gate"]
    passing = [a for a in A if a["passes_gate"]]
    passing.sort(key=lambda a: (a["scan_fraction"], a["cell_target"]))
    C = {x["arm"]: x for x in res["phase_C_engine_71433"]}
    v = {"recall_gate": {
            "passes": bool(passing),
            "n_configs_passing": len(passing),
            "n_configs_tested": len(A),
            "cheapest_passing": ({k: passing[0][k] for k in
                                  ("cell_target", "beam_frac", "recall_at_4_pct",
                                   "delta_vs_exhaustive_pt", "ci95_pt", "scan_fraction")}
                                 if passing else None)}}
    v["arena_latency_71433"] = [
        {"cell_target": x.get("cell_target"), "beam_frac": x.get("beam_frac"),
         "p50_ms": x.get("interleaved", {}).get("exhaustive", {}).get("p50_ms"),
         "router_gather_p50_ms": x.get("interleaved", {}).get("router_gather", {}).get("p50_ms"),
         "router_contiguous_p50_ms": x.get("interleaved", {}).get("router_contiguous", {}).get("p50_ms"),
         "speedup_gather": x.get("speedup_p50", {}).get("router_gather"),
         "speedup_contiguous": x.get("speedup_p50", {}).get("router_contiguous"),
         "fit_seconds": x.get("fit_seconds")}
        for x in res["phase_B_latency_71433"] if not x.get("failed")]
    il = res.get("phase_C_engine_latency_interleaved") or {}
    if "off" in C:
        v["engine_71433"] = [
            {"arm": k, "routing_mode": x["routing_mode"],
             "p50_ms": (il.get("interleaved", {}).get(k) or {}).get("p50_ms"),
             "speedup_vs_off": (il.get("speedup_p50_vs_off") or {}).get(k),
             "ingest_seconds_total": x["ingest_seconds_total"],
             "reopen_seconds": x["reopen_seconds"],
             "peak_rss_delta_mb": x["peak_rss_delta_mb"],
             "recall_at_4_pct": x["recall_at_4_pct"]}
            for k, x in C.items()]
    if res["phase_D_synthetic_214299"]:
        d = res["phase_D_synthetic_214299"][0]
        v["synthetic_214299"] = {
            "documents": d["documents"], "cell_target": d["cell_target"],
            "beam_frac": d["beam_frac"], "fit_seconds": d["fit_seconds"],
            "exhaustive_p50_ms": d["interleaved"]["exhaustive"]["p50_ms"],
            "router_gather_p50_ms": d["interleaved"]["router_gather"]["p50_ms"],
            "router_contiguous_p50_ms": d["interleaved"]["router_contiguous"]["p50_ms"],
            "speedup": d["speedup_p50"],
            "exhaustive_recall_at_4_pct": d["exhaustive_recall_at_4_pct"],
            "router_recall_at_4_pct": d["router_recall_at_4_pct"],
            "router_delta_pt": d["router_delta_vs_exhaustive_pt"],
            "router_ci95_pt": d["router_ci95_pt"]}
    v["below_n_exhaustive_unchanged"] = (
        res["phase_E_below_n_exhaustive"] or {}).get("identical_id_and_cosine_for_every_question")
    conf = res.get("latency_confirmation_quiet_machine")
    if conf:
        v["quiet_machine"] = {
            "load_avg_start": conf.get("load_avg_start"),
            "load_avg_end": conf.get("load_avg_end"),
            "arena_latency_71433": [
                {"cell_target": x.get("cell_target"), "beam_frac": x.get("beam_frac"),
                 "p50_ms": x.get("interleaved", {}).get("exhaustive", {}).get("p50_ms"),
                 "router_gather_p50_ms": x.get("interleaved", {}).get("router_gather", {}).get("p50_ms"),
                 "router_contiguous_p50_ms": x.get("interleaved", {}).get("router_contiguous", {}).get("p50_ms"),
                 "speedup_gather": x.get("speedup_p50", {}).get("router_gather"),
                 "speedup_contiguous": x.get("speedup_p50", {}).get("router_contiguous"),
                 "fit_seconds": x.get("fit_seconds")}
                for x in conf.get("phase_B_latency_71433", []) if not x.get("failed")],
            "engine_71433": _engine_rows(conf.get("phase_C_engine_71433"),
                                         conf.get("phase_C_engine_latency_interleaved")),
            "synthetic_214299": ({
                "documents": conf["phase_D_synthetic_214299"]["documents"],
                "exhaustive_p50_ms": conf["phase_D_synthetic_214299"]["interleaved"]["exhaustive"]["p50_ms"],
                "router_gather_p50_ms": conf["phase_D_synthetic_214299"]["interleaved"]["router_gather"]["p50_ms"],
                "router_contiguous_p50_ms": conf["phase_D_synthetic_214299"]["interleaved"]["router_contiguous"]["p50_ms"],
                "speedup": conf["phase_D_synthetic_214299"]["speedup_p50"],
                "exhaustive_recall_at_4_pct": conf["phase_D_synthetic_214299"]["exhaustive_recall_at_4_pct"],
                "router_recall_at_4_pct": conf["phase_D_synthetic_214299"]["router_recall_at_4_pct"],
                "router_delta_pt": conf["phase_D_synthetic_214299"]["router_delta_vs_exhaustive_pt"],
                "router_ci95_pt": conf["phase_D_synthetic_214299"]["router_ci95_pt"],
                "fit_seconds": conf["phase_D_synthetic_214299"]["fit_seconds"]}
                if conf.get("phase_D_synthetic_214299") else None),
            "below_n_exhaustive_unchanged": (
                conf.get("phase_E_below_n_exhaustive") or {}).get(
                    "identical_id_and_cosine_for_every_question"),
            "failures": conf.get("failures", [])}
    v["decision"] = decision(res)
    return v


def confirm_latency():
    """Re-run ONLY the load-sensitive phases, on a quiet machine, and merge.

    Why this exists. The full run above was taken with a 1-minute load average
    between 19.6 and 25.7 on a 12-core box. The RECALL half of the gate does not
    care -- recall@4 is deterministic arithmetic -- but the half that actually
    decides the default is latency, and a number measured under 2x
    oversubscription is not a number anyone should flip a default on.

    So phases B/C/D/E are run again, unchanged, at whatever the load is now, and
    BOTH readings are kept in the file. Phase A is NOT re-run: it is
    deterministic, and re-rolling it would only invite picking the friendlier of
    two identical answers. If the two readings disagree about which side is
    faster, that disagreement is the finding and the file says so.
    """
    import nanomem
    prep()
    res = json.load(open(OUT))
    cands = res["gate_passing_configs"][:4]
    arms = res["engine_arm_configs"]
    conf = {"why": ("phases B-D re-measured on a quiet machine; the original run was "
                    "taken at 1-min load average 19.6-25.7 on 12 cores, which is not a "
                    "fair floor for a latency verdict. Phase A (recall) is deterministic "
                    "and is NOT re-run."),
            "engine_version": nanomem.ENGINE_VERSION,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "load_avg_start": load_avg(),
            "phase_B_latency_71433": [], "phase_C_engine_71433": [],
            "phase_C_engine_latency_interleaved": None,
            "phase_D_synthetic_214299": None, "phase_E_below_n_exhaustive": None,
            "failures": []}

    print("\nCONFIRM PHASE B -- arena latency at 71,433 (quiet machine)", flush=True)
    for a in cands:
        r = run("latency", {"cell_target": a["cell_target"], "beam_frac": a["beam_frac"]},
                "latency ct=%d bf=%.2f" % (a["cell_target"], a["beam_frac"]))
        (conf["failures"] if r.get("failed") else conf["phase_B_latency_71433"]).append(r)

    print("\nCONFIRM PHASE C -- engine end to end at 71,433 (quiet machine)", flush=True)
    for cfg in arms:
        r = run("engine", cfg, "engine build+score %s" % cfg["arm"])
        (conf["failures"] if r.get("failed") else conf["phase_C_engine_71433"]).append(r)
    if len(conf["phase_C_engine_71433"]) == len(arms):
        conf["phase_C_engine_latency_interleaved"] = run(
            "engine_latency", {"arms": arms}, "engine latency (interleaved)")

    print("\nCONFIRM PHASE D -- synthetic 214,299 (quiet machine)", flush=True)
    best = cands[0]
    r = run("synthetic", {"extra_copies": 2, "cell_target": best["cell_target"],
                          "beam_frac": best["beam_frac"]}, "synthetic 214,299 (interleaved)")
    if r.get("failed"):
        conf["failures"].append(r)
    else:
        conf["phase_D_synthetic_214299"] = r

    print("\nCONFIRM PHASE E -- below n_exhaustive (quiet machine)", flush=True)
    r = run("below", {"docs": 10_000}, "below n_exhaustive (10,000 docs)")
    conf["phase_E_below_n_exhaustive"] = r
    if r.get("failed"):
        conf["failures"].append(r)
    conf["load_avg_end"] = load_avg()

    res["latency_confirmation_quiet_machine"] = conf
    res["verdict"] = verdict(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT, flush=True)
    print(json.dumps(res["verdict"]["decision"], indent=1), flush=True)


def replicate_engine_latency(n_reps=5):
    """Time the three engine arms against each other N separate times.

    Why. The first two readings of this measurement disagreed about its SIGN --
    the loaded run put the routed arm at 0.923x the exhaustive scan, the quiet
    run at 1.134x -- and the quiet run's p95 came back at 3-5x its own p50, which
    is what interference looks like. A default cannot be flipped on a number that
    moves 20% between two runs, in either direction. Each replicate re-opens all
    three vaults in a fresh process and interleaves their queries one at a time,
    so every replicate is an independent draw of the same comparison.
    """
    prep()
    res = json.load(open(OUT))
    arms = res["engine_arm_configs"]
    reps = []
    print("\nREPLICATE -- engine latency at 71,433, %d independent runs" % n_reps, flush=True)
    for i in range(n_reps):
        r = run("engine_latency", {"arms": arms}, "engine latency replicate %d" % (i + 1))
        if not r.get("failed"):
            reps.append(r)
    names = [a["arm"] for a in arms]
    per_arm = {a: [r["interleaved"][a]["p50_ms"] for r in reps if a in r["interleaved"]]
               for a in names}
    ratio = {a: [round(r["interleaved"]["off"]["p50_ms"] / r["interleaved"][a]["p50_ms"], 4)
                 for r in reps] for a in names}
    res["engine_latency_replicates"] = {
        "why": ("the two single readings of phase C disagreed on the sign of the router's "
                "speed-up (0.923x loaded, 1.134x quiet), so the comparison is repeated "
                "as independent runs and the spread is reported"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_replicates": len(reps),
        "load_avg_start": reps[0]["load_avg_start"] if reps else None,
        "load_avg_end": reps[-1]["load_avg_end"] if reps else None,
        "p50_ms_per_replicate": per_arm,
        "speedup_vs_off_per_replicate": ratio,
        "summary": {a: {"p50_median_ms": round(float(np.median(per_arm[a])), 4),
                        "p50_min_ms": round(float(np.min(per_arm[a])), 4),
                        "p50_max_ms": round(float(np.max(per_arm[a])), 4),
                        "speedup_median": round(float(np.median(ratio[a])), 4),
                        "speedup_min": round(float(np.min(ratio[a])), 4),
                        "speedup_max": round(float(np.max(ratio[a])), 4),
                        "faster_than_off_in_n_of_%d" % len(reps):
                            int(sum(1 for x in ratio[a] if x > 1.0))}
                    for a in names if per_arm[a]},
        "replicates": reps}
    # Persist the MEASUREMENTS first. `verdict` only summarises them, and a bug
    # in a summary must never be able to throw away the run that produced it --
    # which is exactly what happened the first time this was run.
    json.dump(res, open(OUT, "w"), indent=1)
    res["verdict"] = verdict(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT, flush=True)
    print(json.dumps(res["engine_latency_replicates"]["summary"], indent=1), flush=True)
    print(json.dumps(res["verdict"]["decision"], indent=1), flush=True)


def _engine_rows(C, il):
    """One row per engine arm from a (phase_C, interleaved-latency) pair."""
    if not C or not il:
        return []
    by = {x["arm"]: x for x in C}
    return [{"arm": k, "routing_mode": x["routing_mode"],
             "p50_ms": (il.get("interleaved", {}).get(k) or {}).get("p50_ms"),
             "p95_ms": (il.get("interleaved", {}).get(k) or {}).get("p95_ms"),
             "speedup_vs_off": (il.get("speedup_p50_vs_off") or {}).get(k),
             "ingest_seconds_total": x["ingest_seconds_total"],
             "reopen_seconds": x["reopen_seconds"],
             "peak_rss_delta_mb": x["peak_rss_delta_mb"],
             "recall_at_4_pct": x["recall_at_4_pct"]}
            for k, x in by.items()]


# The bar the router has to clear to become a DEFAULT, written down before the
# numbers were in and not moved afterwards. DECISIONS #3 states one condition
# (recall); the other two are the conditions any router has to meet to be worth
# turning on at all, and they are stated here so a reader can apply their own.
#
#   RECALL       CI lower bound of (router - exhaustive) recall@4 >= -1.0 pt.
#                This is the written DECISIONS #3 gate, verbatim.
#   REPRODUCIBLE The routed arm must be faster than the exact scan in EVERY
#                independent replicate. A default may not rest on a number whose
#                sign changes between runs; the engine's own docstring already
#                admits ~15% run-to-run drift on this machine.
#   AMORTISED    The router is not free at open: it re-fits its k-means every
#                time, because no centroid is persisted. So the per-query saving
#                has to pay that back inside one process. BREAK_EVEN_QUERIES is
#                how many queries a single open is allowed to need before the
#                router comes out ahead. 10,000 is a generous reading of a
#                long-lived server process; a CLI invocation issues one. The raw
#                break-even count is reported either way, so anyone who prefers
#                a different bar can apply it to the same number.
BREAK_EVEN_QUERIES = 10_000


def decision(res):
    """The DECISIONS #3 call, computed from the file rather than asserted."""
    A = res["phase_A_recall_gate"]
    passing = sorted([a for a in A if a["passes_gate"]],
                     key=lambda a: (a["scan_fraction"], a["cell_target"]))
    conf = res.get("latency_confirmation_quiet_machine") or {}
    quiet = bool(conf.get("phase_C_engine_latency_interleaved"))
    src = conf if quiet else res
    rows = {r["arm"]: r for r in _engine_rows(
        src.get("phase_C_engine_71433"),
        src.get("phase_C_engine_latency_interleaved"))}
    off = rows.get("off")
    routed = [r for k, r in rows.items() if k != "off"]

    d = {"gate_definition": {
            "recall": "95%% CI lower bound of (router - exhaustive) recall@4 >= %.1f pt"
                      % GATE_PT,
            "reproducible": "routed p50 < exhaustive p50 in EVERY replicate",
            "amortised": "break-even queries per open <= %d" % BREAK_EVEN_QUERIES},
         "recall_gate_passes": bool(passing),
         "recall_gate_evidence": (
             "%d of %d swept configs clear CI lower bound >= %.1f pt; cheapest is "
             "cell_target=%d beam_frac=%.2f at %.2f%% recall@4 vs %.2f%% exhaustive, "
             "delta %.2f pt CI %s"
             % (len(passing), len(A), GATE_PT, passing[0]["cell_target"],
                passing[0]["beam_frac"], passing[0]["recall_at_4_pct"],
                res["exhaustive_recall_at_4_pct"], passing[0]["delta_vs_exhaustive_pt"],
                passing[0]["ci95_pt"]) if passing else "no config passes"),
         "latency_source": ("quiet machine, load_avg_start %s" % (conf.get("load_avg_start"),)
                            if quiet else "original run, load average 19.6-25.7")}

    # -- reproducibility -----------------------------------------------------
    rep = res.get("engine_latency_replicates") or {}
    summ = rep.get("summary") or {}
    n_rep = int(rep.get("n_replicates") or 0)
    single = {}
    for label, blob in (("original_loaded_machine", res), ("quiet_machine", conf)):
        r = _engine_rows(blob.get("phase_C_engine_71433"),
                         blob.get("phase_C_engine_latency_interleaved"))
        if r:
            single[label] = {x["arm"]: {"p50_ms": x["p50_ms"],
                                        "speedup_vs_off": x["speedup_vs_off"]} for x in r}
    d["engine_p50_single_readings"] = single
    d["engine_p50_replicates"] = summ
    key = "faster_than_off_in_n_of_%d" % n_rep
    d["reproducible_gate_passes"] = (
        None if not summ else
        any(v.get(key) == n_rep and n_rep > 1 for a, v in summ.items() if a != "off"))
    d["reproducible_gate_evidence"] = (
        "no replicates run" if not summ else
        "; ".join("%s: p50 median %.4f ms (%.4f-%.4f), speed-up vs off median %.3fx "
                  "(%.3fx-%.3fx), faster in %d of %d replicates"
                  % (a, v["p50_median_ms"], v["p50_min_ms"], v["p50_max_ms"],
                     v["speedup_median"], v["speedup_min"], v["speedup_max"],
                     v.get(key, 0), n_rep)
                  for a, v in summ.items()))

    # -- amortisation --------------------------------------------------------
    amort = []
    for r in routed:
        med = (summ.get(r["arm"]) or {}).get("p50_median_ms", r["p50_ms"])
        med_off = (summ.get("off") or {}).get("p50_median_ms",
                                              off["p50_ms"] if off else None)
        if med_off is None:
            continue
        saving_ms = med_off - med
        extra_open_ms = (r["reopen_seconds"] - off["reopen_seconds"]) * 1000.0
        amort.append({
            "arm": r["arm"],
            "p50_median_ms": round(med, 4), "p50_median_off_ms": round(med_off, 4),
            "saving_per_query_ms": round(saving_ms, 4),
            "extra_open_seconds": round(extra_open_ms / 1000.0, 3),
            "extra_ingest_seconds": round(
                r["ingest_seconds_total"] - off["ingest_seconds_total"], 3),
            "extra_peak_rss_mb": round(r["peak_rss_delta_mb"] - off["peak_rss_delta_mb"], 1),
            "recall_delta_pt": round(r["recall_at_4_pct"] - off["recall_at_4_pct"], 4),
            "break_even_queries_per_open": (
                None if saving_ms <= 0 else int(round(extra_open_ms / saving_ms))),
            "never_breaks_even": bool(saving_ms <= 0)})
    d["amortisation_71433"] = amort
    ok = [a for a in amort if a["break_even_queries_per_open"] is not None
          and a["break_even_queries_per_open"] <= BREAK_EVEN_QUERIES]
    d["amortised_gate_passes"] = bool(ok)
    d["amortised_gate_evidence"] = "; ".join(
        ("%s never breaks even: it is %.4f ms SLOWER per query than the scan it replaces"
         % (a["arm"], -a["saving_per_query_ms"])) if a["never_breaks_even"] else
        ("%s saves %.4f ms per query and costs %.3f s extra at every open, so one open "
         "must serve %s queries before it is ahead (bar: %d); it also costs %.3f s more "
         "ingest, %.1f MB more peak RSS and %.2f pt of recall@4"
         % (a["arm"], a["saving_per_query_ms"], a["extra_open_seconds"],
            "{:,}".format(a["break_even_queries_per_open"]), BREAK_EVEN_QUERIES,
            a["extra_ingest_seconds"], a["extra_peak_rss_mb"], a["recall_delta_pt"]))
        for a in amort)

    gates = (d["recall_gate_passes"], d["reproducible_gate_passes"],
             d["amortised_gate_passes"])
    d["router_default_should_be"] = "auto" if all(g is True for g in gates) else "off"
    d["shipped_default"] = "off"
    d["gates_failed"] = [name for name, g in
                         (("recall", d["recall_gate_passes"]),
                          ("reproducible", d["reproducible_gate_passes"]),
                          ("amortised", d["amortised_gate_passes"])) if g is not True]
    d["one_sentence"] = (
        "DECISIONS #3's RECALL gate PASSES -- %d of %d swept budgets are within 1.0 pt "
        "of the exact scan and the cheapest is %.2f%% against %.2f%% -- but the router "
        "still ships OFF, because it fails the %s test%s."
        % (len(passing), len(A), passing[0]["recall_at_4_pct"],
           res["exhaustive_recall_at_4_pct"], " and ".join(d["gates_failed"]),
           "" if len(d["gates_failed"]) == 1 else "s")
        if d["router_default_should_be"] == "off" and d["recall_gate_passes"]
        else "All three gates pass; the router should default to \"auto\".")
    return d


def rerun_phase_a():
    """Re-run PHASE A ONLY, persisting per-question data, and merge it in.

    Why this exists. The gate that decides the default is a paired bootstrap CI,
    and the first version of this file stored only the summary statistic for
    each budget -- ``recall_at_4_pct``, ``delta_vs_exhaustive_pt``, ``ci95_pt``
    -- with no per-question data anywhere in ``phase_A_recall_gate``. Neither
    the recall nor the interval could be recomputed from the file: an auditor
    had to re-derive them from the cached embeddings, which is exactly the work
    the file is supposed to make unnecessary. Phase A is deterministic
    arithmetic, so re-running it costs a k-means fit per cell_target and
    changes nothing -- and the re-run's agreement with the stored numbers is
    itself recorded, under ``phase_A_rerun_check``, so "it reproduced" is a
    measurement in the file rather than a claim in a report.

    Only phase A, its ablation and its random-layout control are re-run. The
    latency phases are NOT touched: they are load-sensitive, the file already
    carries both a loaded and a quiet reading of them, and re-rolling a
    measurement whose answer you already know is how a sweep turns into
    shopping. The verdict is recomputed from the merged file at the end.
    """
    prep()
    res = json.load(open(OUT))
    before = {(a["cell_target"], a["beam_frac"]): a
              for a in res.get("phase_A_recall_gate", [])}

    new_arms, ex_string, ex_pct = [], None, None
    for ct in CELL_TARGETS:
        r = run("recall", {"cell_target": ct, "beam_fracs": BEAM_FRACS},
                "recall cell_target=%d (per-question persisted)" % ct)
        if r.get("failed"):
            res["failures"].append(r)
            continue
        ex_string = r["exhaustive_per_question_gold_hits"]
        ex_pct = r["exhaustive_recall_at_4_pct"]
        new_arms.extend(r["arms"])
    if not new_arms:
        print("phase A re-run produced nothing; file untouched", flush=True)
        return

    import nanomem
    check = {"what": "the re-run against the numbers already in the file. Phase A "
                     "is deterministic, so every delta here should be 0.0; a "
                     "non-zero row is a finding, not a rounding.",
             "rerun_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "rerun_engine_version": nanomem.ENGINE_VERSION,
             "rerun_package_version": getattr(nanomem, "__version__", "?"),
             "rerun_load_avg": load_avg(),
             "note": "phase A exercises routing.GlobalCellRouter and numpy only -- "
                     "no engine, no container -- so the engine version it ran "
                     "under cannot change its arithmetic. It is recorded because "
                     "the top-level engine_version belongs to the ORIGINAL full "
                     "run, not to this re-run.",
             "arms_compared": 0, "max_abs_recall_delta_pt": 0.0,
             "max_abs_ci_delta_pt": 0.0, "rows_that_moved": []}
    for a in new_arms:
        old = before.get((a["cell_target"], a["beam_frac"]))
        if old is None:
            continue
        check["arms_compared"] += 1
        dr = abs(a["recall_at_4_pct"] - old["recall_at_4_pct"])
        dci = max(abs(a["ci95_pt"][0] - old["ci95_pt"][0]),
                  abs(a["ci95_pt"][1] - old["ci95_pt"][1]))
        check["max_abs_recall_delta_pt"] = max(check["max_abs_recall_delta_pt"], dr)
        check["max_abs_ci_delta_pt"] = max(check["max_abs_ci_delta_pt"], dci)
        if dr > 1e-9 or dci > 1e-9 or a["passes_gate"] != old["passes_gate"]:
            check["rows_that_moved"].append(
                {"cell_target": a["cell_target"], "beam_frac": a["beam_frac"],
                 "recall_was": old["recall_at_4_pct"], "recall_now": a["recall_at_4_pct"],
                 "ci_was": old["ci95_pt"], "ci_now": a["ci95_pt"],
                 "passes_gate_was": old["passes_gate"], "passes_gate_now": a["passes_gate"]})
    if ex_pct is not None and "exhaustive_recall_at_4_pct" in res:
        check["exhaustive_was"] = res["exhaustive_recall_at_4_pct"]
        check["exhaustive_now"] = ex_pct
    res["phase_A_rerun_check"] = check

    res["phase_A_recall_gate"] = new_arms
    res["exhaustive_recall_at_4_pct"] = ex_pct
    res["exhaustive_per_question_gold_hits"] = ex_string

    r = run("recall", {"cell_target": 25, "beam_fracs": [0.10, 0.25], "layout": "random"},
            "recall cell_target=25 RANDOM layout")
    if not r.get("failed"):
        res["phase_A_random_layout_control"] = r
    r = run("ablation", {"landmarks": [8, 16], "beam_fracs": [0.10, 0.25]},
            "ablation PageLandmarkRouter")
    if not r.get("failed"):
        res["phase_A_ablation"] = r

    passing = [a for a in res["phase_A_recall_gate"] if a["passes_gate"]]
    passing.sort(key=lambda a: (a["scan_fraction"], a["cell_target"]))
    res["gate_passing_configs"] = [{k: a[k] for k in
                                    ("cell_target", "beam_frac", "recall_at_4_pct",
                                     "delta_vs_exhaustive_pt", "ci95_pt", "scan_fraction")}
                                   for a in passing]
    res["phase_A_self_audit"] = phase_a_self_audit(res)
    res["verdict"] = verdict(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT, flush=True)
    print(json.dumps({"rerun_check": check, "self_audit": res["phase_A_self_audit"]},
                     indent=1), flush=True)


def phase_a_self_audit(res):
    """Recompute phase A's headline numbers FROM THE FILE'S OWN per-question data.

    This is the point of persisting the strings: recall, the paired mean
    difference and the 95% bootstrap interval are all recomputed here from
    ``per_question_gold_hits`` alone, with the same resample count and seed, and
    compared against the stored summary. If the two ever disagree the file is
    lying about itself and this block says by how much.
    """
    ex_s = res.get("exhaustive_per_question_gold_hits")
    if not ex_s:
        return {"ran": False, "why": "no per-question data in the file"}
    ex = hits_from_string(ex_s)
    out = {"ran": True, "recomputed_from": "exhaustive_per_question_gold_hits + "
                                           "each arm's per_question_gold_hits",
           "bootstrap": "%d resamples, seed %d, paired over questions"
                        % (BOOT_ITERS, BOOT_SEED),
           "questions": int(ex.size),
           "exhaustive_recall_at_4_pct_recomputed": round(100 * float(ex.mean()), 4),
           "exhaustive_recall_at_4_pct_stored": res.get("exhaustive_recall_at_4_pct"),
           "arms_checked": 0, "max_abs_recall_delta_pt": 0.0,
           "max_abs_mean_delta_pt": 0.0, "max_abs_ci_delta_pt": 0.0,
           "disagreements": []}
    for a in res.get("phase_A_recall_gate", []):
        s = a.get("per_question_gold_hits")
        if not s:
            continue
        h = hits_from_string(s)
        m, lo, hi = paired_bootstrap((h - ex) * 100.0)
        rec = round(100 * float(h.mean()), 4)
        dr = abs(rec - a["recall_at_4_pct"])
        dm = abs(m - a["delta_vs_exhaustive_pt"])
        dci = max(abs(lo - a["ci95_pt"][0]), abs(hi - a["ci95_pt"][1]))
        out["arms_checked"] += 1
        out["max_abs_recall_delta_pt"] = max(out["max_abs_recall_delta_pt"], dr)
        out["max_abs_mean_delta_pt"] = max(out["max_abs_mean_delta_pt"], dm)
        out["max_abs_ci_delta_pt"] = max(out["max_abs_ci_delta_pt"], dci)
        if max(dr, dm, dci) > 1e-6:
            out["disagreements"].append(
                {"cell_target": a["cell_target"], "beam_frac": a["beam_frac"],
                 "recall_stored": a["recall_at_4_pct"], "recall_recomputed": rec,
                 "ci_stored": a["ci95_pt"], "ci_recomputed": [lo, hi]})
    out["file_is_self_consistent"] = (out["arms_checked"] > 0
                                      and not out["disagreements"])
    return out


if __name__ == "__main__":
    if "--worker" in sys.argv:
        spec = json.loads(sys.argv[sys.argv.index("--worker") + 1])
        print(json.dumps(WORKERS[spec["kind"]](spec["cfg"])))
    elif "--persist-phase-a" in sys.argv:
        rerun_phase_a()
    elif "--audit-phase-a" in sys.argv:
        _res = json.load(open(OUT))
        _res["phase_A_self_audit"] = phase_a_self_audit(_res)
        json.dump(_res, open(OUT, "w"), indent=1)
        print(json.dumps(_res["phase_A_self_audit"], indent=1))
    elif "--confirm-latency" in sys.argv:
        confirm_latency()
    elif "--replicate-engine-latency" in sys.argv:
        i = sys.argv.index("--replicate-engine-latency")
        n = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 5
        replicate_engine_latency(n)
    else:
        main()
