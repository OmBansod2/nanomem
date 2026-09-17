"""
nanomem.routing
~~~~~~~~~~~~~~~
Approximate-search routing primitives. Pure numpy, no nanomem imports, no I/O,
fully deterministic given a seed. Two routers live here:

``GlobalCellRouter`` (SHIPPED, opt-in)
    IVF-style global spherical k-means over the whole corpus. ``k`` is chosen as
    ``ceil(n / cell_target)``; a query scans the ``beam_frac`` fraction of cells
    whose centroid is closest to it. This is the router the engine may engage
    above ``n_exhaustive``, and only on a re-clustered layout (see ``recluster``).

``PageLandmarkRouter`` (ABLATION ONLY)
    Per-page (per-block) landmark table: ``L`` spherical k-means landmarks per
    page, block score = max (or power-mean) over that page's landmark
    similarities. Kept because it is measured in the routing sweep and reported
    as an ablation; it is never the default.

Measured facts that motivate the split. EVERY ROW BELOW IS THE **ABLATION**
(``PageLandmarkRouter``), not the shipped ``GlobalCellRouter``; source:
``scratch/refound/design/core_spec.md`` table at line 473, from the orchestrator's
own sweep over 71,433 HotpotQA paragraphs, 2,000 questions, exhaustive
recall@4 = 62.1:

    layout            router                 beam    recall@4
    insertion order   page-landmark max        25%      61.5
    random order      page-landmark max        15%      39.4
    random order      page-landmark max        25%      47.5
    k-means layout    page-landmark max        15%      61.0
    k-means layout    page-landmark max        25%      61.6
    (FAISS IVFFlat at a matched scan fraction: 58.1 @5%, 61.1 @15%, 61.8 @25%)

DECISIONS #3, SETTLED. The sweep the earlier rounds kept deferring has been run
end to end: ``scratch/refound/sweep_router_gate.py`` ->
``scratch/refound/router_gate_results.json``. 71,433 HotpotQA paragraphs,
inserted in a random order (``default_rng(0).permutation``) and then put through
the engine's own ``compact(recluster=True)``, 1,000 questions drawn with
``default_rng(11)``, 30 (cell_target x beam_frac) budgets, 95% CIs from a paired
bootstrap of 10,000 resamples over questions. Exhaustive recall@4 = 77.30.

    cell_target   beam   scan    recall@4   delta      95% CI         gate
    12            10%    13.2%     77.00    -0.30   [-0.55, -0.10]    pass
    12            25%    31.3%     77.25    -0.05   [-0.15,  0.00]    pass
    25            10%    12.0%     76.60    -0.70   [-1.10, -0.35]    FAIL
    25            15%    17.7%     77.00    -0.30   [-0.60, -0.05]    pass
    25            25%    28.9%     77.05    -0.25   [-0.50, -0.05]    pass   <- shipped budget
    50            15%    16.5%     76.85    -0.45   [-0.80, -0.15]    pass
    100           15%    15.9%     75.90    -1.40   [-2.00, -0.85]    FAIL
    250           10%    10.6%     74.40    -2.90   [-3.75, -2.10]    FAIL

THE RECALL GATE PASSES: 13 of the 30 budgets clear "CI lower bound >= -1.0 pt",
including the shipped ``cell_target=25``/``beam_frac=0.25``. The full 30-row
table is ``phase_A_recall_gate`` in the results file, and every row of it now
carries its PER-QUESTION result (``per_question_gold_hits``: one digit per
question, 0/1/2 of the 2 gold paragraphs), so both the recall and the bootstrap
interval above can be recomputed from the file instead of trusted -- the sweep
does exactly that in ``phase_A_self_audit`` and records the deviation, which is
0.0 on every arm. Re-running the phase against the stored numbers reproduced all
30 budgets to the digit (``phase_A_rerun_check``: max recall delta 0.0 pt, max CI
delta 0.0 pt, no row changed its pass/fail). Two controls from the same
run: the ``PageLandmarkRouter`` ABLATION also passes at L=16/25% (77.15, -0.15
CI [-0.35, 0.00]) and fails at L=8/10% (76.10, -1.20 CI [-1.75, -0.70]); and on
the RANDOM (un-reclustered) layout the GlobalCellRouter scores 77.10 against
77.05 on the reclustered one at the same budget -- confirming that for a router
that clusters VECTORS the re-cluster buys locality, not recall.

AND THE ROUTER STILL SHIPS OFF, because recall was the first of three
conditions and it fails the other two. Both are measured in the same file, at
71,433 documents. This box is shared, so every latency figure below was taken
with the arms INTERLEAVED query by query -- each round charges every arm one
query under the same conditions, which is what makes the RATIOS survive a load
average that ranged from 2.49 to 25.7 across the session. The absolute
milliseconds move with it (``router="off"`` reads 3.3358 ms in the single quiet
run and 2.0305 ms median over the replicates); the ratios do not.

  1. IT IS NOT FASTER. At the arena level (quiet machine, load 2.49) the path
     the engine actually uses -- route to a beam, then fancy-index gather those
     rows -- LOSES to one contiguous 71,433 x 768 matmul at every budget
     measured: 0.852x at ct=12/10%, 0.835x at ct=50/15%, 0.728x at ct=25/15%
     and 0.611x at ct=12/15% (``phase_B_latency_71433``). Inside the engine, over FIVE independent
     replicates that re-open all three vaults and interleave their queries one
     at a time (``engine_latency_replicates``, load average 21.96 -> 16.17):

         arm                             p50 median   vs exhaustive   faster in
         router="off" (exhaustive)         2.0305 ms      1.000x          --
         auto, ct=25 bf=0.25 (shipped)     3.6151 ms      0.562x        0 of 5
         auto, ct=12 bf=0.10 (cheapest)    2.1256 ms      0.958x        1 of 5

     One earlier single reading did put the cheapest budget at 1.134x, and
     another at 0.923x; five replicates put it at 0.958x median, ahead in one
     run out of five and by 0.5% in that one. It is not faster -- the 1.134x was
     noise, and it is the reason this was replicated instead of published.
  2. THERE IS THEREFORE NOTHING TO AMORTISE, and the fixed costs are large
     anyway. No centroid is persisted, so every open re-fits the k-means:
     0.193 s to open a ``router="off"`` vault, 13.982 s at the shipped budget,
     25.796 s at the cheapest gate budget (smaller cells, more of them). Ingest
     goes 1.281 s -> 16.414 s / 43.675 s, peak RSS 849.0 MB -> 1818.8 MB /
     1906.5 MB, and recall@4 gives up 0.6 pt at the engine level (76.1 against
     76.7, 500 questions). A router that saved time could pay those back over a
     long-lived process; this one has no saving to pay them back with.

The same shape holds at 214,299 SYNTHETIC documents (DECISIONS #16 distractor
recipe, ``phase_D``): recall is fine (77.25 against 77.30, -0.05 CI
[-0.15, 0.00]) and the gather is 0.268x the exact scan -- 15.32 ms against
4.11 ms. The router gets RELATIVELY worse as the corpus grows, not better,
because the beam it gathers grows with it while the matmul it replaces stays
perfectly sequential. So "constant-factor win" is not a claim this module can
make in the shape the engine uses it; the honest statement is that the exact
scan is linear and hard to beat with a gather.

THE ONE-TIME RE-CLUSTER IS NOT THE PROBLEM. The permutation itself is 5.031 s
of numpy at 71,433 documents (``protocol.prep.recluster_seconds_numpy_only``),
paid once per file and recorded in the container header. The recurring k-means
fit above is the cost that matters, and it is paid on every open.

WHAT WOULD CHANGE THE ANSWER, with the number that says so. The one arm that
DOES beat the exact scan is the contiguous one -- rows stored in cell order, so
a beam is a set of slices instead of a gather: 1.811x at ct=50/beam 15%
(0.827 ms against 1.498 ms), and that budget passes the recall gate. Reaching it
needs two things nanomem 3.0.4 does not have: centroids persisted in the
container header (so an open does not re-fit), and the arena stored in the
router's own cell order (so ``GlobalCellRouter.adopt_layout`` applies). Until
both exist, ``router="auto"`` is an option that is exact enough and costs more
than it saves, and the default is exhaustive.

BELOW ``n_exhaustive`` NOTHING MOVES. At 10,000 documents, ``router="auto"`` and
``router="off"`` returned identical ids and identical cosines for all 200
questions in both runs of ``phase_E_below_n_exhaustive``; the engine reports
``routing_mode="exhaustive"`` for both.

Neither router is sub-linear; both are constant-factor reductions of an exact
linear scan, and on this machine the reduction is smaller than one.

Everything below takes and returns plain numpy arrays so that an independent
implementation can be compared against it element for element.
"""

import math

import numpy as np

# Defaults the engine reads. `n_exhaustive` is the corpus size below which the
# engine does an exact fp32 scan and no router runs at all. These must equal the
# VaultEngine keyword defaults; tests/test_routing.py checks that against the
# live signature, because 3.0.0 advertised landmarks_per_block=8 here while the
# engine shipped 0 and nothing caught it.
DEFAULTS = {
    "n_exhaustive": 50_000,
    "cell_target": 25,
    "beam_frac": 0.25,
    "beam_min_cells": 4,
    "block_capacity": 50,
    "landmarks_per_block": 0,      # per-block landmark tables are OFF by default:
                                   # only the PageLandmarkRouter ablation reads
                                   # them. Set 8 to reproduce that ablation.
    "pool": "max",
    "kmeans_iters": 10,
    "seed": 0,
}

_EPS = 1e-9

# Rows per block in the k-means assignment step. 8,192 x 2,858 float32 = 94 MB,
# against 816 MB for the whole (n, k) matrix at the 71,433-document corpus.
_ASSIGN_CHUNK = 8192


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def unit_rows(V: np.ndarray, copy: bool = True) -> np.ndarray:
    """Row-normalise ``V`` (n, D) to unit length, as float32."""
    A = np.asarray(V, dtype=np.float32)
    if copy or A.base is not None:
        A = np.array(A, dtype=np.float32, copy=True)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    np.maximum(n, _EPS, out=n)
    return A / n


def kmeans_plusplus_init(V: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding on cosine distance ``d = 1 - max_j cos(v, c_j)``.

    Returns the ``k`` row indices chosen as initial centres. ``V`` must have unit
    rows. Deterministic for a given ``rng`` state.
    """
    n = V.shape[0]
    k = int(min(k, n))
    idx = [int(rng.integers(n))]
    if k == 1:
        return np.asarray(idx, dtype=np.int64)
    best = V @ V[idx[0]]
    for _ in range(k - 1):
        d = np.clip(1.0 - best, _EPS, None).astype(np.float64)
        total = d.sum()
        if not np.isfinite(total) or total <= 0:
            remaining = np.setdiff1d(np.arange(n), np.asarray(idx))
            if remaining.size == 0:
                break
            nxt = int(rng.choice(remaining))
        else:
            nxt = int(rng.choice(n, p=d / total))
        idx.append(nxt)
        np.maximum(best, V @ V[nxt], out=best)
    return np.asarray(idx, dtype=np.int64)


def spherical_kmeans(V: np.ndarray, k: int, *, seed: int = 0, iters: int = 10):
    """Spherical k-means (cosine similarity, unit centroids).

    Parameters
    ----------
    V     : (n, D) array; rows are normalised internally.
    k     : requested number of centroids; capped at ``n``.
    seed  : seeds ``np.random.default_rng`` for k-means++ init.
    iters : Lloyd iterations.

    Returns
    -------
    (centroids, assign) with ``centroids`` (k, D) float32 unit rows and
    ``assign`` (n,) int32 giving each row's nearest centroid. An empty cluster is
    re-seeded with the row currently worst covered by any centroid.
    """
    A = unit_rows(V)
    n = A.shape[0]
    k = int(max(1, min(int(k), n)))
    rng = np.random.default_rng(int(seed))
    M = A[kmeans_plusplus_init(A, k, rng)].copy()
    assign = np.zeros(n, dtype=np.int32)
    for _ in range(int(iters)):
        assign, best = _nearest_centroid(A, M)
        sums, counts = _cluster_sums(A, assign, k)
        for j in range(k):
            if counts[j]:
                M[j] = sums[j]
            else:
                w = int(np.argmin(best))
                M[j] = A[w]
                best[w] = np.inf
        nrm = np.linalg.norm(M, axis=1, keepdims=True)
        np.maximum(nrm, _EPS, out=nrm)
        M /= nrm
    assign, _ = _nearest_centroid(A, M)
    return M.astype(np.float32, copy=False), assign


def spherical_kmeans_landmarks(V: np.ndarray, L: int, seed: int = 0, iters: int = 10) -> np.ndarray:
    """``min(L, n)`` unit landmark vectors summarising one page of vectors.

    When ``n <= L`` the vectors themselves are the landmarks (the caller then
    stores ``m = 0`` and reuses the page's own vectors).
    """
    A = unit_rows(V)
    if A.shape[0] <= int(L):
        return A
    M, _ = spherical_kmeans(A, int(L), seed=int(seed), iters=int(iters))
    return M


def pool(S: np.ndarray, group_start: np.ndarray, p="max") -> np.ndarray:
    """Pool per-landmark similarities ``S`` into one score per group.

    ``group_start`` is a length ``n_groups + 1`` prefix array (like
    ``np.add.reduceat`` boundaries). ``p="max"`` takes the maximum; a finite
    ``p`` takes the power mean ``(mean(relu(s)^p))^(1/p)``, which is the knob the
    sweep varies. Groups with no landmarks score ``-inf``.
    """
    S = np.asarray(S, dtype=np.float32)
    gs = np.asarray(group_start, dtype=np.int64)
    n_groups = gs.size - 1
    out = np.full(n_groups, -np.inf, dtype=np.float32)
    sizes = np.diff(gs)
    nonempty = sizes > 0
    if not nonempty.any():
        return out
    starts = gs[:-1][nonempty]
    if p == "max":
        vals = np.maximum.reduceat(S, starts)
    else:
        pf = float(p)
        R = np.clip(S, 0.0, None) ** pf
        sums = np.add.reduceat(R, starts)
        vals = (sums / sizes[nonempty]) ** (1.0 / pf)
    out[nonempty] = vals.astype(np.float32, copy=False)
    return out


def _nearest_centroid(A: np.ndarray, M: np.ndarray, chunk: int = _ASSIGN_CHUNK):
    """``(assign, best)`` for every row of ``A`` against centroids ``M``.

    Row-blocked so the ``(n, k)`` similarity matrix is never materialised. That
    matrix is 816 MB at n = 71,433 / k = 2,858 and was allocated ten times per
    fit; it is the single largest term in the 3,363 MB peak RSS the standard
    competitor run recorded for the routed arm
    (``scratch/refound/competitors_standard_results.json``).
    """
    n = A.shape[0]
    assign = np.empty(n, dtype=np.int32)
    best = np.empty(n, dtype=np.float32)
    Mt = np.ascontiguousarray(M.T)
    step = int(max(1, chunk))
    for i in range(0, n, step):
        S = A[i:i + step] @ Mt
        a = np.argmax(S, axis=1)
        assign[i:i + step] = a
        best[i:i + step] = S[np.arange(a.size), a]
    return assign, best


def _cluster_sums(A: np.ndarray, assign: np.ndarray, k: int):
    """Per-cluster vector sums and row counts.

    One stable sort plus ``np.add.reduceat`` instead of ``k`` boolean masks over
    the whole corpus. The mask loop is O(n*k) -- 204 million comparisons per
    Lloyd iteration at n = 71,433, k = 2,858 -- and dominated the fit.

    Summation order inside a cluster is unchanged (ascending row index), but
    ``reduceat`` accumulates sequentially where ``ndarray.sum`` pairwise-reduces,
    so centroids can differ in the last float32 ulp from 3.0.3's. Cluster
    membership is not affected at any scale measured
    (``scratch/refound/router_gate_results.json``, ``kmeans_parity``).
    """
    counts = np.bincount(assign, minlength=k).astype(np.int64)
    sums = np.zeros((k, A.shape[1]), dtype=np.float32)
    nonempty = counts > 0
    if nonempty.any():
        order = np.argsort(assign, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
        sums[nonempty] = np.add.reduceat(A[order], starts[nonempty], axis=0)
    return sums, counts


def _expand_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenation of ``range(s, s + l)`` for every ``(s, l)``, without a loop.

    The Python-level ``for cell in beam`` concatenation it replaces ran 715 times
    per query at the shipped ``beam_frac`` and 71,433 documents.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    keep = lengths > 0
    starts = np.asarray(starts, dtype=np.int64)[keep]
    lengths = lengths[keep]
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.cumsum(lengths)
    out = np.ones(total, dtype=np.int64)
    out[0] = starts[0]
    if starts.size > 1:
        out[ends[:-1]] = starts[1:] - starts[:-1] - lengths[:-1] + 1
    return np.cumsum(out)


def beam_size(n_groups: int, beam_frac: float, beam_min: int) -> int:
    """How many groups a query scans: ``clip(ceil(frac * n), min, n)``."""
    if n_groups <= 0:
        return 0
    return int(min(n_groups, max(int(beam_min), math.ceil(float(beam_frac) * n_groups))))


# --------------------------------------------------------------------------
# routers
# --------------------------------------------------------------------------
class GlobalCellRouter:
    """IVF-style router over global spherical k-means cells (the shipped one).

    ``fit`` clusters the corpus into ``ceil(n / cell_target)`` cells. ``route``
    returns the row indices inside the ``beam_frac`` closest cells, which the
    caller then scores exactly.
    """

    def __init__(self, cell_target: int = DEFAULTS["cell_target"],
                 beam_frac: float = DEFAULTS["beam_frac"],
                 beam_min_cells: int = DEFAULTS["beam_min_cells"],
                 seed: int = DEFAULTS["seed"], iters: int = DEFAULTS["kmeans_iters"]):
        self.cell_target = int(cell_target)
        self.beam_frac = float(beam_frac)
        self.beam_min_cells = int(beam_min_cells)
        self.seed = int(seed)
        self.iters = int(iters)
        self.centroids = np.zeros((0, 0), dtype=np.float32)
        self.cell_start = np.zeros(1, dtype=np.int64)
        self.cell_rows = np.zeros(0, dtype=np.int64)
        self.n_rows = 0
        self.contiguous = True

    @property
    def n_cells(self) -> int:
        return int(self.centroids.shape[0])

    def fit(self, V: np.ndarray) -> "GlobalCellRouter":
        """Cluster ``V`` (n, D) and build the inverted lists. Returns self."""
        A = unit_rows(V)
        n = A.shape[0]
        self.n_rows = n
        if n == 0:
            self.centroids = np.zeros((0, A.shape[1] if A.ndim == 2 else 0), dtype=np.float32)
            self.cell_start = np.zeros(1, dtype=np.int64)
            self.cell_rows = np.zeros(0, dtype=np.int64)
            return self
        k = max(1, math.ceil(n / max(1, self.cell_target)))
        self.centroids, assign = spherical_kmeans(A, k, seed=self.seed, iters=self.iters)
        k = self.centroids.shape[0]
        order = np.argsort(assign, kind="stable")
        self.cell_rows = order.astype(np.int64, copy=False)
        counts = np.bincount(assign, minlength=k).astype(np.int64)
        self.cell_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        # True when the caller's rows already sit in cell order, so every cell is
        # a contiguous slice and `scan` can skip the gather entirely.
        self.contiguous = bool(np.all(np.diff(assign.astype(np.int64)) >= 0))
        return self

    @classmethod
    def adopt_layout(cls, centroids: np.ndarray, cell_start: np.ndarray, **kw
                     ) -> "GlobalCellRouter":
        """Router over a corpus whose rows are ALREADY STORED in cell order.

        Re-fitting is not a way to get there: ``fit`` on a cell-ordered copy runs
        k-means++ over the new row order and finds a different partition, so its
        ``assign`` is not sorted and :attr:`contiguous` comes back false. The
        cells have to be carried over, which is what this does -- cell ``c`` owns
        rows ``[cell_start[c], cell_start[c + 1])`` and a beam is a set of slices
        rather than a fancy-index gather.

        The layout it needs is not one nanomem 3.0.4 can persist: the container
        header has no room for centroids, so an open would have to re-fit the
        k-means to recover them (11.1 s at 71,433 documents) and re-write the
        whole file to restore the order. It is here because the sweep measures
        the slice path against the gather path and both must come from the
        shipped module, not from the benchmark
        (``scratch/refound/router_gate_results.json``, phases B and D).
        """
        r = cls(**kw)
        r.centroids = np.asarray(centroids, dtype=np.float32)
        r.cell_start = np.asarray(cell_start, dtype=np.int64)
        r.n_rows = int(r.cell_start[-1]) if r.cell_start.size else 0
        r.cell_rows = np.arange(r.n_rows, dtype=np.int64)
        r.contiguous = True
        return r

    def cell_scores(self, q: np.ndarray) -> np.ndarray:
        """Cosine of the query against every cell centroid (exposed for sweeps)."""
        if self.n_cells == 0:
            return np.zeros(0, dtype=np.float32)
        return (self.centroids @ np.asarray(q, dtype=np.float32).ravel()).astype(np.float32)

    def route_cells(self, q: np.ndarray, beam_frac=None) -> np.ndarray:
        """Indices of the cells inside the beam, best first."""
        k = self.n_cells
        if k == 0:
            return np.zeros(0, dtype=np.int64)
        frac = self.beam_frac if beam_frac is None else float(beam_frac)
        b = beam_size(k, frac, self.beam_min_cells)
        S = self.cell_scores(q)
        if b >= k:
            return np.argsort(-S).astype(np.int64)
        top = np.argpartition(-S, b - 1)[:b]
        return top[np.argsort(-S[top])].astype(np.int64)

    def route(self, q: np.ndarray, beam_frac=None) -> np.ndarray:
        """Candidate row indices for a query (sorted ascending)."""
        cells = self.route_cells(q, beam_frac=beam_frac)
        if cells.size == 0:
            return np.zeros(0, dtype=np.int64)
        if cells.size >= self.n_cells:
            return np.arange(self.n_rows, dtype=np.int64)
        starts = self.cell_start[cells]
        rows = self.cell_rows[_expand_ranges(starts, self.cell_start[cells + 1] - starts)]
        rows.sort()
        return rows

    def route_ranges(self, q: np.ndarray, beam_frac=None) -> np.ndarray:
        """Beam as ``(m, 2)`` ``[start, stop)`` row ranges, ascending by start.

        Only meaningful when :attr:`contiguous` is true, i.e. when the caller's
        rows are already stored in cell order so that a cell IS a slice. Scanning
        slices instead of a fancy-index gather is the whole latency argument for
        the router: measured on the 71,433-document corpus at ``beam_frac`` 0.10,
        0.679 ms of contiguous slices against 1.383 ms of gather against 1.770 ms
        for the full exact scan (``scratch/refound/router_gate_results.json``).
        """
        cells = self.route_cells(q, beam_frac=beam_frac)
        if cells.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        cells = np.sort(cells)
        return np.stack([self.cell_start[cells], self.cell_start[cells + 1]], axis=1)

    def scan(self, V: np.ndarray, q: np.ndarray, beam_frac=None):
        """``(rows, cosines)`` for the beam, by the cheaper of the two paths.

        ``V`` must be the same matrix ``fit`` saw, in the same row order.
        """
        if not self.contiguous:
            rows = self.route(q, beam_frac=beam_frac)
            return rows, np.asarray(V[rows] @ q, dtype=np.float32)
        rng = self.route_ranges(q, beam_frac=beam_frac)
        if rng.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        parts = [V[int(a):int(b)] @ q for a, b in rng if b > a]
        if not parts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        cos = np.concatenate(parts).astype(np.float32, copy=False)
        rows = _expand_ranges(rng[:, 0], rng[:, 1] - rng[:, 0])
        return rows, cos

    def scanned_fraction(self, q: np.ndarray, beam_frac=None) -> float:
        """Fraction of the corpus a query would scan (the sweep's budget metric)."""
        if self.n_rows == 0:
            return 0.0
        return float(self.route(q, beam_frac=beam_frac).size) / float(self.n_rows)


class PageLandmarkRouter:
    """Per-page landmark router -- ABLATION ONLY, never the shipped default.

    Each page (block) contributes ``m <= L`` landmarks. A page's score is the
    pooled similarity of the query to its landmarks; the beam is the top
    ``beam_frac`` of pages. Measured deficit versus ``GlobalCellRouter`` on a
    random-order 71k corpus at a 15% budget: 39.4 vs ~61 recall@4.
    """

    def __init__(self, landmarks: np.ndarray, land_start: np.ndarray, *,
                 p=DEFAULTS["pool"], beam_frac: float = DEFAULTS["beam_frac"],
                 beam_min_blocks: int = DEFAULTS["beam_min_cells"]):
        self.landmarks = np.asarray(landmarks, dtype=np.float32)
        self.land_start = np.asarray(land_start, dtype=np.int64)
        self.p = p
        self.beam_frac = float(beam_frac)
        self.beam_min_blocks = int(beam_min_blocks)

    @classmethod
    def from_pages(cls, pages, L: int = DEFAULTS["landmarks_per_block"], *,
                   p=DEFAULTS["pool"], beam_frac: float = DEFAULTS["beam_frac"],
                   beam_min_blocks: int = DEFAULTS["beam_min_cells"],
                   seed: int = DEFAULTS["seed"]):
        """Build from an iterable of (n_i, D) page matrices."""
        mats, starts, acc = [], [0], 0
        for i, P in enumerate(pages):
            M = spherical_kmeans_landmarks(P, L, seed=seed + i)
            mats.append(M)
            acc += M.shape[0]
            starts.append(acc)
        lm = np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32)
        return cls(lm, np.asarray(starts, dtype=np.int64), p=p,
                   beam_frac=beam_frac, beam_min_blocks=beam_min_blocks)

    @property
    def n_blocks(self) -> int:
        return int(self.land_start.size - 1)

    def block_scores(self, q: np.ndarray) -> np.ndarray:
        """Pooled per-page score (exposed so the sweep can plot it)."""
        if self.n_blocks <= 0 or self.landmarks.size == 0:
            return np.full(max(self.n_blocks, 0), -np.inf, dtype=np.float32)
        S = self.landmarks @ np.asarray(q, dtype=np.float32).ravel()
        return pool(S, self.land_start, self.p)

    def route(self, q: np.ndarray, beam_frac=None) -> np.ndarray:
        """Indices of the pages inside the beam, best first."""
        nb = self.n_blocks
        if nb <= 0:
            return np.zeros(0, dtype=np.int64)
        frac = self.beam_frac if beam_frac is None else float(beam_frac)
        b = beam_size(nb, frac, self.beam_min_blocks)
        B = self.block_scores(q)
        if b >= nb:
            return np.argsort(-B).astype(np.int64)
        top = np.argpartition(-B, b - 1)[:b]
        return top[np.argsort(-B[top])].astype(np.int64)


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
def recluster(vectors: np.ndarray, *, cell_target: int = DEFAULTS["block_capacity"],
              seed: int = DEFAULTS["seed"], iters: int = DEFAULTS["kmeans_iters"]) -> np.ndarray:
    """Row permutation that groups semantically close vectors together.

    Runs a global spherical k-means with ``k = ceil(n / cell_target)`` and orders
    rows by ``(cluster, original row)``. Applying this permutation before writing
    blocks is what makes a randomly ingested corpus routable at all (measured:
    random layout 39.4 -> reclustered 61.0 recall@4 at a 15% beam on 71k docs).

    Returns ``perm`` with ``perm[i]`` = the original row that becomes row ``i``.
    """
    A = unit_rows(vectors)
    n = A.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    k = max(1, math.ceil(n / max(1, int(cell_target))))
    _, assign = spherical_kmeans(A, k, seed=seed, iters=iters)
    return np.lexsort((np.arange(n), assign)).astype(np.int64)


def nn_coverage(V: np.ndarray, router, *, sample: int = 256, seed: int = 0,
                row_group=None) -> float:
    """Fraction of exact nearest neighbours a router's beam would still reach.

    This is the layout-quality metric: sample rows, find each one's exact nearest
    *other* row by brute force, and check whether that row is inside the beam the
    router returns for it. ``row_group`` maps a row to its page index (needed for
    :class:`PageLandmarkRouter`); leave it ``None`` for
    :class:`GlobalCellRouter`, whose ``route`` already returns rows.
    """
    A = unit_rows(V)
    n = A.shape[0]
    if n < 2:
        return 1.0
    rng = np.random.default_rng(int(seed))
    rows = rng.choice(n, size=int(min(sample, n)), replace=False)
    hit = 0
    for r in rows:
        s = A @ A[r]
        s[r] = -np.inf
        nn = int(np.argmax(s))
        reached = router.route(A[r])
        if row_group is None:
            hit += int(np.isin(nn, reached))
        else:
            hit += int(np.isin(int(row_group[nn]), reached))
    return float(hit) / float(len(rows))


__all__ = [
    "DEFAULTS", "unit_rows", "kmeans_plusplus_init", "spherical_kmeans",
    "spherical_kmeans_landmarks", "pool", "beam_size", "GlobalCellRouter",
    "PageLandmarkRouter", "recluster", "nn_coverage",
]
