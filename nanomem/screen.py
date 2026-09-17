"""
nanomem.screen
~~~~~~~~~~~~~~
An **exactness-preserving** candidate screen for the search path. Pure numpy, no
nanomem imports, no I/O, deterministic.

WHAT IT IS. A cheap, *provably admissible* upper bound on every document's
cosine, computed in a ``d_out``-dimensional subspace instead of the full
``embed_dim``. Documents whose upper bound falls below the score the
``top_k``-th best document already has are **provably** outside the top-k and
are never touched again; the survivors are re-scored by the arena's own exact
kernel. The answer is the exhaustive answer -- same ids, same float32 scores --
and the only thing that changes is how many rows the exact kernel has to read.

It is NOT an approximate-nearest-neighbour index. It has no recall knob, no
beam, and no accuracy/speed trade-off: turning it on cannot change a result.
That is also why it was never a "win" under the recall-delta rule
``scratch/refound/exotic_routing_results.json`` pre-registered -- an
exactness-preserving mechanism has a recall delta of identically 0, so a bar
phrased as "CI lower bound > 0 on a recall delta" is unreachable by
construction rather than merely unmet. The bar this mechanism is measured
against instead (exact agreement + a measured p50 speedup + no small-corpus
regression) was pre-registered before the module was written, in
``scratch/refound/pca_screen_results.json``.

THE BOUND, AND WHY IT IS ADMISSIBLE FOR *ANY* MATRIX ``B``
----------------------------------------------------------
Let ``B`` be any ``(d, m)`` real matrix and let ``y`` be **whatever fp32 vector
this module actually stored** for document ``x`` -- not the exact projection,
the stored one, rounding and all. Define ``r = x - B y`` and, for a query ``q``,
``pq = B.T q`` and ``q~ = q - B pq``. Then, as pure algebra with no assumption
about ``B`` and no assumption that ``y`` is a good projection of anything::

    q . x  =  pq . y  +  q . r                                     (definition of r)
    q . r  =  (B pq) . r  +  q~ . r  =  pq . (B.T r)  +  q~ . r    (definition of q~)

Cauchy-Schwarz on the last two terms gives::

    q . x  <=  pq . y  +  ||q~|| * ||r||  +  ||pq|| * ||B.T r||

Every piece is either a per-query scalar (``pq``, ``||q~||``, ``||pq||``) or a
per-document scalar stored once (``||r||`` and a bound on ``||B.T r||``), so the
whole bound is one ``(n, m)`` matvec plus two fused multiply-adds over ``n``.

The two per-document quantities are bounded, not estimated, from things that
are cheap to compute. With ``G = B.T B`` and ``t = B.T x`` the exact projection,
``y = t + dy`` for some rounding error ``dy`` with
``||dy|| <= gamma_d * sqrt(m) * ||x||`` (the textbook fp32 dot-product bound,
``gamma_d = d*u/(1-d*u)``, ``u = 2^-24``)::

    ||r||^2   = ||x||^2 - 2 t.y + y.G y  <=  ||x||^2 - 2||y||^2 + y.G y
                                             + 2 ||dy|| ||y||
    ||B.T r|| = ||t - G y|| = ||(I-G) y + dy||  <=  ||I-G||_2 ||y|| + ||dy||

so the stored ``res`` is that first right-hand side, rounded UP, and the
``||B.T r||`` term is folded into a per-query coefficient on the stored
``||y||`` plus one absolute constant. ``y.G y`` costs one ``(n, m) x (m, m)``
product at build time and is what makes the bound stay correct when ``m``
approaches ``d`` and the residual shrinks to nothing -- the case a
``||x||^2 - ||y||^2`` residual gets wrong (measured: it under-bounds by
3.4e-05 at ``d=128, m=127``, which the width sweep in ``tests/test_screen.py``
catches).

Three consequences that matter to a user:

* **A stale basis is slower, never wrong.** ``B`` fitted on the first 10,000
  rows still bounds row 10,001 correctly. Appends therefore cannot invalidate
  correctness -- they can only make the bound looser. New rows are projected
  with the *existing* ``B`` and appended; nothing is refitted.
* **Orthonormality is an optimisation, not a premise.** When ``B`` has
  orthonormal columns ``G = I``, ``||I-G||_2`` is 0 and the third term
  disappears. nanomem fits ``B`` by an eigendecomposition, so ``G`` differs from
  ``I`` by fp32 rounding and the term is worth about 1e-6 of slack -- but it is
  *computed*, so the guarantee does not rest on an assumption the arithmetic
  does not actually honour. ``tests/test_screen.py`` feeds the screen bases that
  are deliberately not orthonormal (scaled, rank-1, all-ones, all-zero) and
  demands admissibility anyway, and a companion test removes this term and
  asserts the bound then IS violated, so the first test cannot pass for the
  wrong reason.
* **The float arithmetic is inside the envelope, by term and not by fudge.**
  ``pq`` is computed in float64 and rounded once, so its error is a
  representation error of ``u*||pq||`` rather than an accumulation; ``||q~||``
  is computed directly as ``||q - B pq||`` in float64 rather than as
  ``sqrt(||q||^2 - ||pq||^2)``, which would assume orthonormality; the stored
  norms are rounded up; and the fp32 evaluation of ``pq . y`` is charged
  ``gamma_m * ||pq|| * ||y||``. Every approximation here widens the admitted
  set. None of them can shrink it, which is the only direction that could
  produce a wrong answer.

WHAT IT COSTS. ``m * 4 + 8`` bytes per document resident (``1032`` B/doc at the
default ``m = 256``, against the fp32 arena's ``embed_dim * 4`` = 3072 B/doc),
plus ``embed_dim * m * 4`` bytes for the basis itself. Measured totals and the
measured speedup are in :class:`nanomem.engine.VaultEngine`'s ``screen``
argument docstring and in ``scratch/refound/pca_screen_results.json``.
"""

from typing import Iterable, Optional

import numpy as np

#: Default width of the screening subspace. 256 of 768 was the measured knee:
#: at 128 the bound is loose enough that ~15% of the corpus survives and the
#: screen is a net LOSS (0.69x), at 256 ~0.34% survives and it is a net win.
#: Both rows are in ``exotic_routing_results.json :: phaseC_latency.arms``.
DEFAULT_D_OUT = 256

#: Rows exact-scored up front to establish the pruning threshold. Only has to be
#: big enough that the top-k by BOUND contains the top-k by cosine most of the
#: time; when it does not, the threshold is merely lower and more rows survive.
DEFAULT_SEED_POOL = 128

#: Cap on the rows the basis is fitted from. The bound is admissible for any
#: basis, so a sample costs selectivity and nothing else.
DEFAULT_FIT_SAMPLE = 200_000

#: Rows per chunk while accumulating the fit's second-moment matrix.
DEFAULT_FIT_CHUNK = 8192

#: fp32 unit roundoff, 2^-24. Every error term below is written in terms of it.
U32 = 2.0 ** -24

#: Subtracted from the pruning threshold. The survivors are exact-scored by the
#: same arena kernel the full scan uses, and the engine CALIBRATES the gather so
#: that is bitwise true (see ``VaultEngine._calibrate_gather``) -- this is belt
#: and braces against a BLAS whose blocking depends on the number of rows. Same
#: magnitude as the arena's own ``1e-6`` rerank separation.
TAU_EPS = 1e-6

#: Multiplied into the stored norms so they can only be over-estimated. An
#: over-estimated residual norm widens the bound; an under-estimated one would
#: narrow it, and that is the one direction that could lose a document.
_NORM_UP = np.float32(1.0 + 1e-5)
_NORM_FLOOR = np.float32(1e-7)


def _gamma(n: int) -> float:
    """``n*u / (1 - n*u)``: the textbook fp32 error factor for a length-``n``
    dot product. Used as written, i.e. as the WORST case -- blocked and pairwise
    summation do far better, and paying for the worst case costs a little
    selectivity and buys a bound that does not depend on which BLAS is loaded."""
    nu = float(n) * U32
    return nu / (1.0 - nu) if nu < 0.5 else 1.0


class PCAScreen:
    """Projection + Cauchy-Schwarz upper bound over a stored subspace basis.

    Build it with :meth:`fit` (one pass over the corpus to get ``B``) then
    :meth:`extend` (one matmul per batch of rows to get their projections).
    :meth:`bounds` then answers "what is the largest cosine each row could
    possibly have against this query?" for the whole corpus in one ``(n, m)``
    matvec.

    Every array this object owns is plain resident numpy. It never opens a file
    and never imports anything from ``nanomem``.
    """

    __slots__ = ("embed_dim", "d_out", "seed_pool", "fit_sample", "fit_chunk",
                 "basis", "_G", "_nIG", "_Y", "_res", "_yn", "n_rows", "_cap",
                 "fit_rows", "fit_seconds", "captured_energy", "_xmax",
                 "_dy", "_gam_m")

    def __init__(self, embed_dim: int, d_out: int = DEFAULT_D_OUT,
                 seed_pool: int = DEFAULT_SEED_POOL,
                 fit_sample: int = DEFAULT_FIT_SAMPLE,
                 fit_chunk: int = DEFAULT_FIT_CHUNK) -> None:
        self.embed_dim = int(embed_dim)
        self.d_out = max(1, min(int(d_out), self.embed_dim))
        self.seed_pool = max(1, int(seed_pool))
        self.fit_sample = max(self.d_out, int(fit_sample))
        self.fit_chunk = max(64, int(fit_chunk))
        self.basis: Optional[np.ndarray] = None
        self._G: Optional[np.ndarray] = None      # B.T B, float64
        self._nIG = 0.0                           # ||I - B.T B||_2
        self._xmax = 1.0                          # max ||x|| over stored rows
        # ||dy|| <= gamma_d * sqrt(m) * ||x||: the fp32 rounding of the stored
        # projection, and gamma_m for the query-time evaluation of pq . y.
        self._dy = _gamma(self.embed_dim) * float(np.sqrt(self.d_out))
        self._gam_m = _gamma(self.d_out)
        self._Y = np.zeros((0, self.d_out), dtype=np.float32)
        self._res = np.zeros(0, dtype=np.float32)
        self._yn = np.zeros(0, dtype=np.float32)
        self.n_rows = 0
        self._cap = 0
        self.fit_rows = 0
        self.fit_seconds = 0.0
        self.captured_energy = 0.0

    # -- build --------------------------------------------------------------
    def fit(self, chunks: Iterable[np.ndarray]) -> "PCAScreen":
        """Fit the basis from an iterable of ``(rows, embed_dim)`` fp32 blocks.

        The basis is the top-``d_out`` eigenvectors of the **uncentered** second
        moment ``sum_i x_i x_i^T``, accumulated in float64. Uncentered on
        purpose: what makes the bound tight is a small residual ``||x - B B.T
        x||``, which is the energy of the vectors themselves, not their variance
        about a mean the bound never subtracts.

        Raises nothing on a degenerate corpus; a rank-deficient moment matrix
        simply yields a basis whose trailing columns capture no energy, and the
        bound stays admissible because admissibility does not depend on ``B``.
        """
        import time
        t0 = time.perf_counter()
        d = self.embed_dim
        moment = np.zeros((d, d), dtype=np.float64)
        seen = 0
        total_energy = 0.0
        for block in chunks:
            V = np.ascontiguousarray(block, dtype=np.float32)
            if V.ndim != 2 or V.shape[1] != d:
                raise ValueError(f"fit block has shape {V.shape}, expected (*, {d})")
            if V.shape[0] == 0:
                continue
            if seen >= self.fit_sample:
                break
            if seen + V.shape[0] > self.fit_sample:
                V = V[: self.fit_sample - seen]
            V64 = V.astype(np.float64)
            moment += V64.T @ V64
            total_energy += float(np.einsum("ij,ij->", V64, V64))
            seen += V.shape[0]
        self.fit_rows = seen
        if seen == 0:
            # Nothing to fit: an identity-like basis is still admissible.
            self.basis = np.eye(d, self.d_out, dtype=np.float32)
        else:
            vals, vecs = np.linalg.eigh(moment)
            order = np.argsort(vals)[::-1][: self.d_out]
            self.basis = np.ascontiguousarray(vecs[:, order], dtype=np.float32)
            kept = float(np.clip(vals[order], 0.0, None).sum())
            self.captured_energy = kept / total_energy if total_energy > 0 else 0.0
        B = self.basis
        # G = B.T B in float64, plus the spectral norm of (I - G). Both are what
        # let the bound hold WITHOUT assuming B is orthonormal -- and B is only
        # ORTHONORMAL-ISH here, because eigh's float64 vectors were cast to fp32.
        B64 = B.astype(np.float64)
        self._G = np.ascontiguousarray(B64.T @ B64)
        self._nIG = float(np.linalg.norm(np.eye(self.d_out) - self._G, 2))
        self._xmax = 1.0
        self._Y = np.zeros((0, self.d_out), dtype=np.float32)
        self._res = np.zeros(0, dtype=np.float32)
        self._yn = np.zeros(0, dtype=np.float32)
        self.n_rows = 0
        self._cap = 0
        self.fit_seconds = time.perf_counter() - t0
        return self

    @property
    def fitted(self) -> bool:
        return self.basis is not None

    def _ensure(self, need: int) -> None:
        if need <= self._cap:
            return
        self._ensure_exact(max(need, int(self._cap * 1.5) + 1024))

    def reserve(self, n_rows: int) -> None:
        """Pre-size the projection arrays to EXACTLY ``n_rows``.

        The engine knows the row count before it starts projecting, so it says
        so and the screen allocates once. Without this the geometric growth in
        :meth:`_ensure` leaves up to 50% headroom -- measured 95.2 MB against a
        used 74.5 MB at 71,433 rows -- which is memory a user is paying for and
        cannot see.
        """
        need = int(n_rows)
        if need > self._cap:
            self._ensure_exact(need)

    def _ensure_exact(self, cap: int) -> None:
        Y = np.empty((cap, self.d_out), dtype=np.float32)
        res = np.empty(cap, dtype=np.float32)
        yn = np.empty(cap, dtype=np.float32)
        if self.n_rows:
            Y[: self.n_rows] = self._Y[: self.n_rows]
            res[: self.n_rows] = self._res[: self.n_rows]
            yn[: self.n_rows] = self._yn[: self.n_rows]
        self._Y, self._res, self._yn, self._cap = Y, res, yn, cap

    def extend(self, block: np.ndarray) -> int:
        """Project and store one ``(rows, embed_dim)`` fp32 block of NEW rows.

        Uses the basis already fitted. This is the whole of "appends cannot
        invalidate correctness": the bound is admissible for any ``B``, so rows
        appended long after the fit are bounded correctly by the old ``B``.
        """
        if self.basis is None:
            raise RuntimeError("PCAScreen.extend() before fit()")
        V = np.ascontiguousarray(block, dtype=np.float32)
        if V.ndim != 2 or V.shape[1] != self.embed_dim:
            raise ValueError(f"block has shape {V.shape}, "
                             f"expected (*, {self.embed_dim})")
        r = V.shape[0]
        if r == 0:
            return 0
        self._ensure(self.n_rows + r)
        lo, hi = self.n_rows, self.n_rows + r
        Y = np.ascontiguousarray(V @ self.basis, dtype=np.float32)
        self._Y[lo:hi] = Y                      # THIS is the y the bound uses
        Y64 = Y.astype(np.float64)
        yn2 = np.einsum("ij,ij->i", Y64, Y64)
        xn2 = np.einsum("ij,ij->i", V.astype(np.float64), V.astype(np.float64))
        yGy = np.einsum("ij,ij->i", Y64, Y64 @ self._G)
        xn = np.sqrt(np.maximum(xn2, 0.0))
        yn = np.sqrt(np.maximum(yn2, 0.0))
        # ||r||^2 = ||x||^2 - 2 t.y + y.G y with t = y + dy, so substituting the
        # worst dy gives this. It is NOT ||x||^2 - ||y||^2: that form assumes
        # both that B is orthonormal and that y is the exact projection, and it
        # under-bounds by 3.4e-05 at d=128/m=127 (tests/test_screen.py).
        r2 = xn2 - 2.0 * yn2 + yGy + 2.0 * self._dy * xn * yn
        self._yn[lo:hi] = (yn * float(_NORM_UP) + float(_NORM_FLOOR)).astype(np.float32)
        self._res[lo:hi] = (np.sqrt(np.maximum(r2, 0.0)) * float(_NORM_UP)
                            + float(_NORM_FLOOR)).astype(np.float32)
        self._xmax = max(self._xmax, float(xn.max()) if r else 0.0)
        self.n_rows = hi
        return r

    def truncate(self, n: int) -> None:
        """Forget rows at and beyond ``n`` (used when a caller detects a shrink)."""
        self.n_rows = max(0, min(int(n), self.n_rows))

    # -- query --------------------------------------------------------------
    def query_terms(self, q: np.ndarray):
        """``(pq, resq, coeff_y, const)`` -- the four per-query numbers the bound needs.

        * ``pq`` is ``B.T q`` computed in float64 and rounded ONCE to fp32, so
          its only error is a representation error of ``u*|pq|``, not an
          accumulated one.
        * ``resq`` is ``||q - B pq||`` computed DIRECTLY in float64. Deriving it
          as ``sqrt(||q||^2 - ||pq||^2)`` would assume ``B`` is orthonormal,
          which is exactly the assumption this module refuses to make.
        * ``coeff_y`` multiplies the stored ``||y||``: it carries the
          ``||I-G||_2`` term, the fp32 evaluation error of ``pq . y``, and the
          rounding of ``pq`` itself.
        * ``const`` is the one genuinely absolute term, ``||pq|| * ||dy||``.
        """
        if self.basis is None:
            raise RuntimeError("PCAScreen.query_terms() before fit()")
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1)
        q64 = q.astype(np.float64)
        B64 = self.basis.astype(np.float64)
        pq = np.ascontiguousarray((q64 @ B64).astype(np.float32))
        pqn = float(np.linalg.norm(pq.astype(np.float64)))
        resq = float(np.linalg.norm(q64 - B64 @ pq.astype(np.float64)))
        resq = resq * float(_NORM_UP) + 1e-7
        coeff_y = pqn * (self._nIG + self._gam_m + U32) + 1e-9
        const = pqn * self._dy * self._xmax + 1e-9
        return pq, np.float32(resq), np.float32(coeff_y), np.float32(const)

    def bounds(self, q: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """An admissible upper bound on ``cos(q, x)`` for every stored row.

        ``bounds(q)[i] >= float(arena_vector(i) @ q)`` for every ``i``, for every
        ``q``, whatever basis is loaded and whenever those rows were appended.
        """
        n = self.n_rows
        if out is None:
            out = np.empty(n, dtype=np.float32)
        if n == 0:
            return out
        pq, resq, coeff_y, const = self.query_terms(q)
        np.dot(self._Y[:n], pq, out=out)
        out += self._res[:n] * resq
        out += self._yn[:n] * coeff_y
        out += const
        return out

    def survivors(self, ub: np.ndarray, tau: float) -> np.ndarray:
        """Rows whose bound leaves them able to reach a score of ``tau``."""
        return np.flatnonzero(ub >= np.float32(tau - TAU_EPS))

    # -- introspection ------------------------------------------------------
    def resident_bytes(self) -> int:
        """Bytes this screen holds resident, basis included -- ALLOCATED, not used."""
        b = 0 if self.basis is None else self.basis.nbytes
        g = 0 if self._G is None else self._G.nbytes
        return int(b + g + self._Y.nbytes + self._res.nbytes + self._yn.nbytes)

    def used_bytes(self) -> int:
        """Bytes the rows that exist actually occupy, basis included."""
        b = 0 if self.basis is None else self.basis.nbytes
        g = 0 if self._G is None else self._G.nbytes
        return int(b + g + self.n_rows * self.bytes_per_row())

    def bytes_per_row(self) -> int:
        return int(self.d_out * 4 + 8)

    def stats(self) -> dict:
        return {
            "d_out": int(self.d_out),
            "orthonormality_defect": float(self._nIG),
            "rows": int(self.n_rows),
            "fit_rows": int(self.fit_rows),
            "fit_seconds": float(self.fit_seconds),
            "captured_energy": float(self.captured_energy),
            "resident_bytes": self.resident_bytes(),
            "used_bytes": self.used_bytes(),
            "bytes_per_row": self.bytes_per_row(),
            "seed_pool": int(self.seed_pool),
        }


__all__ = ["PCAScreen", "DEFAULT_D_OUT", "DEFAULT_SEED_POOL",
           "DEFAULT_FIT_SAMPLE", "TAU_EPS", "U32"]
