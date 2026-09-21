"""
nanomem.engine
~~~~~~~~~~~~~~
VaultEngine: the storage and retrieval core behind :class:`nanomem.vault.Vault`.

What this actually is, stated without decoration:

* One append-only file per vault (``NANOMEM3`` magic, section-by-section layout
  in :mod:`nanomem.container`). Writes go to an in-RAM memtable and spill one
  block at a time; nothing is ever updated in place. Deletes and updates are a
  full atomic rewrite (``replace_all`` -> temp file -> ``os.replace``).
* Every stored vector is resident in RAM (fp32 by default; ``residency=`` can
  hold them as fp16, as a memory-mapped fp16 sidecar, or as int8 with an exact
  fp32 re-rank -- see :mod:`nanomem.arena`), so a search is one dense matmul. Measured on a RE-OPENED vault -- the shape a caller runs --
  with 768-d vectors and top_k=4: 0.344 ms p50 at 10,000 docs and 1.809 ms at
  71,433 (``evidence/scale_results_v3r3.json``, 500 questions), against a
  raw numpy matmul floor of 0.272 / 1.464 ms on the same run, and 0.053 ms at
  1,190 docs (``headtohead_v3.json``). Those are one machine on one afternoon and
  they move by ~15% between runs; the shape is what matters -- roughly 1.25x an
  exhaustive numpy scan, exact, and LINEAR. It is
  not sub-linear, and nothing here claims a fixed RAM budget: ``stats()``
  reports the bytes actually allocated (~3.1 KB/doc used, more while the arena
  sits between two capacity doublings).
* Above ``n_exhaustive`` (default 50,000) an OPT-IN IVF-style cell router
  (``router="auto"``) narrows the scan to a fraction of the corpus. It ships
  OFF, and the DECISIONS #3 sweep that settled that has now been run
  (``evidence/sweep_router_gate.py`` ->
  ``evidence/router_gate_results.json``; 71,433 reclustered paragraphs,
  1,000 questions, 30 budgets, paired bootstrap). The router PASSES the recall
  gate -- 13 of 30 budgets land within 1.0 pt of the exact scan, and the shipped
  budget (``cell_target=25``, ``beam_frac=0.25``) scores 77.05% recall@4 against
  77.30% exhaustive, -0.25 pt CI [-0.50, -0.05]. It is turned off anyway,
  because narrowing the scan does not make the query cheaper here: measured
  inside the engine at 71,433 documents on a reopened vault, the shipped budget
  is 5.11 ms p50 against 3.34 ms exhaustive, and the cheapest gate-passing
  budget is 2.94 ms -- a saving of 0.40 ms per query bought with 25.6 s of extra
  work at EVERY open (the k-means fit is re-run because no centroid is
  persisted), 1.06 GB of extra peak RSS and 0.6 pt of recall. One open has to
  serve tens of thousands of queries before that trade turns positive. Below
  ``n_exhaustive`` nothing changes at all: ``router="auto"`` returns the same
  ids and the same cosines as ``router="off"``, checked at 10,000 documents.
* An OPT-IN, default-OFF **exactness-preserving screen** (``screen="pca"``)
  makes the exact scan cheaper without making it approximate. It bounds every
  document's cosine from above in a 256-dimensional subspace and skips the rows
  that provably cannot reach the top-k; the survivors are scored by the same
  fp32 kernel the full scan uses. MEASURED end to end through
  :meth:`VaultEngine.search` at 71,433 documents, 200 questions x 5 paired
  interleaved cycles: **p50 1.718x** (1.9495 ms -> 1.1350 ms, bootstrap CI
  [1.695, 1.736]), p95 1.207x, at a cost of **1032 bytes per document** of
  resident RAM (71.6 MiB at 71,433, +34.2% over the 209.3 MiB arena) and a
  one-off 0.191 s build. Exactness is not a measurement but a proof, and it was
  measured anyway: 0 bound violations in 71,433,000 document checks, and over
  3,000 searches at k = 1, 4 and 10, zero with a different id list and zero with
  a non-bitwise-identical score list. Reaching that last figure took a fix: ties
  are now broken on the row id (:func:`_select_top_k`), because two documents
  can hold the identical float32 cosine and ``argpartition`` resolved that tie
  differently depending on how many rows it was handed. Before the fix, 9 of
  3,000 differed -- all of them ties, none of them a miss.
  ``evidence/pca_screen_results.json``. See the ``screen`` argument.
* Encryption is opt-in and off by default, exactly like SQLite. With a password:
  scrypt-derived keys, a SHAKE256 keystream and an HMAC-SHA256 tag per block
  (encrypt-then-MAC). See :mod:`nanomem.crypto`; it is not AES and not a NIST
  AEAD, and ``stats()['cipher']`` says so.
* Scores are cosine similarity plus three documented, BOUNDED, purely ADDITIVE
  boosts, all of which apply only to a personal-memory question
  (``entities.is_personal_query``) against a vault that actually holds tagged
  personal records: ``intent_boost`` (+0.25) when a record's entity matches the
  question's intent, ``group_hoist`` (+0.25) for the revision group the question
  named, and ``revision_lead`` (at most +0.20) for the member of that group the
  question asked for. ``stats()['max_boost']`` publishes the total, 1.10, and it
  is a real bound: for every hit
  ``0 <= hit['score'] - hit['cosine'] <= stats()['max_boost']``. 3.0.1 PERMUTED a
  group's scores instead of adding, so a record could hold another record's
  score -- measured +0.6402 above its own cosine against a published cap of 0.50,
  and other records BELOW their own cosine
  (``evidence/adjacent_attributes_v3r3.json``).
* Two records are only treated as statements of one fact when the question
  cannot tell them apart AND the tagger has not said they are different
  attributes. 3.0.1 used the cosine window alone, so "who is my dentist?"
  returned the doctor and "what is my home address?" returned the work address:
  30 of 40 generic adjacent-attribute probes against 40 of 40 for a plain cosine
  scan, i.e. a net regression against the v2 engine it replaced. This build
  scores 40 of 40 -- identical to plain cosine -- while still scoring 11 of 16 on
  same-attribute revision probes where plain cosine gets 4 (same file).
* An ordinary document query gets no boost and no re-ordering at all, and returns
  exactly what an exhaustive cosine scan returns (measured: 0 of 120 questions
  differ on the 1,190-paragraph validation corpus,
  ``evidence/exactness_v3r2.json``).

v2 files are migrated transparently on open, with the original preserved at
``<path>.v2.bak``.
"""

import hashlib
import json
import os
import re
import threading
import time
import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

from . import container as _c
from . import crypto as _crypto
from . import entities as _ent
from . import routing as _routing
from . import screen as _screen
from . import arena as _arena
from .arena import Arena, Memtable
from .container import (BLOCK_CAPACITY, DEFAULT_EMBED_DIM, FILE_VERSION,
                        LANDMARKS_PER_BLOCK, Container, FileHeader)
from .errors import (ClosedVaultError, ContainerReplacedError, CorruptContainerError,
                     IntegrityError, NanomemError, NotEncryptedError,
                     PasswordRequiredError, ReadOnlyVaultError, WrongPasswordError)

#: Engine build. 3.0.4 is 3.0.3 plus three MEASURED behaviour changes: the
#: temporal direction is read off the question's own wording (so a default
#: `search()` returns a different record for a historically-worded question --
#: evidence/ranking_r5_temporal_layer.json), the arena is pre-sized from
#: the container's header row count (evidence/memory_results.json), and
#: the router gate was re-run and the default confirmed OFF
#: (evidence/router_gate_results.json). `stats()["engine_version"]` is
#: the only way a caller can tell those changes apart from 3.0.3.
#:
#: 3.0.5 is 3.0.4 with the RESIDENT ARENA REBUILT (see :mod:`nanomem.arena`).
#: Nothing about an answer changes -- the score vectors are bitwise identical
#: and all 500 top-10 lists at both corpus sizes are unchanged
#: (evidence/ingest_ram_results.json, `exactness`) -- but three things a
#: caller can observe do. Peak ru_maxrss over a 71,433-document build-then-serve
#: falls from 818.6-819.1 MB to 286.4-286.9 MB, the same harness on the same
#: machine (evidence/memory_results.json, `arms.n71433.fp32_reserved`,
#: against the committed 3.0.4 checkout); `stats()["arena_bytes"]` is now the
#: rows that exist rather than up to 1.83x them; and `arena_reservation_bytes`,
#: `arena_reservation_is_mapped` and `arena_growth_copies` are new keys.
#: `reserve_additional_rows()` is new, and `Arena.vec` is now a read-only
#: property. The version moves because `stats()` is the only way to tell a
#: process that pre-sizes its arena from one that grows into it.
#:
#: 3.2.0: the arena cache stopped DUPLICATING the vault. `arena_cache_vectors`
#: and `arena_cache_records` are new constructor flags whose defaults
#: ("offsets_ram" / "vault") keep offsets into the vault's own regions instead
#: of a second, fp32 copy of them -- sidecar 257.48 -> 4.38 MiB at 71,433 rows,
#: cached vault 406.15 -> 153.05 MiB, scores bitwise identical in 36 of 36
#: comparisons. The CACHE format goes 2 -> 3, so every `.arena` is refused once
#: and rebuilt; the VAULT format is untouched at 3. `stats()["arena_cache"]`
#: carries the two new source fields, and a vault truncated out-of-band under a
#: live engine now raises `VaultShrankError` instead of faulting the process.
#: The version moves because the default layout on disk, and one failure mode,
#: both changed with no argument change.
ENGINE_VERSION = "3.4.6"
MT_BASE = 1 << 40                      # virtual row ids for unflushed records
_KEEP = object()                       # sentinel for "leave this as it is"

# Back-compat aliases so `from .engine import ...` keeps working for old code.
FILE_MAGIC = _c.FILE_MAGIC
COORDINATE_DIM = 64                    # v2 concept; unused in v3, kept for imports
PAGE_ALIGNMENT = _c.ALIGN


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------
def matches_filter(meta: Dict[str, Any], meta_filter: Dict[str, Any]) -> bool:
    """v2 filter semantics, verbatim.

    Every filter key must exist in ``meta``. A scalar matches when ``val == v`` or
    ``str(val) == str(v)``; if either side is a list/tuple/set it is a membership
    test with str coercion. (``Vault.export``/``split`` use a stricter
    ``meta.get(k) == v``; that difference is deliberate and untouched.)
    """
    for k, v in meta_filter.items():
        if k not in meta:
            return False
        val = meta[k]
        if isinstance(v, (list, tuple, set)):
            if val not in v and str(val) not in [str(x) for x in v]:
                return False
        elif isinstance(val, (list, tuple, set)):
            if v not in val and str(v) not in [str(x) for x in val]:
                return False
        else:
            if val != v and str(val) != str(v):
                return False
    return True


def _volatility_entry(key: str, t: np.ndarray, t_now: float,
                      min_revisions: int) -> Optional[Dict[str, Any]]:
    """One :meth:`VaultEngine.volatility` row from a group's SORTED timestamps.

    Split out so the sealed-block path and the still-pending path cannot drift:
    before 0.6.4 there was only the first, and the second did not exist at all.
    """
    if t.size < int(min_revisions):
        return None
    d = np.diff(t)
    parts = key.split(_ent.GROUP_SEP)
    return dict(
        key=key,
        user_id=parts[0] if parts else "",
        project=parts[1] if len(parts) > 1 else "",
        entity=parts[2] if len(parts) > 2 else "",
        n_revisions=int(t.size),
        first_ts=float(t[0]), last_ts=float(t[-1]),
        age=float(t_now - t[-1]),
        intervals=[float(x) for x in d],
        mean_interval=float(d.mean()) if d.size else None,
        median_interval=float(np.median(d)) if d.size else None)


def _top_k_stable(values, k: int) -> np.ndarray:
    """Indices of the ``k`` largest ``values``, ties broken by ascending index.

    Identical to ``np.argsort(-values, kind="stable")[:k]`` and O(N + k log k)
    instead of O(N log N). At 70,000 rows that one argsort was ~90% of
    :meth:`VaultEngine._resolve_revisions`, which was itself 50.1% of a
    personal-memory search -- a full sort of every candidate to keep four of
    them (``window_max``).

    WHY IT IS EXACT. ``argpartition`` locates the k-th largest VALUE in linear
    time; every member of the true top k is at or above it, so the survivors are
    a superset of the answer, including every tie sitting exactly on the
    boundary. Stable-sorting that superset orders ties by ascending position,
    which is what stable-sorting the whole array does, because both are indexed
    in ascending order. ``argpartition``'s own tie order is unspecified and is
    never relied on here: it is read for a value, never to choose between equal
    ones.
    """
    n = values.shape[0]
    k = max(0, min(int(k), n))
    if k == 0:
        return np.zeros(0, dtype=np.intp)
    if k >= n:
        return np.argsort(-values, kind="stable")
    part = np.argpartition(-values, k - 1)[:k]
    cut = values[part].min()
    keep = np.flatnonzero(values >= cut)          # superset of the top k
    return keep[np.argsort(-values[keep], kind="stable")[:k]]


def legacy_score(c):
    """The v2 display calibration ``0.6*cos + 0.4*sign(cos)*|cos|^13``.

    Strictly monotone in ``cos``, so it never changed a ranking (measured: 0 of
    5,000 argsort differences). v3 ranks and thresholds on cosine directly; this
    survives only to convert old hard-coded thresholds.
    """
    arr = np.asarray(c, dtype=np.float64)
    out = 0.60 * arr + 0.40 * np.sign(arr) * np.abs(arr) ** 13
    return float(out) if np.isscalar(c) or out.ndim == 0 else out


def legacy_score_to_cosine(s: float, tol: float = 1e-7) -> float:
    """Invert :func:`legacy_score` by bisection.

    Use it to re-tune a threshold that was calibrated against the v2 scale:
    ``0.25 -> 0.417``, ``0.32 -> 0.533``, ``0.35 -> 0.580``.
    """
    target = float(s)
    if target <= legacy_score(-1.0):
        return -1.0
    if target >= legacy_score(1.0):
        return 1.0
    lo, hi = -1.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if legacy_score(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _select_top_k(final, rows, k):
    """The ``k`` best of ``final``, ties broken by ASCENDING ROW ID.

    WHY THIS IS NOT JUST ``argpartition``. ``np.argpartition``'s order among
    EQUAL elements is unspecified, and which of several tied elements lands in
    the first ``k`` depends on how long the array it was handed is. The
    exhaustive scan hands it every row; the exactness-preserving screen
    (``VaultEngine(screen="pca")``) hands it only the survivors. Two documents
    holding the *same* float32 cosine -- 0.075% of rows in the measured corpus
    have an exact duplicate elsewhere in it,
    ``evidence/pca_screen_results.json :: measurement.duplicate_census``
    -- could therefore come back in a different ORDER, or a different one of the
    two could come back at all, purely because a performance flag was on. The
    scores were bitwise identical either way, but that is not what "identical
    top-k" means, and it measured 2991 of 3000 searches rather than 3000
    (``verdict.C1_exactness_BEFORE_the_fix``; with the fix, 3000 of 3000).

    Breaking ties on the ROW ID instead makes the answer a function of the SET
    of rows scored and nothing else -- not of the order they arrived in, not of
    how many there were, not of which numpy is installed. The screen's survivor
    set provably contains every row tied at the cut: a tied row has
    ``bound >= cosine = tau``, so ``PCAScreen.survivors`` admits it. Both paths
    therefore see the same tied rows and choose the same one, which is what
    turns "exact up to ties" into "exact".

    The expansion pass runs only when a tie actually straddles the cut; the
    comparison that decides that is one pass over ``final``. Measured, paired and
    interleaved in one process at the shapes the engine uses: 354.2 -> 370.3
    microseconds at 71,433 rows (+16.1 us, 0.8% of that arm's 1.96 ms p50) and
    2.4 -> 3.5 microseconds over a 289-row survivor set (+1.2 us, 0.1% of its
    1.15 ms). ``verdict.C1_exactness :: cost_of_determinism``.
    """
    if k >= final.size:
        part = np.arange(final.size)
    else:
        part = np.argpartition(-final, k - 1)[:k]
        thr = final[part].min()
        # A non-finite threshold means fewer than k rows survived scoring at
        # all; there is nothing to disambiguate and no reason to pay for a
        # full-array expansion.
        if np.isfinite(thr) and np.count_nonzero(final >= thr) > k:
            part = np.flatnonzero(final >= thr)
    return part[np.lexsort((rows[part], -final[part]))[:k]]


def _arena_cache_bytes(path: str) -> int:
    """Bytes the arena cache occupies, or 0 when there is none."""
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def _unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / (n + 1e-8)


# ---------------------------------------------------------------------------
# deprecated container view (kept for 3.0, removed in 3.1)
# ---------------------------------------------------------------------------
class _ContainerView:
    """Thin read-only view that keeps ``engine.container.toc`` / ``.read_payload``
    / ``.close`` working for callers written against v2. ``vault.py`` itself now
    uses :meth:`VaultEngine.iter_records` and :meth:`VaultEngine.replace_all`.
    """

    def __init__(self, engine: "VaultEngine"):
        self._engine = engine

    @property
    def toc(self) -> List[Dict[str, Any]]:
        e = self._engine
        with e._lock:
            e._reload_if_modified()
            return [{"id": b.block_id, "doc_count": int(b.n), "offset": int(b.offset),
                     "length": int(b.total_len), "timestamp": float(b.written_unix),
                     "first_row": int(b.first_row)} for b in e._blocks_meta()]

    def read_payload(self, offset: int, length: int, doc_count: int,
                     block_id: Optional[str] = None) -> Dict[str, Any]:
        """Return owned copies of one block's records (never arena views)."""
        return self._engine._read_block_payload(offset, length, doc_count, block_id)

    @property
    def filepath(self) -> str:
        return self._engine.filepath

    @property
    def embed_dim(self) -> int:
        return self._engine.embed_dim

    def reload_if_modified(self) -> None:
        with self._engine._lock:
            self._engine._reload_if_modified()

    def close(self) -> None:
        self._engine.close()


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class VaultEngine:
    """Single-file vector + record store with entity-aware temporal ranking."""

    def __init__(self, filepath: str, embed_dim: int = DEFAULT_EMBED_DIM, *,
                 password: Optional[str] = None, score_mode: str = "cosine",
                 vector_dtype: str = "float16", n_exhaustive: int = 50_000,
                 block_capacity: int = BLOCK_CAPACITY, landmarks_per_block: int = 0,
                 router: str = "off", route_p="max", beam_frac: float = 0.25,
                 beam_min_cells: int = 4, cell_target: int = 25,
                 window_delta: float = _ent.WINDOW_DELTA, window_max: int = _ent.WINDOW_MAX,
                 window_sim: float = _ent.WINDOW_SIM,
                 window_delta_marked: float = _ent.WINDOW_DELTA_MARKED,
                 marker_overrides_tag: bool = True, tag_match_skips_sim: bool = True,
                 marker_skips_sim: bool = False, window_union: bool = False,
                 gate_on_corpus_only: bool = False,
                 floor_skips_single_valued: bool = False,
                 group_floor_sim: float = 0.0,
                 intent_margin: float = 0.15,
                 floor_keeps_current: Optional[bool] = None,
                 current_keep_sim: float = 0.0,
                 use_revision_markers: bool = False, historical_mode: str = "oldest",
                 group_hoist: float = _ent.GROUP_HOIST,
                 revision_lead: float = _ent.REVISION_LEAD,
                 level_group_boosts: bool = True, temporal_prior: float = 0.0,
                 group_cos_delta: float = _ent.GROUP_COS_DELTA,
                 intent_boost: float = _ent.INTENT_BOOST,
                 durable: str = "fsync", on_integrity_error: str = "raise",
                 on_torn_tail: str = "warn",
                 residency: str = "float32", scan_chunk: int = _arena.DEFAULT_SCAN_CHUNK,
                 rerank_pool: int = _arena.DEFAULT_RERANK_POOL, arena_dir=None,
                 arena_cache: str = "map",
                 arena_cache_vectors: str = "offsets_ram",
                 arena_cache_records: str = "vault",
                 arena_cache_refresh_rows: int = _arena.ARENA_CACHE_REFRESH_ROWS,
                 screen: str = "off", screen_dims: int = _screen.DEFAULT_D_OUT,
                 screen_min_rows: int = 20_000, screen_max_frac: float = 0.35,
                 screen_seed_pool: int = _screen.DEFAULT_SEED_POOL,
                 migrate: bool = True, migrate_backup="auto",
                 personal_sources=None, readonly: bool = False):
        """Open or eagerly create the vault at ``filepath``.

        For an existing file the header's ``embed_dim`` / ``block_capacity`` /
        ``landmarks_per_block`` / vector dtype win; a mismatch with the arguments
        raises a :class:`UserWarning`, as v2 did. A v2 file is migrated in place
        (original kept at ``<path>.v2.bak``) unless ``migrate=False`` or the
        directory is not writable, in which case it opens read-only.

        ``migrate_backup="auto"`` keeps ``<path>.v2.bak`` for a plaintext vault
        and refuses to keep one when ``password`` is set -- a v2 file is readable
        by anyone holding this library. Pass ``True`` to keep it anyway.

        ``screen`` -- THE EXACTNESS-PRESERVING SEARCH SCREEN
        ---------------------------------------------------
        ``"off"`` (the default) or ``"pca"``. This is a LATENCY flag, not a
        recall flag. It has no accuracy knob and no quality trade-off, because
        turning it on cannot change an answer: it computes a provably admissible
        upper bound on every document's cosine in a ``screen_dims``-dimensional
        subspace, skips the rows whose bound puts them below the ``top_k``-th
        score already in hand, and scores the survivors with the same fp32
        kernel the full scan uses (:mod:`nanomem.screen` derives the bound).

        WHAT IT BUYS, MEASURED. End to end through :meth:`search`, 71,433
        HotpotQA paragraphs, 200 questions x 5 paired interleaved cycles, one
        engine with the flag toggled per query so both arms see the identical
        arena and the identical machine weather
        (``evidence/pca_screen_results.json :: phaseC_latency_full``,
        re-run against this build):

            arm            p50        p95      mean
            screen="off"   1.9495 ms  2.0898   1.9328
            screen="pca"   1.1350 ms  1.7318   1.1829
            speedup        1.718x     1.207x   1.634x

        The five per-cycle p50 ratios were 1.709-1.731. The bootstrap CI on the
        p50 ratio is [1.695, 1.736]. 1-minute load average 1.28, the quietest
        this has been measured on; the arms are still interleaved query by query
        inside ONE engine with the flag toggled, rather than run one after the
        other, because the RATIO is the measurement and the absolute
        milliseconds are worth less. An earlier run at load 4.5-5.4 read
        1.705x [1.689, 1.725] -- the same answer, which is the point of
        interleaving.

        READ THE PHASE BLOCKS IN THAT FILE, NOT ITS ``verdict``. bench_screen.py
        merges a new run over the old file with ``prior.update(res)``, so keys
        no phase produces -- ``verdict`` among them -- are carried over from
        whichever run last wrote them. ``verdict.C2_speed`` currently holds the
        1.705x run. Every PASS/FAIL in it still holds; the third decimal of the
        latency ratio does not.

        It is EXACT, and that is a proof rather than a measurement -- but it was
        measured too: 0 bound violations in 71,433,000 document checks, and over
        3,000 searches at k = 1, 4 and 10, zero with a different id list, zero
        with a non-bitwise-identical score list, zero real misses.

        That last clause was NOT free, and the price is paid by callers who
        never turn this flag on -- see ``_select_top_k`` and the 0.4.0 entry in
        CHANGELOG.md. 0.075% of sampled rows in this corpus
        have an exact duplicate elsewhere in it, so the exhaustive top-k is
        sometimes not unique, and ``np.argpartition`` picks among tied rows
        differently depending on how long the array is -- which is precisely
        what the screen changes. 9 of 3,000 searches came back with a different
        ORDERED id list -- 1 of the 9 a different id SET, all 9 carrying a
        bitwise-identical score sequence, none of them a real miss. Rather than
        restate the claim as "exact up to ties", the selection was changed to
        break ties on the ROW ID (:func:`_select_top_k`), which makes the result
        a function of the SET of rows scored and nothing else; the survivor set
        provably contains every row tied at the cut. It costs 16 us per search
        on the full scan, 0.8% of its p50, and it is paid whether or not the
        flag is on. ``pca_screen_results.json :: amendment_1_deterministic_ties``
        holds the before-and-after runs.

        WHAT IT COSTS. ``screen_dims * 4 + 8`` bytes per document, resident:
        **1032 B/doc** at the default 256, i.e. 71.6 MiB at 71,433 documents
        against a 209.3 MiB arena (+34.2%), plus 0.79 MB for the basis. Building
        it takes 0.196 s at 71,433 (0.135 s to fit the basis, 0.061 s to project
        the corpus), paid once per process on the first search or on an explicit
        :meth:`build_screen`.

        WHERE THE BASIS LIVES: **nowhere on disk. It is recomputed.** Not a
        container block, not a sidecar. Three reasons, in order of weight. (1)
        The bound is admissible for ANY basis, so a persisted one buys latency
        and nothing else -- it cannot buy correctness, and a stale one cannot
        cost it. (2) 0.196 s is the whole build, against a 0.161 s reopen; a
        sidecar would save 0.135 s of that and add a staleness, permissions and
        concurrency surface to a feature whose entire value is that it cannot be
        wrong. (3) The container format and the open path are owned elsewhere.
        If the build cost ever matters more than those, the basis is 768x256
        fp32 = 786,432 bytes and persisting it is a contained change.

        WHY APPENDS CANNOT BREAK IT. Not "we tested appends" -- they cannot,
        by construction. The bound
        ``pq.y + ||q~||*||r|| + ||pq||*||B.T r||`` is an algebraic identity plus
        Cauchy-Schwarz; no property of the basis is used anywhere in it. A basis
        fitted from rows 0..N therefore bounds row N+1 correctly, and appended
        rows are projected with the EXISTING basis rather than triggering a
        refit. Measured as well, since a proof with a bug in the code is still a
        bug: a basis fitted from 6,000 rows, then 24,000 rows appended, gave 0
        violations in 2,400,000 document checks and 0 score differences over 200
        searches (``phaseE_stale_basis_appends``). What is NOT safe is a screen
        whose row *i* stopped describing arena row *i*; every operation that can
        do that replaces the :class:`~nanomem.arena.Arena` object, and the
        screen is keyed on that object's identity, so all of them rebuild.

        WHEN IT STANDS DOWN, AUTOMATICALLY, AND RUNS THE ORDINARY EXACT SCAN:
        below ``screen_min_rows`` (default 20,000 -- see the sweep below); under
        ``residency="int8"``, where :meth:`Arena.scores` is not a cosine below
        its re-rank pool and so there is no exact scan to agree with; with a
        ``metadata_filter``; on a vault holding entity-tagged records, or for an
        explicit ``temporal_direction="historical"``, because there the ranking
        layer can add up to ``stats()['max_boost']`` to a candidate and a screen
        that prunes on cosine cannot see that coming; when the survivors would
        exceed ``screen_max_frac`` of the corpus, where gathering them costs more
        than the scan they replace; and if the basis is missing or cannot be
        built at all. None of these is an error and none changes a result.

        WHY ``screen_min_rows`` IS 20,000. Measured with the same paired
        protocol, forcing the screen on at every size
        (``phaseD_size_sweep``): 1,000 docs 0.485x, 5,000 0.878x, 10,000 0.966x,
        20,000 1.367x, 40,000 1.451x, 71,433 1.705x. Below ~20,000 the bound
        evaluation costs more than the scan it saves, so the flag declines. At
        the shipped default the measured p50 ratio (on/off) at 1,000 / 5,000 /
        10,000 documents is 1.0000 / 1.0025 / 1.0050 -- turning the flag on
        costs at most 0.5% where it cannot help, which is the cost of asking.

        ``screen_dims`` (default 256) is the subspace width. It is the one knob
        with a real trade-off, and it is NOT accuracy: 128 makes the bound loose
        enough that ~15% of the corpus survives and the screen becomes a net
        loss, 256 leaves 0.405% (measured mean 289.2 of 71,433 rows). Halving it
        halves the memory and roughly doubles the survivor count.

        ``search_batch`` does not use the screen; it is a single matmul over all
        queries at once, which the screen cannot improve.

        ``arena_cache`` -- THE RESIDENT ARENA, CACHED BESIDE THE VAULT
        -------------------------------------------------------------
        ``"map"`` (the default), ``"copy"``, ``"verify"`` or ``"off"``. Opening
        a vault used to mean replaying every block to rebuild the fp32 arena,
        which is O(rows) and was nanomem's worst axis against every competitor.
        A ``<vault>.arena`` sidecar holds that arena in its resident layout, so
        an open becomes a bind plus an ``mmap``.

        WHAT IT BUYS, MEASURED (``evidence/reopen_results.json ::
        headline``, regenerated against this build; in-process median of 25
        opens in a warm interpreter, which is the protocol that produced the
        competitors' reopen column, so the comparison is like for like):

            rows      scan open    mapped open   speedup
            1,190     0.002657 s   0.000186 s     14.3x
            10,000    0.022559 s   0.000206 s    109.5x
            71,433    0.163351 s   0.000193 s    846.4x

        At 71,433 rows that is 0.138x sqlite-vec's 0.0014 s, i.e. nanomem now
        wins reopen outright rather than losing it 115x.

        It also costs less RAM, because the vectors become clean file-backed
        pages the kernel may evict instead of dirty anonymous ones: at 71,433
        rows the serving process's ``phys_footprint`` is 9 MB against 288 MB,
        and its peak ``ru_maxrss`` 258.2 MB against 287.5 MB. Steady-state
        latency does not move -- the paired duel, arms alternated cycle by
        cycle on the same vault, reads mapped/scanned at 0.9983
        (``latency_duel.n71433``).

        THREE THINGS IT COSTS, none of them hidden.
        (1) DISK -- and this is what ``arena_cache_vectors`` below now answers.
        The 3.1.0 sidecar was 257.5 MiB beside a 148.7 MiB vault at 71,433 rows,
        so a vault that kept one occupied 406.2 MiB against sqlite-vec's 258.0
        MiB, where nanomem's 148.7 MiB used to be the smallest of every arm
        measured. 209.3 MiB of that file was an fp32 upcast of vectors the vault
        already holds in fp16 and 43.8 MiB was a verbatim copy of the vault's
        record sections, and neither copy is information: the block table the
        sidecar already stored says where both live. Keeping the OFFSETS instead
        takes the sidecar to 4.38 MiB and the pair to 153.05 MiB -- see the
        ``arena_cache_vectors`` block below for the whole measured trade.
        ``arena_cache="off"`` writes nothing and restores 3.0.5 exactly.
        (2) THE FIRST QUERY, which now pays the page faults the open did not:
        0.0109 s to first answer at 71,433 rows against 0.1687 s without the
        cache -- still 15.4x better, but 15.4x is the honest number and 846x is
        not.
        (3) A FRESH PROCESS BELOW ~10,000 ROWS IS SLOWER, not faster. At 1,190
        rows a brand-new interpreter opens in 0.007643 s with the cache against
        0.003210 s without it. ``ARENA_CACHE_MIN_ROWS`` is 256 and is NOT
        re-tuned here: the crossover is bracketed only by those two measured
        sizes, and picking a floor inside that bracket after seeing the result
        would be choosing a criterion to pass it. Pass ``arena_cache="off"`` for
        a small vault opened by short-lived processes.

        WHAT IT DOES NOT COST IS AN ANSWER. Same build, cache on against cache
        off, 500 queries scored against every row at all three sizes: the
        SHA-256 of the concatenated fp32 score vectors is identical and 0 of 500
        top-10 lists change, for ``map``, ``copy`` and ``verify`` alike
        (``verdict.exactness_n71433_map``, ``post_fix_check``).

        ON INTEGRITY, PLAINLY: none of these modes is a security control, and
        the cache does not make the vault weaker than it already is. A plaintext
        vault's block trailer is an unkeyed SHA-256 that anyone can recompute,
        and a vault forged that way is served by a full ``arena_cache="off"``
        scan at cosine 0.999997 (``cache_integrity.vault_forgery``;
        tests/test_arena_cache.py). ``verify`` checks the cache's own digest and
        re-reads every vault block: it catches CORRUPTION in either file and no
        tampering at all, because that digest is unkeyed and lives in the header
        of the file it describes. See :class:`nanomem.arena.ArenaSnapshot` for
        the full statement and for the one case that was closed rather than
        described -- a vault edited in place now raises in ``map`` as it always
        did in ``off``, at a measured +4.08 us per open (``binding_cost``).

        ``arena_cache_vectors`` / ``arena_cache_records`` -- WHAT IT DUPLICATES
        ----------------------------------------------------------------------
        ``arena_cache`` says whether there is a sidecar and how far to trust it.
        These two say what is IN it, which is the axis that made it 257.5 MiB.

          ``arena_cache_records``  ``"vault"`` (default) reads each block's
                                   record section out of the mapped vault, where
                                   it already is; ``"cache"`` copies them into
                                   the sidecar (3.1.0).
          ``arena_cache_vectors``  ``"offsets_ram"`` (default) stores no vectors
                                   and upcasts the mapped vault into one
                                   anonymous fp32 array at the FIRST vector read
                                   -- lazily, so the open stays O(1) and only a
                                   process that searches pays for it.
                                   ``"offsets"`` stores none and scans the
                                   mapped vault on every query.
                                   ``"cache"`` stores the arena's own vectors,
                                   which is 3.1.0 exactly. An fp16 sidecar is
                                   not a fourth value: it is
                                   ``residency="float16"``, whose arena already
                                   IS fp16.

        MEASURED, all of it, by ``evidence/bench_sidecar_size.py`` ->
        ``sidecar_size_results.json``: 71,433 rows, every arm in its own process
        against a byte-identical copy of the same vault, the sidecar written by
        a THIRD process so its build cannot land on the server's peak RSS, two
        passes, medians. ``p50`` is the paired duel -- both arms alternated cycle
        by cycle inside one process -- because the absolute p50 gate this was
        held to failed for the UNMODIFIED baseline as well (1.2577 and 1.3882 ms
        against the 1.1494 ms it was derived from), and an absolute bar on a box
        that has drifted measures the box. Memory is ``screen="off"``.

          arena_cache_vectors=      sidecar   total    reopen   p50 pca   p50 off   peak   phys
                                      MiB     MiB         s   (paired)  (paired)    MB     MB
          "cache" + records "cache"  (3.1.0)  257.48  406.15  0.000187        --        --    318     44
          "cache" + records "vault"   213.66  362.32  0.000226   x0.9994   x0.9945    326     47
          residency="float16"         109.02  257.69  0.000229   x0.9829   x2.2611    227     57
          "offsets"                     4.38  153.05  0.000220   x1.7325   x2.9793    227     57
          "offsets_ram"                 4.38  153.05  0.000232   x0.9976   x1.0037    331    270
          arena_cache="off"             0.00  148.66  0.174820   x0.9913   x0.9909    351    337

        WHAT TO READ OUT OF IT: there is no arm that wins every column. Disk,
        resident memory and p50 are three corners and a layout gets two.
        ``"cache"`` keeps the p50 and a 44 MB phys_footprint and pays
        406.1 MiB of disk. ``"offsets"`` keeps the disk AND the memory
        (227 MB peak, 57 MB phys -- the lowest of every arm) and pays
        x1.7325 on the p50 with the screen on and x2.9793 with it off.
        ``"offsets_ram"`` keeps the disk and the p50 (x0.9976 paired, with
        the screen on and off alike) and pays the phys_footprint: 270 MB
        against 44 MB, because an upcast array is dirty anonymous memory
        where a mapped sidecar is clean file-backed pages. The first query
        after an open pays the upcast once: 0.0227 s against 0.0117 s.
        It is the default because the disk axis is the one this release
        lost and 406.2 MiB -> 153.0 MiB wins it back outright -- below
        sqlite-vec's 258.0 MiB and 4.4 MiB above a vault with no sidecar
        at all. Set ``"cache"`` to get 3.1.0 back exactly.

        WHAT NONE OF THEM CHANGE IS AN ANSWER. The SHA-256 of the concatenated
        fp32 score vectors of all 500 queries against all 71,433 rows is
        IDENTICAL for every arm above and for ``arena_cache="off"``, and 0 of
        500 top-10 lists differ -- at all three corpus sizes, with the screen on
        and off, in both passes, 36 of 36 comparisons. That is construction, not
        tolerance: a vault-backed scan stages the same ``scan_chunk`` rows into
        the same fp32 buffer and makes the same ``buf @ q`` call the fp16
        residencies already made, and the upcast produces the identical fp32
        array the sidecar used to hold.

        ONE THING GETS STRONGER. Under ``arena_cache="verify"`` the fp32 sidecar
        was checked against a digest in its own header, which anyone who can
        edit the file can recompute. With ``arena_cache_vectors="offsets"`` there
        is nothing in the sidecar to forge: the bytes served are the vault's own,
        and re-reading the vault's block trailers -- which ``verify`` already
        does -- is what covers them.

        ONE THING GETS WEAKER, AND IT IS NOT AN ANSWER -- IT IS A CRASH MODE.
        A layout that reads the vault has a live ``mmap`` of it for as long as
        the engine is open. ``mmap`` binds to a length, not to a file: if
        something OUTSIDE nanomem truncates the vault while that mapping is
        live, the pages past the new end are still mapped and touching one
        raises SIGBUS -- a signal, which kills the process with no traceback and
        nothing a caller can catch. ``"cache"`` survives the same damage because
        it holds a copy. Measured: cut a vault in half under a live engine and
        3.1.0 returns the row, while the offset layouts used to exit 138. They
        now raise :class:`nanomem.errors.VaultShrankError` instead, from one
        ``os.fstat`` per scan, per gather and per record read (0.486 us a call;
        10 records a query is 0.4% of a 1.2 ms query, and the p50 above was
        re-measured with the check in place). The check is against the last byte
        the block table addresses, not against the mapped length, so nanomem's
        OWN torn-tail recovery -- which only removes bytes past the last valid
        block -- does not trip it. It is a guard and not a guarantee: a
        truncation that lands between the check and the page touch still faults,
        and ``"cache"`` is the layout that has no window at all.

        WHAT CHANGING THE KNOB COSTS. A sidecar written for one layout is
        REFUSED by an engine configured for another (``arena_cache_info()``
        reports ``"cache was written for a different sidecar layout"``) and
        rebuilt on that open. One slow open, once. Upgrading from 3.1.0 costs
        the same, because the cache format version moved from 2 to 3.
        """
        self.filepath = os.path.abspath(filepath)
        self.embed_dim = int(embed_dim)
        self.score_mode = "legacy" if str(score_mode).lower() == "legacy" else "cosine"
        self.n_exhaustive = int(n_exhaustive)
        self.router_mode = str(router).lower()
        self.route_p = route_p
        self.beam_frac = float(beam_frac)
        self.beam_min_cells = int(beam_min_cells)
        self.cell_target = int(cell_target)
        self.window_delta = float(window_delta)
        self.window_max = int(window_max)
        self.window_sim = float(window_sim)
        self.use_revision_markers = bool(use_revision_markers)
        self.historical_mode = str(historical_mode)
        self.group_hoist = float(group_hoist)
        self.revision_lead = float(revision_lead)
        self.level_group_boosts = bool(level_group_boosts)
        self.temporal_prior = float(temporal_prior)
        self.group_cos_delta = float(group_cos_delta)
        self.intent_boost = float(intent_boost)
        self.durable = str(durable)
        self.window_delta_marked = float(window_delta_marked)
        self.marker_overrides_tag = bool(marker_overrides_tag)
        self.tag_match_skips_sim = bool(tag_match_skips_sim)
        # FIVE MEASURED-AND-REJECTED GROUPING RULES, kept as knobs so the
        # ablation is reproducible. Each is defensible in the abstract; each was
        # measured against the five round-4 ranking sets (40 adjacent-attribute
        # probes, 16 revision probes, the 3-persona selection chat set, the
        # 3-persona round-4 dev chat set, the migrated golden chat vault) and
        # only kept if it paid. Grid: `ranking_dev_r4_u*.json`. (not published, see evidence/INDEX.md)
        #
        #   marker_skips_sim   let an explicit revision marker bypass
        #                      `window_sim`      -> 3-persona chat 33 -> 32.
        #   window_union       union the cosine window with an EXISTING tagged
        #                      group instead of only falling back to it
        #                      -> 3-persona chat 33 -> 31, golden 12 -> 11.
        #   gate_on_corpus_only  drop `temporal_question`'s WORDING test and let
        #                      the corpus decide alone -> neutral on all five
        #                      sets, and exactness stays 0/120. It is not shipped
        #                      because the only failure it is known to address
        #                      was found by reading the held-out chat set, and a
        #                      change that pays nothing on any set I am allowed
        #                      to tune on is not a change I can justify.
        #   floor_skips_single_valued
        #                      drop `group_relevance_floor` when the group's
        #                      entity is SINGLE-VALUED, on the argument that a
        #                      single-valued attribute has one value at a time so
        #                      two records carrying its tag must be competing
        #                      statements of it. Round 5: it is the rule that
        #                      would fix the six remaining historical failures on
        #                      the temporal benchmark's dev split, where the
        #                      genuine older revision sits 0.007-0.19 cosine
        #                      below the floor -- and it fixes NONE of them
        #                      (59/65 with and without) while costing 5 of 36 on
        #                      the 3-persona chat set (35 -> 30). The floor is
        #                      doing real work on real chat; see
        #                      `ranking_dev_r5_floorskip.json` (not published, see evidence/INDEX.md) vs
        #                      ranking_dev_r5_stage4.json.
        #   group_floor_sim    let a tagged group member survive the relevance
        #                      floor by RESEMBLING the group's best member
        #                      (record-to-record) rather than by answering the
        #                      question as well as it does -- the same evidence
        #                      `_cosine_window` already uses to group untagged
        #                      records; see `_apply_group_floor`. This IS the
        #                      rule that fixes the remaining historical failures
        #                      (temporal dev split 59/65 -> 62/65 at 0.50,
        #                      61/65 at 0.60), and it costs 5 and 2 of 36 on the
        #                      3-persona chat set respectively; at 0.70 and above
        #                      it is a no-op on every set. There is no value that
        #                      pays on both, and the chat set is the realistic
        #                      one, so it ships OFF (0.0) and the ablation stays
        #                      reproducible: `ranking_dev_r5_gfs*.json`. (not published, see evidence/INDEX.md)
        self.marker_skips_sim = bool(marker_skips_sim)
        self.floor_skips_single_valued = bool(floor_skips_single_valued)
        self.group_floor_sim = float(group_floor_sim)
        #: How much RAW cosine the question's wording may overrule. 0.0
        #: keeps 0.7.14 behaviour exactly; see design/intent_margin_spec.md.
        self.intent_margin = float(intent_margin)
        # None = decide per group from tag provenance (the 0.6.5 default);
        # True = always, whoever assigned the tag; False = never, which is
        # exactly 0.6.4 and is what the measurement baseline runs as.
        self.floor_keeps_current = (None if floor_keeps_current is None
                                    else bool(floor_keeps_current))
        self.current_keep_sim = float(current_keep_sim)
        self.window_union = bool(window_union)
        self.gate_on_corpus_only = bool(gate_on_corpus_only)
        self.on_integrity_error = str(on_integrity_error)
        # "warn" (default) reports an unreadable trailing fragment and cuts it on
        # the next append; "raise" refuses to open or append over one; "ignore"
        # is 3.0.1's silence. See nanomem.crypto.THREAT_MODEL.
        self.on_torn_tail = str(on_torn_tail)
        # RESIDENT ARENA SHAPE. "float32" is the shipped default and is the only
        # mode whose query latency is unchanged; the narrower modes hold the SAME
        # fp16 values that are on disk and accumulate in fp32, so "float16" and
        # "float16_mmap" return bit-identical scores and only cost time, while
        # "int8" is approximate below its re-rank pool. Every mode's RSS, p50 and
        # recall at 10,000 and 71,433 documents: evidence/memory_results.json.
        if str(residency).lower() not in _arena.RESIDENCY_MODES:
            raise ValueError(f"residency must be one of {_arena.RESIDENCY_MODES!r}, "
                             f"got {residency!r}")
        self.residency = str(residency).lower()
        # THE ARENA CACHE. A vault used to be rebuilt row by row at every open --
        # O(rows), 0.1634 s at 71,433 documents, against sqlite-vec's 0.0014 s --
        # because the arena is derived data and nothing kept it. It is kept now,
        # in a mappable `<vault>.arena` sidecar, and an open that can use it is
        # a stat, a 4 KiB read, a 32-byte read and one mmap. The container is
        # still the source of truth: a cache that is missing, stale, truncated,
        # from another vault or unparseable is dropped and rebuilt in silence.
        # `"map"` (default) / `"copy"` / `"verify"` / `"off"` -- what each one
        # trades is written out in nanomem.arena.ArenaSnapshot.
        if str(arena_cache).lower() not in _arena.ARENA_CACHE_MODES:
            raise ValueError(f"arena_cache must be one of "
                             f"{_arena.ARENA_CACHE_MODES!r}, got {arena_cache!r}")
        self.arena_cache = str(arena_cache).lower()
        # WHAT THE SIDECAR DUPLICATES, which is the axis `arena_cache` never
        # covered. See nanomem.arena.ARENA_VECTOR_SOURCES / ARENA_RECORD_SOURCES
        # and evidence/sidecar_size_results.json for the measurement that
        # picked these defaults.
        self.arena_cache_vectors = str(arena_cache_vectors or "cache").lower()
        if self.arena_cache_vectors not in _arena.ARENA_VECTOR_SOURCES:
            raise ValueError(f"arena_cache_vectors must be one of "
                             f"{_arena.ARENA_VECTOR_SOURCES!r}, "
                             f"got {arena_cache_vectors!r}")
        self.arena_cache_records = str(arena_cache_records or "cache").lower()
        if self.arena_cache_records not in _arena.ARENA_RECORD_SOURCES:
            raise ValueError(f"arena_cache_records must be one of "
                             f"{_arena.ARENA_RECORD_SOURCES!r}, "
                             f"got {arena_cache_records!r}")
        self.arena_cache_refresh_rows = max(1, int(arena_cache_refresh_rows))
        self._arena_cache_info = {"mode": self.arena_cache, "source": "scan"}
        self.scan_chunk = int(scan_chunk)
        self.rerank_pool = int(rerank_pool)
        # EXACTNESS-PRESERVING SEARCH SCREEN, OPT-IN, DEFAULT OFF.
        # `screen="pca"` turns on nanomem.screen.PCAScreen. See the `screen`
        # paragraph of this method's docstring for the semantics, the measured
        # speedup and the measured memory cost.
        self.screen_mode = str(screen or "off").lower()
        if self.screen_mode not in ("off", "pca"):
            raise ValueError(f'screen must be "off" or "pca", got {screen!r}')
        self.screen_dims = int(screen_dims)
        self.screen_min_rows = int(screen_min_rows)
        self.screen_max_frac = float(screen_max_frac)
        self.screen_seed_pool = int(screen_seed_pool)
        self.arena_dir = arena_dir
        self.personal_sources = frozenset(personal_sources
                                          if personal_sources is not None
                                          else _ent.PERSONAL_SOURCES)
        self._password = password
        self._lock = threading.RLock()
        self._closed = False
        self._file_missing = False
        self._last_entity: Dict[Any, str] = {}
        self._router = None
        self._router_rows = 0
        self._screen = None
        self._screen_arena = None
        self._screen_pad = 0
        self._screen_gather_exact = None
        self._screen_calls = 0
        self._screen_engaged = 0
        self._screen_fallbacks = 0
        self._screen_survivors = 0
        self._self_names = None
        self._self_names_rows = -1
        self._engaging = False
        self._container_view = _ContainerView(self)
        self._migrated_from = None

        version = _c.detect_version(self.filepath)
        self.read_only = bool(readonly)
        self._v2_readonly = False
        if version == 2:
            can_write = os.access(os.path.dirname(self.filepath) or ".", os.W_OK) and not readonly
            if migrate and can_write:
                from .legacy_v2 import migrate_v2_to_v3
                migrate_v2_to_v3(self.filepath, password=password, backup=migrate_backup,
                                 vector_dtype=vector_dtype, block_capacity=block_capacity,
                                 landmarks_per_block=landmarks_per_block,
                                 landmark_fn=self._landmark_fn(landmarks_per_block))
                self._migrated_from = 2
            else:
                self._v2_readonly = True
                self.read_only = True

        if self._v2_readonly:
            if password:
                # A SECURITY PARAMETER MUST NOT BE SILENTLY DROPPED. The v2
                # read-only fallback ignores `password` entirely (a v2 file has no
                # key material at all), so 3.0.2 opened the vault in CLEARTEXT with
                # no warning and no error -- unlike the v3 plaintext+password path,
                # which correctly raises NotEncryptedError. The fallback is also
                # taken AUTOMATICALLY when the directory is not writable, so it was
                # not only an explicit opt-in.
                raise NotEncryptedError(
                    f"{self.filepath} is a v2 file being opened read-only "
                    f"(migrate=False, readonly=True, or an unwritable directory); "
                    f"v2 files hold no key material, so `password` cannot apply. "
                    f"Migrate it first, or open it without a password.")
            self._open_v2_readonly(block_capacity, landmarks_per_block)
        else:
            self._cont = Container(self.filepath, embed_dim=self.embed_dim,
                                   password=password, vector_dtype=vector_dtype,
                                   block_capacity=block_capacity,
                                   landmarks_per_block=landmarks_per_block,
                                   durable=self.durable,
                                   on_integrity_error=self.on_integrity_error,
                                   on_torn_tail=self.on_torn_tail,
                                   readonly=self.read_only)
            h = self._cont.header
            if h.embed_dim != self.embed_dim:
                # SAY WHICH CALLS RAISE, BECAUSE NOT ALL OF THEM DO. "every call
                # will raise" was false and actively misleading: delete, prune,
                # compact and forget_superseded all succeed through a mismatched
                # handle AND REWRITE THE FILE. A reviewer read the old wording,
                # watched prune() take a vault to zero, and reported silent
                # destruction through a handle that supposedly could not touch it.
                # It was prune doing its documented job -- identical through a
                # matched handle -- but the warning is what made it look like a
                # defect. Only calls needing a NEW vector raise.
                warnings.warn(
                    f"{self.filepath} stores {h.embed_dim}-d vectors; requested "
                    f"embed_dim={self.embed_dim} ignored. The file is unchanged and "
                    f"this handle can still READ it, and delete/prune/compact/"
                    f"forget_superseded will rewrite it. Only calls that need a new "
                    f"vector -- add, update(text=), search, history -- raise, until "
                    f"you reopen with a {h.embed_dim}-d model.",
                    UserWarning, stacklevel=2)
                self.embed_dim = int(h.embed_dim)
            self.block_capacity = int(h.block_capacity)
            self.landmarks_per_block = int(h.landmarks_per_block)
            self.arena = self._new_arena()
            self._mt = Memtable(self.embed_dim, self.block_capacity)
            self._load_arena()
        self._maybe_engage_router()

    def reserve_rows(self, n_rows: int) -> None:
        """Pre-size the resident arena for ``n_rows`` documents, exactly.

        A reopen does this for itself (the container counts the rows from block
        headers first). A BULK INGEST cannot -- nothing tells the engine how many
        ``add_fact`` calls are coming -- so a loader that knows its corpus size
        should say so.

        It is worth much less than it used to be, and that is the point. Under
        the capacity doubling this engine shipped with, the hint was the
        difference between 820.5 and 288.5 MB of peak ru_maxrss over a
        71,433-document build-then-serve (medians of four runs each). With the
        reserved arena underneath (see :mod:`nanomem.arena`) the unhinted path
        already measures 298.4 MB and this hint takes it to 289.5 -- 3%, inside
        the 7% the same arm moves between runs
        (evidence/ingest_ram_results.json, ``summary`` and ``replicate``).
        The one thing it still buys outright is the allocation count: the arena
        is sized once, ``stats()["arena_growth_copies"]`` stays 0, and no
        reservation is ever larger than the rows.

        Passing a number smaller than the corpus is harmless: growth still works.
        Use :meth:`reserve_additional_rows` when the number is a batch length
        rather than a total.
        """
        with self._lock:
            self.arena.reserve(int(n_rows))

    def reserve_additional_rows(self, n_more: int) -> None:
        """Pre-size for ``n_more`` documents ON TOP of what the vault already has.

        The count a bulk caller actually has is the length of the batch in its
        hand, not the vault's eventual total, and it may hand over many batches.
        This adds the batch to the rows already resident (including the ones
        still in the memtable) and hints THAT, with
        :meth:`nanomem.arena.Arena.hint_rows` -- so a directory ingest that hints
        once per file does not pay a reallocation per file. The number is
        allowed to be an estimate; nothing downstream trusts it for anything but
        an allocation size.
        """
        n_more = int(n_more)
        if n_more <= 0:
            return
        with self._lock:
            self.arena.hint_rows(self.arena.n_rows + self._mt.n + n_more)

    # -- construction helpers ----------------------------------------------
    def _new_arena(self) -> Arena:
        """A fresh arena in this vault's residency mode (see ``residency=``)."""
        return Arena(self.embed_dim, residency=self.residency,
                     scan_chunk=self.scan_chunk, rerank_pool=self.rerank_pool,
                     sidecar_dir=self.arena_dir or os.path.dirname(self.filepath) or None)

    def arena_cache_path(self) -> str:
        """Where this vault's arena cache lives (whether or not one exists)."""
        return _arena.arena_cache_path(self.filepath)

    def arena_cache_info(self) -> Dict[str, Any]:
        """What the last open did with the cache, measured. Keys: ``mode``,
        ``source`` (``"scan"`` / ``"cache"`` / ``"cache+append"``), ``reason``
        (why a cache was not used), ``rows_from_cache``, ``rows_scanned``,
        ``attach_s``, ``scan_s``, ``verify_s``, ``write``,
        ``vault_changed_since_cache`` (``"unchanged"`` / ``"grew"`` / ``None``)
        and ``vault_blocks_checked`` (``"all"`` / ``"appended tail only"`` /
        ``"none"``) -- the last two so that a caller can ask what this open
        re-read instead of having to read this file to find out."""
        return dict(self._arena_cache_info)

    def _arena_cache_eligible(self) -> Optional[str]:
        """``None`` if this vault may use a cache, else why it may not."""
        if self.arena_cache == "off":
            return "mode=off"
        if self._cont is None:
            return "no container (v2 read-only)"
        if self.header is not None and self.header.encrypted:
            # The cache holds vectors and document text in the clear. Writing one
            # beside an encrypted vault would defeat the encryption, and reading
            # one there would mean the plaintext had already been leaked.
            return "vault is encrypted"
        if self.residency not in _arena.ARENA_CACHE_RESIDENCIES:
            return f"residency={self.residency}"
        return None

    def _scan_tolerating_a_concurrent_rewrite(self, attempts: int = 4) -> None:
        """Scan the container, retrying only when the FILE ITSELF changed under us.

        `router="auto"` rewrites the vault on the open that crosses
        `n_exhaustive` (`_maybe_engage_router` -> `compact(recluster=True)`), and
        that rewrite lands with `os.replace`. A process that is midway through
        `scan()` when the swap happens reads a prefix from one file and a trailer
        from the other, and raises `IntegrityError("trailer mismatch")` from
        inside the CONSTRUCTOR. The racing REWRITERS already cooperate --
        `_maybe_engage_router` catches `ContainerReplacedError` and says only one
        of them can win -- but a concurrent READER had no such handling.
        Measured on the shipped suite before this: 4 concurrent `router="auto"`
        opens of one vault failed 5 times in 15 runs, so
        `tests/test_round4_regressions.py` was not reliably green and neither was
        the "631 tests pass" line in the README.

        The retry is gated on EVIDENCE, not on optimism: the file's identity
        (inode, size, mtime) must actually have changed between the attempt and
        the failure. A vault that is genuinely corrupt has a stable identity, so
        it still raises -- on the first attempt, with the same error as before.
        """
        from .errors import IntegrityError as _IntegrityError

        def _identity():
            try:
                st = os.stat(self.filepath)
                return (st.st_ino, st.st_size, st.st_mtime_ns)
            except OSError:
                return None

        last = None
        for attempt in range(attempts):
            before = _identity()
            try:
                self._cont.scan(self.arena)
                return
            except _IntegrityError as exc:
                last = exc
                if _identity() == before:
                    raise                     # stable file: this is real damage
                # Another process replaced the vault mid-scan. Start over on the
                # file that is there now.
                self.arena.reset() if hasattr(self.arena, "reset") else None
                self._cont.reopen() if hasattr(self._cont, "reopen") else None
        raise last

    def _load_arena(self) -> None:
        """Fill the arena for a fresh container: from the cache if one binds.

        The scan still runs afterwards and is what loads anything appended since
        the cache was written, so a cached open of a vault another process has
        been writing to costs the appended rows and nothing else.
        """
        info = {"mode": self.arena_cache, "source": "scan", "reason": None,
                "vectors": self.arena_cache_vectors,
                "records": self.arena_cache_records,
                "vec_source": None, "rec_source": None,
                "rows_from_cache": 0, "rows_scanned": 0, "path": self.arena_cache_path(),
                "vault_changed_since_cache": None}
        t0 = time.perf_counter()
        covered = self._attach_arena_cache(info)
        info["attach_s"] = round(time.perf_counter() - t0, 6)
        t1 = time.perf_counter()
        self._scan_tolerating_a_concurrent_rewrite()
        info["scan_s"] = round(time.perf_counter() - t1, 6)
        # The moment the scan returns is the moment the arena describes the file,
        # so this is the (size, mtime) the cache binds itself to. Captured HERE
        # rather than at write time: if another process rewrites the prefix in
        # place between the scan and the write, a write-time stat would record
        # the NEW mtime and bless a cache built from the OLD bytes forever.
        self._vault_stat_at_scan = _arena._stat_pair(self.filepath)
        info["rows_from_cache"] = int(covered)
        info["rows_scanned"] = int(self.arena.n_rows - covered)
        if covered:
            info["source"] = "cache" if info["rows_scanned"] == 0 else "cache+append"
        # Not a claim, a fact the caller can act on: were this vault's block
        # trailers re-read on THIS open? A full scan reads them all. A cached
        # open reads only the blocks it had to scan itself, unless the mode is
        # `verify`, which re-reads the lot. Reported because a default that
        # quietly stops checking should at least be answerable when asked.
        info["vault_blocks_checked"] = ("all" if not covered or self.arena_cache == "verify"
                                        else ("appended tail only"
                                              if info["rows_scanned"] else "none"))
        self._arena_cache_info = info
        self._maybe_write_arena_cache(covered)

    def _attach_arena_cache(self, info: Dict[str, Any]) -> int:
        """Map the cache into the arena. Returns the rows it supplied (0 = none).

        Every failure path leaves the engine exactly as a cache-less open would:
        the container's scan state is reset and the arena is a fresh empty one,
        so the scan that follows rebuilds from the file.
        """
        why = self._arena_cache_eligible()
        if why:
            info["reason"] = why
            return 0
        snap = _arena.ArenaSnapshot.open(self.arena_cache_path())
        if snap is None:
            info["reason"] = "no readable cache"
            return 0
        # A cache that duplicates different things than this engine was asked to
        # duplicate is REFUSED rather than silently used: the knob decides what
        # is on disk, so flipping it has to cost one rebuild or it decides
        # nothing. The rebuilt cache is written at the end of this open.
        want_vec = "vault" if self.arena_cache_vectors.startswith("offsets") else "cache"
        if (str(snap.header.get("vec_source", "cache")) != want_vec
                or str(snap.header.get("rec_source", "cache")) != self.arena_cache_records):
            snap.close()
            info["reason"] = "cache was written for a different sidecar layout"
            return 0
        if not self._cont.accept_snapshot(snap):
            snap.close()
            self._cont.reset_scan_state()
            info["reason"] = "cache does not bind to this vault"
            return 0
        # The last binding check, and the only one added after the fact: has the
        # vault been written since the scan this cache was built from? A file
        # that grew was appended to and the tail is scanned below. A file of the
        # same length with a different mtime was rewritten IN PLACE, which is
        # exactly the shape of the case a cached open used to serve in silence
        # while `off` and `verify` raised IntegrityError -- so it is refused
        # here and the scan that follows raises. One stat; see
        # nanomem.arena.vault_changed_since_cache for what it is and is not
        # worth (it is a corruption check: `os.utime` defeats it).
        changed = _arena.vault_changed_since_cache(self.filepath, snap.header)
        info["vault_changed_since_cache"] = changed
        if changed not in ("unchanged", "grew"):
            snap.close()
            self._cont.reset_scan_state()
            info["reason"] = ("vault was rewritten in place after the cache was written"
                              if changed == "rewritten" else
                              "cache does not record the vault's size and mtime")
            return 0
        if self.arena_cache == "verify":
            t = time.perf_counter()
            # BOTH halves are checked because they cover DIFFERENT FILES, and
            # both are CHECKSUMS. Re-reading the vault's blocks catches damage to
            # the vault; the cache's content digest catches damage to the cache,
            # which re-reading the vault cannot see at all (measured: rewriting
            # one row of a cache's `vec` section moves top-1 to a planted row at
            # cosine 1.0 and the vault is still perfectly intact).
            #
            # WHAT THIS IS NOT, corrected after it was claimed to be more: this
            # digest is an unkeyed SHA-256 stored in the header of the file it
            # describes, behind a CRC32, so anyone who can edit the cache can
            # recompute both. A ~10-line forgery makes THIS branch serve the
            # planted row at cosine 1.0 -- measured, and pinned by
            # tests/test_arena_cache.py::
            # test_verify_mode_is_hijacked_by_a_forged_content_digest. Nor is
            # the vault side stronger: a plaintext vault's block trailer is an
            # unkeyed SHA-256 too (nanomem/crypto.py THREAT_MODEL) and a
            # passphrase, which would make it an HMAC, disqualifies the vault
            # from having a cache at all. `verify` detects CORRUPTION in both
            # files. It does not detect TAMPERING; in plaintext mode nothing in
            # nanomem does (reopen_results.json -> `cache_integrity`).
            want = snap.header.get("content_sha256")
            if not want or snap.content_sha256() != want:
                info["verify_s"] = round(time.perf_counter() - t, 6)
                snap.close()
                self._cont.reset_scan_state()
                info["reason"] = ("cache content does not match its own digest"
                                  if want else "cache carries no content digest")
                return 0
            ok = self._cont.reauthenticate()
            info["verify_s"] = round(time.perf_counter() - t, 6)
            if not ok:
                snap.close()
                self._cont.reset_scan_state()
                info["reason"] = "vault failed its block trailer check"
                return 0
        vault = None
        if want_vec == "vault" or self.arena_cache_records == "vault":
            vault = _arena._VaultBacking.open(
                self.filepath, snap.array("blocks"), int(self.header.embed_dim),
                int(snap.header["source_valid_end"]), fp16=bool(self.header.fp16))
            if vault is None:
                snap.close()
                self._cont.reset_scan_state()
                info["reason"] = "vault could not be mapped for an offset cache"
                return 0
        try:
            self.arena.attach_snapshot(
                snap, vault=vault,
                copy_vectors=(self.arena_cache == "copy"
                              or self.arena_cache_vectors == "offsets_ram"))
        except Exception as exc:
            if vault is not None:
                vault.close()
            snap.close()
            self._cont.reset_scan_state()
            self.arena = self._new_arena()
            info["reason"] = f"cache rejected: {exc}"
            return 0
        info["vec_source"] = want_vec
        info["rec_source"] = self.arena_cache_records
        return int(self.arena.n_rows)

    def _maybe_write_arena_cache(self, covered: int) -> None:
        """Write the cache when this open had to build rows the cache did not have.

        Deliberately NOT on the write path: an ingest never pays for this, and
        the first open after one does. A small append leaves the cache in place
        and is re-scanned at every open until the arrears pass
        ``arena_cache_refresh_rows``, because rewriting the whole cache after
        every 50-row block would cost far more than re-reading the blocks.
        """
        if self.read_only or self._arena_cache_eligible():
            return
        n = int(self.arena.n_rows)
        if n < _arena.ARENA_CACHE_MIN_ROWS:
            return
        arrears = n - int(covered)
        if arrears <= 0 or (covered and arrears < self.arena_cache_refresh_rows):
            return
        try:
            state = self._cont.snapshot_state()
            state["engine_version"] = ENGINE_VERSION
            stat = getattr(self, "_vault_stat_at_scan", None)
            if stat is not None:
                state["vault_size_at_scan"] = stat[0]
                state["vault_mtime_ns_at_scan"] = stat[1]
            wrote = _arena.ArenaSnapshot.write(
                self.arena_cache_path(), self.arena, state,
                durable=(self.durable != "none"),
                vectors=self.arena_cache_vectors,
                records=self.arena_cache_records)
            self._arena_cache_info["write"] = {
                "bytes": wrote["bytes"], "write_s": round(wrote["write_s"], 6),
                "durable": wrote["durable"],
                "sections": wrote["sections"],
                "vectors": self.arena_cache_vectors,
                "records": self.arena_cache_records}
        except Exception as exc:
            # A cache that cannot be written is a missed optimisation, never an
            # error: a read-only directory, a full disk or a refused temp file
            # must not stop a vault from opening.
            self._arena_cache_info["write_error"] = repr(exc)

    def _drop_arena_cache(self) -> None:
        """Remove the cache because the vault it described no longer exists.

        Called when the file is rewritten (``replace_all``). Unlinking is safe
        while another process has it mapped -- the inode outlives the name --
        and the binding check would refuse it anyway; this only stops a stale
        copy squatting on disk.
        """
        try:
            os.remove(self.arena_cache_path())
        except OSError:
            pass

    @staticmethod
    def _landmark_fn(L):
        if not L:
            return None
        return lambda V, LL, seq: _routing.spherical_kmeans_landmarks(V, LL, seed=seq)

    def _open_v2_readonly(self, block_capacity, landmarks_per_block):
        """Load an un-migrated v2 file into RAM as synthetic in-memory blocks."""
        from .legacy_v2 import iter_v2_records, read_v2_header, repair_entity
        embed_dim, _cd, _cap = read_v2_header(self.filepath)
        self.embed_dim = int(embed_dim)
        self.block_capacity = int(block_capacity)
        self.landmarks_per_block = int(landmarks_per_block)
        self.arena = self._new_arena()
        self._mt = Memtable(self.embed_dim, self.block_capacity)
        self._cont = None
        self._v2_header = FileHeader(embed_dim=self.embed_dim,
                                     block_capacity=self.block_capacity,
                                     landmarks_per_block=self.landmarks_per_block,
                                     created_unix=time.time(), vault_uuid=b"\x00" * 16)
        self._v2_blocks = []
        batch = []
        seq = 0
        for rec in iter_v2_records(self.filepath):
            batch.append(repair_entity(rec))
            if len(batch) >= self.block_capacity:
                seq = self._ingest_virtual_block(batch, seq)
                batch = []
        if batch:
            self._ingest_virtual_block(batch, seq)

    def _ingest_virtual_block(self, batch, seq):
        V = np.ascontiguousarray([r["embedding"] for r in batch], dtype=np.float32)
        ts, rev, ids, groups, docs = self._record_columns(batch)
        rec_bytes = _c.encode_records(ts, rev, ids, groups, docs)
        meta = _c.BlockMeta(index=len(self._v2_blocks), offset=-(seq + 1), n=len(batch),
                            m=0, payload_len=len(rec_bytes), written_unix=time.time(),
                            seq=seq, rec_section_len=len(rec_bytes),
                            first_row=self.arena.n_rows)
        self.arena.add_block(meta, None, V, rec_bytes)
        self._v2_blocks.append(meta)
        return seq + 1

    @staticmethod
    def _record_columns(records):
        ts, rev, ids, groups, docs = [], [], [], [], []
        for r in records:
            meta = r.get("metadata") or {}
            ts.append(float(r.get("timestamp", 0.0)))
            rev.append(int(r.get("revision", 1)))
            ids.append(str(r.get("id") or meta.get("id") or ""))
            groups.append(_ent.make_group_key(meta.get("user_id"), meta.get("project"),
                                              meta.get("entity")))
            docs.append(json.dumps(
                {"text": r.get("text", ""), "source": r.get("source", "unknown"),
                 "metadata": meta},
                ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
        return ts, rev, ids, groups, docs

    # -- state --------------------------------------------------------------
    @property
    def memtable(self) -> List[Dict[str, Any]]:
        """Live list of unflushed records (internal; no external caller reads it)."""
        return self._mt.items

    @property
    def container(self) -> _ContainerView:
        """Deprecated compatibility view -- ``.toc`` / ``.read_payload`` / ``.close``."""
        return self._container_view

    @property
    def header(self) -> FileHeader:
        return self._cont.header if self._cont is not None else self._v2_header

    def _blocks_meta(self):
        return self._cont.blocks if self._cont is not None else self._v2_blocks

    def count(self) -> int:
        """Flushed rows plus pending memtable rows.

        Reloads first, like every other public read: 3.0.0 did not, so ``count()``
        disagreed with ``stats()`` and ``search()`` on the same object whenever
        another instance had appended.
        """
        with self._lock:
            self._reload_if_modified()
            return int(self.arena.n_rows + self._mt.n)

    # -- reload -------------------------------------------------------------
    def _reload_if_modified(self) -> None:
        if self._cont is None or self._closed:
            return
        state = self._cont.modified()
        if state == "same":
            self._file_missing = False
            return
        if state == "missing":
            self._file_missing = True
            return
        self._file_missing = False
        if state == "appended":
            try:
                self._cont.scan(self.arena)
            except IntegrityError:
                raise
            self._invalidate_router()
        else:
            self._full_reload()

    def _full_reload(self) -> None:
        self._cont.close()
        self._cont = Container(self.filepath, embed_dim=self.embed_dim,
                               password=self._password,
                               vector_dtype=self.header.vector_dtype,
                               block_capacity=self.block_capacity,
                               landmarks_per_block=self.landmarks_per_block,
                               durable=self.durable,
                               on_integrity_error=self.on_integrity_error,
                               on_torn_tail=self.on_torn_tail,
                               readonly=self.read_only)
        self.arena = self._new_arena()
        self._load_arena()
        self._reintern_memtable()
        self._invalidate_router()
        self._maybe_engage_router()

    def _reintern_memtable(self) -> None:
        """Re-resolve pending records against the NEW arena's intern tables.

        ``entity_id`` / ``group_id`` are indexes into ``arena.entity_names`` /
        ``arena.group_keys``. A reload rebuilds those tables from disk, so any id
        cached on a memtable item silently comes to mean a different entity (3.0.0
        did exactly that: two chat records tagged ``phone_number`` and
        ``credential`` both ended up as entity 2). The keys are the durable
        identity, so they are what the memtable stores; the ids are re-derived
        here, and ``note_group`` is replayed so the auto-revision counter does not
        restart at 1.
        """
        for item in self._mt.items:
            gid = self.arena.intern_group(item.get("group_key") or "",
                                          bool(item.get("entity_declared")))
            item["group_id"] = gid
            item["entity_id"] = self.arena.intern_entity(item.get("entity_key") or "")
            self.arena.note_group(gid, int(item.get("revision", 1)),
                                  float(item.get("timestamp", 0.0)))

    # -- write path ---------------------------------------------------------
    def add_fact(self, text: str, embedding, source: str = "user_input",
                 metadata: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[float] = None, revision: Optional[int] = None,
                 id: Optional[str] = None) -> str:
        """Store one record and return its document id.

        O(1) amortised and never reads a block. The policy layer -- chat entity
        tagging, anaphora inheritance and auto-revision -- runs ONLY when
        ``revision is None``; a caller that passes an explicit revision (every
        rebuild, ``add_batch`` of existing records, ``Vault.merge``) gets a
        verbatim store, which is what makes a rebuild idempotent.
        """
        with self._lock:
            self._assert_open("add to")
            if self.read_only:
                raise ReadOnlyVaultError(f"{self.filepath} is open read-only")
            self._reload_if_modified()
            meta = dict(metadata or {})
            # WHO NAMED THE ENTITY. `meta["entity"]` alone cannot answer it: on
            # a replay or rebuild the stored metadata carries an entity the
            # TAGGER inferred on the original write, and reading that as a
            # declaration would hand an inferred group the treatment only a
            # declared one has earned. The marker is authoritative when present;
            # a live caller naming the entity sets it; a record written before
            # the marker existed reads as inferred, which is what the engine did
            # before this and so leaves an old vault ranking exactly as it did.
            declared = (bool(meta.get("entity_declared"))
                        if "entity_declared" in meta
                        else bool(revision is None and meta.get("entity")))
            if declared:
                meta["entity_declared"] = True
            ts = float(timestamp if timestamp is not None else meta.get("timestamp", time.time()))
            doc_id = str(id or meta.get("id") or
                         f"doc_{hashlib.md5(f'{text}_{ts}'.encode()).hexdigest()[:10]}")
            meta["id"] = doc_id
            v = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if v.size != self.embed_dim:
                raise ValueError(
                    f"embedding has {v.size} dims, vault at {self.filepath} has {self.embed_dim}")
            nrm = float(np.linalg.norm(v))
            if not np.isfinite(nrm) or nrm <= 0.0:
                # STORED, AND THEN UNREACHABLE. 3.0.2 normalised a non-finite
                # vector to all-NaN and wrote it: `search` masks it out (NaN fails
                # `base >= min_score`) so the record could never be returned, while
                # `search_batch` returned it with a NaN score. The write path now
                # refuses what the query path already refuses.
                raise ValueError(
                    f"embedding must be finite and non-zero (norm={nrm}); "
                    f"nanomem will not store a record it could never return")
            if not np.isfinite(ts):
                # A NaN timestamp survives into every search result dict, and
                # json.dumps then emits the non-standard token NaN -- which breaks
                # `search()`'s promise that its values round-trip through JSON and
                # produces a malformed body from server.py.
                raise ValueError(f"timestamp must be finite, got {ts}")
            v = v / nrm

            uid = meta.get("user_id")
            if revision is None:
                is_chat = (source in self.personal_sources
                           or meta.get("source") in self.personal_sources)
                if is_chat and not meta.get("entity"):
                    ent = _ent.detect_entity(text)
                    if ent is None and _ent.is_pronoun_led(text) and self._last_entity.get(uid):
                        # WHOSE fact is the follow-up about? `entities` draws that
                        # boundary everywhere else -- `detect_entity` tags somebody
                        # else's attribute `other_<x>` and `entities_match` refuses
                        # to merge across the prefix -- and 3.0.2 crossed it right
                        # here, inheriting the previous subject verbatim. Two chat
                        # turns ("my phone number is ...", then a contact's) put the
                        # contact's number in the USER's revision group as revision
                        # 2, where it won rank 1 for "what is my phone number?" on
                        # the lower cosine. `inherit_entity` takes ownership from
                        # the follow-up when it states one.
                        # Measured: evidence/third_party_v3r4.json.
                        ent = _ent.inherit_entity(self._last_entity[uid], text)
                        if ent:
                            meta["anaphora_resolved"] = True  # metadata only; vector untouched
                    if ent:
                        meta["entity"] = ent
                if meta.get("entity"):
                    self._last_entity[uid] = meta["entity"]
                    gkey = _ent.make_group_key(uid, meta.get("project"), meta["entity"])
                    gid = self.arena.intern_group(gkey, declared)
                    rev = int(self.arena.group_max.get(gid, (0, 0.0))[0]) + 1
                else:
                    rev = int(meta.get("revision", 1))
            else:
                rev = int(revision)
            if not (0 <= rev <= 2_147_483_647):
                # The revision column is int32 on disk. 3.0.1 accepted anything,
                # returned a document id, and then raised OverflowError on the
                # NEXT spill -- from which the memtable never recovered, so every
                # later add_fact, flush() and close() raised the same error and
                # the pending records could never be written at all.
                raise ValueError(
                    f"revision must fit in an int32 (0..2147483647), got {rev}")

            gkey = _ent.make_group_key(uid, meta.get("project"), meta.get("entity"))
            ekey = _ent.normalize_entity(meta.get("entity"))
            gid = self.arena.intern_group(gkey, declared)
            eid = self.arena.intern_entity(ekey)
            self._mt.append({"id": doc_id, "text": text, "source": source,
                             "embedding": v, "metadata": meta, "timestamp": ts,
                             "revision": rev, "entity_id": eid, "group_id": gid,
                             "entity_key": ekey, "group_key": gkey,
                             "entity_declared": declared}, v)
            self.arena.note_group(gid, rev, ts)
            if self._mt.n >= self.block_capacity:
                self._spill()
            return doc_id

    def _spill(self) -> None:
        batch = self._mt.items[:self.block_capacity]
        if not batch:
            return
        if self._file_missing or not os.path.exists(self.filepath):
            raise FileNotFoundError(f"vault file was removed: {self.filepath}")
        V = np.ascontiguousarray(self._mt.vec[:len(batch)], dtype=np.float32)
        ts, rev, ids, groups, docs = self._record_columns(batch)

        built = {"lm": None}

        def make_blob(seq):
            """Seal the block for the position it will actually occupy.

            Called by ``append_block`` under the exclusive lock, after any other
            process's blocks have been read back, so ``seq`` is the real position
            even when several processes are appending to one vault.
            """
            lm = None
            if self.landmarks_per_block and len(batch) > self.landmarks_per_block:
                lm = _routing.spherical_kmeans_landmarks(V, self.landmarks_per_block,
                                                         seed=seq)
            built["lm"] = lm
            return _c.build_block_blob(self._cont.header, self._cont.keys, seq, V, lm,
                                       ts, rev, ids, groups, docs)

        try:
            meta = self._cont.append_block(make_blob, self.arena)
        except ContainerReplacedError:
            self._full_reload()
            meta = self._cont.append_block(make_blob, self.arena)
        lm = built["lm"]
        rec_bytes = _c.encode_records(ts, rev, ids, groups, docs)
        self.arena.add_block(meta, lm, V, rec_bytes)
        self._mt.drop_front(len(batch))
        self._invalidate_router()

    def _assert_open(self, what: str) -> None:
        """Reject a write to a closed engine. See :meth:`Container._assert_open`."""
        if self._closed:
            raise ClosedVaultError(
                f"cannot {what} {self.filepath}: the vault is closed")

    def flush(self) -> None:
        """Spill every pending record to disk. Idempotent on an empty memtable."""
        with self._lock:
            if self._mt.n:
                self._assert_open("flush")
            if self.read_only and self._mt.n:
                raise ReadOnlyVaultError(f"{self.filepath} is open read-only")
            if self._mt.n:
                self._reload_if_modified()
            while self._mt.n:
                self._spill()
            if self.durable == "none" and self._cont is not None:
                self._cont.sync()
            self._maybe_engage_router()

    # -- read path ----------------------------------------------------------
    def _row_record(self, row: int) -> Dict[str, Any]:
        if row >= MT_BASE:
            item = self._mt.items[row - MT_BASE]
            return {"text": item["text"], "source": item["source"],
                    "metadata": item["metadata"]}
        return self.arena.record(row)

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Fetch one record by id (O(1) via the id index), or ``None``."""
        with self._lock:
            self._reload_if_modified()
            key = str(id)
            row = self.arena.id_index.get(key)
            if row is not None:
                rec = self.arena.record(row)
                return {"id": key, "text": rec.get("text", ""),
                        "source": rec.get("source", "unknown"),
                        "metadata": dict(rec.get("metadata") or {}),
                        "timestamp": float(self.arena.ts[row]),
                        "revision": int(self.arena.rev[row]),
                        "embedding": self.arena.vector(row)}
            for item in self._mt.items:
                if item["id"] == key:
                    return {"id": key, "text": item["text"], "source": item["source"],
                            "metadata": dict(item["metadata"]),
                            "timestamp": float(item["timestamp"]),
                            "revision": int(item["revision"]),
                            "embedding": np.array(item["embedding"], copy=True)}
            return None

    def iter_records(self, include_embeddings: bool = True) -> Iterator[Dict[str, Any]]:
        """Flush, then yield every record in file order as owned copies.

        Each row is read under the lock. Capturing the row COUNT under the lock
        and then reading ``self.arena`` outside it is a latent ``IndexError``: a
        concurrent ``_full_reload`` rebinds the arena mid-iteration.
        """
        with self._lock:
            if not self.read_only:
                self.flush()
            n = self.arena.n_rows
        for r in range(n):
            with self._lock:
                if r >= self.arena.n_rows:
                    return
                rec = self.arena.record(r)
                out = {"id": self.arena.ids[r], "text": rec.get("text", ""),
                       "source": rec.get("source", "unknown"),
                       "metadata": dict(rec.get("metadata") or {}),
                       "timestamp": float(self.arena.ts[r]),
                       "revision": int(self.arena.rev[r])}
                if include_embeddings:
                    out["embedding"] = self.arena.vector(r)
            yield out

    def _read_block_payload(self, offset: int, length: int, doc_count: int,
                            block_id=None) -> Dict[str, Any]:
        with self._lock:
            self._reload_if_modified()
            block = None
            for b in self._blocks_meta():
                if int(b.offset) == int(offset):
                    block = b
                    break
            if block is None and block_id is not None:
                for b in self._blocks_meta():
                    if b.block_id == block_id:
                        block = b
                        break
            if block is None:
                raise KeyError(f"no block at offset {offset} in {self.filepath}")
            lo = int(block.first_row)
            hi = lo + int(block.n)
            texts, sources, metas = [], [], []
            for r in range(lo, hi):
                rec = self.arena.record(r)
                meta = dict(rec.get("metadata") or {})
                meta.setdefault("id", self.arena.ids[r])
                texts.append(rec.get("text", ""))
                sources.append(rec.get("source", "unknown"))
                metas.append(meta)
            return {"texts": texts, "sources": sources, "metadatas": metas,
                    "timestamps": self.arena.ts[lo:hi].copy(),
                    "revisions": self.arena.rev[lo:hi].copy(),
                    "values": self.arena.matrix(lo, hi).copy()}

    # -- routing ------------------------------------------------------------
    @property
    def _reclustered(self) -> bool:
        """Is the row order on disk already the k-means layout the router wants?"""
        h = self.header
        return bool(h is not None and getattr(h, "reclustered", False))

    def _invalidate_router(self) -> None:
        self._router = None

    def _maybe_engage_router(self) -> None:
        """Build (and ONCE re-cluster for) the opt-in cell router.

        The re-cluster is what DECISIONS.md mandates before any routing, because a
        random insertion order makes a *block*-based router unusable (measured:
        39.4 vs 62.1 recall@4 at a 15% budget on 71k paragraphs). The shipped
        ``GlobalCellRouter`` builds its own cells over the vectors, so for IT the
        benefit of the re-cluster is memory locality of the candidate gather, not
        recall; the cost is one full rewrite (measured ~3 s at 71k docs), logged
        by ``compact()``.

        THE RE-CLUSTER IS NOT THE EXPENSIVE PART. The k-means fit below is, and
        it is paid on every open, not once per file, because the container header
        has nowhere to store centroids. Measured at 71,433 documents
        (``evidence/router_gate_results.json``, quiet machine): opening a
        ``router="off"`` vault takes 0.193 s; the same vault with
        ``router="auto"`` at the shipped budget takes 13.982 s, and at the
        cheapest gate-passing budget (``cell_target=12``) 25.796 s, because a
        smaller ``cell_target`` means more cells to fit. Ingest goes 1.281 s ->
        16.414 s / 43.675 s, and peak RSS 849.0 MB -> 1818.8 MB / 1906.5 MB. That
        is the bill for a per-query saving of at most 0.40 ms, and it is why the
        default is "off" even though the recall gate passed. Persisting the
        centroids is what would change this answer.

        ONCE means once per FILE, not once per process. "Already reclustered" is a
        header flag (``container.FLAG_RECLUSTERED``), so re-opening a reclustered
        vault is an ordinary open. 3.0.2 kept it in an attribute initialised to
        ``False``, so every open rewrote the whole file, rotated ``vault_uuid``
        and changed the inode: three successive opens produced three inodes, and
        4 concurrent ``router="auto"`` opens of one existing vault failed 9 times
        out of 12 with ``ContainerReplacedError`` raised from inside the
        constructor (measured: evidence/router_persist_v3r4.json).
        """
        if self.router_mode != "auto" or self.read_only or self._engaging:
            return
        n = self.arena.n_rows
        if n <= self.n_exhaustive:
            return
        self._engaging = True                      # compact() re-enters flush()
        try:
            if not self._reclustered:
                try:
                    self.compact(recluster=True)
                except ContainerReplacedError:
                    # Several processes can reach the crossing point together and
                    # only one of them can win the rewrite. Losing it is not an
                    # error: adopt the winner's layout. (The router clusters
                    # VECTORS, not blocks, so it still works on an un-reclustered
                    # layout -- 77.10 recall@4 on the RANDOM layout against 77.05
                    # on the reclustered one, same budget, same 71k corpus:
                    # evidence/router_gate_results.json,
                    # phase_A_random_layout_control -- which is why this degrades
                    # instead of raising out of a constructor, as 3.0.2 did for 9
                    # of 12 concurrent opens.)
                    self._full_reload()
                    if not self._reclustered:
                        warnings.warn(
                            f"nanomem: another writer is re-clustering "
                            f"{self.filepath}; routing this instance on the "
                            f"current layout.", RuntimeWarning, stacklevel=2)
        finally:
            self._engaging = False
        if self._router is None or self._router_rows != self.arena.n_rows:
            self._router = _routing.GlobalCellRouter(
                cell_target=self.cell_target, beam_frac=self.beam_frac,
                beam_min_cells=self.beam_min_cells).fit(self.arena.matrix())
            self._router_rows = self.arena.n_rows

    # -- exactness-preserving screen ----------------------------------------
    def _screen_ready(self):
        """The screen for the CURRENT arena, built or extended as needed.

        Returns ``None`` -- meaning "run the ordinary full exact scan" -- for
        every reason a screen could be unusable, so a missing, stale or
        unbuildable basis costs latency and never correctness (the flag's
        automatic-fallback contract).

        THE STALENESS ARGUMENT, in full, because this is where an exactness
        claim would be lost if it were wrong:

        * The bound is admissible for ANY basis (:mod:`nanomem.screen`), so a
          basis fitted on the first N rows still bounds row N+1 correctly. Rows
          appended after the fit are projected with the existing basis and
          appended to the screen; nothing is refitted and nothing can go stale
          in a way that matters.
        * What WOULD be fatal is a screen whose row *i* no longer describes the
          arena's row *i*. Every operation that can do that -- ``_full_reload``,
          ``replace_all``, ``compact``, ``reset`` -- replaces ``self.arena``
          with a NEW :class:`~nanomem.arena.Arena` object, so the identity test
          below catches all of them at once and rebuilds. Everything that keeps
          the same arena object (``_spill``, an appended-tail ``Container.scan``)
          only ever APPENDS rows: :meth:`Arena.add_block` writes at
          ``n_rows`` and no code path rewrites a row in place. So
          ``screen.n_rows <= arena.n_rows`` with the same object is exactly the
          "some rows are new" case, and extending is sound.
        * ``int8`` residency is refused outright: there ``Arena.scores`` is not
          a cosine below its re-rank pool, so there is no exact scan for the
          screen to agree with.
        """
        if self.screen_mode != "pca":
            return None
        if self.residency == "int8":
            return None
        n = self.arena.n_rows
        if n < max(1, self.screen_min_rows):
            return None
        scr = self._screen
        if scr is not None and self._screen_arena is not self.arena:
            scr = None                                    # rows were renumbered
        if scr is not None and scr.embed_dim != self.embed_dim:
            scr = None
        if scr is not None and scr.n_rows > n:
            scr = None                                    # arena shrank: rebuild
        try:
            if scr is None:
                scr = _screen.PCAScreen(self.embed_dim, d_out=self.screen_dims,
                                        seed_pool=self.screen_seed_pool)
                scr.fit(self._arena_chunks(0, n))
                scr.reserve(n)
                for lo, hi in self._arena_ranges(0, n):
                    scr.extend(self.arena.matrix(lo, hi))
            elif scr.n_rows < n:
                for lo, hi in self._arena_ranges(scr.n_rows, n):
                    scr.extend(self.arena.matrix(lo, hi))
        except Exception:                                 # pragma: no cover
            # A screen is an optimisation. Anything that goes wrong building one
            # falls back to the exact scan rather than failing a search.
            self._screen = None
            self._screen_arena = None
            return None
        if scr.n_rows != n:
            self._screen = None
            self._screen_arena = None
            return None
        if self._screen_gather_exact is None or self._screen is not scr:
            self._screen_pad, self._screen_gather_exact = self._calibrate_gather()
        if not self._screen_gather_exact:
            # This machine's BLAS does not give a gathered sub-scan the same
            # arithmetic as the full scan at any pad size we are willing to pay
            # for, so the screen cannot promise identical scores. It declines.
            self._screen = scr
            self._screen_arena = self.arena
            return None
        self._screen = scr
        self._screen_arena = self.arena
        return scr

    #: Row counts tried when calibrating the survivor gather, smallest first.
    _GATHER_LADDER = (16, 32, 64, 128, 256, 512)

    def _calibrate_gather(self):
        """Smallest gather size whose scores are BITWISE the full scan's.

        WHY THIS EXISTS. ``Arena.scores`` scans every row; the screen scores only
        the survivors, through ``Arena.scores_rows``. Both are fp32 dot products
        of the same two vectors, but BLAS picks its kernel by the number of
        ROWS, and MEASURED on this box (Apple Accelerate, numpy 2.5.3, 768
        dims): a gather of 1-8 rows differs from the full scan in the last bit
        (5.96e-08) while a gather of 16 or more is bitwise identical, at both
        12,000 and 71,433 rows, on every trial. One ULP is far below anything a
        ranking can see -- the ids were identical -- but "the same answer" is
        this flag's whole promise, and a score that differs in its last bit is
        not the same answer.

        So the engine MEASURES the threshold on the machine it is running on
        instead of hard-coding 16, and pads every survivor gather up to it with
        the next-best rows by upper bound (free: the bounds are already
        computed, and a superset of the survivors is still exact). If no size on
        the ladder reproduces the full scan, the screen declines to engage and
        the vault runs the ordinary exact scan -- correct, just not faster.
        """
        n = self.arena.n_rows
        if n < self._GATHER_LADDER[0]:
            return 0, False
        rng = np.random.default_rng(0)
        probes = [self.arena.vector(0)]
        for _ in range(2):
            v = rng.standard_normal(self.embed_dim).astype(np.float32)
            probes.append(v / (np.linalg.norm(v) + 1e-12))
        probes = [np.ascontiguousarray(p / (np.linalg.norm(p) + 1e-12),
                                       dtype=np.float32) for p in probes]
        fulls = [self.arena.scores(p) for p in probes]
        for pad in self._GATHER_LADDER:
            if pad > n:
                break
            ok = True
            for p, full in zip(probes, fulls):
                for trial in range(2):
                    rows = np.sort(np.random.default_rng(trial).choice(
                        n, pad, replace=False)).astype(np.int64)
                    if not np.array_equal(self.arena.scores_rows(rows, p),
                                          np.asarray(full)[rows]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return int(pad), True
        return 0, False

    def _arena_ranges(self, lo: int, hi: int, step: int = 8192):
        for s in range(int(lo), int(hi), int(step)):
            yield s, min(s + int(step), int(hi))

    def _arena_chunks(self, lo: int, hi: int, step: int = 8192):
        for a, b in self._arena_ranges(lo, hi, step):
            yield self.arena.matrix(a, b)

    def _ranking_is_inert(self, temporal_direction, personal: bool) -> bool:
        """True when this search is a pure cosine ranking, so a cosine-exact
        candidate set is a result-exact candidate set.

        The screen prunes on COSINE. The engine's entity/temporal layer can add
        up to ``stats()['max_boost']`` to a candidate and can reorder a revision
        group, and a pruned row cannot be brought back by either -- so the
        screen may only engage where neither can fire. Both gates are read from
        :mod:`nanomem.entities` here rather than reimplemented:

        * ``personal`` is :meth:`_has_personal_records`, and
          :meth:`_max_boost` returns exactly ``0.0`` when it is false, so no
          boost exists to change an order;
        * :meth:`_resolve_revisions` returns its input untouched on a
          non-personal corpus unless the CALLER named a historical direction:
          with ``gate_on_corpus_only`` false it stops at
          ``temporal_question(..., personal_corpus=False)``, which
          :func:`entities.temporal_question` reduces to
          :func:`entities.is_explicit_history`; with it true it stops at the
          ``not personal and not explicit`` guard, the same predicate.

        So ``not personal and not is_explicit_history(direction)`` is exactly
        the region where ``final == base == monotone(cosine)``.
        """
        if personal:
            return False
        return not _ent.is_explicit_history(temporal_direction)

    def _screen_candidates(self, q, top_k: int):
        """``(rows, cos)`` for a provably sufficient superset, or ``None``.

        PROOF THAT THE RESULT IS THE EXHAUSTIVE RESULT. ``ub[i] >= cos[i]`` for
        every arena row (:mod:`nanomem.screen`). ``tau`` is the ``k``-th best
        EXACT cosine of a subset of the arena, so ``tau <= T``, the ``k``-th
        best exact cosine of the whole arena. Any row dropped here has
        ``cos[i] <= ub[i] < tau <= T``, so it is strictly worse than ``k`` rows
        that were kept and cannot appear in a top-``k`` ranked on cosine. The
        caller only reaches this method when :meth:`_ranking_is_inert` says the
        final ranking IS cosine (up to the strictly monotone ``legacy_score``),
        and ``min_score`` is likewise monotone in cosine, so the kept set, the
        order and the scores are the exhaustive ones. Survivors are scored by
        :meth:`Arena.scores_rows`, the same fp32 kernel the full scan uses.
        """
        scr = self._screen_ready()
        if scr is None:
            return None
        n = self.arena.n_rows
        self._screen_calls += 1
        ub = scr.bounds(q)
        k = max(1, int(top_k))
        m0 = int(min(n, max(k, scr.seed_pool)))
        if m0 >= n:
            self._screen_fallbacks += 1
            return None
        seed = np.argpartition(-ub, m0 - 1)[:m0]
        seed_cos = self.arena.scores_rows(seed, q)
        kk = int(min(k, seed_cos.size))
        tau = float(-np.partition(-seed_cos, kk - 1)[kk - 1])
        rows = scr.survivors(ub, tau)
        pad = self._screen_pad
        if 0 < rows.size < pad <= n:
            # Pad up to the calibrated gather size with the next-best rows by
            # UPPER BOUND. Every survivor has a strictly larger bound than every
            # non-survivor, so the top-`pad` rows by bound contain all of them:
            # the scored set stays a superset and the answer stays exact.
            rows = np.sort(np.argpartition(-ub, pad - 1)[:pad]).astype(np.int64)
        if rows.size >= n * self.screen_max_frac:
            # Gathering this many rows costs more than the scan it replaces.
            # Still exact -- just not worth it for this query.
            self._screen_fallbacks += 1
            return None
        self._screen_engaged += 1
        self._screen_survivors += int(rows.size)
        return rows.astype(np.int64, copy=False), self.arena.scores_rows(rows, q)

    def build_screen(self) -> Dict[str, Any]:
        """Build the search screen now instead of on the next search.

        No-op returning ``{"built": False, ...}`` when ``screen="off"``, when the
        arena is below ``screen_min_rows`` or under ``int8`` residency.
        """
        with self._lock:
            self._reload_if_modified()
            scr = self._screen_ready()
            if scr is None:
                return {"built": False, "mode": self.screen_mode,
                        "rows": int(self.arena.n_rows),
                        "min_rows": int(self.screen_min_rows)}
            out = {"built": True, "mode": self.screen_mode}
            out.update(scr.stats())
            return out

    def screen_info(self) -> Dict[str, Any]:
        """What the screen did in this process: engagements, fallbacks, cost."""
        with self._lock:
            scr = self._screen
            info = {
                "mode": self.screen_mode,
                "min_rows": int(self.screen_min_rows),
                "max_frac": float(self.screen_max_frac),
                "built": scr is not None,
                "calls": int(self._screen_calls),
                "engaged": int(self._screen_engaged),
                "fell_back": int(self._screen_fallbacks),
                "mean_survivors": (float(self._screen_survivors) / self._screen_engaged
                                   if self._screen_engaged else 0.0),
                "gather_pad": int(self._screen_pad),
                "gather_bitwise_exact": self._screen_gather_exact,
            }
            if scr is not None:
                info.update(scr.stats())
            return info

    def _mode(self, metadata_filter, force_exact: bool = False) -> str:
        if metadata_filter is not None or force_exact:
            return "exhaustive"
        if self.router_mode == "auto" and self._router is not None \
                and self.arena.n_rows > self.n_exhaustive:
            return "cells"
        return "exhaustive"

    def layout_gate(self) -> Dict[str, Any]:
        """Measured nearest-neighbour coverage of the current router beam."""
        with self._lock:
            if self._router is None or self.arena.n_rows < 2:
                return {"coverage": None, "passed": None,
                        "checked_at_rows": int(self.arena.n_rows)}
            cov = _routing.nn_coverage(self.arena.matrix(), self._router,
                                       sample=min(256, self.arena.n_rows))
            return {"coverage": float(cov), "passed": bool(cov >= 0.90),
                    "checked_at_rows": int(self.arena.n_rows)}

    # -- search -------------------------------------------------------------
    def search(self, query_text: str, query_vec, top_k: int = 3,
               metadata_filter: Optional[Dict[str, Any]] = None,
               min_score: float = 0.0,
               temporal_direction: str = "current",
               as_of: Optional[float] = None) -> List[Dict[str, Any]]:
        """Rank stored records against a query vector.

        Returns at most ``top_k`` fresh dicts sorted by ``score`` descending, with
        keys ``id, doc_id, text, source, metadata, score, cosine, timestamp,
        revision`` and pure-Python values (so ``json.dumps`` works).

        ``score`` is ``cosine`` plus the documented entity boosts, and the bound
        is ``stats()['max_boost']`` (``intent_boost + group_hoist +
        revision_lead + temporal_prior`` = +1.10 at the shipped defaults):
        ``0 <= hit['score'] - hit['cosine'] <= stats()['max_boost']`` for every
        hit. Boosts apply only to a personal-memory question against a vault that
        holds tagged personal records. (This docstring said +0.50 through 3.0.2
        while the engine really applied up to 0.70 -- an independent 19,516-hit
        fuzz measured +0.5599 against the promised 0.50.)

        THAT BOUND HOLDS ONLY IN THE DEFAULT ``score_mode="cosine"``. Under
        ``score_mode="legacy"`` the base is :func:`legacy_score`,
        ``0.6*c + 0.4*sign(c)*|c|^13``, which is BELOW ``c`` for every
        ``|c| < 1`` -- so ``score - cosine`` is NEGATIVE for essentially every
        hit and the lower clause does not hold at all. It is a change of SCALE,
        not a boost, and the difference is not comparable to ``max_boost``. This
        was stated here without exception through 3.3.0, and an interaction
        audit measured 43 violations in a 50-hit sample. A caller on the legacy
        scale must compare against legacy-calibrated thresholds or convert one
        with :func:`legacy_score_to_cosine`; ``score_mode="legacy"`` exists for
        that compatibility alone and is not recommended for new code.

        ``min_score`` is compared against the PRE-boost score, which is the v2
        rule. Unflushed records are included.

        ``temporal_direction`` says WHICH revision of a fact the caller wants,
        and only two kinds of value are an instruction:

        * ``"historical"`` (the v2/3.0.x spelling, unchanged), ``"oldest"`` or
          ``"previous"`` -- the caller is asking for a superseded value, and the
          last two name which one; and
        * ``"present"`` / ``"newest"`` / ``"latest"`` -- the newest value,
          whatever the question's wording says.

        ``"current"``, the signature default, means UNSPECIFIED, and the
        direction is then read off the question itself
        (:func:`entities.query_direction`): "what was my ORIGINAL address?" is
        answered with the first value, "the one just BEFORE this one" with the
        one before the newest, and anything else with the newest. Through 3.0.3
        the default was read as an instruction, so every caller that does not
        pass the argument -- ``chat.py``, ``cli.py``, ``Vault.search`` -- got the
        CURRENT value for a question that explicitly asked for an earlier one.
        Measured on the 326-question temporal benchmark: 9.0% top-1 on historical
        questions, against 22.0% for a plain cosine scan with no temporal logic
        at all (``evidence/temporal_bench_results.json``). Inference
        happens behind the same gates as the rest of the layer, so a document
        corpus is still ranked on pure cosine however a question is worded.

        A ``metadata_filter`` disables routing and walks candidates in score
        order until no unexamined candidate could still reach the top ``top_k``
        even with the maximum boost -- so the filtered result is exactly the
        filtered exhaustive result. (3.0.0 stopped after ``max(64, 8*top_k)``
        matches measured on cosine ALONE, which dropped boosted records that
        should have led.)
        """
        with self._lock:
            self._reload_if_modified()
            if top_k is None or int(top_k) <= 0:
                return []
            top_k = int(top_k)
            q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
            if q.size != self.embed_dim:
                raise ValueError(
                    f"query has {q.size} dims, vault at {self.filepath} has {self.embed_dim}")
            nrm = float(np.linalg.norm(q))
            if not np.isfinite(nrm) or nrm <= 0.0:
                return []
            q = q / nrm

            personal = self._has_personal_records()
            intent, intent_ids = self._resolve_intent(query_text) if personal else (None, None)

            screen_ok = (self.screen_mode == "pca"
                         and metadata_filter is None
                         and as_of is None
                         and self.arena.n_rows >= self.screen_min_rows
                         and self._ranking_is_inert(temporal_direction, personal))
            rows, cos = self._candidates(q, metadata_filter, intent_ids,
                                         top_k=top_k, screen_ok=screen_ok,
                                         force_exact=as_of is not None)
            if rows.size == 0:
                return []

            ent, rv, tsv = self._columns(rows)
            base = legacy_score(cos) if self.score_mode == "legacy" else cos.astype(np.float64)
            mask = base >= float(min_score)
            if as_of is not None:
                # EXACT BY CONSTRUCTION, not by filtering a shortlist. `as_of`
                # forced `_candidates` exhaustive above, so `rows` is every row
                # (engine.py `_candidates`: `rows = np.arange(N)`) plus the
                # memtable, and masking here is therefore identical to running
                # this engine against a vault holding only the rows at or before
                # `as_of`. Masking a ROUTED or SCREENED shortlist would not be:
                # a newer record can crowd an older one out of the shortlist
                # before the mask is ever applied, and the older one is exactly
                # what an as-of query is asking for.
                mask = mask & (tsv <= float(as_of))
            if not mask.any():
                return []
            # A REJECTED INTENT MUST NOT BOOST EITHER.
            # Correcting `top_entity` alone fixed `history` and left `search`
            # wrong: `intent_boost` is applied from `intent_ids` here, before
            # `_resolve_revisions` runs, so a +0.25 still landed on the records
            # of an intent the margin had already judged untrustworthy. The test
            # runs once, and a failing intent is dropped for everything
            # downstream -- the boost, the group, and the revision layer.
            if (self.intent_margin > 0.0 and intent_ids is not None
                    and np.size(intent_ids)):
                mine = mask & np.isin(ent, intent_ids)
                if mine.any() and mask.any():
                    gap = float(cos[mask].max()) - float(cos[mine].max())
                    if gap > self.intent_margin:
                        intent, intent_ids = None, None
            max_boost = self._max_boost(intent_ids, personal)
            if metadata_filter is not None:
                mask = self._apply_filter(rows, base, mask, metadata_filter,
                                          top_k, max_boost)
                if not mask.any():
                    return []

            boost = _ent.apply_intent_boost(cos, ent, intent_ids, mask, self.intent_boost)
            final = np.where(mask, base + boost, -np.inf)
            final = self._resolve_revisions(final, rows, cos, ent, rv, tsv, mask,
                                            query_text, temporal_direction, intent,
                                            personal, base)

            k = int(min(top_k, int(mask.sum())))
            if k <= 0:
                return []
            idx = _select_top_k(final, rows, k)

            hits = []
            for i in idx:
                r = int(rows[i])
                rec = self._row_record(r)
                doc_id = (self.arena.ids[r] if r < MT_BASE
                          else self._mt.items[r - MT_BASE]["id"])
                hits.append({"id": str(doc_id), "doc_id": str(doc_id),
                             "text": rec.get("text", ""),
                             "source": rec.get("source", "unknown"),
                             "metadata": dict(rec.get("metadata") or {}),
                             "score": float(final[i]), "cosine": float(cos[i]),
                             "timestamp": float(tsv[i]), "revision": int(rv[i])})
            return hits

    def _resolve_intent(self, query_text):
        """Pick the question's intent USING THE VAULT'S OWN ENTITY TABLE.

        ``entities.query_intents`` returns every attribute a question could be
        asking about, best guess first; a question that names two ("which <A>
        should go on my <B> now?") is ambiguous on wording alone. The candidate
        this vault actually holds records for is the one it can be asking about,
        so ambiguity is resolved from DATA rather than from another rule. 3.0.2
        took the first guess unconditionally and answered `<B>`.
        """
        cands = _ent.query_intents(query_text, self._names_the_owner(query_text))
        if not cands:
            return None, None
        names = self.arena.entity_names
        for c in cands:
            ids = _ent.matching_ids(c, names)
            if ids.size:
                return c, ids
        return cands[0], _ent.matching_ids(cands[0], names)

    def _names_the_owner(self, query_text) -> bool:
        """Is this third-person question actually about the vault's OWNER?

        "What is <owner> allergic to?" and "What keyboard does <owner> use?" are
        first-person questions written in the third person. They carry no
        possessive, so neither ``is_personal_query`` nor ``is_third_party`` fires
        and 3.0.2's whole entity/temporal layer stayed silent for them -- the
        measured boost was +0.000 and search fell back to raw cosine, which
        reliably returns the "my name is <owner>" record because that row is the
        only one carrying the name.

        The answer comes from the VAULT, not from a list of names: whatever the
        owner's own ``name``-tagged records say their name is. A document corpus
        has no ``name`` record, so this can never fire there.
        """
        toks = self._self_name_tokens()
        if not toks:
            return False
        low = _ent._norm_text(query_text).lower()
        return bool(toks & set(re.findall(r"[a-z\u00c0-\u024f]{3,}", low)))

    def _self_name_tokens(self) -> frozenset:
        """Cached tokens of the owner's own name, from records tagged ``name``.

        Rebuilt whenever the row count changes; ``name`` records are a handful at
        most, so this is a few record decodes once per vault generation.
        """
        n = int(self.arena.n_rows) + int(self._mt.n)
        if self._self_names is not None and self._self_names_rows == n:
            return self._self_names
        toks = set()
        names = self.arena.entity_names
        want = [i for i, nm in enumerate(names)
                if _ent.normalize_entity(nm) == "name"]
        if want and self.arena.n_rows:
            col = self.arena.entity_id[:self.arena.n_rows]
            for r in np.flatnonzero(np.isin(col, np.asarray(want, dtype=np.int32))):
                toks |= _ent.self_name_tokens(
                    self.arena.record(int(r)).get("text", ""))
        for it in self._mt.items:
            if _ent.normalize_entity(it.get("entity_key")) == "name":
                toks |= _ent.self_name_tokens(it["text"])
        self._self_names = frozenset(toks)
        self._self_names_rows = n
        return self._self_names

    def _has_personal_records(self) -> bool:
        """True when this vault holds at least one entity-tagged record.

        The entity/temporal layer is meaningless -- and measurably harmful -- on a
        document corpus, where "is this statement a revision of that one?" has no
        answer. Gating on the CORPUS as well as on the question is what makes the
        exactness guarantee hold regardless of how a question is worded: a vault
        with no tagged records has no revision groups, so nothing can be permuted.
        """
        if self.arena.entity_names:
            return True
        return any(it.get("entity_key") for it in self._mt.items)

    def _max_boost(self, intent_ids=None, personal: bool = True) -> float:
        """Largest value the boosts can ADD to a candidate's score.

        Every entity term is additive (3.0.1 permuted instead, which is how a
        record ended up +0.6402 above its own cosine against a published cap of
        0.50), so this is a true bound: for a cosine-scored vault
        ``abs(score - cosine) <= max_boost`` holds for every hit, and
        ``score >= cosine`` always. Used both to keep a filtered search exact and
        to publish the cap in ``stats()``.

        ``intent_boost`` and ``group_hoist`` need a matching entity intent;
        ``revision_lead`` does not (the vocabulary-free cosine window can fire
        without one), so it counts whenever the vault holds personal records.
        """
        if not personal:
            return 0.0
        total = float(max(0.0, self.revision_lead)) + float(max(0.0, self.temporal_prior))
        if intent_ids is not None and len(intent_ids):
            total += float(self.intent_boost) + float(max(0.0, self.group_hoist))
        return total

    def _resolve_revisions(self, final, rows, cos, ent, rv, tsv, mask, query_text,
                           temporal_direction, intent, personal=True, base=None):
        """Make the requested revision of a fact lead its own group.

        Two ways of recognising "these candidates are the same fact restated":

        1. the tagger gave them the same entity (``phone_number``, ``location``,
           ...), which is exact but only as good as the tagger's recall; or
        2. they sit within ``window_delta`` cosine of the current top hit AND the
           tagger has not positively said they are different attributes -- see
           :meth:`_cosine_window`. This needs no vocabulary of its own, so it
           works in any language and on attributes the tagger never heard of.

        The entity boosts are then LEVELLED across the group (the evidence
        belongs to the fact, not to whichever member happened to carry the tag)
        and the wanted revision is raised just above the best score in its own
        group by ``entities.apply_revision_lead``. Every step is an addition
        bounded by ``stats()['max_boost']``.

        MEASURED, top-1 accuracy on a RE-OPENED vault (the deployed shape; the
        in-session arm is identical on every set). Round 5 first, then the
        round-4 sets it must not regress:

        * 326-question TEMPORAL benchmark, 10 synthetic users, no argument passed
          (the shipped default path) -- 90.5% top-1 / 98.8% top-3, from 44.2% /
          88.0%. The benchmark's own held-out 8 users, measured once after the
          code was frozen, give 90.4%, so the gain is not tuning: per type,
          current 100.0% (was 91.0), historical 75.0% (was 9.0), previous 98.2%
          (was 1.8), adjacent 90.0% (was 57.5), third-party 96.7% (was 66.7).
          A plain exhaustive cosine scan over the same vectors scores 47.9%.
          ``evidence/ranking_r5_temporal_layer.json``
        * 3-persona SELECTION chat set, n=36 -- 97.2% (top-3 100.0%), from 91.7%.
          End-to-end 97.2%. ``evidence/clean_chat_results_r5_3p.json``
        * round-4 DEV chat set, 3 new personas, n=24 -- 95.8% (top-3 100.0%),
          unchanged. ``ranking_dev_r5_final.json` (not published, see evidence/INDEX.md)`
        * 40 generic adjacent-attribute probes 40/40 (exactly plain cosine);
          16 generic revision probes 14/16 current (plain cosine 3/16) and
          16/16 historical; migrated golden chat vault 12/12 -- all unchanged
          from round 4. ``ranking_dev_r5_final.json` (not published, see evidence/INDEX.md)`

        WHAT IS STILL WRONG. On the temporal benchmark the layer is 36/40 on
        adjacent attributes where a plain cosine scan is 40/40, so it is still
        NET NEGATIVE on that one axis; and ``historical_value`` for an address is
        0/10, because the oldest revision ("I live at <a>.") sits 0.18 cosine
        below the newest ("my home address is now <b>.") for a question worded
        "home address" and :func:`entities.group_relevance_floor` drops it. Three
        separate fixes for that were measured and all three cost more on the
        3-persona chat set than they gained; they ship off as knobs
        (``floor_skips_single_valued``, ``group_floor_sim``) with the numbers in
        ``ranking_r5_temporal_layer.json``.
        """
        if not mask.any():
            return final
        top_entity = _ent.resolve_top_entity(
            final, ent, self.arena.entity_names, intent,
            cos=cos, margin=self.intent_margin)
        explicit = _ent.is_explicit_history(temporal_direction)
        by_wording = _ent.temporal_question(query_text, temporal_direction,
                                            intent, personal_corpus=personal)
        # HOW the layer was reached decides whether the newest revision is also
        # exempt from the relevance floor. Reached by wording, it is, exactly as
        # before. Reached ONLY because the group is declared, it is not: the
        # question did not say it was about this fact, and the floor is the
        # check that it is. Without this split, a declared group's newest
        # revision could take rank 1 on a question it barely matches --
        # measured on `temporal_baseline.py`, a document query promoted a
        # revision at cosine 0.016 over the leader's 0.110.
        reached_by_wording = bool(self.gate_on_corpus_only or explicit or by_wording)
        if (not self.gate_on_corpus_only and not by_wording
                and not self._declared_group_present(ent, mask, top_entity, rows)):
            # A DECLARED group is exempt from the wording test; see
            # `_declared_group_present`. Everything below is unchanged and still
            # describes why the test exists for every other write.
            #
            # 3.0.2 additionally required the WORDING to look personal or
            # temporal, so "what's the gate code at the wharf?" -- asked of a
            # chat vault holding two revisions of exactly that -- engaged nothing
            # at all. The decisive guard was always the DATA (`personal_corpus`
            # here, and `_is_revisable` per record), which is what keeps a
            # document corpus exact however a question is worded; measured 0/120
            # top-4 differences either way on the 1,190-paragraph validation
            # corpus (evidence/exactness_v3r2.json).
            return final
        if self.gate_on_corpus_only and not personal and not explicit:
            return final

        # WHICH revision the question asks for. `temporal_direction` is only an
        # instruction when the caller spelled one out ("historical" / "oldest" /
        # "previous", or "present" to force the newest); its default value
        # "current" means UNSPECIFIED, and the direction is then read off the
        # question's own wording by `entities.query_direction`.
        #
        # 3.0.3 read the default as an instruction, so a vault asked "what was my
        # ORIGINAL address?" through `chat.py` -- which never passes the argument
        # -- ordered the group newest-first and answered with the value that
        # question explicitly excludes. That is why the layer scored BELOW a plain
        # cosine scan overall (44.2% vs 47.9%) and less than half of it on
        # historical questions (9.0% vs 22.0%) on the temporal benchmark
        # (evidence/temporal_bench_results.json -> arms.nanomem_default).
        #
        # `hist_mode` is per query for the same reason: "my ORIGINAL number" and
        # "the one just BEFORE this one" are both past, but they name different
        # members of the chain (the first, and the one before the newest).
        # `self.historical_mode` remains the fallback for an anterior question
        # whose wording does not say which -- and for an explicit
        # `temporal_direction="historical"`, so that caller is unaffected.
        explicit_before = explicit
        explicit, hist_mode = _ent.resolve_direction(
            temporal_direction, query_text, self.historical_mode)
        if explicit and not explicit_before and not personal:
            # Wording alone never engages the layer on a document corpus: that is
            # the guard `temporal_question`'s `personal_corpus` argument exists
            # for, and it must hold for the inferred direction too.
            return final

        if self.temporal_prior > 0.0:
            # A personal memory is a log of revisions: later statements are more
            # likely to be "now" and earlier ones more likely to be "before".
            # This is a small rank-normalised prior, so it only decides near-ties;
            # it is off for ordinary document queries (the gate above).
            idx = np.flatnonzero(mask)
            if idx.size > 1:
                order = np.lexsort((tsv[idx], rv[idx]))
                ranks = np.empty(idx.size, dtype=np.float64)
                ranks[order] = np.arange(idx.size) / float(idx.size - 1)
                final[idx] += self.temporal_prior * ((1.0 - ranks) if explicit else ranks)

        group = self._tagged_group(ent, mask, top_entity, rows, cos,
                                   protect_current=reached_by_wording)
        if intent and group.size and self.group_hoist:
            # The question named an attribute and we hold records tagged with it:
            # direct evidence, worth a bounded lift over untagged candidates.
            final[group] += self.group_hoist
        group = self._widen_group(group, cos, mask, rows, ent)
        if group.size < 2:
            return final
        # LEVEL THE ENTITY BOOSTS ACROSS THE GROUP. These records have just been
        # declared competing statements of ONE fact, so the evidence that the
        # question is about that fact belongs to all of them -- it is an accident
        # of the tagger which member carries the tag ("I sold the hatchback and
        # bought a grey estate car" inherits nothing, the sentence before it is
        # tagged `car`). Without this the boosts themselves opened a 0.50 gap
        # INSIDE a group, which is wider than the cosine window that formed it,
        # and the revision lead then had to be large enough to close a gap the
        # ranking layer had created itself. Each member is raised to the best
        # boost any member holds, so the spread left inside the group is just the
        # cosine spread (at most ``window_delta``) and ``revision_lead`` only ever
        # has to cover that.
        if base is not None and self.level_group_boosts:
            b = final[group] - base[group]
            final[group] = base[group] + float(np.max(b))
        marks = None
        if self.use_revision_markers:
            marks = np.zeros(final.shape[0], dtype=np.int8)
            for g in group:
                marks[int(g)] = _ent.has_revision_marker(
                    self._row_record(int(rows[int(g)])).get("text", ""))
        # A REVISION NUMBER IS AN ORDINAL WITHIN ONE GROUP, NOT A GLOBAL CLOCK.
        # `add_fact` assigns it as `group_max[(user_id, project, entity)] + 1`, and
        # an untagged record simply gets 1. So two records that do NOT share a
        # group key carry revision numbers from different counters and comparing
        # them means nothing -- yet `temporal_order` sorts on `(revision,
        # timestamp)` with the revision PRIMARY. For the tagged path that is
        # right and is what makes v2's junk groups resolve. For a group the
        # vocabulary-free cosine window assembled, whose members may carry
        # different tags or none at all, it silently ordered the group by which
        # counter each member happened to come from: a record tagged `licence`
        # with revision 1 was ranked "older" than a record tagged `phone_number`
        # with revision 2 that was written a second EARLIER, and asking for the
        # first of something promoted the wrong one
        # (`ranking_dev_r5_*.json` (not published, see evidence/INDEX.md); the case that exposed it is a
        # dev-persona chat log whose group members carried two different tags).
        # When the revisions are not comparable the timestamp is the only order
        # there is, so the revision key is dropped rather than trusted.
        rv_cmp = rv if self._revisions_comparable(rows, group) else np.zeros_like(rv)
        # THE CAP AND THE FLOOR ARE ONE RULE. `floor_keeps_current` deliberately
        # keeps the newest member of a DECLARED group however far below the
        # group's best it sits, so a declared group's internal cosine spread is
        # no longer bounded by `window_delta` -- and the comment above, which
        # justifies the lead's cap by that bound, stopped being true when that
        # exemption shipped. A cap of 0.20 then silently failed to lift the
        # member the floor had just protected: the third black-box review found
        # a declared three-revision chain whose current value needed a lift of
        # 0.2850 and got 0.20, so `search[0]` returned a value two revisions old
        # while `history()` returned the right one
        # (`evidence/revision_lead_cap_results.json`, arm G).
        #
        # The cap is NOT bounded by the group's own spread, though that fixes
        # every measured arm identically: it would leave the total boost bounded
        # by no constant, and `_apply_filter` proves the exactness of FILTERED
        # search by assuming `max_boost` bounds it. Measured excess under that
        # candidate: 0.7968 against a published 0.70. The candidate was
        # withdrawn rather than void a proof that holds
        # (design/revision_lead_cap_spec.md, amendment 1).
        return _ent.apply_revision_lead(final, group, rv_cmp, tsv, explicit, marks,
                                        hist_mode, self.revision_lead)

    def _tagged_group(self, ent, mask, top_entity, rows, cos,
                      apply_floor: bool = True,
                      protect_current: bool = True) -> np.ndarray:
        """Candidates carrying the question's attribute tag, after the floor.

        Split out of :meth:`_resolve_revisions` unchanged so that
        :meth:`history` can form the SAME group the ranker forms without
        applying, or even computing, any boost. The order of the two steps is
        load-bearing and is preserved: the relevance floor runs FIRST, because a
        record that merely carries the tag but sits far below the group's best is
        not an answer to this question, and 3.0.0 hoisted it anyway because the
        floor ran after the addition.
        """
        group = np.zeros(0, dtype=np.int64)
        if top_entity:
            ids = _ent.matching_ids(top_entity, self.arena.entity_names)
            if ids.size:
                group = np.flatnonzero(mask & np.isin(ent, ids))
        if apply_floor and not (self.floor_skips_single_valued
                                and _ent.is_single_valued(top_entity)):
            group = self._apply_group_floor(group, rows, cos,
                                            protect_current=protect_current)
        return group

    def _widen_group(self, group, cos, mask, rows, ent) -> np.ndarray:
        """Add restatements the tagger never labelled, via the cosine window.

        UNION, NOT FALLBACK. 3.0.2 consulted the cosine window only when the
        TAGGED group had fewer than two members, so a newer restatement the
        tagger happened to label differently ("New spot is ..." tagged ``spot``
        beside two records tagged ``location``) could never join a group that
        already existed, and the superseded value kept rank 1 -- even though the
        window, with its marker exemption, would have admitted it. The window
        applies the same tag check either way, so this only ever adds records the
        window itself would have grouped.
        """
        if self.window_delta > 0 and (group.size < 2 or self.window_union):
            win = self._cosine_window(cos, mask, rows, ent)
            if group.size < 2:
                group = win
            elif win.size:
                group = np.union1d(group, win)
        return group

    def _apply_group_floor(self, group, rows, cos,
                           protect_current: bool = True) -> np.ndarray:
        """Which members of a TAGGED group are competing statements of the fact.

        ``entities.group_relevance_floor`` asks it one way: does this member
        answer THIS QUESTION about as well as the best member does? That is the
        right question for a record that merely carries the tag (a commute note
        tagged `location` is not a revision of an address), but it is the wrong
        one for a genuine older revision phrased differently from the question --
        "I live at <a>." sits 0.18 cosine below "my home address is now <b>."
        when the question says "home address", so the layer never saw the value
        it was being asked for. Measured before this rule: 0/10 on
        ``historical_value/home_address``.

        ``group_floor_sim`` asks it the other way, and a member only has to pass
        ONE of the two: does this member look like the group's best member? Two
        statements of one fact resemble each other whatever the question was
        worded like -- which is exactly the evidence :meth:`_cosine_window` uses
        to group records the tagger never labelled, so the tagged path is being
        given the test the untagged path already has, not a new one. Set it to 0
        to get 3.0.3's query-relative floor alone.

        WHEN TO TURN IT ON, MEASURED. It ships at 0.0 and the default does not
        move, but that default is right for exactly one of the two ways entities
        get assigned, and which one you are on decides the answer:

        * YOU DECLARE THE ENTITY (``metadata={"entity": ...}`` on every write --
          an application with its own attribute schema). ``group_floor_sim=0.45``
          is worth **+19.0 points** of top-1 on revision chains whose newest
          statement is phrased furthest from the question, the case where the
          query-relative floor alone deletes the current value from its own group
          and answers with a superseded one. It costs nothing elsewhere:
          canonical phrasing +1.0, sibling/adjacent probes +0.7.
        * THE TAGGER INFERS IT (plain ``source="chat_session"`` writes, which is
          what ``chat.py`` and the MCP server do). The same 0.45 costs **-13.9
          points** on the 3-persona chat set. 0.65 and 0.70 cost nothing there
          and buy +2.0 and -1.0 respectively, i.e. nothing.

        A single global threshold cannot serve both, because the floor is
        compensating for TAGGER PRECISION -- and a caller that declares its own
        schema has no imprecision to compensate for.

        0.6.5 STOPPED ASKING THE CALLER TO KNOW THIS. The engine already has the
        answer at write time: either the caller named the entity or the lexical
        tagger guessed it. That bit is now stored per group
        (:meth:`_group_is_declared`), and a DECLARED group's newest revision is
        exempt from the query-relative floor while an inferred group's is not.
        Measured over six arms against the 0.6.4 default
        (``design/floor_current_value_spec.md``,
        ``floor_current_value_results.json``):

        ==================================  =======  =======
        arm                                   0.6.4    0.6.5
        ==================================  =======  =======
        A drifting phrasing, declared          52.0     71.0
        B canonical phrasing, declared         89.0     90.0
        C sibling/adjacent probes              68.6     69.3
        D 3-persona chat, tagger-inferred      97.2     97.2
        E historical_value, declared           64.0     64.0
        F previous_value, declared             92.9     92.9
        ==================================  =======  =======

        and `search("where do I work")` returns the current employer instead of
        ranking it third. Two cheaper fixes were measured first and rejected:
        exempting the newest member unconditionally costs -19.4 on arm D, and
        gating that exemption on how much it resembles the group (sweep 0.40 to
        0.80) found no threshold that helped A without costing D the same.

        ``group_floor_sim`` remains a knob and remains 0.0. It is a different
        question -- it tests EVERY member, not the one the revision counter
        already calls current. ``evidence/floor_retune_results.json``,
        ``floor_chatcheck_results.json``.
        """
        group = np.asarray(group, dtype=np.int64)
        kept = _ent.group_relevance_floor(group, cos, self.group_cos_delta)
        if not protect_current:
            # Reached only because the group is DECLARED, not because the
            # question said it was temporal or personal. See `_resolve_revisions`.
            return kept
        keep_current = self.floor_keeps_current
        # ONE KEY, not concordance. This asked `_revisions_comparable`, and when
        # 0.7.4 added the concordance condition to that predicate the protection
        # below silently switched off for every chain whose arrival order and
        # timestamps disagree -- exactly the chains 0.7.4 was written for. The
        # floor then dropped the current value and a superseded one answered.
        # Found by the operation fuzzer on a chain with revisions 5, 3, 4 in
        # time order: search returned the MIDDLE record of three.
        if (kept.size != group.size and group.size >= 2
                and self._same_group_key(rows, group)):
            if keep_current is None:
                keep_current = self._group_is_declared(rows, group)
        else:
            keep_current = False
        if keep_current or (self.current_keep_sim > 0.0 and kept.size != group.size
                            and group.size >= 2
                            and self._same_group_key(rows, group)):
            # THE NEWEST REVISION IS NOT A PRECISION RISK. The floor drops
            # records that carry the tag without restating the fact; the
            # maximum-revision member is the one `add_fact` numbered
            # `group_max + 1`, which is the engine's own assertion at write time
            # that this record restates this fact. If including it is wrong the
            # GROUPING is wrong, and the floor is the wrong place to correct it.
            # Gated on `_revisions_comparable` because a revision number orders
            # nothing across group keys.
            # WHICH record is current follows the ranker's rule: the latest
            # timestamp, with the arrival counter only breaking a tie. Taking
            # `rv.max()` alone protected the last-WRITTEN record, which on an
            # out-of-order chain is not the current one.
            _e, rv, tsv = self._columns(np.asarray(rows)[group])
            latest = tsv.max()
            at_latest = np.flatnonzero(tsv == latest)
            best_rev = rv[at_latest].max()
            newest = group[at_latest[rv[at_latest] == best_rev]]
            if self.current_keep_sim > 0.0:
                # ...AND IT STILL HAS TO LOOK LIKE THE FACT. Protecting the
                # newest member unconditionally helps where the group is real
                # (+19.0 on drifting phrasing with declared tags) and hurts where
                # the tagger built a bad one (-19.4 on the 3-persona chat set):
                # the newest member of a group that is not a chain is junk being
                # promoted. Which case this is can be read off the vectors
                # without knowing who assigned the tag -- two statements of one
                # fact resemble each other however the question was worded, which
                # is the evidence `_cosine_window` uses to group untagged records
                # in the first place. `group_floor_sim` applies that test to every
                # member; this applies it to the one member the revision counter
                # already calls current.
                lead = int(group[int(np.argmax(cos[group]))])
                vl = self._vectors_for(np.asarray(rows)[[lead]])[0]
                Vn = self._vectors_for(np.asarray(rows)[newest])
                newest = newest[(Vn @ vl) >= float(self.current_keep_sim)]
            kept = np.union1d(kept, newest)
        if kept.size == group.size or self.group_floor_sim <= 0.0 or group.size < 2:
            return kept
        lead = int(group[int(np.argmax(cos[group]))])
        V = self._vectors_for(np.asarray(rows)[group])
        vl = self._vectors_for(np.asarray(rows)[[lead]])[0]
        near = (V @ vl) >= float(self.group_floor_sim)
        return group[near | np.isin(group, kept)]

    def _group_is_declared(self, rows, group) -> bool:
        """Did the CALLER name this group's entity, or did the tagger infer it?

        This is what decides whether the newest revision is protected from the
        relevance floor, and it is the whole difference between a fix and a
        regression. Measured on the six arms of
        ``design/floor_current_value_spec.md``, protecting it UNCONDITIONALLY is
        worth +19.0 points of top-1 where the entity was declared and costs
        -19.4 where it was inferred -- near-symmetric, because the newest member
        of a group the tagger built wrongly is the wrong record to promote. The
        seven regressions it caused on the 3-persona chat set are all on
        `phone number` and `address`, the two attributes with SIBLINGS, which is
        that failure exactly.

        No vector test separates the two cases: candidate F3 gated the same rule
        on how much the newest member resembles the group and found no threshold
        that helped one path without costing the other by the same amount
        (``floor_current_value_results.json``). Provenance is the signal, so
        provenance is what is stored.
        """
        gids = self._group_ids(np.asarray(rows)[np.asarray(group)])
        if gids.size == 0 or gids[0] < 0:
            return False
        decl = self.arena.group_declared
        g = int(gids[0])
        return bool(0 <= g < len(decl) and decl[g])

    def _declared_group_present(self, ent, mask, top_entity, rows) -> bool:
        """Does this question resolve to a group the CALLER's schema declared?

        The wording gate below compensates for TAGGER IMPRECISION: the entity
        layer must not fire on a document corpus just because a question happens
        to contain "first" or "new". A caller that declares its own entities has
        no imprecision to compensate for, which is the same argument 0.6.5
        already accepted for the relevance floor in :meth:`_apply_group_floor`.

        Measured before this exemption, with the entity pinned and only the
        QUESTION's wording differing (`evidence/usecases_findings.json`):
        third-person `as_of` accuracy was 54.5% on config drift, 53.3% on policy
        versions and 64.0% on 1,000-entity fleet state, against 100.0% for the
        same probes with the word "current" in them. An application with an
        attribute schema does not phrase its queries as a person talking about
        themselves, so the layer it adopted nanomem FOR never ran.

        Only consulted when the wording test has already failed, so it costs
        nothing on a query that engages anyway, and it requires two members --
        one record is not a revision chain and has nothing to resolve.
        """
        if not top_entity:
            return False
        ids = _ent.matching_ids(top_entity, self.arena.entity_names)
        if not ids.size:
            return False
        group = np.flatnonzero(mask & np.isin(ent, ids))
        if group.size < 2:
            return False
        return self._group_is_declared(rows, group)

    def _same_group_key(self, rows, group) -> bool:
        """Every member of ``group`` shares one ``(user_id, project, entity)``.

        Split back out of :meth:`_revisions_comparable` when that gained its
        concordance condition. The two answer different questions and gate
        different things: "do these rows belong to one chain" is what decides
        whether the chain HAS a current value, and "does the counter agree with
        the clock" is what decides whether the counter may ORDER it. Fusing them
        turned the floor's current-value protection off on any chain written out
        of order -- see :meth:`_apply_group_floor`.
        """
        g = np.asarray(group, dtype=np.int64)
        if g.size < 2:
            return True
        gids = self._group_ids(np.asarray(rows)[g])
        return bool(gids[0] >= 0 and np.all(gids == gids[0]))

    def _revisions_comparable(self, rows, group) -> bool:
        """True when ``group``'s revision numbers actually order it.

        Two conditions, and both must hold.

        ONE KEY. Revision numbers are per ``(user_id, project, entity)`` (see
        :meth:`add_fact`), so across keys they order nothing.

        AGREEING WITH TIME. ``add_fact`` numbers a new record ``group_max + 1``,
        which is INSERTION order -- it cannot be anything else, because the
        counter is what breaks ties between records written at the same instant.
        That is right for a chat log, where the two orders are the same. It is
        wrong the moment a caller BACKFILLS: importing history, replaying a log,
        migrating from another store or syncing out of order all write an older
        record after a newer one, and insertion order then says the oldest fact
        is the newest revision. Measured before this rule, on three addresses
        written newest-middle-oldest:

            current value  -> "3 Elm Lane"   (the OLDEST, by 350 days)
            history        -> Oak, Pine, Elm (not time order either)

        So when the caller's own timestamps contradict the counter, the counter
        is the thing that is wrong: a timestamp is a statement about when the
        fact was true, and insertion order is an artefact of how it arrived.
        Concordance is checked rather than assumed, and on disagreement the
        revision key is dropped exactly as it is for a multi-key group -- the
        existing rule, applied to the other way revisions can fail to order.

        THIS IS A NO-OP WHENEVER THE TWO ORDERS AGREE, which is every write that
        arrives in the order it happened.
        """
        g = np.asarray(group, dtype=np.int64)
        if g.size < 2:
            return True
        if not self._same_group_key(rows, group):
            return False
        sel = np.asarray(rows)[g]
        _ent_col, rv, tsv = self._columns(sel)
        order = np.argsort(rv, kind="stable")
        ts_by_rev = tsv[order]
        # ties on revision cannot invert anything, and equal timestamps are the
        # case the counter exists to break, so only a strict inversion counts.
        return bool(np.all(np.diff(ts_by_rev) >= 0.0))

    def _group_ids(self, rows) -> np.ndarray:
        """The interned revision-group id of each row (``-1`` when ungrouped)."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.full(rows.size, -1, dtype=np.int64)
        arena_mask = rows < MT_BASE
        ar = rows[arena_mask]
        if ar.size:
            out[arena_mask] = self.arena.group_id[ar]
        for j in np.flatnonzero(~arena_mask):
            out[j] = int(self._mt.items[int(rows[j]) - MT_BASE].get("group_id", -1))
        return out

    def _is_revisable(self, row: int, entity_id: int) -> bool:
        """Can this record be a revision of another one?

        Only if it is entity-tagged, or its source is one nanomem treats as a
        personal log (``entities.PERSONAL_SOURCES``; override with the
        ``personal_sources`` constructor argument). A paragraph of an ingested
        document carries the file name as its source, and is not a revision of
        another paragraph:
        permuting two of them by timestamp is exactly the re-ordering that broke
        exactness on the validation corpus. This is a property of the DATA, so it
        holds however the question happens to be worded. Costs at most
        ``window_max`` record decodes per query.
        """
        if entity_id is not None and int(entity_id) >= 0:
            return True
        return self._row_record(int(row)).get("source") in self.personal_sources

    def _cosine_window(self, cos, mask, rows=None, ent=None):
        """Candidates that answer the question about as well as the best one does.

        Most of the time no tagged revision group exists -- free-form chat does
        not announce which attribute it is updating -- so the records within
        ``window_delta`` cosine of the top hit, capped at ``window_max``, are
        treated as competing statements of the same fact. Three further
        conditions, each of which exists because of a measured failure:

        * NO CONTRADICTED TAG. A candidate whose entity is known and does NOT
          match the leader's is dropped, however close its cosine. The window is
          for records the tagger had no opinion about; when the tagger DID have
          one and the two disagree, it has positively said these are different
          facts. 3.0.1 skipped this, so "my dentist"/"my doctor",
          "my home address"/"my work address" and "my primary email"/"my backup
          email" were all declared restatements of one another and the later,
          unrelated record was promoted: 30/40 generic adjacent-attribute probes
          against 40/40 for a plain cosine scan
          (``evidence/adjacent_attributes_v3r3.json``).
        * ``window_sim`` requires them to look like each other (measured: without
          it, an unrelated but later record displaced a correctly tagged answer on
          the golden chat vault).
        * :meth:`_is_revisable` requires them to be the kind of record that CAN
          supersede another.

        Needs no vocabulary of its own, so it works in any language; the tag
        check only ever REMOVES candidates, so an untagged corpus behaves exactly
        as before.
        """
        cand = np.flatnonzero(mask)
        if cand.size < 2:
            return np.zeros(0, dtype=np.int64)
        order = cand[_top_k_stable(cos[cand], int(self.window_max))]
        best = float(cos[order[0]])
        wide = max(float(self.window_delta), float(self.window_delta_marked))
        near = order[cos[order] >= best - wide]
        marked = self._revision_marks(rows, near) if rows is not None \
            else np.zeros(near.size, dtype=bool)
        keep = (cos[near] >= best - float(self.window_delta)) | marked
        order, marked = near[keep], marked[keep]
        if order.size > 1 and ent is not None:
            k = self._tag_compatible(ent, order, marked)
            order, marked = order[k], marked[k]
        if order.size > 1 and rows is not None:
            k = np.asarray([self._is_revisable(int(rows[int(j)]),
                                               None if ent is None else ent[int(j)])
                            for j in order], dtype=bool)
            order, marked = order[k], marked[k]
        if order.size > 1 and self.window_sim > 0 and rows is not None:
            V = self._vectors_for(rows[order])
            k = (V @ V[0]) >= float(self.window_sim)
            if self.marker_skips_sim:
                # Same argument as the tag match below: `window_sim` is a proxy
                # for "these are statements of one fact", and a sentence that
                # SAYS it replaces an earlier one is direct evidence. A
                # re-worded revision is exactly the case where the proxy is
                # weakest (a restatement that opens "Switched <thing>. The old
                # one is retired; as of yesterday I'm on ..." sits below 0.60 of
                # the statement it supersedes).
                k = k | marked
            if self.tag_match_skips_sim and ent is not None:
                # A POSITIVE TAG MATCH BEATS THE LOOK-ALIKE PROXY. `window_sim`
                # exists to stop an unrelated record joining a group; when the
                # TAGGER has already said two records state the same attribute
                # that question is answered, and vector similarity is only a
                # proxy for it. 3.0.2 applied the proxy anyway, so "I live at
                # <a>." / "I moved to <b> last month." -- both tagged `location`,
                # cosine 0.56 apart in wording -- never formed a group and the
                # superseded address won. Measured: the 16 generic revision
                # probes score 14 with this rule and 13 without, and the
                # adjacent probes, both chat sets and the golden vault are
                # unchanged (`ranking_dev_r4_shipped.json` (not published, see evidence/INDEX.md) vs
                # ranking_dev_B_no_tagsim.json).
                k = k | self._tag_matches_leader(ent, order)
            order, marked = order[k], marked[k]
        return order

    def _revision_marks(self, rows, order) -> np.ndarray:
        """Does each candidate's own text ANNOUNCE that it supersedes something?

        ``entities.has_revision_marker`` over at most ``window_max`` record
        decodes. Measured on the generic probes: 15 of 16 real revisions carry a
        marker on the newer statement and 0 of 40 adjacent-attribute pairs carry
        one at all (``evidence/marker_separation_v3r4.json``), which is
        what makes it safe to let a marker override the tag check below.
        """
        out = np.zeros(int(np.asarray(order).size), dtype=bool)
        for i, j in enumerate(np.asarray(order)):
            txt = self._row_record(int(rows[int(j)])).get("text", "")
            out[i] = _ent.has_revision_marker(txt)
        return out

    def _tag_matches_leader(self, ent, order) -> np.ndarray:
        """Boolean mask: candidates whose entity POSITIVELY matches the leader's."""
        names = self.arena.entity_names
        out = np.zeros(order.size, dtype=bool)
        lead = int(ent[int(order[0])])
        if lead < 0 or lead >= len(names):
            return out
        ok = set(int(i) for i in _ent.matching_ids(names[lead], names))
        for i in range(order.size):
            out[i] = int(ent[int(order[i])]) in ok
        return out

    def _tag_compatible(self, ent, order, marked=None) -> np.ndarray:
        """Boolean mask over ``order``: keep the leader, and every candidate whose
        entity is unknown, matches the leader's, or explicitly announces a revision.

        ``entities_match`` is the same relation the tagged path uses, so the two
        paths can never disagree about whether two records state one attribute.

        THE MARKER EXEMPTION. A tagger disagreement is evidence that two records
        are different attributes -- that is the whole reason this check exists --
        but it is weaker evidence than a sentence that says in words that it
        replaces an earlier one. A revision is commonly re-worded ("my office is
        on the fourth floor" -> "we moved; my office is on the seventh floor
        now", which the tagger reads as `office` and `location`), and 3.0.2
        refused to group every such pair, so the superseded value kept winning:
        4 of the 5 remaining failures on the 16 generic revision probes. The
        exemption is safe because the two families separate almost perfectly on
        this signal -- 15 of 16 revisions carry a marker, 0 of 40
        adjacent-attribute pairs do (``evidence/marker_separation_v3r4.json``)
        -- and the rule is measured both ways: with it the 16 generic revision
        probes score 14, without it 13, and the adjacent probes, the 3-persona
        chat set, the round-4 dev chat set and the golden vault are all unchanged
        (``ranking_dev_r4_shipped.json` (not published, see evidence/INDEX.md)` vs
        ``ranking_dev_A_no_marker_tag.json``).
        """
        names = self.arena.entity_names
        lead = int(ent[int(order[0])])
        keep = np.ones(order.size, dtype=bool)
        if lead < 0 or lead >= len(names):
            return keep                      # the leader is untagged: no opinion
        ok = _ent.matching_ids(names[lead], names)
        for i in range(1, order.size):
            e = int(ent[int(order[i])])
            if 0 <= e < len(names) and e not in ok:
                if self.marker_overrides_tag and marked is not None and bool(marked[i]):
                    continue
                keep[i] = False
        return keep

    def _vectors_for(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((rows.size, self.embed_dim), dtype=np.float32)
        for i, r in enumerate(rows):
            r = int(r)
            out[i] = (self._mt.vec[r - MT_BASE] if r >= MT_BASE else self.arena.vector(r))
        return out

    def _candidates(self, q, metadata_filter, intent_ids, top_k=1,
                    screen_ok: bool = False, force_exact: bool = False):
        N = self.arena.n_rows
        mode = self._mode(metadata_filter, force_exact)
        if N == 0:
            rows = np.zeros(0, dtype=np.int64)
            cos = np.zeros(0, dtype=np.float32)
        elif mode == "exhaustive":
            got = (self._screen_candidates(q, top_k)
                   if screen_ok and metadata_filter is None else None)
            if got is not None:
                rows, cos = got
            else:
                rows = np.arange(N, dtype=np.int64)
                cos = self.arena.scores(q)
        else:
            rows = self._router.route(q)
            if intent_ids is not None and len(intent_ids):
                extra = np.setdiff1d(self.arena.rows_with_entity_ids(intent_ids), rows)
                if extra.size:
                    rows = np.concatenate([rows, extra])
            cos = self.arena.scores_rows(rows, q)
        m = self._mt.n
        if m:
            mt_rows = MT_BASE + np.arange(m, dtype=np.int64)
            mt_cos = self._mt.vec[:m] @ q
            rows = np.concatenate([rows, mt_rows])
            cos = np.concatenate([cos, mt_cos])
        return rows, np.asarray(cos, dtype=np.float32)

    def _columns(self, rows):
        arena_mask = rows < MT_BASE
        n = rows.size
        ent = np.full(n, -1, dtype=np.int32)
        rv = np.ones(n, dtype=np.int32)
        tsv = np.zeros(n, dtype=np.float64)
        ar = rows[arena_mask]
        if ar.size:
            ent[arena_mask] = self.arena.entity_id[ar]
            rv[arena_mask] = self.arena.rev[ar]
            tsv[arena_mask] = self.arena.ts[ar]
        mt_idx = np.flatnonzero(~arena_mask)
        for j in mt_idx:
            item = self._mt.items[int(rows[j]) - MT_BASE]
            ent[j] = item["entity_id"]
            rv[j] = item["revision"]
            tsv[j] = item["timestamp"]
        return ent, rv, tsv

    def _entity_ids_containing(self, index, norm):
        """Every interned entity id whose key has ``norm`` as one of its parts.

        A composite tag (`{"entity": ["work", "job"]}`) interns as `work_job`, so
        the superset above has to admit it -- but walking all 5,000 entity names
        on EVERY filtered query cost 0.4 ms at that size and grew linearly. The
        parts map is built once and rebuilt only when the vocabulary grows.
        """
        cache = getattr(self, "_entity_parts_cache", None)
        if cache is None or cache[0] is not index or cache[1] != len(index):
            parts = {}
            for name, i in index.items():
                for piece in {str(name)} | set(str(name).split("_")):
                    parts.setdefault(piece, []).append(i)
            cache = (index, len(index), parts)
            self._entity_parts_cache = cache
        return cache[2].get(norm, [])

    def _apply_filter(self, rows, base, mask, metadata_filter, top_k, max_boost):
        """Lazy metadata walk in score order, stopped by a boost-safe bound.

        Walking descending ``base``, once ``top_k`` candidates have matched, any
        unexamined candidate ``j`` has ``base[j] <= base[i]``, so its best
        possible final score is ``base[i] + max_boost``. If that is still below
        the ``top_k``-th score already kept, ``j`` cannot enter the result and the
        walk can stop. The kept set is therefore exactly what a full filtered scan
        would return -- which is what the docstring has always claimed.

        The revision permutation cannot defeat the bound: a candidate only joins
        a revision group if it is within ``group_cos_delta`` (0.06) or
        ``window_delta`` (0.16) of the group's leader, and anything the bound
        prunes is more than ``max_boost`` (1.10 at the shipped defaults, and the
        caller passes the value ``stats()`` publishes) below the leader.
        """
        # CHEAP SUPERSET FIRST, when the filter names an entity.
        # The walk below decodes one record per step and its early-stop bound
        # cannot fire until `top_k` matches are in hand, so a selective filter
        # walked the WHOLE corpus: measured on 15,000 records with 5,000
        # entities, 15,001 record decodes and 33.70 ms against 0.70 ms
        # unfiltered, 48x.
        #
        # `entity_id` is the interned `normalize_entity(...)` of the record's
        # tag, and `matches_filter` compares the RAW metadata, so the two are
        # not interchangeable. They do not have to be: normalization is a
        # function, so raw == filter implies normalize(raw) == normalize(filter),
        # and restricting to that id can only drop rows that could never have
        # matched. Every surviving row is still put through `matches_filter`
        # unchanged, so this narrows the walk without touching its semantics.
        # ...BUT THE SUPERSET HAS TO BE A SUPERSET. The argument above holds for a
        # SCALAR tag, where `matches_filter` tests equality. For a LIST-valued
        # tag it does not: `{"entity": ["work", "job"]}` interns as `work_job`,
        # `matches_filter` accepts it for `{"entity": "work"}` by membership, and
        # restricting to the id `work` dropped it. Measured: `search(filter=
        # {"entity": "work"})` returned one row where `get_all_records(where=...)`
        # returned two -- and only once ANOTHER row carried the plain tag, so the
        # exactness `_apply_filter` documents failed on a corpus-dependent trigger.
        #
        # Composite ids are joined with "_" by `normalize_entity`, so every id
        # that could match by membership has the wanted id as one of its parts.
        # Taking all of them keeps this a superset; `matches_filter` below still
        # decides. A scalar tag that merely contains the part (`home_address` for
        # `home`) is admitted too, which costs a few extra decodes and changes no
        # result.
        want_ent = metadata_filter.get("entity") if metadata_filter else None
        if isinstance(want_ent, (str, bytes)) and want_ent:
            index = getattr(self.arena, "entity_index", None)
            if index:
                norm = _ent.normalize_entity(want_ent)
                eids = self._entity_ids_containing(index, norm)
                if eids:
                    ent_col = self._columns(np.asarray(rows))[0]
                    mask = mask & np.isin(ent_col, np.asarray(eids, dtype=ent_col.dtype))
                    if not mask.any():
                        return mask
        keep = np.zeros(mask.shape, dtype=bool)
        order = np.argsort(-base, kind="stable")
        kept = []
        k = max(1, int(top_k))
        for i in order:
            if not mask[i]:
                continue
            if len(kept) >= k and float(base[i]) + max_boost < kept[k - 1]:
                break
            meta = self._row_record(int(rows[i])).get("metadata") or {}
            if matches_filter(meta, metadata_filter):
                keep[i] = True
                kept.append(float(base[i]))
        return mask & keep

    def history(self, query_text: str, query_vec, max_len: Optional[int] = None,
                min_score: float = 0.0) -> List[Dict[str, Any]]:
        """The revision chain for the fact a query names, OLDEST FIRST.

        Every value the fact has held, not just the one that wins the ranking.
        Each element carries ``id, doc_id, text, source, metadata, cosine,
        timestamp, revision, superseded``; the LAST element is the current value
        and is the only one with ``superseded == False``.

        This is the group :meth:`_resolve_revisions` has always assembled in
        order to decide WHICH member to surface -- the same
        :meth:`_tagged_group` and :meth:`_widen_group`, in the same order -- but
        returned instead of consumed. No boost is computed and none is applied,
        so ``cosine`` here is the raw similarity and nothing has been permuted.

        It differs from ``search(..., temporal_direction="historical")`` in what
        the caller is asking for. That call asks the RANKER for one superseded
        value and is gated on the question's wording, because a document corpus
        must stay exact however a question is phrased. Calling this method IS the
        instruction, so the wording gate does not apply; the DATA gate still
        does, which is the guard that matters (see :meth:`_resolve_revisions`).
        On a vault holding no personal records there are no revisions to report
        and the chain is the single best match.

        A fact with no restatements has a one-element history. That is a real
        answer, not an empty one: it says the value has never changed.

        ``max_len`` keeps the ``max_len`` MOST RECENT entries, because a chain is
        truncated from its old end.
        """
        with self._lock:
            self._reload_if_modified()
            q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
            if q.size != self.embed_dim:
                raise ValueError(
                    f"query has {q.size} dims, vault at {self.filepath} has {self.embed_dim}")
            nrm = float(np.linalg.norm(q))
            if not np.isfinite(nrm) or nrm <= 0.0:
                return []
            q = q / nrm
            personal = self._has_personal_records()
            intent, intent_ids = self._resolve_intent(query_text) if personal else (None, None)
            rows, cos = self._candidates(q, None, intent_ids, top_k=1,
                                         screen_ok=False, force_exact=True)
            if rows.size == 0:
                return []
            ent, rv, tsv = self._columns(rows)
            base = cos.astype(np.float64)
            mask = base >= float(min_score)
            if not mask.any():
                return []
            scored = np.where(mask, base, -np.inf)
            top_entity = _ent.resolve_top_entity(
                scored, ent, self.arena.entity_names, intent,
                cos=cos, margin=self.intent_margin)
            # NO RELEVANCE FLOOR HERE, deliberately. `group_relevance_floor`
            # answers "is this member a plausible answer to THIS QUESTION",
            # which is the ranker's question: a record that merely carries the
            # tag and sits far below the group's best must not be promoted over
            # one that answers what was asked. `history` asks a different
            # question -- "is this a statement of the same fact" -- and the tag
            # is exactly that evidence.
            #
            # Applying the floor here does not merely shorten the chain, it
            # CORRUPTS it. Measured on four restatements at cosine 0.80 / 0.77 /
            # 0.74 / 0.71 to their own question, GROUP_COS_DELTA = 0.06 drops the
            # fourth -- and the chain then reports the THIRD as the current
            # value, with `superseded=False`, while a newer one exists. A memory
            # that answers "what has this been" by silently deleting the newest
            # entry and relabelling an old one as current is worse than one that
            # declines to answer.
            #
            # The cost is the opposite error: a record carrying the tag without
            # being a restatement (the commute note tagged `location` in
            # `_apply_group_floor`) now appears in the chain. That is the safer
            # direction for an audit surface -- the entry arrives with its own
            # text and timestamp for the caller to judge, and nothing is hidden.
            # The cosine WINDOW below keeps its bound either way, because
            # similarity is the only evidence an untagged candidate has.
            # ...AND `min_score` MUST NOT DELETE MEMBERS OF THE CHAIN EITHER.
            # `mask` is `cosine >= min_score`, default 0.0, so a revision worded
            # unlike the question -- which is the ordinary case for an old value,
            # and the whole reason this method exists -- was dropped before the
            # group was formed. Measured on this chain with the offline encoder,
            # "Moved down to the annexe at Larkfield." scores -0.0298 against
            # "where is my desk" and vanished, leaving a 2-entry history of a
            # 3-entry chain. It is the same failure as the intent bug above and
            # it survived that fix.
            #
            # `min_score` still decides WHETHER there is a fact here and which
            # one: `top_entity` is resolved from the masked scores, untouched.
            # What it no longer does is delete records the tagger has already
            # said are statements of that same fact. The cosine WINDOW keeps the
            # original mask, because similarity is the only evidence an untagged
            # candidate has.
            group_mask = mask
            if top_entity:
                tagged_ids = _ent.matching_ids(top_entity, self.arena.entity_names)
                if tagged_ids.size:
                    group_mask = mask | np.isin(ent, tagged_ids)
            group = self._tagged_group(ent, group_mask, top_entity, rows, cos,
                                       apply_floor=False)
            group = self._widen_group(group, cos, mask, rows, ent)
            if group.size == 0:
                group = np.asarray([int(np.argmax(scored))], dtype=np.int64)
            # ONE CHAIN BELONGS TO ONE GROUP KEY, NOT TO ONE ENTITY NAME.
            # `tagged_ids` above are interned from `normalize_entity(...)`, which
            # is the entity alone -- but a revision group is keyed on
            # (user_id, project, entity), which is why two tenants' first records
            # are both revision 1. So `history("where do I work")` returned
            # Alice's two employers AND Bob's, interleaved by timestamp and
            # numbered as if they were one person's chain. Measured: a 3-entry
            # history spanning two user_ids.
            #
            # Restrict to the anchor's own group. A row with no group key (an
            # untagged record) keeps the old behaviour, so this narrows only the
            # case that was wrong.
            gids = self._group_ids(np.asarray(rows)[group])
            anchor = int(np.argmax(scored))
            anchor_gid = self._group_ids(np.asarray(rows)[[anchor]])
            if anchor_gid.size and anchor_gid[0] >= 0:
                same = group[gids == anchor_gid[0]]
                if same.size:
                    group = same
            # Same guard as the ranker: a revision number is an ordinal within
            # ONE group key, so across keys the timestamp is the only order there
            # is (see :meth:`_revisions_comparable`).
            rv_cmp = rv if self._revisions_comparable(rows, group) else np.zeros_like(rv)
            order = _ent.temporal_order(rv_cmp, tsv, group, historical=True,
                                        historical_mode="oldest")
            order = np.asarray(order, dtype=np.int64)
            if max_len is not None and int(max_len) >= 0:
                order = order[-int(max_len):] if int(max_len) else order[:0]
            last = order.size - 1
            out = []
            for j, i in enumerate(order):
                i = int(i)
                r = int(rows[i])
                rec = self._row_record(r)
                doc_id = (self.arena.ids[r] if r < MT_BASE
                          else self._mt.items[r - MT_BASE]["id"])
                out.append({"id": str(doc_id), "doc_id": str(doc_id),
                            "text": rec.get("text", ""),
                            "source": rec.get("source", "unknown"),
                            "metadata": dict(rec.get("metadata") or {}),
                            "cosine": float(cos[i]),
                            "timestamp": float(tsv[i]), "revision": int(rv[i]),
                            "superseded": bool(j != last)})
            return out

    def changes(self, since: float, until: Optional[float] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Records asserted in ``(since, until]``, oldest first.

        What the vault learned in a window, with no query vector: the half-open
        interval excludes ``since`` so that ``changes(t)`` after
        ``search(as_of=t)`` reports exactly the rows that view could not see.

        Each element carries ``id, text, source, metadata, timestamp, revision,
        entity``, where ``entity`` is the attribute tag the write path assigned
        or ``None`` for an untagged record. ``revision > 1`` means the record
        SUPERSEDED an earlier value of the same attribute at write time.

        This reads the resident ``ts`` column and never scores anything, so it
        costs a comparison over the column and a sort over what it selects.
        """
        with self._lock:
            self._reload_if_modified()
            lo = float(since)
            hi = float(until) if until is not None else float("inf")
            if not (hi >= lo):
                raise ValueError(f"until ({hi}) is before since ({lo})")
            n = int(self.arena.n_rows)
            names = self.arena.entity_names
            picked = []
            if n:
                ts = np.asarray(self.arena.ts[:n])
                for r in np.flatnonzero((ts > lo) & (ts <= hi)):
                    picked.append((float(ts[int(r)]), int(r)))
            for k, item in enumerate(self._mt.items[:self._mt.n]):
                t = float(item["timestamp"])
                if lo < t <= hi:
                    picked.append((t, MT_BASE + k))
            picked.sort(key=lambda p: (p[0], p[1]))
            if limit is not None and int(limit) >= 0:
                picked = picked[:int(limit)]
            out = []
            for t, r in picked:
                rec = self._row_record(r)
                if r < MT_BASE:
                    doc_id = self.arena.ids[r]
                    eid = int(self.arena.entity_id[r])
                    rev = int(self.arena.rev[r])
                else:
                    item = self._mt.items[r - MT_BASE]
                    doc_id, eid, rev = item["id"], int(item["entity_id"]), int(item["revision"])
                out.append({"id": str(doc_id), "doc_id": str(doc_id),
                            "text": rec.get("text", ""),
                            "source": rec.get("source", "unknown"),
                            "metadata": dict(rec.get("metadata") or {}),
                            "timestamp": float(t), "revision": rev,
                            "entity": names[eid] if 0 <= eid < len(names) else None})
            return out

    def volatility(self, min_revisions: int = 2, now: Optional[float] = None
                   ) -> List[Dict[str, Any]]:
        """MEASURED revision statistics per fact -- how often each one changes.

        One entry per revision group that has been restated at least
        ``min_revisions`` times, ordered by ``age`` descending (the longest
        unconfirmed first). Each carries ``entity, user_id, project,
        n_revisions, first_ts, last_ts, age, intervals, mean_interval,
        median_interval``.

        Nothing here is modelled: these are the timestamps the log already
        holds, differenced. :meth:`staleness` is the layer that turns them into
        a probability, and it is separate precisely so a caller can use the
        measurements without buying the model.

        This reads the resident ``ts`` / ``rev`` / ``group_id`` columns and the
        pending memtable, and scores no vectors, so it does not touch the search
        path. Records written but not yet flushed ARE counted: `volatility()`
        and `search()` never disagree about what has been written.

        Records with no entity share the empty group key and are excluded: they
        are not restatements of one fact, they are everything the tagger did not
        recognise, and pooling them would report one enormous fake fact.
        """
        with self._lock:
            self._reload_if_modified()
            t_now = float(time.time() if now is None else now)
            keys = self.arena.group_keys
            n = int(self.arena.n_rows)

            # Records still in the memtable are part of the log. `search` and
            # `history` have always scored them and `stats()` counts them as
            # `memtable_pending`; until 0.6.4 this method read `self.arena`
            # alone, so it saw SEALED 50-row blocks only. Under `block_capacity`
            # writes it returned [], and past that it answered off a stale
            # prefix: measured on 0.6.3, 120 writes with the newest made TODAY
            # reported `n_revisions=100, age=20 days`. Reporting a fact restated
            # today as three weeks unconfirmed is the exact answer `staleness()`
            # exists to get right. Every test flushed first, so the suite agreed.
            pending: Dict[int, List[float]] = {}
            for it in self._mt.items[:self._mt.n]:
                g = int(it.get("group_id", -1))
                if g >= 0:
                    pending.setdefault(g, []).append(float(it["timestamp"]))

            out = []
            if n:
                gid = np.asarray(self.arena.group_id[:n])
                ts = np.asarray(self.arena.ts[:n], dtype=np.float64)
                order = np.argsort(gid, kind="stable")
                gsorted = gid[order]
                bounds = np.flatnonzero(np.diff(gsorted)) + 1
                for lo, hi in zip(np.concatenate([[0], bounds]),
                                  np.concatenate([bounds, [gsorted.size]])):
                    g = int(gsorted[lo])
                    key = keys[g] if 0 <= g < len(keys) else ""
                    if not key:
                        continue
                    # `pop`, so a group with both sealed and pending records is
                    # counted once, by this branch, with all of its timestamps.
                    t = np.concatenate([
                        ts[order[lo:hi]],
                        np.asarray(pending.pop(g, ()), dtype=np.float64)])
                    e = _volatility_entry(key, np.sort(t), t_now, min_revisions)
                    if e is not None:
                        out.append(e)

            # Groups with no sealed record at all -- which is every group in a
            # vault younger than one block.
            for g, times in pending.items():
                key = keys[g] if 0 <= g < len(keys) else ""
                if not key:
                    continue
                e = _volatility_entry(
                    key, np.sort(np.asarray(times, dtype=np.float64)),
                    t_now, min_revisions)
                if e is not None:
                    out.append(e)

            out.sort(key=lambda r: -r["age"])
            return out

    def staleness(self, now: Optional[float] = None, shrink: float = 1.0,
                  min_revisions: int = 2, assume_memoryless: bool = False
                  ) -> List[Dict[str, Any]]:
        """MODELLED probability that each fact's current value is out of date.

        ``p_superseded`` IS None UNLESS ``assume_memoryless=True``. That is not
        caution for its own sake -- it is the pre-registered consequence of a
        gate this model did not clear
        (``evidence/staleness_calibration.json``,
        ``design/staleness_spec.md``). Measured, held-out last interval:

        ============================  =====  =====  ==================
        corpus                          ECE  Brier  constant baseline
        ============================  =====  =====  ==================
        uniform intervals             .0771  .1916  .1909  (ties/wins)
        exponential intervals         .0593  .1704  .1779  (model wins)
        ============================  =====  =====  ==================

        Calibration is fine in both (bar was ECE <= 0.15). What fails is the
        second clause: against a single corpus-wide rate, the PER-FACT rate wins
        only on the corpus whose intervals were generated to match the model's
        own assumption. Where the process is not memoryless it is a tie, and a
        tie means the per-fact rate earned nothing. The bar said such a model
        does not ship on by default, so it does not.

        Set ``assume_memoryless=True`` only if the domain actually justifies it.
        ``volatility()`` returns the measured statistics with no model at all
        and is the surface to prefer.

        Adds ``rate``, ``p_superseded`` and ``confidence`` to every entry
        :meth:`volatility` returns.

        THE MODEL, and its limits, because a caller will act on this number.
        A fact is assumed to change as a memoryless (exponential) process with
        rate ``lambda = 1 / mean(intervals)``, so

            p_superseded = 1 - exp(-lambda * age)

        * MEMORYLESS IS WRONG for anything with a natural period -- a two-year
          lease, an annual renewal. It is the weakest defensible assumption, not
          the best available model, and a fact whose changes are regular will be
          reported as more uncertain than it is.
        * A fact restated ONCE has a single interval and essentially no rate
          information of its own. Its estimate is carried by the pooled prior,
          and ``confidence`` says ``"prior"`` rather than implying otherwise.
        * A fact never restated has no interval at all. It is not returned: an
          unchanging fact and an unobserved one are indistinguishable from the
          log, and guessing between them is what this method refuses to do.

        The per-fact rate is shrunk toward the pooled rate of every fact sharing
        the same ``entity`` name, with weight ``shrink`` (1.0 = one pseudo-
        observation of the pool). Calibration is measured, not asserted:
        ``evidence/staleness_calibration.json``.
        """
        rows = self.volatility(min_revisions=min_revisions, now=now)
        if not rows:
            return []
        pool = {}
        for r in rows:
            pool.setdefault(r["entity"], []).extend(r["intervals"])
        pool_rate = {k: (1.0 / float(np.mean(v)) if v and np.mean(v) > 0 else None)
                     for k, v in pool.items()}
        allv = [x for v in pool.values() for x in v]
        global_rate = 1.0 / float(np.mean(allv)) if allv and np.mean(allv) > 0 else None
        for r in rows:
            d = r["intervals"]
            own = 1.0 / float(np.mean(d)) if d and np.mean(d) > 0 else None
            prior = pool_rate.get(r["entity"]) or global_rate
            n_i = len(d)
            if own is not None and prior is not None:
                rate = (n_i * own + shrink * prior) / (n_i + shrink)
            else:
                rate = own if own is not None else prior
            r["rate"] = float(rate) if rate else None
            r["p_superseded"] = (
                float(1.0 - np.exp(-rate * max(0.0, r["age"])))
                if (rate and assume_memoryless) else None)
            r["model"] = ("exponential" if assume_memoryless else
                          "suppressed: did not beat a constant rate off its own "
                          "assumption (staleness_calibration.json)")
            r["confidence"] = ("prior" if n_i <= 1 else
                               "weak" if n_i <= 3 else "observed")
        rows.sort(key=lambda r: -(r["p_superseded"] if r["p_superseded"] is not None
                                  else r["age"] * (r["rate"] or 0.0)))
        return rows

    def search_batch(self, query_vecs, top_k: int = 10):
        """Cosine-only top-k ids for many queries in one matmul (no boosts, no filters)."""
        with self._lock:
            self._reload_if_modified()
            Q = np.atleast_2d(np.asarray(query_vecs, dtype=np.float32))
            Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
            N = self.arena.n_rows
            if N == 0:
                return [[] for _ in range(Q.shape[0])]
            S = self.arena.score_matrix(Q)
            k = int(min(max(1, top_k), N))
            out = []
            for j in range(Q.shape[0]):
                col = S[:, j]
                idx = np.argpartition(-col, k - 1)[:k]
                idx = idx[np.argsort(-col[idx])]
                out.append([(self.arena.ids[int(i)], float(col[int(i)])) for i in idx])
            return out

    # -- rewrite ------------------------------------------------------------
    def replace_all(self, records: Iterable[Dict[str, Any]], *, password=_KEEP,
                    vector_dtype=_KEEP, order=None, reclustered: bool = False) -> int:
        """Atomically rewrite the vault from ``records``. Returns the count.

        Temp file -> fsync -> exclusive lock -> ``os.replace`` -> directory fsync,
        then reload. A crash before the replace leaves the old file untouched.
        Records are stored verbatim; the policy layer never runs here.

        CONCURRENCY. The whole rewrite runs under the container's exclusive lock,
        which ``append_block`` also takes, so a rebuild and an append can no
        longer interleave. If another process appended between the snapshot and
        the lock, those blocks are read back under the lock and APPENDED to
        ``records`` (they arrived after the caller's snapshot, so no filter or
        edit of this rebuild applies to them) -- 3.0.0 took no lock here at all
        and silently discarded them, along with the ``add_fact`` that returned a
        document id for them. If the file was replaced outright by somebody else,
        :class:`~nanomem.errors.ContainerReplacedError` is raised instead of
        guessing.
        """
        with self._lock:
            self._assert_open("rewrite")
            if self.read_only:
                raise ReadOnlyVaultError(f"{self.filepath} is open read-only")
            recs = list(records)
            if order is not None:
                # APPLIED TO THE CALLER'S SNAPSHOT ONLY. 3.0.2 appended the
                # unflushed records first and then re-indexed with the caller's
                # order, which truncated the list back to its original length and
                # dropped every pending record -- silently defeating the guard
                # immediately below it, which exists for exactly that reason
                # (measured: 8 records in, 5 out, `get(id)` None for 3 ids
                # `add_fact` had already returned).
                idx = [int(i) for i in order]
                if sorted(idx) != list(range(len(recs))):
                    raise ValueError(
                        f"order must be a permutation of range({len(recs)}), "
                        f"got {len(idx)} indices")
                recs = [recs[i] for i in idx]
            if self._mt.n:
                # PENDING RECORDS ARE NOT THE CALLER'S TO DISCARD. `add_fact` has
                # already returned a document id for each of them and `count()`
                # already includes them; 3.0.1 called `self._mt.clear()` here and
                # they vanished (measured: 65 records in, 60 out, `get(id)` None
                # for the other 5). They post-date whatever snapshot the caller
                # built -- `Vault._rebuild` and friends flush first, so this only
                # ever fires for a direct `VaultEngine.replace_all` -- so they are
                # appended to the rewrite, exactly like blocks another process
                # appended under the lock below.
                recs = recs + [self._pending_record(i) for i in range(self._mt.n)]
            for r in recs:
                rv = int(r.get("revision", 1))
                if not (0 <= rv <= 2_147_483_647):
                    # The revision column is int32 on disk. 3.0.2 let numpy raise a
                    # bare OverflowError from inside the rewrite, which escapes the
                    # single `except NanomemError` guard `errors` promises -- the
                    # same defect `add_fact`'s range check was added to close --
                    # and it accepted NEGATIVE revisions that `add_fact` rejects,
                    # so the verbatim path and the write path disagreed.
                    raise ValueError(
                        f"revision must fit in an int32 (0..2147483647), got {rv} "
                        f"for record {r.get('id')!r}")
            new_password = self._password if password is _KEEP else password
            old = self.header
            dtype = old.vector_dtype if vector_dtype is _KEEP else vector_dtype
            header = Container._new_header(self.embed_dim, dtype, self.block_capacity,
                                           self.landmarks_per_block, new_password,
                                           reclustered=bool(reclustered))
            if new_password and self._password and new_password == self._password \
                    and old.kdf_id == _crypto.KDF_SCRYPT:
                header.kdf_salt = old.kdf_salt           # same password keeps working
                header.scrypt_log2_n = old.scrypt_log2_n
                header.scrypt_r, header.scrypt_p = old.scrypt_r, old.scrypt_p
            keys = None
            if new_password:
                keys = _crypto.derive_keys(new_password, header.kdf_salt,
                                           header.scrypt_log2_n, header.scrypt_r,
                                           header.scrypt_p)
            lm_fn = self._landmark_fn(self.landmarks_per_block)

            def blobs():
                seq = 0
                for i in range(0, len(recs), self.block_capacity):
                    batch = recs[i:i + self.block_capacity]
                    V = np.ascontiguousarray(
                        [_unit(r["embedding"]) for r in batch], dtype=np.float32)
                    ts, rev, ids, groups, docs = self._record_columns(batch)
                    lm = (lm_fn(V, self.landmarks_per_block, seq)
                          if lm_fn is not None and len(batch) > self.landmarks_per_block
                          else None)
                    yield _c.build_block_blob(header, keys, seq, V, lm, ts, rev,
                                              ids, groups, docs)
                    seq += 1

            tmp = _c.temp_path_for(self.filepath)
            try:
                rows_before = self.arena.n_rows
                with self._cont.exclusive(sink=self.arena) as (lock_f, state):
                    if state == "replaced":
                        raise ContainerReplacedError(
                            f"{self.filepath} was replaced by another writer; "
                            f"reload and retry the rebuild")
                    if state == "appended" and self.arena.n_rows > rows_before:
                        recs = recs + [
                            self._arena_record(r)
                            for r in range(rows_before, self.arena.n_rows)]
                    self._cont.write_new_file(tmp, header, keys, blobs())
                    # `lock_f` is the exclusive handle ON THE FILE BEING
                    # REPLACED. Windows will not replace a path anything holds
                    # open, so `replace_with` closes it there and leaves it
                    # alone everywhere else.
                    self._cont.replace_with(tmp, holding=lock_f)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            if keys is not None:
                keys.wipe()              # 3.0.1 left this derived key alive
            self._password = new_password
            self._mt.clear()
            self._drop_arena_cache()     # it describes a file that is now gone
            self._full_reload()
            return len(recs)

    def _pending_record(self, i: int) -> Dict[str, Any]:
        """One unflushed memtable row as an owned record dict."""
        it = self._mt.items[i]
        return {"id": it["id"], "text": it["text"], "source": it["source"],
                "metadata": dict(it["metadata"]), "timestamp": float(it["timestamp"]),
                "revision": int(it["revision"]),
                "embedding": np.array(self._mt.vec[i], copy=True)}

    def _arena_record(self, r: int) -> Dict[str, Any]:
        """One flushed row as an owned record dict (used to absorb concurrent appends)."""
        rec = self.arena.record(r)
        return {"id": self.arena.ids[r], "text": rec.get("text", ""),
                "source": rec.get("source", "unknown"),
                "metadata": dict(rec.get("metadata") or {}),
                "timestamp": float(self.arena.ts[r]),
                "revision": int(self.arena.rev[r]),
                "embedding": self.arena.vector(r)}

    def compact(self, recluster: bool = False) -> Dict[str, Any]:
        """Coalesce under-filled blocks; ``recluster=True`` also re-orders rows.

        The re-cluster is a global spherical k-means over every vector (measured
        ~3 s at 71k docs). It is the only measured remedy for an unroutable
        random-insertion layout, and it changes file order -- so
        ``get_all_records`` / ``/dump`` order changes with it.
        """
        with self._lock:
            before_blocks = len(self._blocks_meta())
            recs = list(self.iter_records(include_embeddings=True))
            order = None
            t0 = time.perf_counter()
            if recluster and recs:
                V = np.ascontiguousarray([r["embedding"] for r in recs], dtype=np.float32)
                order = _routing.recluster(V, cell_target=self.block_capacity)
            n = self.replace_all(recs, order=order, reclustered=bool(recluster and recs))
            return {"records": n, "blocks_before": before_blocks,
                    "blocks_after": len(self._blocks_meta()),
                    "reclustered": bool(recluster),
                    "elapsed_s": round(time.perf_counter() - t0, 3)}

    def rekey(self, new_password: Optional[str]) -> None:
        """Re-encrypt under a new password, or decrypt to plaintext with ``None``."""
        with self._lock:
            recs = list(self.iter_records(include_embeddings=True))
            self.replace_all(recs, password=new_password)

    # -- stats --------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Measured, JSON-serialisable vault statistics (never constants)."""
        with self._lock:
            self._reload_if_modified()
            res = self.arena.resident_bytes()
            mt_bytes = self._mt.nbytes()
            heap = (res["arena_bytes"] + res["landmark_bytes"] + res["column_bytes"]
                    + res["record_bytes"] + res["index_bytes_estimated"] + mt_bytes)
            tm = None
            try:
                import tracemalloc
                if tracemalloc.is_tracing():
                    tm = round(tracemalloc.get_traced_memory()[0] / 1024.0, 1)
            except Exception:
                tm = None
            try:
                size_mb = os.path.getsize(self.filepath) / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            h = self.header
            encrypted = bool(h.encrypted)
            cell = self._router
            out = {
                # --- the eight legacy keys, with honest values ---
                "file_path": self.filepath,
                "file_size_mb": round(size_mb, 3),
                "total_documents": int(self.arena.n_rows + self._mt.n),
                "compacted_blocks": len(self._blocks_meta()),
                "memtable_pending": int(self._mt.n),
                "active_heap_ram_kb": round(heap / 1024.0, 1),
                "active_heap_ram_method": (
                    "sum of the numpy buffers this vault has ALLOCATED (capacity, "
                    "not just the used slice) + record bytes + a 64 B/row index "
                    "estimate; never a constant. It is what the VAULT holds, not "
                    "what the PROCESS costs: the interpreter, numpy itself and "
                    "transient copies are not in it. MEASURED GAP, in a process "
                    "that does nothing but open a vault it did not write and "
                    "read stats once: 1.10x at 71,433 docs (288.9 MB of RSS "
                    "growth against 262.0 MB reported) and 1.38x at 10,000 "
                    "(50.7 MB against 36.8 MB) -- "
                    "evidence/stats_rss_gap_v3r5.json. Both sides of that "
                    "gap moved when Arena.reserve became the default: the older "
                    "figure (1.52x, 612.3 MB against 401.6 MB, "
                    "evidence/rss_v3r4.json) was measured on an arena "
                    "that grew by doubling. Use 'process_rss_kb' for the process "
                    "figure. A CACHED OPEN (`arena_source == 'cache'`) holds "
                    "almost none of this in the heap at all: the vectors, the "
                    "columns, the record sections and the id table are clean, "
                    "evictable, file-backed pages of `<vault>.arena`, counted in "
                    "'mapped_bytes' instead. That is why this key can read ~0 on "
                    "a 71,433-document vault -- the bytes did not vanish, they "
                    "moved to a page the OS may drop."),
                "tracemalloc_current_kb": tm,
                "encrypted_at_rest": encrypted,
                "cipher": _crypto.CIPHER_LABEL_SHAKE if encrypted else _crypto.CIPHER_LABEL_NONE,
                # --- v3 additions ---
                "engine_version": ENGINE_VERSION,
                "format_version": 2 if self._v2_readonly else FILE_VERSION,
                "embed_dim": int(self.embed_dim),
                "vector_dtype": h.vector_dtype,
                "vector_dtype_on_disk": h.vector_dtype,
                "block_capacity": int(self.block_capacity),
                "landmarks_per_block": int(self.landmarks_per_block),
                "score_scale": self.score_mode,
                "router": self.router_mode,
                "screen": self.screen_mode,
                "screen_built": self._screen is not None,
                "screen_dims": int(self.screen_dims),
                "screen_min_rows": int(self.screen_min_rows),
                "screen_resident_mb": round(
                    (self._screen.resident_bytes() if self._screen else 0)
                    / (1024 * 1024), 3),
                "screen_exact": True,
                "routing_mode": "cells" if self._mode(None) == "cells" else "exhaustive",
                "n_exhaustive": int(self.n_exhaustive),
                "beam_fraction": float(self.beam_frac),
                "beam_cells": int(_routing.beam_size(cell.n_cells, self.beam_frac,
                                                     self.beam_min_cells)) if cell else 0,
                "resident_arena_mb": round(res["arena_bytes"] / (1024 * 1024), 3),
                "resident_arena_used_mb": round(res["arena_used_bytes"] / (1024 * 1024), 3),
                "resident_arena_mapped_mb": round(res["arena_mapped_bytes"] / (1024 * 1024), 3),
                "arena_residency": res["arena_residency"],
                "arena_residency_exact": res["arena_residency"] != "int8",
                "arena_dtype": res["arena_dtype"],
                "arena_reserved_rows": int(self.arena.reserved_rows),
                "arena_cache": self.arena_cache,
                "arena_cache_vectors": self.arena_cache_vectors,
                "arena_cache_records": self.arena_cache_records,
                "arena_source": self._arena_cache_info.get("source", "scan"),
                # what THIS open re-read of the vault: "all" / "appended tail
                # only" / "none". Here as well as in arena_cache_info() because
                # a default that stops checking should be visible where an
                # operator already looks.
                "arena_vault_blocks_checked": self._arena_cache_info.get(
                    "vault_blocks_checked", "all"),
                "arena_cache_path": self.arena_cache_path(),
                "arena_cache_bytes": _arena_cache_bytes(self.arena_cache_path()),
                "arena_from_cache": bool(res.get("arena_from_cache")),
                "arena_vectors_mapped": bool(res.get("arena_vectors_mapped")),
                "mapped_bytes": int(res.get("arena_cache_mapped_bytes", 0)
                                    + res.get("record_mapped_bytes", 0)
                                    + res.get("id_table_mapped_bytes", 0)
                                    + res.get("column_mapped_bytes", 0)),
                "record_mapped_bytes": int(res.get("record_mapped_bytes", 0)),
                "column_mapped_bytes": int(res.get("column_mapped_bytes", 0)),
                "id_table_mapped_bytes": int(res.get("id_table_mapped_bytes", 0)),
                "max_boost": round(float(self.intent_boost)
                                   + float(max(0.0, self.group_hoist))
                                   + float(max(0.0, self.revision_lead))
                                   + float(max(0.0, self.temporal_prior)), 4),
                "intent_boost": float(self.intent_boost),
                "group_hoist": float(self.group_hoist),
                "revision_lead": float(self.revision_lead),
                "group_cos_delta": float(self.group_cos_delta),
                "window_delta": float(self.window_delta),
                "arena_bytes": res["arena_bytes"],
                "arena_used_bytes": res["arena_used_bytes"],
                # Address space reserved behind the vectors so that growth never
                # copies. Untouched pages of an anonymous mapping are NOT
                # resident, so this is not RAM and must not be added to
                # `arena_bytes`; `arena_growth_copies` is how many times the
                # arena has been copied since it was created, which is the
                # number the reservation exists to keep at zero.
                "arena_reservation_bytes": res["arena_reservation_bytes"],
                "arena_reservation_is_mapped": res["arena_reservation_is_mapped"],
                "arena_growth_copies": res["arena_growth_copies"],
                "landmark_bytes": res["landmark_bytes"],
                "column_bytes": res["column_bytes"],
                "record_bytes": res["record_bytes"],
                "memtable_bytes": mt_bytes,
                "index_bytes_estimated": res["index_bytes_estimated"],
                "process_peak_rss_kb": _c.peak_rss_kb(),
                "process_rss_kb": _c.current_rss_kb(),
                "integrity_errors": list(self._cont.integrity_errors) if self._cont else [],
                "truncated_tail_bytes": int(self._cont.truncated_tail_bytes) if self._cont else 0,
                "read_only": bool(self.read_only),
                "durable": self.durable,
                "kdf": _crypto.kdf_label(h.kdf_id, h.scrypt_log2_n, h.scrypt_r, h.scrypt_p),
                "vault_uuid": bytes(h.vault_uuid).hex(),
            }
            return out

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        """Idempotent. Wipes key material and drops the passphrase; no OS handle
        is held to release. Any write after this raises
        :class:`~nanomem.errors.ClosedVaultError`.

        ``close()`` does NOT flush -- an explicit close of a half-built batch is
        a discard, and that is deliberate. ``flush()`` first, or use the context
        manager (``with VaultEngine(...) as e``), which flushes for you;
        ``Vault.close()`` already does.
        """
        with self._lock:
            if self._cont is not None:
                self._cont.close()
            # Under a mapped/int8 residency the arena owns an unlinked sidecar
            # fd; dropping it here is what makes the temp file's blocks free
            # instead of waiting on the garbage collector.
            if getattr(self, "arena", None) is not None:
                self.arena.close()
            self._password = None        # 3.0.1 left the passphrase alive as a str
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        """Flush, then close. A clean exit never drops a record.

        3.0.2 called ``close()`` alone, so the idiomatic ``with VaultEngine(...)``
        block DISCARDED every unflushed record although ``add_fact`` had returned
        a document id for each of them and ``count()`` had included them
        (measured: 1 in, 0 out, ``get(id)`` None). ``Vault.close()`` flushes, so
        the high-level API was always safe; the engine's own context manager --
        new in v3 -- was not. On an exception the pending records are still
        written, because they are data the caller was told nanomem holds; if the
        flush itself fails the vault is closed anyway and the flush error
        propagates.
        """
        try:
            if exc_type is None or self._mt.n:
                self.flush()
        finally:
            self.close()

    _matches_filter = staticmethod(matches_filter)


__all__ = [
    "VaultEngine", "ENGINE_VERSION", "matches_filter", "legacy_score",
    "legacy_score_to_cosine", "NanomemError", "CorruptContainerError",
    "IntegrityError", "WrongPasswordError", "PasswordRequiredError",
    "ReadOnlyVaultError", "ContainerReplacedError", "BLOCK_CAPACITY",
    "DEFAULT_EMBED_DIM", "FILE_MAGIC", "PAGE_ALIGNMENT",
]
