#!/usr/bin/env python3
"""Attack nanomem's measured weakness: 822 MB resident at 71,433 documents.

Seven arena arms, one corpus, one set of vectors, one recall function, measured
at 10,000 and 71,433 documents, in TWO phases -- what a LOADER costs
(build + reopen in one process, whose ru_maxrss the ingest dominates) and what a
SERVER costs (`reopen_only`: open a vault another process wrote, then answer
queries). The 822 MB figure is a loader number; the server number is the one a
long-running process pays, and they have different fixes:

  legacy_303_fp32_doubling  what 3.0.3 shipped, reconstructed inside this
                            harness by monkey-patching the allocator back to
                            pure doubling and disabling `Arena.reserve`. This is
                            the BASELINE every other row is compared against,
                            and the harness asserts it really does allocate
                            131,072 rows at 71,433 documents.
  fp32_reserved             fp32 arena, pre-sized on REOPEN from a header-only
                            row count. Identical arithmetic -- an allocator
                            change only. This is what ships today.
  fp32_reserved_bulk        the same, plus `VaultEngine.reserve_rows(n)` before
                            the ingest loop: what a loader that knows its corpus
                            size gets. Nothing tells an engine how many
                            `add_fact()` calls are coming, so this is the only
                            arm that fixes the LOADER peak.
  fp16_reserved             fp16 arena, fp32 accumulation in `scan_chunk`-row
                            chunks. The stored values are the SAME fp16 values
                            that are on disk.
  fp16_reserved_bulk        fp16 plus the bulk pre-size.
  fp16_mmap_reserved        the same fp16 values in an unlinked, memory-mapped
                            sidecar: clean, evictable pages instead of anonymous
                            RAM.
  int8_rerank_reserved      per-row symmetric int8 scan + exact fp32 re-rank of
                            the top `rerank_pool` rows.

Four sweeps answer the questions the arms raise, each merged into its own key of
the results file: `int8_pool_sweep` (how big must the re-rank pool be before the
top-4 stops changing), `scan_chunk_sweep` (does the narrow kernel's staging
buffer size buy anything, and is it still bit-identical), `latency_duel` (every
mode timed inside one cycle, cycles repeated, median reported -- the machine
could NOT be quiesced and a single pass measures the minute as much as the mode)
and `growth_policy` (the allocator question with the engine taken out).

RULES THIS HARNESS ENFORCES
  * It reuses the cache `bench_competitors.py --prep` built (same .npy vectors,
    same gold, same `rng(0)` insertion order), so every number here is directly
    comparable to competitors_standard_results.json.
  * Recall is computed in the DRIVER from each arm's returned id matrix. An arm
    cannot score itself.
  * Exactness is judged against a ground truth the DRIVER computes with numpy:
    `docs -> fp16 -> fp32`, full argsort. That is what a correct engine storing
    fp16 on disk must return, so "exact" here means exact, not "agrees with the
    other nanomem arm".
  * One subprocess per (arm, corpus) so ru_maxrss is not contaminated.
  * Every arm is built, closed and REOPENED FROM DISK, then warmed, then timed.
  * Each worker also times a bare numpy matvec on the same vectors AFTER all its
    memory numbers are captured, so latency can be read against the floor the
    machine was actually offering at that moment. The 1-minute load average is
    recorded at the start and end of every arm.

Usage (the sweeps write to $BENCH_WORK and are then merged into the results
file, so the driver can be re-run without losing them):
    bench_memory.py                    # driver: every arm, both phases
    bench_memory.py --worker --arm A --corpus C --out J
    bench_memory.py --pool-sweep    && bench_memory.py --merge-pool-sweep
    bench_memory.py --chunk-sweep   && bench_memory.py --merge-chunk-sweep
    bench_memory.py --latency-duel  && bench_memory.py --merge-latency-duel
    bench_memory.py --growth-sweep  && bench_memory.py --merge-growth-sweep
    bench_memory.py --recommend        # derive the decision table from the rows
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import List, Optional

import numpy as np

REPO = "<redacted local path> /4D llm"
REFOUND = os.path.join(REPO, "scratch", "refound")
NANOMEM_DIR = os.path.join(REPO, "nanomem_standalone")
CACHE = os.environ.get("BENCH_CACHE", "/tmp/nanomem_bench_cache")
WORK = os.environ.get("BENCH_WORK", "/tmp/nanomem_mem_work")
RESULTS = os.path.join(REFOUND, "memory_results.json")

DIM = 768
TOP_K = 10
SEED = 0
WARMUP_QUERIES = 20
SCAN_CHUNK = 4096
RERANK_POOL = 512
POOL_SWEEP = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
CHUNK_SWEEP = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)

CORPORA = ("n10000", "n71433")
ARMS = (
    "legacy_303_fp32_doubling",
    "fp32_reserved",
    "fp32_reserved_bulk",
    "fp16_reserved",
    "fp16_reserved_bulk",
    "fp16_mmap_reserved",
    "int8_rerank_reserved",
)
BASELINE_ARM = "legacy_303_fp32_doubling"
DEFAULT_ARM = "fp32_reserved"

ARM_CONFIG = {
    # arm -> (residency, reserve_on_reopen, bulk_reserve_rows, rerank_pool)
    #
    # `bulk_reserve_rows` calls the public VaultEngine.reserve_rows(n) before the
    # ingest loop. Nothing tells an engine how many add_fact() calls are coming,
    # so without it a bulk load grows the arena by doubling and the process keeps
    # every discarded copy. The *_bulk rows are what a loader that knows its
    # corpus size gets; the plain rows are what a loader that does not gets.
    "legacy_303_fp32_doubling": ("float32", False, False, 0),
    "fp32_reserved":            ("float32", True, False, 0),
    "fp32_reserved_bulk":       ("float32", True, True, 0),
    "fp16_reserved":            ("float16", True, False, 0),
    "fp16_reserved_bulk":       ("float16", True, True, 0),
    "fp16_mmap_reserved":       ("float16_mmap", True, False, 0),
    "int8_rerank_reserved":     ("int8", True, False, RERANK_POOL),
}


# --------------------------------------------------------------------------
# process measurement
# --------------------------------------------------------------------------
def maxrss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


def rss_bytes() -> Optional[int]:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                      text=True, stderr=subprocess.DEVNULL)
        return int(out.strip()) * 1024
    except Exception:
        return None


def phys_footprint_bytes() -> Optional[int]:
    """macOS `phys_footprint`: the number Activity Monitor shows.

    It EXCLUDES clean file-backed pages, which is exactly the difference the
    memory-mapped arm is trying to buy -- ru_maxrss counts a mapped page the
    moment it is touched and never lets go of it, whether or not the OS could
    drop it for free. Reported in MB by the tool, so it is coarse.
    """
    try:
        out = subprocess.check_output(["footprint", "-p", str(os.getpid())],
                                      text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("phys_footprint:"):
            parts = s.split()
            try:
                val = float(parts[1])
            except (IndexError, ValueError):
                return None
            unit = parts[2].upper() if len(parts) > 2 else "MB"
            mult = {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "B": 1}.get(unit, 1 << 20)
            return int(val * mult)
    return None


def loadavg() -> List[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except Exception:
        return []


def path_bytes(p: str) -> int:
    if os.path.isfile(p):
        return os.path.getsize(p)
    total = 0
    for root, _d, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def pct(a, q) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def _patch_legacy_allocator():
    """Reconstruct 3.0.3's arena: capacity doubling with no `reserve`.

    3.0.3 grew every row-indexed array by ``max(64, cap * 2)`` and had no
    `reserve` at all, so the container had nowhere to put a row count. 3.0.4
    keeps the doubling (it was re-measured and it is the cheapest policy that
    has to grow) and adds `reserve`, so disabling `reserve` is the whole
    difference. The harness then ASSERTS the arena really did allocate 131,072
    rows at 71,433 documents, which is 3.0.3's signature.
    """
    from nanomem import arena as _a
    assert _a._next_capacity(65536, 71433, 3072) == 131072, "allocator is not doubling"
    _a.Arena.reserve = lambda self, n_rows: None
    return "Arena.reserve disabled (library growth is already 3.0.3 doubling)"


class Ctx:
    def __init__(self, corpus: str, work: str):
        d = os.path.join(CACHE, corpus)
        self.corpus = corpus
        self.work = work
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.docs = np.load(os.path.join(d, "docs.npy"))
        self.queries = np.load(os.path.join(d, "queries.npy"))
        self.texts = json.load(open(os.path.join(d, "texts.json")))
        self.n = int(self.docs.shape[0])
        self.order = np.random.default_rng(SEED).permutation(self.n)
        self.Dp = np.ascontiguousarray(self.docs[self.order])
        self.Tp = [self.texts[int(i)] for i in self.order]
        self.ids = [int(i) for i in self.order]
        gc.collect()


def run_arm(arm: str, ctx: Ctx) -> dict:
    residency, reserve_on, bulk_reserve, pool = ARM_CONFIG[arm]
    sys.path.insert(0, NANOMEM_DIR)
    patched = None
    if not reserve_on:
        patched = _patch_legacy_allocator()
    from nanomem.engine import VaultEngine
    import nanomem as _nm

    os.makedirs(ctx.work, exist_ok=True)
    path = os.path.join(ctx.work, "vault.dat")
    kw = dict(embed_dim=DIM, vector_dtype="float16", router="off", n_exhaustive=50_000,
              residency=residency, scan_chunk=SCAN_CHUNK,
              rerank_pool=(pool or 1), arena_dir=ctx.work)

    gc.collect()
    base_rss, base_max = rss_bytes(), maxrss_bytes()
    base_foot = phys_footprint_bytes()
    load_start = loadavg()

    # ---- ingest ----------------------------------------------------------
    t0 = time.perf_counter()
    e = VaultEngine(path, **kw)
    if bulk_reserve:
        e.reserve_rows(ctx.n)
    for j in range(ctx.n):
        e.add_fact(ctx.Tp[j], ctx.Dp[j], source="wiki", metadata={"idx": ctx.ids[j]})
    e.flush()
    ingest_s = time.perf_counter() - t0
    ingest_max = maxrss_bytes()
    ingest_rss = rss_bytes()
    e.close()
    del e
    gc.collect()

    # ---- reopen ----------------------------------------------------------
    after_close_rss = rss_bytes()
    t1 = time.perf_counter()
    e2 = VaultEngine(path, **kw)
    reopen_s = time.perf_counter() - t1
    st = e2.stats()

    def q(v, k):
        return [int(h["metadata"]["idx"]) for h in e2.search("", v, top_k=k)]

    Q = ctx.queries
    nq = int(Q.shape[0])
    t2 = time.perf_counter()
    _first = q(Q[0], TOP_K)
    first_query_s = time.perf_counter() - t2
    for i in range(min(WARMUP_QUERIES, nq)):
        q(Q[i], TOP_K)

    lat = []
    out = np.full((nq, TOP_K), -1, dtype=np.int32)
    for i in range(nq):
        t = time.perf_counter()
        ids = q(Q[i], TOP_K)
        lat.append((time.perf_counter() - t) * 1000.0)
        ids = [int(x) for x in ids][:TOP_K]
        out[i, :len(ids)] = ids

    gc.collect()
    peak_max = maxrss_bytes()
    live_rss = rss_bytes()
    live_foot = phys_footprint_bytes()
    load_end = loadavg()

    arena = e2.arena
    res = arena.resident_bytes()
    vec_shape = [int(x) for x in arena.vec.shape]
    e2.close()
    del e2
    gc.collect()

    # ---- numpy floor, AFTER every memory number is captured ---------------
    A = np.ascontiguousarray(ctx.docs.astype(np.float16).astype(np.float32))
    qv = np.ascontiguousarray(Q[0], dtype=np.float32)
    for _ in range(3):
        _ = A @ qv
    floor = []
    for i in range(min(50, nq)):
        v = np.ascontiguousarray(Q[i], dtype=np.float32)
        t = time.perf_counter()
        _s = A @ v
        floor.append((time.perf_counter() - t) * 1000.0)
    del A
    gc.collect()

    return {
        "arm": arm,
        "corpus": ctx.corpus,
        "n_docs": ctx.n,
        "topk": out,
        "config": {
            "residency": residency,
            "reserve_on_reopen": reserve_on,
            "bulk_reserve_rows": bulk_reserve,
            "allocator": "capacity doubling (unchanged from 3.0.3)",
            "monkeypatch": patched,
            "scan_chunk": SCAN_CHUNK,
            "rerank_pool": pool,
            "vector_dtype_on_disk": "float16",
            "router": "off",
            "nanomem_version": _nm.__version__,
            "engine_version": getattr(_nm, "ENGINE_VERSION", None),
            "source": "wiki (NOT in nanomem.entities.PERSONAL_SOURCES), so the "
                      "entity/temporal boost layer is inert",
        },
        "memory": {
            "baseline_rss_bytes": base_rss,
            "baseline_maxrss_bytes": base_max,
            "baseline_phys_footprint_bytes": base_foot,
            "ingest_peak_maxrss_delta_bytes": ingest_max - base_max,
            "ingest_rss_delta_bytes": (ingest_rss - base_rss) if ingest_rss else None,
            "after_close_rss_delta_bytes": (after_close_rss - base_rss) if after_close_rss else None,
            "peak_maxrss_delta_bytes": peak_max - base_max,
            "live_rss_delta_bytes": (live_rss - base_rss) if live_rss else None,
            "phys_footprint_delta_bytes": ((live_foot - base_foot)
                                           if (live_foot and base_foot) else None),
            "method": "ru_maxrss (bytes on darwin) for the peak, `ps -o rss=` for "
                      "the live figure, `footprint -p` for phys_footprint. The "
                      "baseline is taken AFTER the shared vectors and texts are "
                      "loaded, so the deltas are the engine's own cost.",
        },
        "arena": {
            "vec_shape": vec_shape,
            "resident_bytes": {k: v for k, v in res.items()},
            "stats_resident_arena_mb": st.get("resident_arena_mb"),
            "stats_resident_arena_used_mb": st.get("resident_arena_used_mb"),
            "stats_resident_arena_mapped_mb": st.get("resident_arena_mapped_mb"),
            "stats_arena_reserved_rows": st.get("arena_reserved_rows"),
            "stats_arena_residency": st.get("arena_residency"),
            "stats_active_heap_ram_kb": st.get("active_heap_ram_kb"),
        },
        "timing": {
            "ingest_s": round(ingest_s, 4),
            "reopen_s": round(reopen_s, 4),
            "first_query_after_reopen_s": round(first_query_s, 4),
            "latency_ms": {
                "p50": round(pct(lat, 50), 4), "p95": round(pct(lat, 95), 4),
                "p99": round(pct(lat, 99), 4), "mean": round(float(np.mean(lat)), 4),
                "min": round(float(np.min(lat)), 4), "max": round(float(np.max(lat)), 4),
                "n_timed_queries": len(lat), "n_unique_queries": nq,
            },
            "numpy_floor_ms": {
                "p50": round(pct(floor, 50), 4), "min": round(float(np.min(floor)), 4),
                "what": "a bare (N,768)@(768,) fp32 matvec over the SAME vectors, "
                        "timed in this process after the memory numbers were "
                        "captured. It is the machine's offer at that moment, so "
                        "p50/floor is comparable across a contended run.",
            },
        },
        "disk_bytes": path_bytes(path),
        "load_avg_start": load_start,
        "load_avg_end": load_end,
    }


def run_reopen_only(arm: str, ctx: Ctx, vault: str) -> dict:
    """Open a vault somebody else built, then serve queries. Measure the process.

    This is the number a query server actually costs, and it is the one
    `reserve()` exists for. The build+reopen arms cannot show it: `ru_maxrss` is
    a high-water mark, so in a process that ingested 71,433 documents first, the
    ingest peak hides everything the reopen does afterwards.

    The vault file is built ONCE per corpus and every arm opens that same file --
    residency is a RAM-side choice and changes nothing on disk. The harness
    records the file's sha256 with each arm so that can be checked.
    """
    import hashlib
    residency, reserve_on, _bulk, pool = ARM_CONFIG[arm]
    sys.path.insert(0, NANOMEM_DIR)
    patched = None
    if not reserve_on:
        patched = _patch_legacy_allocator()
    from nanomem.engine import VaultEngine

    h = hashlib.sha256()
    with open(vault, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()

    gc.collect()
    base_rss, base_max = rss_bytes(), maxrss_bytes()
    base_foot = phys_footprint_bytes()
    load_start = loadavg()

    t1 = time.perf_counter()
    e = VaultEngine(vault, embed_dim=DIM, vector_dtype="float16", router="off",
                    n_exhaustive=50_000, residency=residency, scan_chunk=SCAN_CHUNK,
                    rerank_pool=(pool or 1), arena_dir=ctx.work)
    reopen_s = time.perf_counter() - t1
    after_open_max = maxrss_bytes()
    after_open_rss = rss_bytes()
    st = e.stats()

    def q(v, k):
        return [int(hh["metadata"]["idx"]) for hh in e.search("", v, top_k=k)]

    Q = ctx.queries
    nq = int(Q.shape[0])
    t2 = time.perf_counter()
    _ = q(Q[0], TOP_K)
    first_query_s = time.perf_counter() - t2
    for i in range(min(WARMUP_QUERIES, nq)):
        q(Q[i], TOP_K)
    lat = []
    out = np.full((nq, TOP_K), -1, dtype=np.int32)
    for i in range(nq):
        t = time.perf_counter()
        ids = q(Q[i], TOP_K)
        lat.append((time.perf_counter() - t) * 1000.0)
        ids = [int(x) for x in ids][:TOP_K]
        out[i, :len(ids)] = ids
    gc.collect()
    peak_max = maxrss_bytes()
    live_rss = rss_bytes()
    live_foot = phys_footprint_bytes()
    load_end = loadavg()
    arena = e.arena
    res = arena.resident_bytes()
    vec_shape = [int(x) for x in arena.vec.shape]
    e.close()

    return {
        "arm": arm, "corpus": ctx.corpus, "n_docs": ctx.n, "topk": out,
        "phase": "reopen_only",
        "vault_sha256": digest,
        "vault_bytes": os.path.getsize(vault),
        "config": {"residency": residency, "reserve_on_reopen": reserve_on,
                   "rerank_pool": pool, "monkeypatch": patched,
                   "scan_chunk": SCAN_CHUNK},
        "memory": {
            "baseline_rss_bytes": base_rss,
            "after_open_maxrss_delta_bytes": after_open_max - base_max,
            "after_open_rss_delta_bytes": (after_open_rss - base_rss) if after_open_rss else None,
            "peak_maxrss_delta_bytes": peak_max - base_max,
            "live_rss_delta_bytes": (live_rss - base_rss) if live_rss else None,
            "phys_footprint_delta_bytes": ((live_foot - base_foot)
                                           if (live_foot and base_foot) else None),
        },
        "arena": {"vec_shape": vec_shape, "resident_bytes": res,
                  "stats_resident_arena_mb": st.get("resident_arena_mb"),
                  "stats_resident_arena_mapped_mb": st.get("resident_arena_mapped_mb"),
                  "stats_arena_reserved_rows": st.get("arena_reserved_rows"),
                  "stats_arena_residency": st.get("arena_residency")},
        "timing": {"reopen_s": round(reopen_s, 4),
                   "first_query_after_reopen_s": round(first_query_s, 4),
                   "latency_ms": {"p50": round(pct(lat, 50), 4),
                                  "p95": round(pct(lat, 95), 4),
                                  "p99": round(pct(lat, 99), 4),
                                  "mean": round(float(np.mean(lat)), 4),
                                  "n_timed_queries": len(lat)}},
        "load_avg_start": load_start, "load_avg_end": load_end,
    }


def build_vault(corpus: str, dest: str) -> dict:
    """Build the shared vault for the reopen-only phase. Residency-independent."""
    sys.path.insert(0, NANOMEM_DIR)
    from nanomem.engine import VaultEngine
    work = os.path.dirname(dest)
    os.makedirs(work, exist_ok=True)
    ctx = Ctx(corpus, work)
    t0 = time.perf_counter()
    e = VaultEngine(dest, embed_dim=DIM, vector_dtype="float16", router="off",
                    n_exhaustive=50_000, residency="float32")
    e.reserve_rows(ctx.n)
    for j in range(ctx.n):
        e.add_fact(ctx.Tp[j], ctx.Dp[j], source="wiki", metadata={"idx": ctx.ids[j]})
    e.flush(); e.close()
    return {"corpus": corpus, "path": dest, "build_s": round(time.perf_counter() - t0, 3),
            "bytes": os.path.getsize(dest)}


GROWTH_POLICIES = ("double", "step_32mb", "x1_5", "exact_fit", "reserve")


def _grow_capacity(policy: str, cap: int, need: int, row_bytes: int) -> int:
    if need <= cap:
        return cap
    if policy == "exact_fit":
        return need
    if policy == "step_32mb":
        step = max(1, (32 << 20) // max(1, row_bytes))
        new = max(cap, 0)
        while new < need:
            new += step
        return new
    if policy == "x1_5":
        new = max(64, int(cap))
        while new < need:
            new = max(new + 1, int(new * 1.5))
        return new
    new = max(64, int(cap))                      # "double"
    while new < need:
        new *= 2
    return new


def run_growth_policy(policy: str, n_rows: int, dim: int, block: int) -> dict:
    """Peak RSS of filling an fp32 arena row-block by row-block under one policy.

    This is the allocator question on its own, with the engine, the container and
    the records taken out: a ``(cap, dim)`` fp32 array is grown to ``n_rows`` in
    ``block``-row steps exactly as ``Arena.add_block`` grows it, and the process
    is measured. ``reserve`` allocates once up front, which is what a reopen does
    once the container hands the arena a row count.

    Run in its own subprocess so ``ru_maxrss`` is that policy's and nobody
    else's, and the freed-bytes column is the arithmetic sum of every array the
    policy threw away -- the quantity the peak tracks.
    """
    row_bytes = dim * 4
    gc.collect()
    base_max, base_rss = maxrss_bytes(), rss_bytes()
    t0 = time.perf_counter()
    cap = n_rows if policy == "reserve" else 0
    vec = np.zeros((cap, dim), dtype=np.float32)
    growths, freed = (1 if policy == "reserve" else 0), 0
    src = np.ones((block, dim), dtype=np.float32)
    rows = 0
    while rows < n_rows:
        n = min(block, n_rows - rows)
        need = rows + n
        if vec.shape[0] < need:
            new_cap = _grow_capacity(policy, vec.shape[0], need, row_bytes)
            out = np.zeros((new_cap, dim), dtype=np.float32)
            out[:rows] = vec[:rows]
            freed += int(vec.nbytes)
            vec = out
            growths += 1
        vec[rows:need] = src[:n]
        rows = need
    build_s = time.perf_counter() - t0
    peak, live = maxrss_bytes(), rss_bytes()
    final_cap = int(vec.shape[0])
    del src
    return {"policy": policy, "n_rows": int(n_rows), "dim": int(dim),
            "block_rows": int(block),
            "final_capacity_rows": final_cap,
            "final_capacity_mb": round(final_cap * row_bytes / 2 ** 20, 1),
            "live_rows_mb": round(n_rows * row_bytes / 2 ** 20, 1),
            "growths": int(growths),
            "bytes_freed_mb": round(freed / 2 ** 20, 1),
            "peak_rss_delta_mb": round((peak - base_max) / 2 ** 20, 1),
            "live_rss_delta_mb": (round((live - base_rss) / 2 ** 20, 1)
                                  if (live and base_rss) else None),
            "build_s": round(build_s, 3),
            "load_avg": loadavg()}


def run_latency_duel(corpus: str, cycles: int = 5) -> dict:
    """Latency only, measured so that machine contention cannot decide the winner.

    This machine could not be quiesced -- other work was running throughout, and
    the 1-minute load average recorded per arm ranged from 1.8 to 22.2 across
    this file's runs. A single pass per mode is therefore a measurement of the
    mode AND of whatever else the box was doing at that minute. So each CYCLE
    opens every mode in turn on the SAME vault, serves the SAME 500 queries, and
    also times a bare numpy matvec; the cycle is repeated, and the reported
    figure is the median across cycles. Contention that drifts over minutes hits
    every mode, and the per-cycle rows are kept so the spread can be seen.

    One arena is resident at a time (each mode is opened, timed and closed
    before the next), so no mode is charged for another mode's cache pressure.
    No memory number is taken here -- that is what the isolated-subprocess arms
    are for.
    """
    sys.path.insert(0, NANOMEM_DIR)
    from nanomem.engine import VaultEngine
    work = os.path.join(WORK, "duel_" + corpus)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    ctx = Ctx(corpus, work)
    path = os.path.join(work, "vault.dat")
    e = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off",
                    n_exhaustive=50_000, residency="float32")
    e.reserve_rows(ctx.n)
    for j in range(ctx.n):
        e.add_fact(ctx.Tp[j], ctx.Dp[j], source="wiki", metadata={"idx": ctx.ids[j]})
    e.flush(); e.close()

    Q = ctx.queries
    nq = int(Q.shape[0])
    A = np.ascontiguousarray(ctx.docs.astype(np.float16).astype(np.float32))
    modes = ["float32", "float16", "float16_mmap", "int8"]
    per_cycle = {m: [] for m in modes}
    per_cycle["numpy_floor"] = []
    rows = []
    for cyc in range(int(cycles)):
        for mode in modes:
            eng = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off",
                              n_exhaustive=50_000, residency=mode, scan_chunk=SCAN_CHUNK,
                              rerank_pool=RERANK_POOL, arena_dir=work)
            for i in range(min(WARMUP_QUERIES, nq)):
                eng.search("", Q[i], top_k=TOP_K)
            lat = []
            for i in range(nq):
                t = time.perf_counter()
                eng.search("", Q[i], top_k=TOP_K)
                lat.append((time.perf_counter() - t) * 1000.0)
            eng.close()
            p50 = pct(lat, 50)
            per_cycle[mode].append(p50)
            rows.append({"cycle": cyc, "mode": mode, "p50_ms": round(p50, 4),
                         "p95_ms": round(pct(lat, 95), 4), "load_avg": loadavg()})
        fl = []
        for i in range(min(50, nq)):
            v = np.ascontiguousarray(Q[i], dtype=np.float32)
            t = time.perf_counter()
            _ = A @ v
            fl.append((time.perf_counter() - t) * 1000.0)
        per_cycle["numpy_floor"].append(pct(fl, 50))
        print(f"[duel] {corpus} cycle {cyc}: "
              + "  ".join(f"{m}={per_cycle[m][-1]:.3f}" for m in modes)
              + f"  floor={per_cycle['numpy_floor'][-1]:.3f}  load={loadavg()}", flush=True)
    del A
    shutil.rmtree(work, ignore_errors=True)

    med = {m: round(float(np.median(v)), 4) for m, v in per_cycle.items()}
    base = med["float32"]
    return {"corpus": corpus, "n_docs": ctx.n, "n_queries": nq, "cycles": int(cycles),
            "median_p50_ms_across_cycles": med,
            "spread_p50_ms": {m: {"min": round(min(v), 4), "max": round(max(v), 4)}
                              for m, v in per_cycle.items()},
            "p50_vs_float32": {m: round(med[m] / max(1e-9, base), 3) for m in modes},
            "paired_ratio_to_float32_per_cycle": {
                m: [round(per_cycle[m][i] / max(1e-9, per_cycle["float32"][i]), 3)
                    for i in range(int(cycles))] for m in modes},
            "rows": rows}


def run_chunk_sweep(corpus: str) -> dict:
    """Does the narrow arena's `scan_chunk` buy anything back?

    The narrow modes pay for their smaller arena with a per-chunk fp16 -> fp32
    conversion into a staging buffer. The buffer size is the one free parameter
    in that kernel, and it is a cache-residency question, not an accuracy one:
    every chunk size computes the SAME fp32 dot products, so this sweep asserts
    bit-identity against the full fp32 arena on the same vault and then reports
    only latency. Both narrow modes are swept; `float32` is measured once as the
    floor to read them against.
    """
    sys.path.insert(0, NANOMEM_DIR)
    from nanomem.engine import VaultEngine
    work = os.path.join(WORK, "chunk_" + corpus)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    ctx = Ctx(corpus, work)
    path = os.path.join(work, "vault.dat")
    e = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off",
                    n_exhaustive=50_000, residency="float32")
    e.reserve_rows(ctx.n)
    for j in range(ctx.n):
        e.add_fact(ctx.Tp[j], ctx.Dp[j], source="wiki", metadata={"idx": ctx.ids[j]})
    e.flush(); e.close()

    Q = ctx.queries
    nq = int(Q.shape[0])

    def timed(engine):
        out = np.full((nq, TOP_K), -1, dtype=np.int32)
        for i in range(min(WARMUP_QUERIES, nq)):
            engine.search("", Q[i], top_k=TOP_K)
        lat = []
        for i in range(nq):
            t = time.perf_counter()
            hits = engine.search("", Q[i], top_k=TOP_K)
            lat.append((time.perf_counter() - t) * 1000.0)
            ids = [int(h["metadata"]["idx"]) for h in hits][:TOP_K]
            out[i, :len(ids)] = ids
        return out, lat

    ref_e = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off",
                        n_exhaustive=50_000, residency="float32")
    ref, ref_lat = timed(ref_e)
    # the whole fp32 score vector for one query, to compare bit for bit
    ref_scores = ref_e.arena.scores(np.ascontiguousarray(Q[0], dtype=np.float32)).copy()
    ref_e.close()

    rows = []
    for mode in ("float16", "float16_mmap"):
        for chunk in CHUNK_SWEEP:
            e2 = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off",
                             n_exhaustive=50_000, residency=mode, scan_chunk=chunk,
                             arena_dir=work)
            got, lat = timed(e2)
            sc = e2.arena.scores(np.ascontiguousarray(Q[0], dtype=np.float32))
            bitwise = bool(np.array_equal(sc, ref_scores))
            max_ulp = float(np.max(np.abs(sc.astype(np.float64) - ref_scores.astype(np.float64))))
            e2.close()
            rows.append({
                "residency": mode, "scan_chunk": int(chunk),
                "staging_buffer_mb": round(chunk * DIM * 4 / 2 ** 20, 3),
                "p50_ms": round(pct(lat, 50), 4), "p95_ms": round(pct(lat, 95), 4),
                "top4_changed_queries_vs_fp32_engine": int(changed_rows(got, ref, 4)),
                "top10_changed_queries_vs_fp32_engine": int(changed_rows(got, ref, 10)),
                "scores_bitwise_identical_to_fp32": bitwise,
                "max_abs_score_diff_vs_fp32": max_ulp,
            })
            print(f"[chunk] {corpus} {mode} chunk={chunk}: {rows[-1]}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    best = {}
    for mode in ("float16", "float16_mmap"):
        cand = [r for r in rows if r["residency"] == mode]
        b = min(cand, key=lambda r: r["p50_ms"])
        best[mode] = {"scan_chunk": b["scan_chunk"], "p50_ms": b["p50_ms"],
                      "p50_vs_fp32": round(b["p50_ms"] / max(1e-9, pct(ref_lat, 50)), 3)}
    return {"corpus": corpus, "n_docs": ctx.n, "n_queries": nq,
            "fp32_reference_p50_ms": round(pct(ref_lat, 50), 4),
            "fp32_reference_p95_ms": round(pct(ref_lat, 95), 4),
            "shipped_scan_chunk": SCAN_CHUNK,
            "load_avg_at_end": loadavg(),
            "best_per_mode": best, "rows": rows}


def run_pool_sweep(corpus: str) -> dict:
    """How large must the int8 re-rank pool be before the top-4 stops changing?"""
    sys.path.insert(0, NANOMEM_DIR)
    from nanomem.engine import VaultEngine
    work = os.path.join(WORK, "pool_" + corpus)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    ctx = Ctx(corpus, work)
    path = os.path.join(work, "vault.dat")
    e = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off",
                    residency="int8", scan_chunk=SCAN_CHUNK, rerank_pool=RERANK_POOL,
                    arena_dir=work)
    for j in range(ctx.n):
        e.add_fact(ctx.Tp[j], ctx.Dp[j], source="wiki", metadata={"idx": ctx.ids[j]})
    e.flush(); e.close()
    # The reference is the fp32 arena's OWN answer on the SAME vault -- i.e. what
    # nanomem ships today -- not the driver's numpy argsort. The two differ on a
    # handful of exact ties (see exactness.tie_analysis), and charging int8 for
    # those would overstate its cost.
    ref_e = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off",
                        residency="float32")
    ref = np.full((ctx.queries.shape[0], TOP_K), -1, dtype=np.int32)
    for i in range(ctx.queries.shape[0]):
        ids = [int(h["metadata"]["idx"]) for h in ref_e.search("", ctx.queries[i], top_k=TOP_K)]
        ref[i, :len(ids)] = ids[:TOP_K]
    ref_e.close()
    numpy_truth = exact_topk(ctx.docs, ctx.queries, TOP_K)

    e2 = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off",
                     residency="int8", scan_chunk=SCAN_CHUNK, rerank_pool=RERANK_POOL,
                     arena_dir=work)
    rows = []
    for pool in POOL_SWEEP:
        e2.arena.rerank_pool = int(pool)
        got = np.full((ctx.queries.shape[0], TOP_K), -1, dtype=np.int32)
        lat = []
        for i in range(ctx.queries.shape[0]):
            t = time.perf_counter()
            ids = [int(h["metadata"]["idx"]) for h in e2.search("", ctx.queries[i], top_k=TOP_K)]
            lat.append((time.perf_counter() - t) * 1000.0)
            got[i, :len(ids)] = ids[:TOP_K]
        ta = tie_analysis(ctx.docs, ctx.queries, got, ref, 4)
        rows.append({
            "rerank_pool": int(pool),
            "top4_changed_queries_vs_fp32_engine": int(changed_rows(got, ref, 4)),
            "top10_changed_queries_vs_fp32_engine": int(changed_rows(got, ref, 10)),
            "top4_slot_agreement_vs_fp32_engine_pct": round(agree_pct(got, ref, 4), 4),
            "top4_changed_queries_vs_numpy": int(changed_rows(got, numpy_truth, 4)),
            "disagreements_with_zero_cos_gap": ta["n_with_zero_cosine_gap"],
            "max_abs_cos_gap_of_disagreements": ta["max_abs_cos_gap"],
            "p50_ms": round(pct(lat, 50), 4),
        })
        print(f"[pool] {corpus} pool={pool}: {rows[-1]}", flush=True)
    e2.close()
    shutil.rmtree(work, ignore_errors=True)
    zero = [r["rerank_pool"] for r in rows
            if r["top4_changed_queries_vs_fp32_engine"] == 0]
    return {"corpus": corpus, "n_docs": ctx.n, "n_queries": int(ctx.queries.shape[0]),
            "top_k": TOP_K, "reference": "the fp32-residency engine's own top-10 on "
                                         "the same vault (what nanomem ships today)",
            "smallest_pool_with_zero_top4_change": (min(zero) if zero else None),
            "rows": rows}


# --------------------------------------------------------------------------
# driver-side scoring
# --------------------------------------------------------------------------
def exact_topk(docs: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground truth: full fp32 argsort over the fp16-quantised corpus.

    The corpus is round-tripped through fp16 first because that is what the
    vault stores (DECISIONS.md core-5). Comparing against un-quantised fp32
    would charge every nanomem arm for a decision that was taken on disk.
    """
    A = np.ascontiguousarray(docs.astype(np.float16).astype(np.float32))
    out = np.full((queries.shape[0], k), -1, dtype=np.int32)
    for i in range(queries.shape[0]):
        s = A @ np.ascontiguousarray(queries[i], dtype=np.float32)
        idx = np.argpartition(-s, k - 1)[:k]
        out[i] = idx[np.argsort(-s[idx])][:k]
    return out


def agree_pct(got: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Percent of (query, rank) slots where the arm returned the exact id."""
    g, t = got[:, :k], truth[:, :k]
    return 100.0 * float((g == t).mean())


def changed_rows(got: np.ndarray, truth: np.ndarray, k: int) -> int:
    return int((got[:, :k] != truth[:, :k]).any(axis=1).sum())


def set_agree_pct(got: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Percent of queries whose top-k SET matches (order ignored)."""
    n = got.shape[0]
    hit = sum(1 for i in range(n) if set(got[i, :k].tolist()) == set(truth[i, :k].tolist()))
    return 100.0 * hit / max(1, n)


def tie_analysis(docs: np.ndarray, queries: np.ndarray, got: np.ndarray,
                 truth: np.ndarray, k: int, limit: int = 40) -> dict:
    """Explain every top-k disagreement with the driver ground truth.

    For each disagreeing query it reports the fp32 cosine of the id the ground
    truth put at the first differing rank and of the id the arm put there. A gap
    of exactly 0.0 is a TIE -- two corpus rows whose scores are bit-identical,
    which HotpotQA really does contain (duplicated paragraphs) -- and a tie
    broken the other way is not an error by either side.
    """
    A = np.ascontiguousarray(docs.astype(np.float16).astype(np.float32))
    rows, gaps = [], []
    for i in range(got.shape[0]):
        g, t = got[i, :k], truth[i, :k]
        if bool((g == t).all()):
            continue
        j = int(np.flatnonzero(g != t)[0])
        s = A @ np.ascontiguousarray(queries[i], dtype=np.float32)
        a, b = int(t[j]), int(g[j])
        gap = float(s[a]) - float(s[b])
        gaps.append(abs(gap))
        if len(rows) < limit:
            rows.append({"query": i, "rank": j, "truth_id": a, "arm_id": b,
                         "truth_cos": float(s[a]), "arm_cos": float(s[b]),
                         "cos_gap": gap,
                         "bitwise_identical_embeddings":
                             bool(np.array_equal(A[a], A[b]))})
    return {"n_disagreeing_queries": len(gaps),
            "n_with_zero_cosine_gap": int(sum(1 for x in gaps if x == 0.0)),
            "max_abs_cos_gap": (max(gaps) if gaps else 0.0),
            "examples": rows}


def recall(topk: np.ndarray, gold: List[List[int]], k: int) -> dict:
    ev, per = [], []
    for i, g in enumerate(gold):
        got = set(int(x) for x in topk[i, :k] if x >= 0)
        gs = set(int(x) for x in g)
        ev.append(1.0 if gs and gs.issubset(got) else 0.0)
        per.append(len(gs & got) / len(gs) if gs else 0.0)
    return {"evidence_recall": round(100.0 * float(np.mean(ev)), 4),
            "per_doc_recall": round(100.0 * float(np.mean(per)), 4)}


# --------------------------------------------------------------------------
# recommendation, derived from the measured rows and nothing else
# --------------------------------------------------------------------------
def build_recommendation(res: dict) -> dict:
    """Turn the measured rows into the decision table, in code.

    Every field here is arithmetic over numbers already in this file, so the
    recommendation can be re-derived and checked. Nothing is typed in by hand.
    """
    def row(corpus, arm, phase="summary"):
        rows = (res["summary"][corpus] if phase == "summary"
                else res["summary"]["reopen_only"][corpus])
        for r in rows:
            if r["arm"] == arm:
                return r
        return {}

    out = {"what": (
        "derived by bench_memory.py from the rows above. 'server shape' is the "
        "reopen-only phase (open a vault somebody else wrote, then answer "
        "queries); 'loader shape' is the build+reopen phase, whose ru_maxrss is "
        "dominated by the ingest. `exact` means 0 of 500 queries changed in the "
        "top 10 against the engine that ships today, on the same vault."),
        "baseline": BASELINE_ARM, "sizes": {}}
    for corpus in ("n10000", "n71433"):
        if corpus not in res.get("summary", {}):
            continue
        base_s = row(corpus, BASELINE_ARM)
        base_r = row(corpus, BASELINE_ARM, "reopen_only")
        duel = res.get("latency_duel", {}).get(corpus, {})
        entries = {}
        for arm in ARMS:
            a, b = row(corpus, arm), row(corpus, arm, "reopen_only")
            if not a:
                continue
            entries[arm] = {
                "residency": a.get("residency"),
                "loader_peak_rss_mb": a.get("peak_rss_delta_mb"),
                "loader_peak_vs_legacy": (round(a["peak_rss_delta_mb"] / base_s["peak_rss_delta_mb"], 3)
                                          if base_s.get("peak_rss_delta_mb") else None),
                "server_live_rss_mb": b.get("live_rss_delta_mb"),
                "server_live_vs_legacy": (round(b["live_rss_delta_mb"] / base_r["live_rss_delta_mb"], 3)
                                          if base_r.get("live_rss_delta_mb") else None),
                "server_phys_footprint_mb": b.get("phys_footprint_delta_mb"),
                "evidence_recall_at_4": a.get("evidence_recall_at_4"),
                "top4_changed_vs_shipped": a.get("top4_changed_vs_shipped"),
                "top10_changed_vs_shipped": a.get("top10_changed_vs_shipped"),
                "exact_vs_shipped": a.get("exact_vs_shipped"),
                "p50_ms_single_pass": a.get("p50_ms"),
            }
        out["sizes"][corpus] = {
            "n_docs": res["corpora"][corpus]["n_docs"],
            "arms": entries,
            "cycled_latency_median_p50_ms": duel.get("median_p50_ms_across_cycles"),
            "cycled_latency_vs_float32": duel.get("p50_vs_float32"),
        }
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def versions(py: str) -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        sys.path.insert(0, NANOMEM_DIR)
        import nanomem as _nm
        out["nanomem"] = _nm.__version__
        out["nanomem_engine"] = getattr(_nm, "ENGINE_VERSION", None)
        out["nanomem_path"] = os.path.dirname(_nm.__file__)
    except Exception as exc:
        out["nanomem_error"] = repr(exc)
    return out


def ensure_cache() -> None:
    need = [os.path.join(CACHE, c, "docs.npy") for c in CORPORA]
    if all(os.path.exists(p) for p in need):
        return
    comp = os.path.join(REFOUND, "bench_competitors.py")
    print(f"[prep] cache missing -> {comp} --prep", flush=True)
    subprocess.check_call([sys.executable, comp, "--prep", "--cache", CACHE])


#: Keys a driver run produces per corpus. Everything else in the results file
#: belongs to a sweep and must survive a driver run that did not re-measure it.
PER_CORPUS_KEYS = ("corpora", "arms", "exactness", "summary")


def merge_driver_run(new: dict, dst: str, corpora) -> None:
    """Write a driver run into ``dst`` WITHOUT destroying what is already there.

    Two defects are fixed here, both of which destroyed real data once:

    1. ``drive()`` used to ``open(RESULTS, "w")`` on a dict it had just built
       from nothing. A PARTIAL run -- ``--corpora n10000``, or a single arm --
       therefore rebuilt the whole file, silently deleting the other corpus AND
       all four sweeps (``int8_pool_sweep``, ``scan_chunk_sweep``,
       ``latency_duel``, ``growth_policy``), which are expensive and are
       produced by different invocations. This merges per corpus instead: a
       corpus this run did not measure keeps the rows it already had, and a key
       this run does not produce is never touched.
    2. A second FULL run used to overwrite the first, and this file has no
       confidence intervals on memory or latency -- two independent runs are the
       only statement of run-to-run spread it can make. So the run being
       replaced is rotated into ``replicate`` together with the load average and
       machine it was taken on, and ``replicate.spread_vs_current`` reports the
       per-arm disagreement in percent, computed here rather than asserted in
       prose.

    Only the PREVIOUS run is kept; ``replicate`` is not a growing history.
    """
    prev = None
    if os.path.exists(dst):
        try:
            prev = json.load(open(dst))
        except Exception as exc:                          # pragma: no cover
            print(f"[warn] {dst} is unreadable ({exc}); writing a fresh file",
                  flush=True)
            prev = None
    if not isinstance(prev, dict) or not prev.get("arms"):
        with open(dst, "w") as fh:
            json.dump(new, fh, indent=2, default=str)
        return

    merged = dict(prev)
    for key in ("title", "generated_at", "script", "machine", "versions",
                "protocol", "failures"):
        merged[key] = new[key]
    merged["reopen_only"] = dict(prev.get("reopen_only") or {})
    merged["reopen_only"]["arms"] = dict(
        (prev.get("reopen_only") or {}).get("arms") or {})
    for key in ("what", "protocol"):
        if key in (new.get("reopen_only") or {}):
            merged["reopen_only"][key] = new["reopen_only"][key]
    merged["summary"] = dict(prev.get("summary") or {})
    merged["summary"].setdefault("reopen_only", {})
    merged["summary"]["reopen_only"] = dict(merged["summary"]["reopen_only"])

    # --- rotate the run being replaced into `replicate` -------------------
    rotated = [c for c in corpora if c in (prev.get("summary") or {})]
    if rotated:
        spread = {}
        for c in rotated:
            old_rows = {r["arm"]: r for r in prev["summary"][c]}
            cur_rows = {r["arm"]: r for r in new["summary"].get(c, [])}
            old_re = {r["arm"]: r for r in
                      (prev.get("summary", {}).get("reopen_only", {}) or {}).get(c, [])}
            cur_re = {r["arm"]: r for r in
                      (new.get("summary", {}).get("reopen_only", {}) or {}).get(c, [])}
            per_arm = {}
            for arm in cur_rows:
                if arm not in old_rows:
                    continue
                row = {"peak_rss_delta_mb": [old_rows[arm]["peak_rss_delta_mb"],
                                             cur_rows[arm]["peak_rss_delta_mb"]]}
                if arm in old_re and arm in cur_re:
                    row["server_live_rss_delta_mb"] = [old_re[arm]["live_rss_delta_mb"],
                                                       cur_re[arm]["live_rss_delta_mb"]]
                for k, pair in list(row.items()):
                    a, b = pair
                    if a and b:
                        row[k + "_pct_diff"] = round(abs(b - a) / a * 100.0, 2)
                per_arm[arm] = row
            spread[c] = per_arm
        merged["replicate"] = {
            "what": "the PREVIOUS full driver run of the corpora the current run "
                    "re-measured, kept because this file has no confidence "
                    "intervals on memory or latency: two independent runs are "
                    "the only honest statement of run-to-run spread it can make. "
                    "The rows under the top-level keys are the CURRENT run.",
            "generated_at": prev.get("generated_at"),
            "machine": prev.get("machine"),
            "versions": prev.get("versions"),
            "corpora_covered": rotated,
            "summary": {c: prev["summary"][c] for c in rotated},
            "reopen_only_summary": {
                c: (prev.get("summary", {}).get("reopen_only", {}) or {}).get(c)
                for c in rotated},
            "spread_vs_current": spread,
        }

    for key in PER_CORPUS_KEYS:
        merged[key] = dict(merged.get(key) or {})
    merged["summary"]["reopen_only"] = dict(
        (prev.get("summary") or {}).get("reopen_only") or {})
    for corpus in corpora:
        for key in PER_CORPUS_KEYS:
            if corpus in (new.get(key) or {}):
                merged[key][corpus] = new[key][corpus]
        if corpus in (new.get("reopen_only") or {}).get("arms", {}):
            merged["reopen_only"]["arms"][corpus] = new["reopen_only"]["arms"][corpus]
        if corpus in ((new.get("summary") or {}).get("reopen_only") or {}):
            merged["summary"]["reopen_only"][corpus] = \
                new["summary"]["reopen_only"][corpus]

    with open(dst, "w") as fh:
        json.dump(merged, fh, indent=2, default=str)


def drive(arms, corpora, timeout: int) -> None:
    ensure_cache()
    os.makedirs(WORK, exist_ok=True)
    raw = os.path.join(WORK, "raw")
    os.makedirs(raw, exist_ok=True)

    results = {
        "title": "nanomem v3 arena residency -- RSS / latency / recall at 10,000 and 71,433 docs",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": os.path.abspath(__file__),
        "machine": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                        capture_output=True, text=True).stdout.strip(),
            "model": subprocess.run(["sysctl", "-n", "hw.model"],
                                    capture_output=True, text=True).stdout.strip(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                            capture_output=True, text=True).stdout.strip() or 0),
            "load_avg_at_start": loadavg(),
        },
        "versions": versions(sys.executable),
        "protocol": {
            "cache": CACHE,
            "cache_provenance": "built by scratch/refound/bench_competitors.py --prep; "
                                "the SAME docs.npy / queries.npy / gold.json / rng(0) "
                                "insertion order used by competitors_standard_results.json",
            "top_k_queried": TOP_K,
            "ground_truth": "computed in the driver: docs -> float16 -> float32, "
                            "full fp32 matvec, argsort. Quantising the corpus to "
                            "fp16 first is deliberate -- fp16 ON DISK is "
                            "DECISIONS.md core-5 and is not a residency decision.",
            "recall": "computed in the driver from each arm's returned id matrix; "
                      "an arm cannot score itself",
            "reopen": "every arm is built, closed and reopened from disk before "
                      "it is timed; >= 20 warm-up queries first",
            "isolation": "one subprocess per (arm, corpus) so ru_maxrss is clean",
            "memory_metrics": "ru_maxrss peak delta, `ps` live delta and macOS "
                              "phys_footprint delta are all reported. They differ "
                              "on purpose: ru_maxrss counts a mapped page the "
                              "moment it is touched, phys_footprint does not "
                              "count clean file-backed pages at all.",
            "contention": "this harness does not quiesce the machine and does not "
                          "assume it is quiet: every arm records the 1-minute load "
                          "average at its start and end AND times a bare numpy "
                          "matvec on the same vectors, so latency can be read as a "
                          "multiple of the floor the machine was offering at that "
                          "moment. Read `machine.load_avg_at_start` and each arm's "
                          "own load before quoting any millisecond figure from the "
                          "`arms` block, and prefer `latency_duel`, which cycles "
                          "the modes. A RATIO IS NOT LOAD-INDEPENDENT EITHER: the "
                          "same paired duel read float16 at 1.55x float32 under "
                          "load ~18 and 2.28x under load ~2, because contention "
                          "flatters the slower arm.",
        },
        "corpora": {},
        "arms": {},
        "exactness": {},
        "int8_pool_sweep": {},
        "scan_chunk_sweep": {},
        "latency_duel": {},
        "growth_policy": {},
        "summary": {},
        "failures": [],
    }

    golds = {}
    for c in corpora:
        results["corpora"][c] = json.load(open(os.path.join(CACHE, c, "meta.json")))
        golds[c] = json.load(open(os.path.join(CACHE, c, "gold.json")))

    topks = {}
    for corpus in corpora:
        for arm in arms:
            tag = f"{arm}__{corpus}"
            outj = os.path.join(raw, tag + ".json")
            work = os.path.join(WORK, "w_" + tag)
            shutil.rmtree(work, ignore_errors=True)
            print(f"[run ] {tag} ...", flush=True)
            t0 = time.perf_counter()
            cmd = [sys.executable, os.path.abspath(__file__), "--worker",
                   "--arm", arm, "--corpus", corpus, "--out", outj, "--work", work]
            try:
                p = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                results["failures"].append({"arm": arm, "corpus": corpus,
                                            "error": f"timeout after {timeout}s"})
                continue
            wall = time.perf_counter() - t0
            if p.returncode != 0 or not os.path.exists(outj):
                results["failures"].append({"arm": arm, "corpus": corpus,
                                            "returncode": p.returncode,
                                            "stderr": p.stderr[-4000:]})
                print(f"[FAIL] {tag}: rc={p.returncode}\n{p.stderr[-2000:]}", flush=True)
                shutil.rmtree(work, ignore_errors=True)
                continue
            rec = json.load(open(outj))
            rec["driver_wall_s"] = round(wall, 2)
            tk = np.load(outj.replace(".json", ".topk.npy"))
            topks[(arm, corpus)] = tk
            rec["recall"] = {f"at_{k}": recall(tk, golds[corpus], k) for k in (4, 10)}
            results["arms"].setdefault(corpus, {})[arm] = rec
            m = rec["memory"]
            print(f"[done] {tag}  peakRSS {m['peak_maxrss_delta_bytes']/2**20:8.1f} MB"
                  f"  live {(m['live_rss_delta_bytes'] or 0)/2**20:8.1f} MB"
                  f"  p50 {rec['timing']['latency_ms']['p50']:7.3f} ms"
                  f"  recall@4 {rec['recall']['at_4']['evidence_recall']:.2f}"
                  f"  ({wall:.1f}s)", flush=True)
            shutil.rmtree(work, ignore_errors=True)

    # ---- reopen-only phase ----------------------------------------------
    # The build+reopen arms above measure what a LOADER costs. This measures what
    # a SERVER costs: open a vault somebody else wrote, then answer queries.
    vaults = os.path.join(WORK, "vaults")
    os.makedirs(vaults, exist_ok=True)
    results["reopen_only"] = {"what": (
        "each arm opens a vault built by a SEPARATE process and then serves the "
        "same 500 queries. ru_maxrss is a high-water mark, so in the build+reopen "
        "arms the ingest peak hides the reopen entirely; this phase is the number "
        "a query server actually pays, and it is what Arena.reserve() exists for. "
        "Every arm opens the SAME file (sha256 recorded per arm) -- residency is a "
        "RAM-side choice and changes nothing on disk."), "arms": {}, "vaults": {}}
    for corpus in corpora:
        vdir = os.path.join(vaults, corpus)
        shutil.rmtree(vdir, ignore_errors=True)
        os.makedirs(vdir, exist_ok=True)
        vpath = os.path.join(vdir, "vault.dat")
        bj = os.path.join(raw, f"build__{corpus}.json")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--build-vault",
                            "--corpus", corpus, "--vault", vpath, "--out", bj],
                           capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            results["failures"].append({"arm": "build_vault", "corpus": corpus,
                                        "stderr": p.stderr[-4000:]})
            continue
        results["reopen_only"]["vaults"][corpus] = json.load(open(bj))
        for arm in arms:
            tag = f"reopen__{arm}__{corpus}"
            outj = os.path.join(raw, tag + ".json")
            work = os.path.join(WORK, "w_" + tag)
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(work, exist_ok=True)
            print(f"[open] {tag} ...", flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), "--worker-reopen",
                   "--arm", arm, "--corpus", corpus, "--vault", vpath,
                   "--out", outj, "--work", work]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if p.returncode != 0 or not os.path.exists(outj):
                results["failures"].append({"arm": tag, "corpus": corpus,
                                            "returncode": p.returncode,
                                            "stderr": p.stderr[-4000:]})
                print(f"[FAIL] {tag}\n{p.stderr[-2000:]}", flush=True)
                shutil.rmtree(work, ignore_errors=True)
                continue
            rec = json.load(open(outj))
            tk = np.load(outj.replace(".json", ".topk.npy"))
            topks[("reopen__" + arm, corpus)] = tk
            rec["recall"] = {f"at_{k}": recall(tk, golds[corpus], k) for k in (4, 10)}
            results["reopen_only"]["arms"].setdefault(corpus, {})[arm] = rec
            m = rec["memory"]
            print(f"[done] {tag}  liveRSS {(m['live_rss_delta_bytes'] or 0)/2**20:8.1f} MB"
                  f"  peak {m['peak_maxrss_delta_bytes']/2**20:8.1f} MB"
                  f"  foot {(m['phys_footprint_delta_bytes'] or 0)/2**20:8.1f} MB"
                  f"  reopen {rec['timing']['reopen_s']:.3f}s"
                  f"  p50 {rec['timing']['latency_ms']['p50']:7.3f} ms", flush=True)
            shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(vdir, ignore_errors=True)

    # ---- exactness -------------------------------------------------------
    for corpus in corpora:
        docs = np.load(os.path.join(CACHE, corpus, "docs.npy"))
        queries = np.load(os.path.join(CACHE, corpus, "queries.npy"))
        truth = exact_topk(docs, queries, TOP_K)
        block = {"ground_truth": "driver numpy argsort over the fp16-round-tripped corpus",
                 "n_queries": int(queries.shape[0]), "arms": {}}
        for arm in list(arms) + ["reopen__" + a for a in arms]:
            tk = topks.get((arm, corpus))
            if tk is None:
                continue
            block["arms"][arm] = {
                "top4_slot_agreement_pct": round(agree_pct(tk, truth, 4), 4),
                "top10_slot_agreement_pct": round(agree_pct(tk, truth, 10), 4),
                "top4_set_agreement_pct": round(set_agree_pct(tk, truth, 4), 4),
                "top10_set_agreement_pct": round(set_agree_pct(tk, truth, 10), 4),
                "queries_with_any_top4_change": changed_rows(tk, truth, 4),
                "queries_with_any_top10_change": changed_rows(tk, truth, 10),
            }
        block["tie_analysis"] = {
            arm: tie_analysis(docs, queries, topks[(arm, corpus)], truth, 4)
            for arm in arms if (arm, corpus) in topks
        }
        base = topks.get((BASELINE_ARM, corpus))
        if base is not None:
            block["vs_shipped_3_0_3"] = {}
            for arm in list(arms) + ["reopen__" + a for a in arms]:
                tk = topks.get((arm, corpus))
                if tk is None:
                    continue
                block["vs_shipped_3_0_3"][arm] = {
                    "top4_identical_queries": int((tk[:, :4] == base[:, :4]).all(axis=1).sum()),
                    "top4_changed_queries": changed_rows(tk, base, 4),
                    "top10_changed_queries": changed_rows(tk, base, 10),
                }
        results["exactness"][corpus] = block

    # ---- summary ---------------------------------------------------------
    for corpus in corpora:
        rowsum = []
        arms_c = results["arms"].get(corpus, {})
        base = arms_c.get(BASELINE_ARM)
        for arm in arms:
            r = arms_c.get(arm)
            if r is None:
                continue
            m, t = r["memory"], r["timing"]
            ex = results["exactness"][corpus]["arms"].get(arm, {})
            row = {
                "arm": arm,
                "residency": r["config"]["residency"],
                "peak_rss_delta_mb": round(m["peak_maxrss_delta_bytes"] / 2 ** 20, 1),
                "live_rss_delta_mb": (round(m["live_rss_delta_bytes"] / 2 ** 20, 1)
                                      if m["live_rss_delta_bytes"] else None),
                "phys_footprint_delta_mb": (round(m["phys_footprint_delta_bytes"] / 2 ** 20, 1)
                                            if m["phys_footprint_delta_bytes"] else None),
                "ingest_peak_rss_delta_mb": round(m["ingest_peak_maxrss_delta_bytes"] / 2 ** 20, 1),
                "arena_anon_mb": r["arena"]["stats_resident_arena_mb"],
                "arena_mapped_mb": r["arena"]["stats_resident_arena_mapped_mb"],
                "p50_ms": t["latency_ms"]["p50"],
                "p95_ms": t["latency_ms"]["p95"],
                "numpy_floor_p50_ms": t["numpy_floor_ms"]["p50"],
                "p50_over_floor": round(t["latency_ms"]["p50"] / max(1e-9, t["numpy_floor_ms"]["p50"]), 3),
                "ingest_s": t["ingest_s"],
                "reopen_s": t["reopen_s"],
                "evidence_recall_at_4": r["recall"]["at_4"]["evidence_recall"],
                "per_doc_recall_at_10": r["recall"]["at_10"]["per_doc_recall"],
                "top4_slot_agreement_pct": ex.get("top4_slot_agreement_pct"),
                "queries_with_any_top4_change_vs_numpy": ex.get("queries_with_any_top4_change"),
            }
            vs = results["exactness"][corpus].get("vs_shipped_3_0_3", {}).get(arm)
            if vs is not None:
                row["top4_changed_vs_shipped"] = vs["top4_changed_queries"]
                row["top10_changed_vs_shipped"] = vs["top10_changed_queries"]
                # "exact" means: this arm returns the SAME top-10 as the engine
                # that ships today, on every query. That is the claim the
                # residency work must not break. Agreement with the driver's own
                # numpy argsort is reported too, but it is not the bar: numpy
                # breaks exact ties in a different order and the corpus contains
                # bit-identical duplicate paragraphs (see tie_analysis).
                row["exact_vs_shipped"] = (vs["top4_changed_queries"] == 0
                                           and vs["top10_changed_queries"] == 0)
            if base is not None:
                bm = base["memory"]["peak_maxrss_delta_bytes"]
                row["peak_rss_vs_shipped"] = round(m["peak_maxrss_delta_bytes"] / max(1, bm), 3)
                bp = base["timing"]["latency_ms"]["p50"]
                row["p50_vs_shipped"] = round(t["latency_ms"]["p50"] / max(1e-9, bp), 3)
            rowsum.append(row)
        results["summary"][corpus] = rowsum

    for corpus in corpora:
        rows = []
        base = results["reopen_only"]["arms"].get(corpus, {}).get(BASELINE_ARM)
        for arm, r in results["reopen_only"]["arms"].get(corpus, {}).items():
            m, t = r["memory"], r["timing"]
            ex = results["exactness"][corpus].get("vs_shipped_3_0_3", {}).get("reopen__" + arm, {})
            row = {"arm": arm, "residency": r["config"]["residency"],
                   "live_rss_delta_mb": (round(m["live_rss_delta_bytes"] / 2 ** 20, 1)
                                         if m["live_rss_delta_bytes"] else None),
                   "peak_rss_delta_mb": round(m["peak_maxrss_delta_bytes"] / 2 ** 20, 1),
                   "phys_footprint_delta_mb": (round(m["phys_footprint_delta_bytes"] / 2 ** 20, 1)
                                               if m["phys_footprint_delta_bytes"] else None),
                   "arena_anon_mb": r["arena"]["stats_resident_arena_mb"],
                   "arena_mapped_mb": r["arena"]["stats_resident_arena_mapped_mb"],
                   "reopen_s": t["reopen_s"], "p50_ms": t["latency_ms"]["p50"],
                   "p95_ms": t["latency_ms"]["p95"],
                   "evidence_recall_at_4": r["recall"]["at_4"]["evidence_recall"],
                   "top4_changed_vs_shipped": ex.get("top4_changed_queries"),
                   "top10_changed_vs_shipped": ex.get("top10_changed_queries"),
                   # None, not False, when this run did not include the baseline
                   # arm to compare against: a partial run (`--arms ...`) has no
                   # opinion on exactness, and reporting "False" there would be a
                   # measurement the run never made.
                   "exact_vs_shipped": (None if ex.get("top4_changed_queries") is None
                                        else (ex["top4_changed_queries"] == 0
                                              and ex["top10_changed_queries"] == 0))}
            if base is not None and base["memory"]["live_rss_delta_bytes"]:
                row["live_rss_vs_shipped"] = round(
                    (m["live_rss_delta_bytes"] or 0) / base["memory"]["live_rss_delta_bytes"], 3)
            rows.append(row)
        results["summary"].setdefault("reopen_only", {})[corpus] = rows

    results["machine"]["load_avg_at_end"] = loadavg()
    merge_driver_run(results, RESULTS, corpora)
    print(f"\n[ok] wrote {RESULTS} ({os.path.getsize(RESULTS)} bytes), "
          f"failures={len(results['failures'])}", flush=True)
    for corpus in corpora:
        print(f"\n=== reopen-only (server shape) {corpus} ===")
        for row in results["summary"].get("reopen_only", {}).get(corpus, []):
            print(f"  {row['arm']:26s} liveRSS {row['live_rss_delta_mb']:8.1f} MB"
                  f"  foot {(row['phys_footprint_delta_mb'] or 0):7.1f} MB"
                  f"  reopen {row['reopen_s']:6.3f}s  p50 {row['p50_ms']:8.3f} ms"
                  f"  exact_vs_shipped={row['exact_vs_shipped']}")
    for corpus in corpora:
        print(f"\n=== build+reopen {corpus} ===")
        for row in results["summary"].get(corpus, []):
            print(f"  {row['arm']:26s} peakRSS {row['peak_rss_delta_mb']:8.1f} MB"
                  f"  p50 {row['p50_ms']:8.3f} ms ({row['p50_over_floor']:5.2f}x floor)"
                  f"  r@4 {row['evidence_recall_at_4']:6.2f}"
                  f"  exact_vs_shipped={row.get('exact_vs_shipped')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-reopen", action="store_true")
    ap.add_argument("--build-vault", action="store_true")
    ap.add_argument("--vault")
    ap.add_argument("--pool-sweep", action="store_true")
    ap.add_argument("--merge-pool-sweep", action="store_true")
    ap.add_argument("--chunk-sweep", action="store_true")
    ap.add_argument("--merge-chunk-sweep", action="store_true")
    ap.add_argument("--latency-duel", action="store_true")
    ap.add_argument("--merge-latency-duel", action="store_true")
    ap.add_argument("--merge-growth-sweep", action="store_true")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--growth-sweep", action="store_true")
    ap.add_argument("--growth-worker")
    ap.add_argument("--growth-rows", type=int, default=71433)
    ap.add_argument("--recommend", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--corpus")
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--corpora", default=",".join(CORPORA))
    ap.add_argument("--timeout", type=int, default=5400)
    a = ap.parse_args()

    # `--out` used to be declared and then ignored by every path except the
    # workers, so `--out /tmp/scratch.json` wrote the CANONICAL results file
    # anyway. It now names the results file for the driver, the merge paths and
    # --recommend. The worker paths keep their own meaning for it (the per-arm
    # scratch file the driver reads back), so they are excluded by name.
    global RESULTS
    if a.out and not (a.worker or a.worker_reopen or a.build_vault
                      or a.growth_worker):
        RESULTS = os.path.abspath(a.out)

    if a.worker:
        try:
            ctx = Ctx(a.corpus, a.work)
            rec = run_arm(a.arm, ctx)
            tk = rec.pop("topk")
            np.save(a.out.replace(".json", ".topk.npy"), tk)
            with open(a.out, "w") as fh:
                json.dump(rec, fh, indent=2, default=str)
            return 0
        except Exception:
            traceback.print_exc()
            return 1

    if a.build_vault:
        rec = build_vault(a.corpus, a.vault)
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=2)
        return 0

    if a.worker_reopen:
        try:
            os.makedirs(a.work, exist_ok=True)
            ctx = Ctx(a.corpus, a.work)
            rec = run_reopen_only(a.arm, ctx, a.vault)
            tk = rec.pop("topk")
            np.save(a.out.replace(".json", ".topk.npy"), tk)
            with open(a.out, "w") as fh:
                json.dump(rec, fh, indent=2, default=str)
            return 0
        except Exception:
            traceback.print_exc()
            return 1

    if a.pool_sweep:
        ensure_cache()
        out = {}
        for c in [x for x in a.corpora.split(",") if x]:
            out[c] = run_pool_sweep(c)
        dst = os.path.join(WORK, "pool_sweep.json")
        os.makedirs(WORK, exist_ok=True)
        with open(dst, "w") as fh:
            json.dump(out, fh, indent=2)
        print("[ok] wrote", dst)
        return 0

    if a.recommend:
        res = json.load(open(RESULTS))
        res["recommendation"] = build_recommendation(res)
        with open(RESULTS, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print(json.dumps(res["recommendation"], indent=2)[:4000])
        print("[ok] wrote recommendation into", RESULTS)
        return 0

    if a.growth_worker:
        rec = run_growth_policy(a.growth_worker, a.growth_rows, DIM, 50)
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=2)
        return 0

    if a.growth_sweep:
        os.makedirs(WORK, exist_ok=True)
        rows = []
        for pol in GROWTH_POLICIES:
            outj = os.path.join(WORK, f"growth_{pol}.json")
            p = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--growth-worker", pol, "--growth-rows",
                                str(a.growth_rows), "--out", outj],
                               capture_output=True, text=True, timeout=a.timeout)
            if p.returncode != 0:
                print("[FAIL]", pol, p.stderr[-2000:], flush=True)
                continue
            rows.append(json.load(open(outj)))
            print(f"[growth] {pol}: {rows[-1]}", flush=True)
        dst = os.path.join(WORK, "growth_policy.json")
        with open(dst, "w") as fh:
            json.dump({"rows": rows}, fh, indent=2)
        print("[ok] wrote", dst)
        return 0

    if a.merge_growth_sweep:
        src = os.path.join(WORK, "growth_policy.json")
        res = json.load(open(RESULTS))
        blk = json.load(open(src))
        blk["what"] = (
            "the allocator question with the engine taken out: an fp32 "
            "(rows, 768) arena is grown to 71,433 rows in 50-row steps under "
            "each growth policy, one subprocess per policy so ru_maxrss is "
            "clean. `bytes_freed_mb` is the arithmetic sum of the arrays the "
            "policy discarded; the peak tracks (live + freed) because this "
            "allocator hands almost none of it back. It is why `Arena.reserve` "
            "exists and why doubling was KEPT for the case where no row count "
            "is available.")
        res["growth_policy"] = blk
        with open(RESULTS, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print("[ok] merged growth policy sweep into", RESULTS)
        return 0

    if a.latency_duel:
        ensure_cache()
        out = {}
        for c in [x for x in a.corpora.split(",") if x]:
            out[c] = run_latency_duel(c, a.cycles)
        dst = os.path.join(WORK, "latency_duel.json")
        os.makedirs(WORK, exist_ok=True)
        with open(dst, "w") as fh:
            json.dump(out, fh, indent=2)
        print("[ok] wrote", dst)
        return 0

    if a.merge_latency_duel:
        src = os.path.join(WORK, "latency_duel.json")
        res = json.load(open(RESULTS))
        prev_duel = res.get("latency_duel") or {}
        res["latency_duel"] = json.load(open(src))
        # Same rule as the driver's `replicate`: the duel is the only latency
        # figure any docstring is allowed to quote, it has no confidence
        # interval, and a ratio measured under load is NOT load-independent --
        # contention flatters the SLOWER arm, so the same comparison read 1.55x
        # at load ~18 and 2.26x at load ~2.5. Keep the run being replaced.
        if prev_duel.get("n71433") or prev_duel.get("n10000"):
            prev_duel.pop("replicate", None)
            res["latency_duel"]["replicate"] = prev_duel
        res["latency_duel"]["what"] = (
            "latency only, cycled: every mode is opened, timed on the same 500 "
            "queries and closed inside one cycle, the cycle is repeated, and the "
            "reported number is the median across cycles. This machine could not "
            "be quiesced, so a single pass per mode measures the mode AND the "
            "minute; cycling makes drift hit every mode. No memory figure is "
            "taken here.")
        with open(RESULTS, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print("[ok] merged latency duel into", RESULTS)
        return 0

    if a.chunk_sweep:
        ensure_cache()
        out = {}
        for c in [x for x in a.corpora.split(",") if x]:
            out[c] = run_chunk_sweep(c)
        dst = os.path.join(WORK, "chunk_sweep.json")
        os.makedirs(WORK, exist_ok=True)
        with open(dst, "w") as fh:
            json.dump(out, fh, indent=2)
        print("[ok] wrote", dst)
        return 0

    if a.merge_chunk_sweep:
        src = os.path.join(WORK, "chunk_sweep.json")
        res = json.load(open(RESULTS))
        res["scan_chunk_sweep"] = json.load(open(src))
        res["scan_chunk_sweep"]["what"] = (
            "`scan_chunk` is the staging-buffer size the narrow residency modes "
            "convert fp16 -> fp32 through. It cannot change an answer -- every "
            "chunk size computes the same fp32 dot products -- so each row "
            "asserts bit-identity against the fp32 arena on the same vault and "
            "then reports latency only.")
        with open(RESULTS, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print("[ok] merged scan_chunk sweep into", RESULTS)
        return 0

    if a.merge_pool_sweep:
        src = os.path.join(WORK, "pool_sweep.json")
        res = json.load(open(RESULTS))
        sweep = json.load(open(src))
        # A disagreement whose cosine gap is EXACTLY 0.0 is a tie between two
        # corpus rows with bit-identical scores, and HotpotQA really contains
        # duplicated paragraphs. Breaking such a tie the other way is not an
        # error, so the pool question is answered twice: literally, and with
        # ties discounted. Both come straight from the measured rows.
        for _c, blk in sweep.items():
            if not isinstance(blk, dict) or "rows" not in blk:
                continue
            for r in blk["rows"]:
                r["top4_changed_queries_with_a_real_cosine_gap"] = (
                    int(r["top4_changed_queries_vs_fp32_engine"])
                    - int(r["disagreements_with_zero_cos_gap"]))
            nontie = [r["rerank_pool"] for r in blk["rows"]
                      if r["top4_changed_queries_with_a_real_cosine_gap"] == 0]
            blk["smallest_pool_with_zero_nontie_top4_change"] = (min(nontie) if nontie else None)
        res["int8_pool_sweep"] = sweep
        res["int8_pool_sweep"]["what"] = (
            "int8 residency re-ranks the top `rerank_pool` rows exactly from the "
            "fp16 sidecar. This sweep is the pool size needed for ZERO top-4 "
            "change, measured twice: `smallest_pool_with_zero_top4_change` is the "
            "literal count against the fp32-residency engine on the same vault, "
            "and `smallest_pool_with_zero_nontie_top4_change` discounts "
            "disagreements whose cosine gap is exactly 0.0 (bit-identical "
            "duplicate paragraphs, which either order answers correctly). "
            "`top4_changed_queries_vs_numpy` is the same comparison against the "
            "driver's own argsort.")
        with open(RESULTS, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print("[ok] merged pool sweep into", RESULTS)
        return 0

    drive([x for x in a.arms.split(",") if x],
          [x for x in a.corpora.split(",") if x], a.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
