#!/usr/bin/env python3
"""Attack nanomem's last measured defeat: reopen, 0.1609 s against sqlite-vec's
0.0014 s at 71,433 documents (competitors_standard_results.json, ``winners``).

THE DEFEAT IS STRUCTURAL. Opening a vault rebuilt the resident arena by
replaying the container: every block read, its HMAC recomputed, its vectors
upcast, its records parsed, one Python dict entry made per row. That is O(rows)
and it is paid at every open, so it cannot be tuned away -- sqlite-vec and
Chroma map or lazily page their storage and nanomem rebuilt its own. This
harness measures the fix (a mappable `<vault>.arena` cache, see
nanomem/arena.py::ArenaSnapshot) against the build it replaces, at 1,190 /
10,000 / 71,433 documents.

ARMS. One package per arm so the comparison is a package comparison, not a flag
comparison inside one build:

  baseline_e617ba3   the committed engine (git archive of the commit under
                     test), which has no cache at all. Every other number is
                     read against this one.
  cache_off          the new package with ``arena_cache="off"``. It exists to
                     show the change costs nothing when it is not used.
  cache_map          the new package's DEFAULT: map the cache, bind it to the
                     vault, do not re-authenticate.
  cache_copy         map the cache, then copy the vectors into anonymous RAM
                     once -- the "does a search read the mapping or a copy"
                     question, measured instead of argued.
  cache_verify       map the cache, then re-authenticate every block before
                     serving. Keeps the integrity property a scan has.

PRE-REGISTERED BEFORE ANY NUMBER WAS TAKEN (the criterion, and the bar):
  * PRIMARY: reopen wall time at 71,433 documents, same-process protocol,
    warm page cache -- which is exactly the protocol that produced sqlite-vec's
    0.0014 s and Chroma's 0.0017 s in competitors_standard_results.json. nanomem
    "wins reopen" iff cache_map's median reopen is <= 0.0014 s there.
  * SECONDARY, and reported whether or not the primary passes: TIME TO FIRST
    ANSWER = reopen + first query. A mapped arena moves work from the open to
    the first page fault, so reporting the open alone would be exactly the kind
    of win-by-choosing-the-metric this project has already been burned by. Both
    numbers are printed for every arm at every size.
  * EXACTNESS IS A GATE, NOT A METRIC: the SHA-256 of the concatenated fp32
    score vectors of EVERY query against EVERY row must equal the baseline's at
    every size, and 0 of the top-10 lists may change. An arm that fails this is
    reported as a failure whatever its latency.

WHAT THIS HARNESS CANNOT DO. It cannot drop the OS page cache: ``purge`` needs
root on this machine and this process does not have it ("Unable to purge disk
buffers: Operation not permitted"). Every number here is therefore
WARM-PAGE-CACHE, which is the same condition competitors_standard_results.json
measured its arms under, so the comparison is like for like -- but it means the
cold cost of faulting a 209 MiB mapping in from disk is NOT measured here. What
is measured is where that cost lands when the pages are available: the
first-query-after-reopen column.

Usage:
    bench_reopen.py --prep            # build one vault per corpus (baseline pkg)
    bench_reopen.py                   # every arm, every corpus -> results json
    bench_reopen.py --worker ...      # one (arm, corpus, phase) in its own process
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import List, Optional

import numpy as np

REPO = "<redacted local path> /4D llm"
REFOUND = os.path.join(REPO, "scratch", "refound")
NEW_DIR = os.path.join(REPO, "nanomem_standalone")
BASE_COMMIT = "e617ba3"
BASE_DIR = "/tmp/nanomem_baseline_%s/nanomem_standalone" % BASE_COMMIT
CACHE = os.environ.get("BENCH_CACHE", "/tmp/nanomem_bench_cache")
WORK = os.environ.get("BENCH_WORK", "/tmp/nanomem_reopen_work")
VAULTS = os.path.join(WORK, "vaults")
RESULTS = os.path.join(REFOUND, "reopen_results.json")
COMPETITORS = os.path.join(REFOUND, "competitors_standard_results.json")

DIM = 768
TOP_K = 10
SEED = 0
WARMUP_QUERIES = 20
REOPEN_REPEATS = 25          # in-process reopens; the median is the headline
CORPORA = ("val1190", "n10000", "n71433")

# arm -> (package dir, arena_cache kwarg or None for a package that has none)
ARMS = {
    "baseline_e617ba3": (BASE_DIR, None),
    "cache_off":        (NEW_DIR, "off"),
    "cache_map":        (NEW_DIR, "map"),
    "cache_copy":       (NEW_DIR, "copy"),
    "cache_verify":     (NEW_DIR, "verify"),
}
BASELINE_ARM = "baseline_e617ba3"
CACHED_ARMS = ("cache_map", "cache_copy", "cache_verify")

# The bar, fixed here before the first measurement.
TARGETS = {"sqlitevec_bruteforce": 0.0014, "chroma_hnsw_default": 0.0017}


# --------------------------------------------------------------------------
# process measurement (same definitions bench_memory.py uses, so the numbers
# are comparable to memory_results.json)
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
    """macOS ``phys_footprint``: the number Activity Monitor shows.

    It EXCLUDES clean file-backed pages, which is the whole point of mapping the
    arena: ``ru_maxrss`` counts a mapped page the moment it is touched and never
    gives it back, whether or not the OS could drop it for free.
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


def pct(a, q) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def path_bytes(p: str) -> int:
    try:
        return int(os.path.getsize(p))
    except OSError:
        return 0


# --------------------------------------------------------------------------
# corpora / vaults
# --------------------------------------------------------------------------
def corpus_dir(corpus: str) -> str:
    return os.path.join(CACHE, corpus)


def load_corpus(corpus: str):
    d = corpus_dir(corpus)
    docs = np.load(os.path.join(d, "docs.npy"))
    queries = np.load(os.path.join(d, "queries.npy"))
    texts = json.load(open(os.path.join(d, "texts.json")))
    meta = json.load(open(os.path.join(d, "meta.json")))
    return docs, queries, texts, meta


def vault_path(corpus: str) -> str:
    return os.path.join(VAULTS, corpus + ".dat")


def cache_path(corpus: str) -> str:
    return vault_path(corpus) + ".arena"


def prep() -> None:
    """Build ONE vault per corpus with the BASELINE package.

    Every arm then opens the same bytes, written by the build under test's own
    predecessor, so nothing in the comparison depends on who wrote the file.
    Insertion order is ``default_rng(0).permutation``, source ``wiki`` and
    metadata ``{"idx": row}`` -- the same conventions bench_memory.py and
    bench_ingest_ram.py use, so these vaults are the vaults those files measured.
    """
    os.makedirs(VAULTS, exist_ok=True)
    sys.path.insert(0, BASE_DIR)
    from nanomem.engine import VaultEngine
    out = {}
    for corpus in CORPORA:
        docs, _q, texts, _m = load_corpus(corpus)
        n = int(docs.shape[0])
        order = np.random.default_rng(SEED).permutation(n)
        p = vault_path(corpus)
        for stale in (p, p + ".arena"):
            if os.path.exists(stale):
                os.remove(stale)
        t0 = time.perf_counter()
        e = VaultEngine(p, embed_dim=int(docs.shape[1]), durable="none")
        e.reserve_rows(n)
        for i in order:
            i = int(i)
            e.add_fact(texts[i], docs[i], source="wiki", metadata={"idx": i})
        e.flush()
        e.close()
        out[corpus] = {"rows": n, "build_s": round(time.perf_counter() - t0, 3),
                       "vault_bytes": path_bytes(p)}
        print("[prep] %-8s %6d rows  %8.2f MiB  %.1fs" % (
            corpus, n, out[corpus]["vault_bytes"] / 1048576, out[corpus]["build_s"]),
            flush=True)
    return out


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def engine_kwargs(arm: str, corpus: str) -> dict:
    kw = dict(embed_dim=DIM, vector_dtype="float16", router="off",
              n_exhaustive=50_000, durable="none")
    mode = ARMS[arm][1]
    if mode is not None:
        kw["arena_cache"] = mode
    return kw


def open_engine(VaultEngine, arm, corpus):
    return VaultEngine(vault_path(corpus), **engine_kwargs(arm, corpus))


def run_serve(arm: str, corpus: str) -> dict:
    """Open a vault this process did not write, then serve every query.

    ``reopen_s`` is taken twice on purpose:

      * ``in_process``: the median of REOPEN_REPEATS opens in one warmed
        interpreter. This is the protocol competitors_standard_results.json used
        (it builds, closes and reopens inside one process), so it is the number
        that may be compared with sqlite-vec's 0.0014 s.
      * ``fresh_process``: the single open this process does first, before
        anything else has run. It includes whatever the open itself imports
        lazily, which is what a CLI invocation really pays.
    """
    pkg = ARMS[arm][0]
    sys.path.insert(0, pkg)
    from nanomem.engine import VaultEngine
    import nanomem as _nm

    docs, Q, texts, cmeta = load_corpus(corpus)
    nq = int(Q.shape[0])
    gc.collect()
    base_rss, base_max = rss_bytes(), maxrss_bytes()
    base_foot = phys_footprint_bytes()
    load_start = loadavg()

    t0 = time.perf_counter()
    e = open_engine(VaultEngine, arm, corpus)
    reopen_fresh = time.perf_counter() - t0
    after_open_max = maxrss_bytes()
    after_open_rss = rss_bytes()

    def q(v, k=TOP_K):
        return [int(h["metadata"]["idx"]) for h in e.search("", v, top_k=k)]

    t1 = time.perf_counter()
    _first = q(Q[0])
    first_query_s = time.perf_counter() - t1
    after_first_foot = phys_footprint_bytes()
    after_first_rss = rss_bytes()

    for i in range(min(WARMUP_QUERIES, nq)):
        q(Q[i])
    lat = []
    topk = np.full((nq, TOP_K), -1, dtype=np.int32)
    for i in range(nq):
        t = time.perf_counter()
        ids = q(Q[i])
        lat.append((time.perf_counter() - t) * 1000.0)
        topk[i, :len(ids)] = ids[:TOP_K]

    h = hashlib.sha256()
    for i in range(nq):
        v = np.ascontiguousarray(Q[i], dtype=np.float32)
        h.update(np.ascontiguousarray(e.arena.scores(v), dtype=np.float32).tobytes())
    scores_sha = h.hexdigest()

    gc.collect()
    peak_max = maxrss_bytes()
    live_rss = rss_bytes()
    live_foot = phys_footprint_bytes()
    st = e.stats()
    info = e.arena_cache_info() if hasattr(e, "arena_cache_info") else {"source": "scan"}
    e.close()
    del e
    gc.collect()

    # repeated opens in this (now warm) process: the steady-state open cost
    reopens = []
    for _ in range(REOPEN_REPEATS):
        t = time.perf_counter()
        e2 = open_engine(VaultEngine, arm, corpus)
        reopens.append(time.perf_counter() - t)
        e2.close()
        del e2
    firsts = []
    for _ in range(5):
        e3 = open_engine(VaultEngine, arm, corpus)
        t = time.perf_counter()
        _ = [int(x["metadata"]["idx"]) for x in e3.search("", Q[1], top_k=TOP_K)]
        firsts.append(time.perf_counter() - t)
        e3.close()
        del e3

    # the machine's offer at this moment, captured after every memory number
    A = np.ascontiguousarray(docs.astype(np.float16).astype(np.float32))
    qv = np.ascontiguousarray(Q[0], dtype=np.float32)
    for _ in range(3):
        _ = A @ qv
    floor = []
    for i in range(min(50, nq)):
        t = time.perf_counter()
        _s = A @ np.ascontiguousarray(Q[i], dtype=np.float32)
        floor.append((time.perf_counter() - t) * 1000.0)
    del A
    gc.collect()

    return {
        "arm": arm, "corpus": corpus, "n_docs": int(docs.shape[0]), "n_queries": nq,
        "topk": topk,
        "config": {"package": pkg, "arena_cache": ARMS[arm][1],
                   "nanomem_version": getattr(_nm, "__version__", None),
                   "engine_version": getattr(_nm, "ENGINE_VERSION", None),
                   "source": "wiki (not a PERSONAL_SOURCE, so the entity/temporal "
                             "boost layer is inert)"},
        "arena_cache_info": {k: v for k, v in info.items() if k != "path"},
        "timing": {
            "reopen_s_fresh_process": round(reopen_fresh, 6),
            "reopen_s_in_process_median": round(float(np.median(reopens)), 6),
            "reopen_s_in_process_min": round(float(np.min(reopens)), 6),
            "reopen_s_in_process_p95": round(pct(reopens, 95), 6),
            "reopen_s_in_process_all": [round(x, 6) for x in reopens],
            "first_query_after_reopen_s": round(first_query_s, 6),
            "first_query_after_reopen_s_median_of_5": round(float(np.median(firsts)), 6),
            "time_to_first_answer_s": round(float(np.median(reopens))
                                            + float(np.median(firsts)), 6),
            "latency_ms": {"p50": round(pct(lat, 50), 4), "p95": round(pct(lat, 95), 4),
                           "p99": round(pct(lat, 99), 4),
                           "mean": round(float(np.mean(lat)), 4),
                           "n_timed_queries": len(lat)},
            "numpy_floor_ms": {"p50": round(pct(floor, 50), 4),
                               "min": round(float(np.min(floor)), 4)},
        },
        "memory": {
            "baseline_rss_bytes": base_rss, "baseline_maxrss_bytes": base_max,
            "baseline_phys_footprint_bytes": base_foot,
            "after_open_maxrss_delta_bytes": after_open_max - base_max,
            "after_open_rss_delta_bytes": (after_open_rss - base_rss) if after_open_rss else None,
            "after_first_query_rss_delta_bytes": (after_first_rss - base_rss) if after_first_rss else None,
            "after_first_query_phys_footprint_delta_bytes": (
                (after_first_foot - base_foot) if (after_first_foot and base_foot) else None),
            "peak_maxrss_delta_bytes": peak_max - base_max,
            "live_rss_delta_bytes": (live_rss - base_rss) if live_rss else None,
            "phys_footprint_delta_bytes": ((live_foot - base_foot)
                                           if (live_foot and base_foot) else None),
            "method": "ru_maxrss for the peak, `ps -o rss=` live, `footprint -p` "
                      "for phys_footprint (which excludes clean file-backed "
                      "pages). Baseline taken after the corpus arrays are loaded.",
        },
        "disk": {"vault_bytes": path_bytes(vault_path(corpus)),
                 "arena_cache_bytes": path_bytes(cache_path(corpus))},
        "exactness": {"scores_sha256": scores_sha},
        "stats": {k: st.get(k) for k in (
            "total_documents", "active_heap_ram_kb", "resident_arena_mb",
            "resident_arena_mapped_mb", "arena_source", "arena_cache",
            "arena_cache_bytes", "mapped_bytes", "record_bytes",
            "record_mapped_bytes", "column_bytes", "column_mapped_bytes",
            "id_table_mapped_bytes", "index_bytes_estimated", "arena_growth_copies",
            "process_rss_kb", "process_peak_rss_kb", "file_size_mb")},
        "load_avg_start": load_start, "load_avg_end": loadavg(),
    }


def run_rebuild(arm: str, corpus: str, repeats: int = 3) -> dict:
    """What a STALE cache costs: delete it, then time the open that rebuilds it.

    This is the slow path the design has to pay for, and it is the honest other
    half of the headline: every cached open is free because one open was not.
    Repeated, because the first of these on a large corpus also has the just
    unlinked 270 MB cache being reclaimed underneath it and reads high.
    """
    pkg = ARMS[arm][0]
    sys.path.insert(0, pkg)
    from nanomem.engine import VaultEngine
    p = cache_path(corpus)
    gc.collect()
    base_max = maxrss_bytes()
    runs = []
    info = {}
    n = 0
    for _ in range(repeats):
        if os.path.exists(p):
            os.remove(p)
        t0 = time.perf_counter()
        e = open_engine(VaultEngine, arm, corpus)
        total = time.perf_counter() - t0
        info = e.arena_cache_info() if hasattr(e, "arena_cache_info") else {}
        n = e.count()
        e.close()
        del e
        gc.collect()
        runs.append({"open_s": round(total, 6), "scan_s": info.get("scan_s"),
                     "write_s": (info.get("write") or {}).get("write_s")})
    return {"arm": arm, "corpus": corpus, "n_docs": int(n), "runs": runs,
            "rebuild_open_s": round(float(np.median([r["open_s"] for r in runs])), 6),
            "scan_s": round(float(np.median([r["scan_s"] for r in runs
                                             if r["scan_s"] is not None])), 6)
            if any(r["scan_s"] is not None for r in runs) else None,
            "attach_s": info.get("attach_s"),
            "write": info.get("write"), "write_error": info.get("write_error"),
            "cache_bytes": path_bytes(p),
            "peak_maxrss_delta_bytes": maxrss_bytes() - base_max,
            "load_avg": loadavg()}


def run_append(arm: str, corpus: str) -> dict:
    """What an open costs when another process appended after the cache was
    written -- the case the prefix binding exists for. 200 rows appended, which
    is under ``arena_cache_refresh_rows``, so the cache is KEPT and the tail is
    rescanned."""
    pkg = ARMS[arm][0]
    sys.path.insert(0, pkg)
    from nanomem.engine import VaultEngine
    docs, Q, texts, _m = load_corpus(corpus)
    src = vault_path(corpus)
    dst = os.path.join(WORK, "append", corpus + ".dat")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for stale in (dst, dst + ".arena"):
        if os.path.exists(stale):
            os.remove(stale)
    shutil.copyfile(src, dst)
    if os.path.exists(cache_path(corpus)):
        shutil.copyfile(cache_path(corpus), dst + ".arena")
    kw = engine_kwargs(arm, corpus)
    e = VaultEngine(dst, **kw)
    n0 = e.count()
    rng = np.random.default_rng(7)
    for i in range(200):
        v = rng.normal(size=DIM).astype(np.float32)
        e.add_fact("appended row %d" % i, v / np.linalg.norm(v), source="wiki",
                   metadata={"idx": -1 - i})
    e.flush()
    e.close()
    del e
    gc.collect()
    times = []
    info = {}
    for _ in range(7):
        t = time.perf_counter()
        e2 = VaultEngine(dst, **kw)
        times.append(time.perf_counter() - t)
        info = e2.arena_cache_info() if hasattr(e2, "arena_cache_info") else {}
        n = e2.count()
        e2.close()
        del e2
    return {"arm": arm, "corpus": corpus, "rows_before": int(n0), "rows_after": int(n),
            "appended": 200,
            "reopen_s_median": round(float(np.median(times)), 6),
            "source": info.get("source"), "rows_from_cache": info.get("rows_from_cache"),
            "rows_scanned": info.get("rows_scanned"),
            "rewrote_cache": bool(info.get("write")), "load_avg": loadavg()}


def _writable_copy(corpus: str, arm: str, tag: str) -> str:
    """A private copy of the vault (and of its cache) this arm may write to."""
    dst = os.path.join(WORK, tag, "%s.%s.dat" % (corpus, arm))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for stale in (dst, dst + ".arena"):
        if os.path.exists(stale):
            os.remove(stale)
    # copy2, not copyfile: the cache binds to the vault's (size, mtime) as of the
    # scan it was built from, so a copy that resets mtime is refused -- correctly,
    # but it would make this helper measure a scan while claiming a cached open.
    shutil.copy2(vault_path(corpus), dst)
    if os.path.exists(cache_path(corpus)):
        shutil.copy2(cache_path(corpus), dst + ".arena")
    return dst


def run_scanduel(arm: str, corpus: str) -> dict:
    """Open the vault REOPEN_REPEATS times and report the median.

    Run by the driver alternately for ``baseline_e617ba3`` and ``cache_off``
    over several cycles, so the question "does the new package cost anything on
    the path it did not change" is answered by a paired comparison on the same
    minute of the same machine rather than by two numbers taken an hour apart.
    """
    pkg = ARMS[arm][0]
    sys.path.insert(0, pkg)
    from nanomem.engine import VaultEngine
    e0 = open_engine(VaultEngine, arm, corpus)      # warm the interpreter
    n = e0.count()
    e0.close()
    del e0
    gc.collect()
    ts = []
    for _ in range(REOPEN_REPEATS):
        t = time.perf_counter()
        e = open_engine(VaultEngine, arm, corpus)
        ts.append(time.perf_counter() - t)
        e.close()
        del e
    return {"arm": arm, "corpus": corpus, "n_docs": int(n),
            "open_s_median": round(float(np.median(ts)), 6),
            "open_s_min": round(float(np.min(ts)), 6),
            "load_avg": loadavg()}


def run_firstwrite(arm: str, corpus: str) -> dict:
    """What the FIRST write after an open costs.

    A cached open defers the id index, the intern tables and the columns' move
    out of the mapping; a writer pays for all three at once. That is the bill
    laziness runs up and it is measured here rather than asserted to be small.
    """
    pkg = ARMS[arm][0]
    sys.path.insert(0, pkg)
    from nanomem.engine import VaultEngine
    dst = _writable_copy(corpus, arm, "firstwrite")
    kw = engine_kwargs(arm, corpus)
    gc.collect()
    base_max = maxrss_bytes()
    e = VaultEngine(dst, **kw)
    info = e.arena_cache_info() if hasattr(e, "arena_cache_info") else {}
    rng = np.random.default_rng(11)
    v = rng.normal(size=DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    t0 = time.perf_counter()
    e.add_fact("one appended row", v, source="wiki", metadata={"idx": -1})
    first_add_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    e.flush()
    flush_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    for i in range(50):
        w = rng.normal(size=DIM).astype(np.float32)
        e.add_fact("another appended row %d" % i, w / np.linalg.norm(w),
                   source="wiki", metadata={"idx": -2 - i})
    e.flush()
    next_50_s = time.perf_counter() - t2
    st = e.stats()
    n = e.count()
    e.close()
    return {"arm": arm, "corpus": corpus, "source": info.get("source"),
            "rows_after": int(n),
            "first_add_fact_s": round(first_add_s, 6),
            "first_flush_s": round(flush_s, 6),
            "next_50_rows_s": round(next_50_s, 6),
            "arena_growth_copies": st.get("arena_growth_copies"),
            "peak_maxrss_delta_bytes": maxrss_bytes() - base_max,
            "load_avg": loadavg()}


def run_latduel(arm: str, corpus: str) -> dict:
    """p50/p95 of the query path, plus the machine's bare-matvec floor, in one
    process. The driver alternates arms cycle by cycle so the question "does
    scanning a MAPPED arena cost anything per query" is answered paired, on the
    same minute -- nanomem/arena.py already carries a warning about what an
    unpaired ratio on a loaded box is worth.
    """
    pkg = ARMS[arm][0]
    sys.path.insert(0, pkg)
    from nanomem.engine import VaultEngine
    docs, Q, _t, _m = load_corpus(corpus)
    nq = int(Q.shape[0])
    e = open_engine(VaultEngine, arm, corpus)
    for i in range(min(WARMUP_QUERIES, nq)):
        e.search("", Q[i], top_k=TOP_K)
    lat = []
    for i in range(nq):
        t = time.perf_counter()
        e.search("", Q[i], top_k=TOP_K)
        lat.append((time.perf_counter() - t) * 1000.0)
    src_name = (e.arena_cache_info().get("source")
                if hasattr(e, "arena_cache_info") else "scan")
    e.close()
    del e
    A = np.ascontiguousarray(docs.astype(np.float16).astype(np.float32))
    for _ in range(3):
        _ = A @ np.ascontiguousarray(Q[0], dtype=np.float32)
    floor = []
    for i in range(min(50, nq)):
        t = time.perf_counter()
        _ = A @ np.ascontiguousarray(Q[i], dtype=np.float32)
        floor.append((time.perf_counter() - t) * 1000.0)
    del A
    return {"arm": arm, "corpus": corpus, "source": src_name,
            "p50_ms": round(pct(lat, 50), 4), "p95_ms": round(pct(lat, 95), 4),
            "floor_p50_ms": round(pct(floor, 50), 4), "load_avg": loadavg()}


def run_ties(arm: str, corpus: str) -> dict:
    """Prove that every top-10 disagreement with the BASELINE package is a tie.

    A changed top-10 list is only harmless if the two documents that swapped
    hold the SAME float32 cosine. That is a claim about numbers, so it is
    checked as one: both packages are run on the same vault, every position
    where their lists differ is located, and the two documents at that position
    are scored. ``all_disagreements_are_exact_ties`` is true only if every one
    of those gaps is exactly 0.0 -- a single non-zero gap is a real ranking
    change, and this phase reports ``ok: False`` so the driver records a
    failure instead of a footnote.

    Both packages are loaded in SEPARATE subprocesses, because they are two
    different ``nanomem`` packages and only one can be imported per process.
    """
    p, cdir = vault_path(corpus), corpus_dir(corpus)
    queries = np.load(os.path.join(cdir, "queries.npy"))
    tops = {}
    for tag, pkg in (("base", BASE_DIR), ("new", NEW_DIR)):
        fd, tmpf = tempfile.mkstemp(suffix=".npy"); os.close(fd)
        src = (
            "import sys, numpy as np\n"
            "sys.path.insert(0, %r)\n"
            "from nanomem.engine import VaultEngine\n"
            "Q = np.load(%r)\n"
            "kw = {} if %r == 'base' else {'arena_cache': 'off'}\n"
            "e = VaultEngine(%r, embed_dim=%d, **kw)\n"
            "out = np.full((Q.shape[0], %d), -1, dtype=np.int64)\n"
            "for i, q in enumerate(Q):\n"
            "    ids = [int(h['metadata']['idx']) for h in e.search('', q, top_k=%d)]\n"
            "    out[i, :len(ids)] = ids[:%d]\n"
            "e.close()\n"
            "np.save(%r, out)\n"
            % (pkg, os.path.join(cdir, "queries.npy"), tag, p, DIM,
               TOP_K, TOP_K, TOP_K, tmpf))
        r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("tie worker (%s) failed: %s" % (tag, r.stderr[-2000:]))
        tops[tag] = np.load(tmpf)
        os.remove(tmpf)

    base_tk, new_tk = tops["base"], tops["new"]
    rows_changed = [int(i) for i in np.flatnonzero(np.any(base_tk != new_tk, axis=1))]

    # score the disagreeing positions with the NEW package, in yet another
    # process, and report the largest gap found.
    fd, gapf = tempfile.mkstemp(suffix=".json"); os.close(fd)
    src = (
        "import sys, json\n"
        "import numpy as np\n"
        "sys.path.insert(0, %r)\n"
        "from nanomem.engine import VaultEngine\n"
        "Q = np.load(%r); base = np.load(%r); new = np.load(%r)\n"
        "rows = %r\n"
        "e = VaultEngine(%r, embed_dim=%d, arena_cache='off')\n"
        "row_of = {}\n"
        "for r in range(e.arena.n_rows):\n"
        "    row_of[int(e.arena.record(r)['metadata']['idx'])] = r\n"
        "gaps = []\n"
        "for i in rows:\n"
        "    s = np.asarray(e.arena.scores(Q[i]), dtype=np.float32)\n"
        "    for pos in range(new.shape[1]):\n"
        "        a, b = int(new[i, pos]), int(base[i, pos])\n"
        "        if a != b and a in row_of and b in row_of:\n"
        "            gaps.append(abs(float(s[row_of[a]]) - float(s[row_of[b]])))\n"
        "e.close()\n"
        "json.dump({'gaps': gaps}, open(%r, 'w'))\n"
        % (NEW_DIR, os.path.join(cdir, "queries.npy"),
           os.path.join(cdir, "_base.npy"), os.path.join(cdir, "_new.npy"),
           rows_changed, p, DIM, gapf))
    np.save(os.path.join(cdir, "_base.npy"), base_tk)
    np.save(os.path.join(cdir, "_new.npy"), new_tk)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("tie scorer failed: %s" % r.stderr[-2000:])
    gaps = json.load(open(gapf))["gaps"]
    os.remove(gapf)
    for f in ("_base.npy", "_new.npy"):
        try:
            os.remove(os.path.join(cdir, f))
        except OSError:
            pass

    worst = max(gaps) if gaps else 0.0
    ok = (worst == 0.0)
    return {"n_queries": int(base_tk.shape[0]),
            "top10_lists_changed_vs_baseline": len(rows_changed),
            "queries_changed": rows_changed,
            "positions_compared": len(gaps),
            "max_abs_score_gap_at_a_changed_position": worst,
            "all_disagreements_are_exact_ties": bool(ok),
            "verdict": ("every disagreement is between two documents holding the "
                        "SAME float32 cosine, so both orders are correct answers"
                        if ok else
                        "A DISAGREEMENT IS NOT A TIE -- this is a real ranking change"),
            "cause": ("engine._select_top_k, new in this working tree from the "
                      "PCA-screen workstream: it replaces argpartition's "
                      "unspecified order among equal scores with a defined "
                      "ascending-row-id tie-break. Not the arena cache -- "
                      "cache_off shows the same lists as cache_map."),
            "ok": bool(ok)}


PHASES = {"serve": run_serve, "rebuild": run_rebuild, "append": run_append,
          "scanduel": run_scanduel, "firstwrite": run_firstwrite,
          "latduel": run_latduel, "ties": run_ties}


def latency_duel(cycles: int = 5) -> None:
    """Merge a paired query-latency duel into an existing results file."""
    doc = json.load(open(RESULTS))
    tmp = os.path.join(WORK, "_json")
    os.makedirs(tmp, exist_ok=True)
    out_doc = {}
    for corpus in CORPORA:
        rows = {"baseline_e617ba3": [], "cache_map": []}
        floors = {"baseline_e617ba3": [], "cache_map": []}
        for cycle in range(cycles):
            for arm in ("baseline_e617ba3", "cache_map"):
                out = os.path.join(tmp, "lat.%s.%s.%d.json" % (arm, corpus, cycle))
                spawn(["--worker", "--arm", arm, "--corpus", corpus,
                       "--phase", "latduel", "--out", out])
                rec = json.load(open(out))
                if rec.get("ok"):
                    rows[arm].append(rec["p50_ms"])
                    floors[arm].append(rec["floor_p50_ms"])
                print("[lat ] %-18s %-8s cycle %d p50=%s floor=%s" % (
                    arm, corpus, cycle, rec.get("p50_ms"), rec.get("floor_p50_ms")),
                    flush=True)
        pair = [b / a for a, b in zip(rows["baseline_e617ba3"], rows["cache_map"]) if a]
        out_doc[corpus] = {
            "baseline_e617ba3_p50_ms": rows["baseline_e617ba3"],
            "cache_map_p50_ms": rows["cache_map"],
            "baseline_floor_p50_ms": floors["baseline_e617ba3"],
            "cache_map_floor_p50_ms": floors["cache_map"],
            "paired_ratio_mapped_over_scanned": [round(x, 4) for x in pair],
            "median_ratio": round(float(np.median(pair)), 4) if pair else None,
            "what": "does scanning a mapped fp32 arena cost anything per query? "
                    "Same vault, same 500 queries, arms alternated cycle by "
                    "cycle, bare-matvec floor measured in every cycle.",
        }
        print("[lat ] %-8s median ratio %s" % (corpus, out_doc[corpus]["median_ratio"]),
              flush=True)
    doc["latency_duel"] = out_doc
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)


def worker(arm: str, corpus: str, phase: str, out: str) -> int:
    rec = {"arm": arm, "corpus": corpus, "phase": phase, "ok": False}
    try:
        got = PHASES[phase](arm, corpus)
        tk = got.pop("topk", None)
        if tk is not None:
            np.save(out.replace(".json", ".topk.npy"), tk)
        rec.update(got)
        rec["ok"] = True
    except Exception:
        rec["error"] = traceback.format_exc()
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    return 0 if rec["ok"] else 1


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def spawn(args: List[str], timeout: int = 1800) -> None:
    cmd = [sys.executable, os.path.abspath(__file__)] + args
    subprocess.run(cmd, timeout=timeout, check=False)


def warm_cache(corpus: str) -> dict:
    """Create the arena cache the cached arms read, in its own process.

    Deliberately a separate process: it means no cached arm ever measures an
    open that also WROTE the thing it is opening, and it is also how the cache
    really comes to exist (some earlier process opened the vault).
    """
    out = os.path.join(WORK, "_json", "warm.%s.json" % corpus)
    spawn(["--worker", "--arm", "cache_map", "--corpus", corpus,
           "--phase", "rebuild", "--out", out])
    try:
        return json.load(open(out))
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def versions() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "baseline_commit": BASE_COMMIT}


def competitors_reopen() -> dict:
    """The competitors' own reopen numbers, read out of their results file."""
    try:
        doc = json.load(open(COMPETITORS))
    except Exception:
        return {}
    out = {}
    for corpus in CORPORA:
        row = {}
        for arm, rec in (doc.get("arms") or {}).items():
            got = (rec or {}).get(corpus) or {}
            if "reopen_s" in got:
                row[arm] = {"reopen_s": got.get("reopen_s"),
                            "first_query_after_reopen_s": got.get("first_query_after_reopen_s")}
        out[corpus] = row
    return out


def drive() -> None:
    os.makedirs(os.path.join(WORK, "_json"), exist_ok=True)
    if not os.path.isdir(BASE_DIR):
        raise SystemExit("baseline package missing: %s\n"
                         "  cd %r && git archive %s nanomem_standalone/nanomem "
                         "| tar -x -C %s" % (BASE_DIR, REPO, BASE_COMMIT,
                                             os.path.dirname(os.path.dirname(BASE_DIR))))
    for corpus in CORPORA:
        if not os.path.exists(vault_path(corpus)):
            raise SystemExit("no vault for %s -- run --prep first" % corpus)

    doc = {
        "title": "nanomem reopen: rebuilding the arena vs mapping it",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": os.path.abspath(__file__),
        "versions": versions(),
        "protocol": {
            "vaults": "one per corpus, built ONCE by the baseline package "
                      "(rng(0) insertion order, source 'wiki', metadata "
                      "{'idx': row}); every arm opens the same bytes",
            "arena_cache": "created by a separate process before any cached arm "
                           "runs, so no measured open also wrote the cache",
            "reopen_in_process": "median of %d opens in one warm interpreter -- "
                                 "the protocol competitors_standard_results.json "
                                 "used, and therefore the one compared against "
                                 "sqlite-vec 0.0014 s / Chroma 0.0017 s" % REOPEN_REPEATS,
            "reopen_fresh_process": "the first open of a brand-new interpreter, "
                                    "which additionally pays whatever the open "
                                    "imports lazily",
            "page_cache": "WARM for every arm. `purge` requires root here and was "
                          "refused, so a cold-cache reopen is NOT measured; the "
                          "competitors' numbers were taken warm too.",
            "latency": "500 queries (120 at val1190) after %d warm-ups, p50/p95" % WARMUP_QUERIES,
            "exactness": "SHA-256 over the concatenated fp32 score vectors of "
                         "every query against every row, plus the top-10 id "
                         "matrix, both compared against the baseline arm",
            "memory": "one subprocess per (arm, corpus, phase); ru_maxrss, live "
                      "RSS and phys_footprint",
        },
        "pre_registered": {
            "primary": "cache_map reopen_s_in_process_median at n71433 <= 0.0014 s "
                       "(sqlite-vec's measured reopen) => nanomem wins reopen",
            "secondary": "time to first answer = reopen + first query, reported "
                         "for every arm at every size whether or not the primary "
                         "passes",
            "gate": "identical score digests and 0 changed top-10 lists at every "
                    "size, or the arm is a failure regardless of its latency",
            "gate_amended_after_it_failed": (
                "STATED PLAINLY BECAUSE IT IS AN AMENDMENT MADE AFTER SEEING THE "
                "RESULT. The gate above compares against the committed package, "
                "and the working tree under test contains a SECOND, unrelated "
                "change -- engine._select_top_k, the PCA-screen workstream's "
                "ascending-row-id tie-break -- so the gate cannot attribute a "
                "changed list to the arena cache. It failed: 8 of 500 lists at "
                "n71433, 1 of 500 at n10000. Rather than redefine it, BOTH are "
                "now reported: the original gate (still failing, with every "
                "disagreement mechanically proved to be a bitwise tie by "
                "`tie_forensics`, which fails loudly if one is not) and an "
                "isolating control against the same build with the cache off, "
                "which is the only comparison in which the arena cache is the "
                "sole variable."),
        },
        "corpora": {}, "arms": {}, "rebuild": {}, "append": {},
        "competitors": competitors_reopen(),
        "failures": [],
    }
    for corpus in CORPORA:
        doc["corpora"][corpus] = json.load(open(os.path.join(corpus_dir(corpus), "meta.json")))
        doc["corpora"][corpus]["vault_bytes"] = path_bytes(vault_path(corpus))

    tmp = os.path.join(WORK, "_json")
    topks, digests = {}, {}
    for corpus in CORPORA:
        print("[warm] %s" % corpus, flush=True)
        doc.setdefault("warm", {})[corpus] = warm_cache(corpus)
        for arm in ARMS:
            out = os.path.join(tmp, "serve.%s.%s.json" % (arm, corpus))
            print("[run ] %-18s %-8s serve" % (arm, corpus), flush=True)
            spawn(["--worker", "--arm", arm, "--corpus", corpus,
                   "--phase", "serve", "--out", out])
            try:
                rec = json.load(open(out))
            except Exception as exc:
                doc["failures"].append({"arm": arm, "corpus": corpus,
                                        "phase": "serve", "error": repr(exc)})
                continue
            if not rec.get("ok"):
                doc["failures"].append({"arm": arm, "corpus": corpus,
                                        "phase": "serve", "error": rec.get("error")})
                continue
            doc["arms"].setdefault(arm, {})[corpus] = rec
            digests[(arm, corpus)] = rec["exactness"]["scores_sha256"]
            tkf = out.replace(".json", ".topk.npy")
            if os.path.exists(tkf):
                topks[(arm, corpus)] = np.load(tkf)

        for arm in CACHED_ARMS + (BASELINE_ARM, "cache_off"):
            out = os.path.join(tmp, "append.%s.%s.json" % (arm, corpus))
            spawn(["--worker", "--arm", arm, "--corpus", corpus,
                   "--phase", "append", "--out", out])
            try:
                rec = json.load(open(out))
                if rec.get("ok"):
                    doc["append"].setdefault(arm, {})[corpus] = rec
                else:
                    doc["failures"].append(rec)
            except Exception as exc:
                doc["failures"].append({"arm": arm, "corpus": corpus,
                                        "phase": "append", "error": repr(exc)})

        for arm in ("cache_map", "cache_off", BASELINE_ARM):
            out = os.path.join(tmp, "rebuild.%s.%s.json" % (arm, corpus))
            spawn(["--worker", "--arm", arm, "--corpus", corpus,
                   "--phase", "rebuild", "--out", out])
            try:
                rec = json.load(open(out))
                if rec.get("ok"):
                    doc["rebuild"].setdefault(arm, {})[corpus] = rec
                else:
                    doc["failures"].append(rec)
            except Exception as exc:
                doc["failures"].append({"arm": arm, "corpus": corpus,
                                        "phase": "rebuild", "error": repr(exc)})
        # Every rebuild arm DELETES the cache (that is what it measures), and
        # `cache_off`/`baseline` never write one back, so the cache is restored
        # here before any later phase that needs it.
        doc.setdefault("rewarm", {})[corpus] = warm_cache(corpus)

    # ---- paired scan-path duel: does the new package cost anything? ------
    duel = {}
    for corpus in CORPORA:
        rows = {"baseline_e617ba3": [], "cache_off": []}
        for cycle in range(5):
            for arm in ("baseline_e617ba3", "cache_off"):
                out = os.path.join(tmp, "duel.%s.%s.%d.json" % (arm, corpus, cycle))
                spawn(["--worker", "--arm", arm, "--corpus", corpus,
                       "--phase", "scanduel", "--out", out])
                try:
                    rec = json.load(open(out))
                    if rec.get("ok"):
                        rows[arm].append(rec["open_s_median"])
                except Exception:
                    pass
        pair = [b / a for a, b in zip(rows["baseline_e617ba3"], rows["cache_off"])
                if a]
        duel[corpus] = {
            "baseline_e617ba3_open_s": rows["baseline_e617ba3"],
            "cache_off_open_s": rows["cache_off"],
            "paired_ratio_new_over_baseline": [round(x, 4) for x in pair],
            "median_ratio": round(float(np.median(pair)), 4) if pair else None,
            "what": "same vault, same protocol, arms alternated cycle by cycle. "
                    "The cache is not read or written in either arm, so a ratio "
                    "away from 1.0 is the new package's cost on the path it did "
                    "not change.",
        }
    doc["scan_path_duel"] = duel

    # ---- what laziness costs the first writer ----------------------------
    fw = {}
    for corpus in CORPORA:
        for arm in ("cache_off", "cache_map"):
            out = os.path.join(tmp, "fw.%s.%s.json" % (arm, corpus))
            spawn(["--worker", "--arm", arm, "--corpus", corpus,
                   "--phase", "firstwrite", "--out", out])
            try:
                rec = json.load(open(out))
                if rec.get("ok"):
                    fw.setdefault(arm, {})[corpus] = rec
                else:
                    doc["failures"].append(rec)
            except Exception as exc:
                doc["failures"].append({"arm": arm, "corpus": corpus,
                                        "phase": "firstwrite", "error": repr(exc)})
    doc["first_write_after_open"] = fw

    # ---- exactness, judged here, never by an arm -------------------------
    #
    # TWO references, because this working tree contains TWO independent
    # changes and one reference cannot separate them:
    #
    #   vs baseline_e617ba3  the committed package. This is the pre-registered
    #                        gate, and it is NOT clean: the same working tree
    #                        also carries `engine._select_top_k`, which is the
    #                        PCA-screen workstream's deliberate replacement of
    #                        `argpartition`'s unspecified order among EQUAL
    #                        scores with a defined ascending-row-id tie-break.
    #                        That moves which of two bit-identical duplicate
    #                        paragraphs fills a slot, on a corpus where 0.075%
    #                        of rows have an exact duplicate. It has nothing to
    #                        do with the arena cache.
    #   vs cache_off         the SAME build with the cache switched off. This is
    #                        the control that isolates THIS change -- the only
    #                        difference between it and the cached arms is
    #                        whether the arena was mapped or rebuilt -- and it
    #                        is the number that answers "does mapping the arena
    #                        change an answer".
    #
    # Both are reported. `tie_forensics` then proves mechanically that every
    # disagreement against the baseline is a bitwise tie, and FAILS LOUDLY if
    # one of them is not, so "they are only ties" is checked, never asserted.
    ex = {}
    for corpus in CORPORA:
        ref_sha = digests.get((BASELINE_ARM, corpus))
        ref_tk = topks.get((BASELINE_ARM, corpus))
        off_tk = topks.get(("cache_off", corpus))
        off_sha = digests.get(("cache_off", corpus))
        rows = {}
        for arm in ARMS:
            sha = digests.get((arm, corpus))
            tk = topks.get((arm, corpus))
            changed = changed_off = None
            if ref_tk is not None and tk is not None and tk.shape == ref_tk.shape:
                changed = int(np.sum(np.any(tk != ref_tk, axis=1)))
            if off_tk is not None and tk is not None and tk.shape == off_tk.shape:
                changed_off = int(np.sum(np.any(tk != off_tk, axis=1)))
            rows[arm] = {"scores_sha256": sha,
                         "identical_to_baseline": bool(sha is not None and sha == ref_sha),
                         "identical_to_cache_off": bool(sha is not None and sha == off_sha),
                         "top10_lists_changed_vs_baseline": changed,
                         "top10_lists_changed_vs_same_build_cache_off": changed_off,
                         "n_queries": int(ref_tk.shape[0]) if ref_tk is not None else None}
        ex[corpus] = rows
    doc["exactness"] = ex

    # ---- are those disagreements ties, or are they losses? ---------------
    tf = {}
    for corpus in CORPORA:
        out = os.path.join(tmp, "ties.%s.json" % corpus)
        spawn(["--worker", "--arm", "cache_off", "--corpus", corpus,
               "--phase", "ties", "--out", out])
        try:
            tf[corpus] = json.load(open(out))
        except Exception as exc:
            tf[corpus] = {"ok": False, "error": repr(exc)}
        if not tf[corpus].get("all_disagreements_are_exact_ties"):
            doc["failures"].append({"corpus": corpus, "phase": "ties",
                                    "detail": tf[corpus]})
    doc["tie_forensics"] = tf

    doc["headline"] = build_headline(doc)
    doc["verdict"] = build_verdict(doc)
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)
    print(json.dumps(doc["verdict"], indent=2))


def build_headline(doc: dict) -> dict:
    out = {}
    for corpus in CORPORA:
        row = {}
        for arm in ARMS:
            rec = (doc["arms"].get(arm) or {}).get(corpus)
            if not rec:
                continue
            t, m, d = rec["timing"], rec["memory"], rec["disk"]
            row[arm] = {
                "reopen_s": t["reopen_s_in_process_median"],
                "reopen_s_fresh_process": t["reopen_s_fresh_process"],
                "first_query_s": t["first_query_after_reopen_s_median_of_5"],
                "time_to_first_answer_s": t["time_to_first_answer_s"],
                "p50_ms": t["latency_ms"]["p50"], "p95_ms": t["latency_ms"]["p95"],
                "numpy_floor_p50_ms": t["numpy_floor_ms"]["p50"],
                "peak_rss_mb": round(m["peak_maxrss_delta_bytes"] / 1048576, 1),
                "live_rss_mb": (round(m["live_rss_delta_bytes"] / 1048576, 1)
                                if m["live_rss_delta_bytes"] else None),
                "phys_footprint_mb": (round(m["phys_footprint_delta_bytes"] / 1048576, 1)
                                      if m["phys_footprint_delta_bytes"] else None),
                "vault_mib": round(d["vault_bytes"] / 1048576, 1),
                "cache_mib": round(d["arena_cache_bytes"] / 1048576, 1),
                "arena_source": rec["arena_cache_info"].get("source"),
                "load_avg_start": rec["load_avg_start"],
            }
        out[corpus] = row
    return out


def build_verdict(doc: dict) -> dict:
    base = ((doc["arms"].get(BASELINE_ARM) or {}).get("n71433") or {})
    new = ((doc["arms"].get("cache_map") or {}).get("n71433") or {})
    if not base or not new:
        return {"status": "incomplete"}
    b, n = base["timing"]["reopen_s_in_process_median"], new["timing"]["reopen_s_in_process_median"]
    target = TARGETS["sqlitevec_bruteforce"]
    ex = doc["exactness"].get("n71433", {}).get("cache_map", {})
    arms = ("cache_off", "cache_map", "cache_copy", "cache_verify")

    # The gate as it was WRITTEN: identical digests and zero changed top-10
    # lists against the committed package. It does NOT pass, and it is reported
    # not passing rather than redefined into passing.
    exact_everywhere = all(
        doc["exactness"][c][a]["identical_to_baseline"]
        and (doc["exactness"][c][a]["top10_lists_changed_vs_baseline"] in (0, None))
        for c in CORPORA for a in arms if a in doc["exactness"].get(c, {}))

    # The gate that ISOLATES this change: the same build, cache on against cache
    # off. Every difference above is carried by code this workstream did not
    # write (see `exactness`' comment and `tie_forensics`), and this is the
    # comparison in which only the arena cache varies.
    exact_vs_self = all(
        doc["exactness"][c][a]["identical_to_cache_off"]
        and (doc["exactness"][c][a]["top10_lists_changed_vs_same_build_cache_off"]
             in (0, None))
        for c in CORPORA for a in arms if a in doc["exactness"].get(c, {}))

    tf = doc.get("tie_forensics", {})
    all_ties = all(bool(tf.get(c, {}).get("all_disagreements_are_exact_ties"))
                   for c in CORPORA if c in tf)
    changed_vs_base = {c: doc["exactness"][c]["cache_map"]
                       ["top10_lists_changed_vs_baseline"] for c in CORPORA
                       if "cache_map" in doc["exactness"].get(c, {})}
    return {
        "reopen_before_s": b, "reopen_after_s": n,
        "speedup": round(b / n, 1) if n else None,
        "target_sqlitevec_s": target,
        "wins_reopen": bool(n <= target),
        "margin_vs_sqlitevec": round(n / target, 2) if target else None,
        "time_to_first_answer_before_s": base["timing"]["time_to_first_answer_s"],
        "time_to_first_answer_after_s": new["timing"]["time_to_first_answer_s"],
        "exactness_gate_passed": bool(exact_everywhere),
        "exactness_gate_passed_note": (
            "This is the gate AS PRE-REGISTERED -- zero changed top-10 lists "
            "against the committed package -- and it does not pass: %s. Every "
            "one of those disagreements is proved by `tie_forensics` to be "
            "between two documents holding the SAME float32 cosine (max gap "
            "0.0), and it is carried by engine._select_top_k, which this "
            "working tree gained from the PCA-screen workstream and which "
            "deliberately replaced argpartition's unspecified tie order with "
            "an ascending-row-id one. It is not the arena cache: cache_off, "
            "which never reads or writes a cache, shows exactly the same lists "
            "as cache_map." % (changed_vs_base,)),
        "exactness_gate_vs_same_build_cache_off_passed": bool(exact_vs_self),
        "exactness_gate_vs_same_build_cache_off_note": (
            "The isolating control: same build, same vault, same queries, the "
            "ONLY difference being whether the arena was mapped from the cache "
            "or rebuilt by a scan. Mapping the arena changes no score and no "
            "top-10 list at any size."),
        "all_baseline_disagreements_are_exact_ties": bool(all_ties),
        "exactness_n71433_map": ex,
    }


def _forge_a_row_in_the_vault(path: str, victim: int, q, VaultEngine, C, crypto) -> int:
    """Rewrite one row's fp16 vector inside its block body and recompute the
    block's UNKEYED trailer. Returns ``(honest top-1, the victim row's own id)``,
    both as the corpus indices an answer carries."""
    e = VaultEngine(path, embed_dim=DIM, arena_cache="off")
    honest = int(e.search("", q, top_k=1)[0]["metadata"]["idx"])
    # `victim` is an ARENA ROW; the answer carries the corpus index the row was
    # written with, and the two differ because the vault was built in a shuffled
    # insertion order. Comparing the wrong one reports a real hijack as "safe".
    victim_idx = int(e.get(e.arena.ids[victim])["metadata"]["idx"])
    blk = int(e.arena.row_block[victim])
    meta = list(e._cont.blocks)[blk]
    off_in_blk = victim - int(e.arena.block_start[blk])
    uuid = bytes(e.header.vault_uuid)
    e.close()
    with open(path, "r+b") as f:
        f.seek(int(meta.offset))
        hdr_bytes = f.read(C.BLOCK_HEADER_SIZE)
        body = bytearray(f.read(int(meta.payload_len)))
        vecs = np.frombuffer(bytes(body), dtype=np.float16, count=meta.n * DIM,
                             offset=0).reshape(meta.n, DIM).copy()
        vecs[off_in_blk] = np.asarray(q, dtype=np.float16)
        body[:meta.n * DIM * 2] = vecs.tobytes()
        f.seek(int(meta.offset) + C.BLOCK_HEADER_SIZE)
        f.write(bytes(body))
        f.write(crypto.block_tag(None, uuid, hdr_bytes[:C.BLOCK_AUTH_LEN], bytes(body)))
    return honest, victim_idx


def _edit_a_block_in_place(path: str) -> None:
    """Flip one byte of the first block's payload and leave its trailer stale --
    the damage a scan exists to catch. Same length, so no length check sees it."""
    off = 256 + 96 + 16                      # FILE_HEADER + BLOCK_HEADER + 16
    with open(path, "r+b") as f:
        f.seek(off)
        b = f.read(1)
        f.seek(off)
        f.write(bytes([b[0] ^ 0xFF]))


def cache_integrity(repeats: int = 5) -> dict:
    """Who can make this vault answer a query wrong, and what does checking cost?

    REWRITTEN AFTER A REVIEW, and the rewrite is the point. The first version of
    this phase planted a row in the cache and reported "map HIJACKED, copy
    HIJACKED, verify SAFE", from which the workstream concluded it had found and
    fixed a security defect. That conclusion is WITHDRAWN. The phase had one
    arm too few: it never let the attacker update the digest it was being caught
    by. Two arms are added here and both invert the conclusion.

      ``cache_tamper.forged_digest``  plant the row, then recompute the cache's
          own SHA-256 and the header CRC32 -- ~10 lines, no key needed, because
          the digest is unkeyed and lives inside the file it describes. All
          three modes serve the planted row, ``verify`` included.
      ``vault_forgery``  the same hijack with NO CACHE ON DISK AT ALL and
          ``arena_cache="off"``, i.e. the full scan that re-reads every block.
          A plaintext vault's trailer is an unkeyed SHA-256 (crypto.py's own
          THREAT_MODEL), and a vault with a passphrase -- whose trailer would be
          a real HMAC -- is refused a cache outright. So the sidecar was never
          "a strictly weaker path to the same answers": it is exactly as weak as
          the file it sits beside.

    What survives is a corruption story, and one real fix inside it:
    ``vault_inplace_edit`` is a block edited in place with its trailer left
    stale, which ``off`` and ``verify`` refuse with ``IntegrityError`` and which
    ``map`` -- the DEFAULT -- used to serve in silence. A one-stat (size, mtime)
    binding now refuses that cache so the scan runs and raises. Its own limit is
    measured in the same table: restore the mtime with ``os.utime`` and ``map``
    serves it again.
    """
    sys.path.insert(0, NEW_DIR)
    from nanomem import arena as A
    from nanomem import container as C
    from nanomem import crypto
    from nanomem.engine import VaultEngine
    from nanomem.errors import IntegrityError

    def top1(engine, q):
        hits = engine.search("", q, top_k=10)
        return int(hits[0]["metadata"]["idx"]) if hits else None

    def open_and_ask(path, mode, q, honest, row=0):
        """Open in ``mode`` and report what it answered, or what it refused."""
        try:
            e = VaultEngine(path, embed_dim=DIM, arena_cache=mode)
        except IntegrityError as exc:
            return {"served_from": None, "reason": None, "raised": type(exc).__name__,
                    "raised_text": str(exc), "top1_returned": None,
                    "top1_honest": honest, "answer_hijacked": False}
        info = e.arena_cache_info()
        got = top1(e, q)
        score0 = float(np.asarray(e.arena.scores(q))[int(row)])
        out = {"served_from": info.get("source"), "reason": info.get("reason"),
               "raised": None,
               "vault_changed_since_cache": info.get("vault_changed_since_cache"),
               "vault_blocks_checked": info.get("vault_blocks_checked"),
               "planted_row_score": round(score0, 6), "scored_row": int(row),
               "top1_returned": got, "top1_honest": honest,
               "answer_hijacked": bool(got != honest)}
        e.close()
        return out

    out = {
        "what": ("ask every mode what it answers under four attacks, and time what "
                 "the checking costs"),
        "withdrawn": [
            "\"verify SAFE\" -- verify is hijacked by a forged content digest "
            "(cache_tamper.forged_digest), because the digest is unkeyed and "
            "stored inside the file it authenticates behind a CRC32.",
            "\"the sidecar is a strictly weaker, unauthenticated path to the same "
            "answers\" -- the vault is forgeable too and by the same means "
            "(vault_forgery); a cache is only ever written beside a plaintext "
            "vault, whose block trailer is an unkeyed SHA-256.",
            "\"a real security defect, found and fixed\" -- the defect is real, "
            "the fix is a corruption check. Nothing here resists an attacker who "
            "can write the vault's directory, with or without a cache.",
        ],
        "supersedes": ("the version of this phase that reported 'map HIJACKED, copy "
                       "HIJACKED, verify SAFE'. It had one arm too few: it never let "
                       "the attacker update the digest it was being caught by."),
        "corpora": {},
    }
    for corpus in CORPORA:
        p = vault_path(corpus)
        cpath = cache_path(corpus)
        docs, queries, _texts, _meta = load_corpus(corpus)
        q0 = np.ascontiguousarray(queries[0], dtype=np.float32)

        if os.path.exists(cpath):
            os.remove(cpath)
        VaultEngine(p, embed_dim=DIM, arena_cache="map").close()   # write the cache
        clean_bytes = path_bytes(cpath)

        e = VaultEngine(p, embed_dim=DIM, arena_cache="off")
        honest = top1(e, q0)
        e.close()

        # --- what each half of `verify` costs, on the CLEAN cache -------------
        digest_s, reauth_s, verify_open_s, off_open_s = [], [], [], []
        for _ in range(repeats):
            snap = A.ArenaSnapshot.open(cpath)
            t = time.perf_counter()
            got = snap.content_sha256()
            digest_s.append(time.perf_counter() - t)
            assert got == snap.header["content_sha256"]
            snap.close()
            t = time.perf_counter()
            e = VaultEngine(p, embed_dim=DIM, arena_cache="verify")
            verify_open_s.append(time.perf_counter() - t)
            info = e.arena_cache_info()
            assert info["source"] == "cache", info
            reauth_s.append(float(info["verify_s"]) - digest_s[-1])
            e.close()
            t = time.perf_counter()
            e = VaultEngine(p, embed_dim=DIM, arena_cache="off")
            off_open_s.append(time.perf_counter() - t)
            e.close()

        # --- attack 1+2: the cache, with and without a re-forged digest -------
        raw = open(cpath, "rb").read()
        hdr = A._parse_cache_header(raw[:A.ARENA_CACHE_HEADER])
        spec = hdr["sections"]["vec"]
        rb = int(spec["shape"][1]) * np.dtype(spec["dtype"]).itemsize
        buf = bytearray(raw)
        buf[int(spec["off"]):int(spec["off"]) + rb] = np.ascontiguousarray(
            q0, dtype=np.dtype(spec["dtype"])).tobytes()
        body_only = bytes(buf)
        assert len(body_only) == len(raw)

        forged = bytearray(body_only)
        h2 = A._parse_cache_header(bytes(forged[:A.ARENA_CACHE_HEADER]))
        h2["content_sha256"] = hashlib.sha256(
            bytes(forged[A.ARENA_CACHE_HEADER:int(h2["total_bytes"])])).hexdigest()
        forged[:A.ARENA_CACHE_HEADER] = A._pack_cache_header(h2)
        forged = bytes(forged)
        assert len(forged) == len(raw)

        cache_tamper = {}
        for label, poisoned in (("body_only", body_only), ("forged_digest", forged)):
            modes = {}
            for mode in ("map", "copy", "verify"):
                with open(cpath, "wb") as fh:       # undo any self-repair
                    fh.write(poisoned)
                modes[mode] = open_and_ask(p, mode, q0, honest)
                modes[mode]["cache_repaired_by_this_open"] = bool(
                    path_bytes(cpath) == clean_bytes
                    and open(cpath, "rb").read() != poisoned)
            cache_tamper[label] = modes

        # leave a clean cache behind
        if os.path.exists(cpath):
            os.remove(cpath)
        VaultEngine(p, embed_dim=DIM, arena_cache="map").close()

        # --- attack 3: forge the VAULT, with no cache in sight ----------------
        dst = _writable_copy(corpus, "forge", "attack")
        os.remove(dst + ".arena")
        e = VaultEngine(dst, embed_dim=DIM, arena_cache="off")
        victim = int(np.argmin(np.asarray(e.arena.scores(q0))))    # the worst row
        e.close()
        honest_v, victim_idx = _forge_a_row_in_the_vault(
            dst, victim, q0, VaultEngine, C, crypto)
        vf = open_and_ask(dst, "off", q0, honest_v, row=victim)
        vf["victim_row"] = victim
        vf["victim_idx"] = victim_idx
        vf["cache_on_disk"] = os.path.exists(dst + ".arena")
        vf["answer_hijacked"] = bool(vf["top1_returned"] == victim_idx)
        vf["note"] = ("no sidecar exists here and mode=off never looks for one: this is "
                      "the full scan that re-reads every block and recomputes every "
                      "trailer, and it accepts a row forged with crypto.block_tag(None, ...)")

        # --- attack 4: edit a block in place and leave its trailer stale ------
        dst = _writable_copy(corpus, "inplace", "attack")
        st = os.stat(dst)
        _edit_a_block_in_place(dst)
        inplace = {}
        for mode in ("off", "verify", "map"):
            inplace[mode] = open_and_ask(dst, mode, q0, honest)
        os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))        # the whole attack
        inplace["map_mtime_restored"] = open_and_ask(dst, "map", q0, honest)
        inplace["map_mtime_restored"]["note"] = (
            "this row IS the pre-fix behaviour of the default, reproduced by putting "
            "the mtime back: a hard IntegrityError becomes a successful open")
        shutil.rmtree(os.path.join(WORK, "attack"), ignore_errors=True)

        out["corpora"][corpus] = {
            "cache_bytes": clean_bytes,
            "cache_mib": round(clean_bytes / 1048576, 1),
            "vault_mib": round(path_bytes(p) / 1048576, 1),
            "content_digest_s_median": round(float(np.median(digest_s)), 6),
            "vault_reauth_s_median": round(float(np.median(reauth_s)), 6),
            "verify_open_s_median": round(float(np.median(verify_open_s)), 6),
            "off_open_s_median": round(float(np.median(off_open_s)), 6),
            "verify_vs_scan_x": round(float(np.median(verify_open_s))
                                      / float(np.median(off_open_s)), 3),
            "digest_throughput_mib_s": round(
                clean_bytes / 1048576 / float(np.median(digest_s)), 1),
            "cache_tamper": cache_tamper,
            "vault_forgery": vf,
            "vault_inplace_edit": inplace,
        }
        print("[integrity] %-8s digest %.4fs reauth %.4fs verify %.4fs scan %.4fs (%.2fx)"
              % (corpus, np.median(digest_s), np.median(reauth_s),
                 np.median(verify_open_s), np.median(off_open_s),
                 np.median(verify_open_s) / np.median(off_open_s)), flush=True)
        for label, modes in cache_tamper.items():
            print("            cache/%-13s %s" % (label, ", ".join(
                "%s:%s" % (m, "HIJACKED" if v["answer_hijacked"] else "safe")
                for m, v in modes.items())), flush=True)
        print("            vault_forgery(off)  %s" % (
            "HIJACKED" if vf["answer_hijacked"] else "safe"), flush=True)
        print("            inplace_edit        %s" % (", ".join(
            "%s:%s" % (m, v["raised"] or ("HIJACKED" if v["answer_hijacked"]
                                          else "served %s" % v["served_from"]))
            for m, v in inplace.items())), flush=True)

    out["finding"] = (
        "CORRECTED. (1) A forged content digest defeats `verify` in every corpus: "
        "the digest is unkeyed, it is stored in the header of the file it "
        "describes, and that header is protected by a CRC32, so the same editor "
        "recomputes both. (2) The vault is forgeable by the same means with no "
        "cache involved at all, because a plaintext vault's block trailer is an "
        "unkeyed SHA-256 and an encrypted vault is refused a cache -- so the "
        "sidecar inherits the vault's threat model rather than lowering it. "
        "Nothing here is tamper resistance; all of it is corruption detection. "
        "(3) The one real improvement is narrow and is measured as such: a "
        "(size, mtime) binding makes `map` refuse a vault that was rewritten in "
        "place, where it used to serve one silently while `off` and `verify` "
        "raised -- and `os.utime` defeats that binding, which is the "
        "`map_mtime_restored` row.")
    return out


NOBIND_DIR = "/tmp/nanomem_nobinding/nanomem_standalone"

_REOPEN_PROBE = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from nanomem.engine import VaultEngine
p, mode, reps = sys.argv[2], sys.argv[3], int(sys.argv[4])
VaultEngine(p, embed_dim=768, vector_dtype="float16", router="off",
            n_exhaustive=50000, durable="none", arena_cache=mode).close()
ts, srcs = [], []
for _ in range(reps):
    t = time.perf_counter()
    e = VaultEngine(p, embed_dim=768, vector_dtype="float16", router="off",
                    n_exhaustive=50000, durable="none", arena_cache=mode)
    ts.append(time.perf_counter() - t)
    srcs.append(e.arena_cache_info().get("source"))
    e.close()
print(json.dumps({"median": float(np.median(ts)), "min": float(min(ts)),
                  "sources": sorted(set(srcs))}))
"""


def _make_nobinding_build() -> dict:
    """A copy of the package under test with ONE thing removed: the (size, mtime)
    binding check in the open path. Everything else -- arena.py, container.py,
    the cache format, the header fields -- is byte-identical, so the pair
    isolates what the fix costs rather than what the session changed.
    """
    dst = os.path.dirname(NOBIND_DIR)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(NEW_DIR, NOBIND_DIR,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    p = os.path.join(NOBIND_DIR, "nanomem", "engine.py")
    src = open(p, encoding="utf-8").read()
    start = src.index("        changed = _arena.vault_changed_since_cache(")
    end = src.index("        if self.arena_cache == \"verify\":", start)
    removed = src[start:end]
    assert "return 0" in removed and len(removed) < 900, len(removed)
    src = src[:start] + "        info[\"vault_changed_since_cache\"] = None\n" + src[end:]
    open(p, "w", encoding="utf-8").write(src)
    import hashlib as _h
    def sha(path):
        return _h.sha256(open(path, "rb").read()).hexdigest()[:16]
    same = {f: sha(os.path.join(NEW_DIR, "nanomem", f)) == sha(os.path.join(NOBIND_DIR, "nanomem", f))
            for f in ("arena.py", "container.py", "screen.py", "vault.py")}
    return {"dir": NOBIND_DIR, "removed_chars": len(removed),
            "other_modules_identical": same}


def binding_cost(cycles: int = 6, reps: int = 25) -> dict:
    """What does the (size, mtime) binding cost the open it was added to?

    One extra ``os.stat`` per cached open, and one per cache write. Measured as
    a PAIRED, INTERLEAVED comparison against the same package with that check
    deleted (see ``_make_nobinding_build``), in fresh subprocesses, alternating
    which build goes first so neither always pays for a cold page cache.
    """
    build = _make_nobinding_build()
    out = {"what": ("paired A/B of the same package with and without the "
                    "(size, mtime) binding check, in-process reopen median of "
                    "%d, %d interleaved cycles" % (reps, cycles)),
           "control_build": build, "cycles": {}, "corpora": {}}
    arms = {"with_binding": NEW_DIR, "no_binding": NOBIND_DIR}
    for corpus in ("n10000", "n71433"):
        p = vault_path(corpus)
        if not os.path.exists(cache_path(corpus)):
            sys.path.insert(0, NEW_DIR)
            from nanomem.engine import VaultEngine
            VaultEngine(p, embed_dim=DIM, arena_cache="map").close()
        per = {k: [] for k in arms}
        for c in range(cycles):
            order = list(arms) if c % 2 == 0 else list(arms)[::-1]
            for arm in order:
                r = subprocess.run([sys.executable, "-c", _REOPEN_PROBE, arms[arm],
                                    p, "map", str(reps)],
                                   capture_output=True, text=True, timeout=1800)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-2000:])
                got = json.loads(r.stdout.strip().splitlines()[-1])
                assert got["sources"] == ["cache"], got
                per[arm].append(float(got["median"]))
        med = {k: float(np.median(v)) for k, v in per.items()}
        out["corpora"][corpus] = {
            "with_binding_s": round(med["with_binding"], 8),
            "no_binding_s": round(med["no_binding"], 8),
            "delta_us": round((med["with_binding"] - med["no_binding"]) * 1e6, 2),
            "ratio": round(med["with_binding"] / med["no_binding"], 4),
            "per_cycle": {k: [round(x, 8) for x in v] for k, v in per.items()},
        }
        print("[binding] %-8s with %.6fs  without %.6fs  %+.2f us (%.3fx)" % (
            corpus, med["with_binding"], med["no_binding"],
            out["corpora"][corpus]["delta_us"], out["corpora"][corpus]["ratio"]),
            flush=True)
    out["loadavg"] = loadavg()
    return out


def post_fix_check(cycles: int = 3) -> dict:
    """Did the binding change any ANSWER or any query LATENCY? It should not.

    The check only decides whether a cache is used; when one is used the bytes
    served are the bytes that were always served. That is an argument, so this
    measures it instead: a SHA-256 over the full fp32 score vector of 200
    queries against every row, per mode, plus a paired p50 in the same process.
    """
    sys.path.insert(0, NEW_DIR)
    from nanomem.engine import VaultEngine
    out = {"what": ("score-vector digest and p50 per mode on the build WITH the "
                    "binding, so the isolating control is cache-on vs cache-off "
                    "inside one build"), "corpora": {}}
    for corpus in ("n10000", "n71433"):
        p_ = vault_path(corpus)
        _docs, queries, _t, _m = load_corpus(corpus)
        Q = [np.ascontiguousarray(queries[i], dtype=np.float32)
             for i in range(min(200, len(queries)))]
        rec = {}
        for mode in ("off", "map", "copy", "verify"):
            e = VaultEngine(p_, embed_dim=DIM, vector_dtype="float16", router="off",
                            n_exhaustive=50_000, durable="none", arena_cache=mode)
            h = hashlib.sha256()
            for q in Q:
                h.update(np.ascontiguousarray(e.arena.scores(q), dtype=np.float32).tobytes())
            for q in Q[:20]:
                e.search("", q, top_k=TOP_K)
            lat = []
            for _ in range(cycles):
                for q in Q:
                    t = time.perf_counter()
                    e.search("", q, top_k=TOP_K)
                    lat.append((time.perf_counter() - t) * 1e3)
            info = e.arena_cache_info()
            rec[mode] = {"scores_sha256": h.hexdigest(),
                         "source": info.get("source"),
                         "vault_changed_since_cache": info.get("vault_changed_since_cache"),
                         "vault_blocks_checked": info.get("vault_blocks_checked"),
                         "p50_ms": round(pct(lat, 50), 4), "n_queries": len(Q)}
            e.close()
        digests = {m: r["scores_sha256"] for m, r in rec.items()}
        rec["all_modes_identical"] = len(set(digests.values())) == 1
        out["corpora"][corpus] = rec
        print("[postfix] %-8s identical=%s  p50 off %.4f map %.4f verify %.4f" % (
            corpus, rec["all_modes_identical"], rec["off"]["p50_ms"],
            rec["map"]["p50_ms"], rec["verify"]["p50_ms"]), flush=True)
    out["loadavg"] = loadavg()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--latency-duel", action="store_true")
    ap.add_argument("--cache-integrity", action="store_true")
    ap.add_argument("--binding-cost", action="store_true")
    ap.add_argument("--post-fix-check", action="store_true")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--corpus")
    ap.add_argument("--phase")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        return worker(a.arm, a.corpus, a.phase, a.out)
    if a.prep:
        print(json.dumps(prep(), indent=2))
        return 0
    if a.latency_duel:
        latency_duel(a.cycles)
        return 0
    if a.post_fix_check:
        got = post_fix_check()
        doc = {}
        if os.path.exists(RESULTS):
            with open(RESULTS) as fh:
                doc = json.load(fh)
        doc["post_fix_check"] = got
        with open(RESULTS, "w") as fh:
            json.dump(doc, fh, indent=2, default=str)
        print(json.dumps(got["corpora"], indent=2)[:1500])
        return 0
    if a.binding_cost:
        got = binding_cost(a.cycles)
        doc = {}
        if os.path.exists(RESULTS):
            with open(RESULTS) as fh:
                doc = json.load(fh)
        # keep what earlier paired runs measured: the delta here is ~1 us on a
        # ~175 us open, so run-to-run spread is the honest context for it.
        prev = doc.get("binding_cost")
        if prev:
            got["previous_runs"] = (prev.pop("previous_runs", []) +
                                    [{"corpora": prev.get("corpora"),
                                      "loadavg": prev.get("loadavg")}])
        doc["binding_cost"] = got
        with open(RESULTS, "w") as fh:
            json.dump(doc, fh, indent=2, default=str)
        print(json.dumps(got["corpora"], indent=2))
        return 0
    if a.cache_integrity:
        got = cache_integrity(a.cycles)
        doc = {}
        if os.path.exists(RESULTS):
            with open(RESULTS) as fh:
                doc = json.load(fh)
        doc["cache_integrity"] = got
        with open(RESULTS, "w") as fh:
            json.dump(doc, fh, indent=2, default=str)
        print(json.dumps({k: v for k, v in got["corpora"]["n71433"].items()
                          if k in ("cache_tamper", "vault_forgery",
                                   "vault_inplace_edit")}, indent=2)[:2000])
        return 0
    drive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
