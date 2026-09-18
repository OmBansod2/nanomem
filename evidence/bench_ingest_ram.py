#!/usr/bin/env python3
"""nanomem v3 -- the INGEST path's peak RSS, before and after the reserved arena.

The axis this measures is the one nanomem lost on: `competitors_standard_results.json`
reads 826.4 MB of peak ru_maxrss to build-then-serve 71,433 documents, against
230.6 MB for FAISS IndexFlatIP and 290.5 MB for Chroma's default HNSW.

Protocol, deliberately the same as bench_competitors.py so the numbers are
comparable to the ones already published:

  * The SAME shared cache (`/tmp/nanomem_bench_cache`, built by
    `bench_competitors.py --prep`): same vectors, same rng(0) insertion order,
    same 500 queries, same gold.
  * One SUBPROCESS per (arm, corpus, phase) so `ru_maxrss` -- a monotonic
    high-water mark -- is never contaminated by the arm before it.
  * The baseline arm imports the 3.0.3 package as it is COMMITTED at git
    6aa6923, extracted to a temp directory, so "before" is measured on this
    machine today rather than quoted from an older run.
  * Two phases per arm. `build` is the loader shape: ingest, flush, close,
    reopen, query -- exactly what bench_competitors.py charges. `serve` is the
    server shape: a fresh process opens the file the build left behind and
    answers the same queries.
  * Recall and top-10 identity are computed in the DRIVER from each arm's
    returned id matrix. An arm cannot score itself.

Usage:
    bench_ingest_ram.py                  # everything
    bench_ingest_ram.py --worker ...     # internal
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
NEW_DIR = os.path.join(REPO, "nanomem_standalone")
BASE_COMMIT = "6aa6923"
BASE_DIR = os.path.join("/tmp", "nanomem_baseline_%s" % BASE_COMMIT, "nanomem_standalone")
CACHE = "/tmp/nanomem_bench_cache"
WORK = "/tmp/nanomem_ingest_ram"
RESULTS = os.path.join(REFOUND, "ingest_ram_results.json")

DIM = 768
TOP_K = 10
RECALL_KS = (4, 10)
MIN_LATENCY_QUERIES = 200
WARMUP_QUERIES = 20
SEED = 0

CORPORA = ("n10000", "n71433")

# arm -> (package, how the rows are fed, what the arm is told up front)
#
# The baseline is the package AS COMMITTED at git 6aa6923, which reports itself
# as nanomem 0.3.1 / engine 3.0.4. Its reopen path already pre-sizes the arena
# from the block headers (that is what 3.0.4 was); its INGEST path is the one
# that still grew by capacity doubling, and that is what these arms are about.
ARMS = {
    "committed_baseline_loop": (
        "baseline", "add_fact", None,
        "The committed engine, unchanged: add_fact() in a loop, nothing told to "
        "it. This is the arm competitors_standard_results.json calls "
        "nanomem_v3_exact, re-measured here on today's machine."),
    "committed_baseline_hint": (
        "baseline", "add_fact", "reserve_rows",
        "The committed engine plus the escape hatch that already existed and "
        "that nothing in the library called: VaultEngine.reserve_rows(n) before "
        "the loop."),
    "reserved_view_loop": (
        "new", "add_fact", None,
        "The reserved-view arena: add_fact() in a loop, still nothing told to "
        "the engine. This is the fix that needs no caller cooperation."),
    "reserved_view_hint": (
        "new", "add_fact", "reserve_rows",
        "The reserved-view arena plus the exact row count."),
    "vault_add_batch": (
        "new", "add_batch", "wired",
        "Through the public Vault.add_batch, which now hints the batch length "
        "for itself -- the wiring, end to end, with no benchmark-only call."),
}


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
    try:
        out = subprocess.check_output(
            ["footprint", "-p", str(os.getpid())], text=True,
            stderr=subprocess.DEVNULL, timeout=20)
        for line in out.splitlines():
            if "phys_footprint" in line.lower():
                for tok in line.replace("=", " ").split():
                    try:
                        return int(float(tok.replace(",", "")) * 1024)
                    except ValueError:
                        continue
    except Exception:
        return None
    return None


def pct(a: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def loadavg() -> List[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except Exception:
        return []


class Ctx:
    """Identical to bench_competitors.Ctx: same cache, same order, same queries."""

    def __init__(self, corpus: str):
        d = os.path.join(CACHE, corpus)
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.docs = np.load(os.path.join(d, "docs.npy"))
        self.queries = np.load(os.path.join(d, "queries.npy"))
        self.texts = json.load(open(os.path.join(d, "texts.json")))
        self.gold = json.load(open(os.path.join(d, "gold.json")))
        self.n = int(self.docs.shape[0])
        self.order = np.random.default_rng(SEED).permutation(self.n)
        self.Dp = np.ascontiguousarray(self.docs[self.order])
        self.Tp = [self.texts[int(i)] for i in self.order]
        self.ids = [int(i) for i in self.order]
        gc.collect()


def measure_queries(query_fn, Q: np.ndarray, top_k: int) -> dict:
    nq = int(Q.shape[0])
    query_fn(Q[0], top_k)
    for i in range(min(WARMUP_QUERIES, nq)):
        query_fn(Q[i], top_k)
    reps = 1 if nq >= MIN_LATENCY_QUERIES else int(np.ceil(MIN_LATENCY_QUERIES / nq))
    lat: List[float] = []
    out = np.full((nq, top_k), -1, dtype=np.int32)
    for r in range(reps):
        for i in range(nq):
            t = time.perf_counter()
            ids = query_fn(Q[i], top_k)
            lat.append((time.perf_counter() - t) * 1000.0)
            if r == 0:
                ids = [int(x) for x in ids][:top_k]
                out[i, :len(ids)] = ids
    return {"topk": out,
            "latency_ms": {"p50": round(pct(lat, 50), 4), "p95": round(pct(lat, 95), 4),
                           "p99": round(pct(lat, 99), 4),
                           "mean": round(float(np.mean(lat)), 4),
                           "n_timed_queries": len(lat), "passes": reps}}


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def engine_kw() -> dict:
    """The shipped defaults bench_competitors.py used for nanomem_v3_exact."""
    return dict(embed_dim=DIM, vector_dtype="float16", router="off",
                n_exhaustive=50_000)


def run_build(arm: str, ctx: Ctx, vault_path: str) -> dict:
    pkg, feed, hint, _doc = ARMS[arm]
    from nanomem.engine import VaultEngine

    rec = {}
    t0 = time.perf_counter()
    if feed == "add_batch":
        from nanomem.vault import Vault
        v = Vault(vault_path)
        recs = [{"id": "doc_%d" % ctx.ids[j], "text": ctx.Tp[j],
                 "embedding": ctx.Dp[j], "source": "wiki",
                 "metadata": {"idx": ctx.ids[j]}} for j in range(ctx.n)]
        # The record list itself is the caller's, not the engine's; charge it
        # honestly by taking the peak AFTER it exists but report both.
        rec["records_built_peak_mb"] = round(maxrss_bytes() / 2**20, 1)
        v.add_batch(recs, batch_size=64)
        del recs
        e = v.engine
    else:
        e = VaultEngine(vault_path, **engine_kw())
        if hint == "reserve_rows":
            e.reserve_rows(ctx.n)
        for j in range(ctx.n):
            e.add_fact(ctx.Tp[j], ctx.Dp[j], source="wiki",
                       metadata={"idx": ctx.ids[j]})
        e.flush()
    rec["ingest_s"] = round(time.perf_counter() - t0, 3)
    rec["ingest_peak_mb"] = round(maxrss_bytes() / 2**20, 1)
    rec["ingest_rss_mb"] = round((rss_bytes() or 0) / 2**20, 1)
    st_hot = e.stats()
    e.close()
    del e
    if feed == "add_batch":
        del v
    gc.collect()
    rec["after_close_rss_mb"] = round((rss_bytes() or 0) / 2**20, 1)

    t_ro = time.perf_counter()
    e2 = VaultEngine(vault_path, **engine_kw())
    rec["reopen_s"] = round(time.perf_counter() - t_ro, 4)

    def q(v_, k):
        return [int(h["metadata"]["idx"]) for h in e2.search("", v_, top_k=k)]

    res = measure_queries(q, ctx.queries, TOP_K)
    rec["topk"] = res.pop("topk")
    rec.update(res)
    st = e2.stats()
    e2.close()
    rec["index_bytes"] = os.path.getsize(vault_path)
    rec["engine_stats_reopened"] = {k: st[k] for k in sorted(st)
                                    if not isinstance(st[k], (list, dict))}
    rec["engine_stats_after_ingest"] = {
        k: st_hot[k] for k in sorted(st_hot)
        if k.startswith("arena") or k.startswith("resident") or k == "active_heap_ram_kb"}
    return rec


def run_serve(arm: str, ctx: Ctx, vault_path: str) -> dict:
    from nanomem.engine import VaultEngine
    rec = {}
    t_ro = time.perf_counter()
    e = VaultEngine(vault_path, **engine_kw())
    rec["reopen_s"] = round(time.perf_counter() - t_ro, 4)

    def q(v_, k):
        return [int(h["metadata"]["idx"]) for h in e.search("", v_, top_k=k)]

    res = measure_queries(q, ctx.queries, TOP_K)
    rec["topk"] = res.pop("topk")
    rec.update(res)
    rec["server_rss_mb"] = round((rss_bytes() or 0) / 2**20, 1)
    rec["server_peak_mb"] = round(maxrss_bytes() / 2**20, 1)
    fp = phys_footprint_bytes()
    rec["server_phys_footprint_mb"] = round(fp / 2**20, 1) if fp else None
    st = e.stats()
    e.close()
    rec["engine_stats"] = {k: st[k] for k in sorted(st)
                           if not isinstance(st[k], (list, dict))}
    return rec


def run_worker(arm: str, corpus: str, phase: str, out: str) -> int:
    pkg = ARMS[arm][0]
    sys.path.insert(0, NEW_DIR if pkg == "new" else BASE_DIR)
    work = os.path.join(WORK, arm, corpus)
    os.makedirs(work, exist_ok=True)
    vault_path = os.path.join(work, "vault.dat")
    rec = {"arm": arm, "corpus": corpus, "phase": phase, "ok": False,
           "package_dir": NEW_DIR if pkg == "new" else BASE_DIR,
           "load_avg_start": loadavg()}
    try:
        if phase == "build" and os.path.exists(vault_path):
            os.remove(vault_path)
        ctx = Ctx(corpus)
        import nanomem as _nm
        rec["nanomem_version"] = _nm.__version__
        rec["engine_version"] = getattr(_nm, "ENGINE_VERSION", None)
        rec["nanomem_file"] = _nm.__file__
        gc.collect()
        base_peak, base_cur = maxrss_bytes(), rss_bytes()
        t0 = time.perf_counter()
        res = run_build(arm, ctx, vault_path) if phase == "build" else run_serve(arm, ctx, vault_path)
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        end_peak, end_cur = maxrss_bytes(), rss_bytes()
        topk = res.pop("topk")
        np.save(out.replace(".json", ".topk.npy"), topk)
        rec.update(res)
        rec["memory"] = {
            "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss (bytes on darwin) "
                      "for the peak, `ps -o rss=` for the resident value; baseline "
                      "taken after the shared vectors/texts are loaded, so the delta "
                      "is the engine's own footprint. Identical to the accounting in "
                      "bench_competitors.py.",
            "baseline_peak_bytes": base_peak, "peak_bytes": end_peak,
            "peak_delta_mb": round((end_peak - base_peak) / 2**20, 1),
            "baseline_rss_bytes": base_cur, "rss_bytes": end_cur,
            "rss_delta_mb": (round((end_cur - base_cur) / 2**20, 1)
                             if (base_cur is not None and end_cur is not None) else None),
        }
        rec["ok"] = True
    except Exception:
        rec["error"] = traceback.format_exc()
    rec["load_avg_end"] = loadavg()
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    return 0 if rec["ok"] else 1


# --------------------------------------------------------------------------
# allocator microbenchmark: the growth_policy table, with the new policy in it
# --------------------------------------------------------------------------
def run_policy_worker(policy: str, n_rows: int, out: str) -> int:
    sys.path.insert(0, NEW_DIR)
    rec = {"policy": policy, "n_rows": n_rows, "ok": False}
    try:
        import numpy as _np
        from nanomem.arena import _VectorStore, _next_capacity
        D, BLOCK = DIM, 50
        gc.collect()
        base = maxrss_bytes()
        t0 = time.perf_counter()
        freed = 0
        if policy in ("double", "exact_fit"):
            vec = _np.zeros((0, D), dtype=_np.float32)
            n = 0
            while n < n_rows:
                k = min(BLOCK, n_rows - n)
                need = n + k
                if vec.shape[0] < need:
                    cap = (_next_capacity(vec.shape[0], need, D * 4)
                           if policy == "double" else need)
                    out_a = _np.zeros((cap, D), dtype=_np.float32)
                    out_a[:n] = vec[:n]
                    freed += vec.nbytes
                    vec = out_a
                vec[n:need] = 1.0
                n = need
            cap_rows = vec.shape[0]
        else:
            st = _VectorStore(D, _np.float32)
            if policy == "reserve":
                st.reserve(n_rows, 0)
            n = 0
            while n < n_rows:
                k = min(BLOCK, n_rows - n)
                need = n + k
                st.ensure(need, n)
                st.array[n:need] = 1.0
                n = need
            cap_rows = st.array.shape[0]
            rec["copies"] = st.copies
            rec["reservation_mb"] = round(st.reservation_bytes() / 2**20, 1)
            rec["reservation_is_mapped"] = st.mapped_reservation
        rec.update({
            "build_s": round(time.perf_counter() - t0, 3),
            "final_capacity_rows": int(cap_rows),
            "final_capacity_mb": round(cap_rows * D * 4 / 2**20, 1),
            "live_rows_mb": round(n_rows * D * 4 / 2**20, 1),
            "bytes_freed_mb": round(freed / 2**20, 1),
            "peak_rss_delta_mb": round((maxrss_bytes() - base) / 2**20, 1),
            "ok": True,
        })
    except Exception:
        rec["error"] = traceback.format_exc()
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    return 0 if rec["ok"] else 1


# --------------------------------------------------------------------------
# the directory-ingest estimate, measured on real trees
# --------------------------------------------------------------------------
def directory_estimate() -> dict:
    sys.path.insert(0, NEW_DIR)
    from nanomem.vault import Vault
    exts = {".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".go", ".cpp", ".c", ".h",
            ".hpp", ".java", ".cs", ".rb", ".php", ".swift", ".kt", ".sh", ".bash",
            ".sql", ".md", ".json", ".yaml", ".yml", ".toml", ".html", ".css",
            ".txt", ".csv", ".tsv", ".log", ".rst", ".transcript"}
    ignored = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
               ".idea", ".vscode", "dist", "build", "target", ".next", ".nuxt",
               "coverage", ".pytest_cache"}
    rows = []
    for root in (NEW_DIR, os.path.join(REPO, "pure_latent_4d_mind"),
                 os.path.join(REFOUND, "design"), REPO):
        tb = tc = nf = 0
        for r, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ignored and not d.startswith(".")]
            for f in files:
                if os.path.splitext(f)[1].lower() not in exts or f.startswith("."):
                    continue
                fp = os.path.join(r, f)
                try:
                    b = os.path.getsize(fp)
                    with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                        lines = fh.readlines()
                except Exception:
                    continue
                if not lines:
                    continue
                c = 0
                for s in range(0, len(lines), 40):
                    sl = lines[s:s + 50]
                    if not sl:
                        break
                    if "".join(sl).strip():
                        c += 1
                tb += b
                tc += c
                nf += 1
        if tc:
            rows.append({"tree": root, "files": nf, "bytes": tb, "chunks": tc,
                         "bytes_per_chunk": round(tb / tc, 1),
                         "estimated_chunks": tb // Vault.BYTES_PER_CHUNK_ESTIMATE,
                         "estimate_over_actual": round(
                             (tb // Vault.BYTES_PER_CHUNK_ESTIMATE) / tc, 3)})
    return {"constant_bytes_per_chunk": Vault.BYTES_PER_CHUNK_ESTIMATE,
            "split": "the shipped ingest_file default: 50-line chunks, 10-line overlap",
            "what": "the hint ingest_directory uses before it has read anything: "
                    "total bytes / BYTES_PER_CHUNK_ESTIMATE. Rows are measured by "
                    "running the real split over real trees.",
            "rows": rows}


# --------------------------------------------------------------------------
# headline: the one table, derived from the rows above, never typed
# --------------------------------------------------------------------------
def headline() -> None:
    doc = json.load(open(RESULTS))
    ref = doc["reference_points"]
    out = {"what": "derived from `summary` and `replicate` in this same file and "
                   "from the peak_rss_delta_mb of competitors_standard_results.json. "
                   "Every nanomem figure is the pair of runs, not a single reading.",
           "metric": "peak ru_maxrss delta over build + reopen + 500 queries, MB",
           "corpora": {}}
    for corpus in CORPORA:
        refkey = "%s_peak_rss_delta_mb" % corpus
        rows = {}
        for arm in ARMS:
            rep = doc.get("replicate", {}).get("corpora", {}).get(corpus, {}).get(arm, {})
            runs = rep.get("runs_mb") or [doc["summary"][corpus][arm]["loader_peak_rss_mb"]]
            rows[arm] = {"runs_mb": runs,
                         "median_mb": round(float(np.median(runs)), 1),
                         "range_mb": [min(runs), max(runs)]}
        base = rows["committed_baseline_loop"]["median_mb"]
        for arm in rows:
            rows[arm]["vs_committed_baseline"] = round(rows[arm]["median_mb"] / base, 3)
        comp = {k: v for k, v in ref.get(refkey, {}).items()
                if not k.startswith("nanomem")}
        best = rows["reserved_view_loop"]["median_mb"]
        out["corpora"][corpus] = {
            "nanomem": rows,
            "competitors_from_competitors_standard_results": comp,
            "shipped_default_now": best,
            "nanomem_as_published": ref.get(refkey, {}).get("nanomem_v3_exact"),
            "vs_competitors": {k: round(best / v, 3) for k, v in comp.items()},
            "arms_now_beaten": sorted(k for k, v in comp.items() if best < v),
            "arms_still_ahead": sorted(k for k, v in comp.items() if best >= v),
        }
    doc["headline"] = out
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)
    for c in out["corpora"]:
        h = out["corpora"][c]
        print("%s: shipped default %.1f MB (%.3fx the committed baseline); beats %s; "
              "still behind %s" % (c, h["shipped_default_now"],
                                   h["nanomem"]["reserved_view_loop"]["vs_committed_baseline"],
                                   ", ".join(h["arms_now_beaten"]) or "nothing",
                                   ", ".join(h["arms_still_ahead"]) or "nothing"))


# --------------------------------------------------------------------------
# replicate: the same build, measured a second time
# --------------------------------------------------------------------------
def replicate(extra: int = 3) -> None:
    """Re-run every build phase ``extra`` more times so the SPREAD is on record.

    ru_maxrss is a high-water mark of a process that is doing other things too
    (reading a 219 MB npy, holding the corpus text) on a machine that is not
    quiet. One reading quoted to four digits would be over-claiming. This
    records every reading and the median and range across them, and the report
    quotes those.
    """
    tmp = os.path.join(WORK, "_json")
    doc = json.load(open(RESULTS))
    out_doc = {"what": "`runs_mb` is the loader peak of run 1 (the `summary` above) "
                       "followed by %d independent re-runs of the same build. "
                       "Quote `median_mb` and `range_mb`, not a single reading." % extra,
               "extra_runs": extra, "corpora": {}}
    for corpus in CORPORA:
        rows = {}
        for arm in ARMS:
            runs = [doc["summary"][corpus][arm]["loader_peak_rss_mb"]]
            lat, loads = [doc["summary"][corpus][arm]["p50_ms_build_phase"]], []
            for k in range(extra):
                o = os.path.join(tmp, "rep%d.%s.%s.build.json" % (k, arm, corpus))
                print("[replicate %d] %-24s %s" % (k + 1, arm, corpus), flush=True)
                spawn(["--worker", "--arm", arm, "--corpus", corpus,
                       "--phase", "build", "--out", o])
                r = json.load(open(o))
                if not r.get("ok"):
                    continue
                runs.append(r["memory"]["peak_delta_mb"])
                lat.append(r["latency_ms"]["p50"])
                loads.append(r["load_avg_start"])
            rows[arm] = {
                "runs_mb": runs,
                "median_mb": round(float(np.median(runs)), 1),
                "range_mb": [min(runs), max(runs)],
                "spread_pct": round(100.0 * (max(runs) - min(runs)) / max(min(runs), 1e-9), 2),
                "p50_ms_per_run": lat,
                "load_avg_per_rerun": loads,
            }
        out_doc["corpora"][corpus] = rows
        print("-- %s --" % corpus)
        for arm, v in rows.items():
            print("  %-24s median %7.1f MB  runs %s  spread %.1f%%"
                  % (arm, v["median_mb"], v["runs_mb"], v["spread_pct"]))
    doc["replicate"] = out_doc
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)


# --------------------------------------------------------------------------
# the alternative that was NOT taken: one arena per fixed-size chunk
# --------------------------------------------------------------------------
def run_chunked_worker(corpus: str, out: str) -> int:
    """Price the other way to make growth free: store the arena as CHUNKS.

    Fixed-size chunks never copy either -- a new chunk is appended and nothing
    moves -- and they need no address-space reservation at all. What they cost
    is the scan: one ``(n, D) @ (D,)`` BLAS call becomes ceil(n/chunk) of them.
    This measures exactly that, on the real corpus, paired in cycles against the
    contiguous scan the engine actually does, and checks that the score vectors
    are bit-identical either way (they are: chunking changes which rows share a
    call, not the arithmetic in one).
    """
    rec = {"corpus": corpus, "ok": False}
    try:
        d = os.path.join(CACHE, corpus)
        D = np.load(os.path.join(d, "docs.npy")).astype(np.float16).astype(np.float32)
        Q = np.load(os.path.join(d, "queries.npy"))
        n = D.shape[0]
        sizes = [4096, 8192, 16384, 32768]
        chunked = {c: [np.ascontiguousarray(D[i:i + c]) for i in range(0, n, c)]
                   for c in sizes}
        out_buf = np.empty(n, dtype=np.float32)

        def contiguous(q):
            return D @ q

        def do_chunks(q, c):
            s0 = 0
            for blk in chunked[c]:
                e0 = s0 + blk.shape[0]
                np.dot(blk, q, out=out_buf[s0:e0])
                s0 = e0
            return out_buf

        def timed(fn):
            for i in range(20):
                fn(Q[i % Q.shape[0]])
            lat = []
            for i in range(200):
                q = Q[i % Q.shape[0]]
                t = time.perf_counter()
                fn(q)
                lat.append((time.perf_counter() - t) * 1000.0)
            return round(pct(lat, 50), 4)

        cycles = {"contiguous": [], **{str(c): [] for c in sizes}}
        for _ in range(3):
            cycles["contiguous"].append(timed(contiguous))
            for c in sizes:
                cycles[str(c)].append(timed(lambda q, c=c: do_chunks(q, c)))
        med = {k: round(float(np.median(v)), 4) for k, v in cycles.items()}
        ref = contiguous(Q[7])
        identical = {str(c): bool(np.array_equal(ref, do_chunks(Q[7], c).copy()))
                     for c in sizes}
        rec.update({
            "n_rows": int(n), "dim": int(D.shape[1]),
            "p50_ms_median_of_3_cycles": med,
            "p50_ms_per_cycle": cycles,
            "vs_contiguous": {k: round(v / med["contiguous"], 3)
                              for k, v in med.items() if k != "contiguous"},
            "scores_bitwise_identical_to_contiguous": identical,
            "chunk_count": {str(c): len(chunked[c]) for c in sizes},
            "load_avg": loadavg(), "ok": True})
    except Exception:
        rec["error"] = traceback.format_exc()
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    return 0 if rec["ok"] else 1


def chunked_probe() -> None:
    tmp = os.path.join(WORK, "_json")
    os.makedirs(tmp, exist_ok=True)
    doc = json.load(open(RESULTS))
    out_doc = {"what": "the design alternative the reserved view was chosen over: "
                       "holding the arena as fixed-size chunks. Same RAM story "
                       "(growth never copies), no reservation needed, and the scan "
                       "pays for it. Measured on the same corpora, paired in cycles.",
               "corpora": {}}
    for corpus in CORPORA:
        o = os.path.join(tmp, "chunked.%s.json" % corpus)
        print("[chunked] %s" % corpus, flush=True)
        spawn(["--chunked-worker", "--corpus", corpus, "--out", o])
        r = json.load(open(o))
        out_doc["corpora"][corpus] = r
        if r.get("ok"):
            print("   contiguous %.4f ms; chunked %s"
                  % (r["p50_ms_median_of_3_cycles"]["contiguous"],
                     {k: v for k, v in r["vs_contiguous"].items()}))
    doc["chunked_alternative"] = out_doc
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)


# --------------------------------------------------------------------------
# bitwise exactness: the scan arithmetic, with the machine taken out
# --------------------------------------------------------------------------
def run_exact_worker(arm: str, corpus: str, out: str) -> int:
    """SHA-256 over the FULL score vector of every query, not just the top-10.

    A latency comparison on a loaded box is noise; this is not. The arms hold
    the same vectors in memory that came from the same fp16 bytes, and the scan
    kernel is untouched, so every one of the 500 (n_rows,) fp32 score vectors
    must be bit-for-bit identical. A digest catches a change in any row, not
    just in the ten that happen to win.
    """
    import hashlib
    pkg = ARMS[arm][0]
    sys.path.insert(0, NEW_DIR if pkg == "new" else BASE_DIR)
    rec = {"arm": arm, "corpus": corpus, "ok": False}
    try:
        from nanomem.engine import VaultEngine
        d = os.path.join(CACHE, corpus)
        Q = np.load(os.path.join(d, "queries.npy"))
        vault_path = os.path.join(WORK, arm, corpus, "vault.dat")
        e = VaultEngine(vault_path, **engine_kw())
        h = hashlib.sha256()
        for i in range(Q.shape[0]):
            v = np.ascontiguousarray(Q[i], dtype=np.float32)
            sc = np.ascontiguousarray(e.arena.scores(v), dtype=np.float32)
            h.update(sc.tobytes())
        rec.update({"n_rows": int(e.arena.n_rows), "n_queries": int(Q.shape[0]),
                    "scores_sha256": h.hexdigest(), "ok": True})
        e.close()
    except Exception:
        rec["error"] = traceback.format_exc()
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    return 0 if rec["ok"] else 1


def exactness() -> None:
    tmp = os.path.join(WORK, "_json")
    arms = ["committed_baseline_loop", "committed_baseline_hint", "reserved_view_loop",
            "reserved_view_hint", "vault_add_batch"]
    doc = json.load(open(RESULTS))
    out_doc = {"what": "SHA-256 over the concatenated fp32 score vectors of every "
                       "query against every row -- the whole scan, not the top-10. "
                       "Equal digests mean the reserved arena changed no number "
                       "anywhere, which a latency measurement on a loaded machine "
                       "cannot show.",
               "corpora": {}}
    for corpus in CORPORA:
        rows = {}
        for a in arms:
            o = os.path.join(tmp, "exact.%s.%s.json" % (a, corpus))
            spawn(["--exact-worker", "--arm", a, "--corpus", corpus, "--out", o])
            r = json.load(open(o))
            rows[a] = r.get("scores_sha256") or r.get("error")
        ref = rows["committed_baseline_loop"]
        out_doc["corpora"][corpus] = {
            "scores_sha256": rows,
            "all_identical_to_committed_baseline": all(v == ref for v in rows.values()),
        }
        print("[exact] %s all_identical=%s" % (
            corpus, out_doc["corpora"][corpus]["all_identical_to_committed_baseline"]))
    doc["exactness"] = out_doc
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)


# --------------------------------------------------------------------------
# paired latency duel
# --------------------------------------------------------------------------
def run_floor_worker(corpus: str, out: str) -> int:
    """The bare numpy matvec the machine is offering RIGHT NOW.

    A ratio measured under load is not load-independent (nanomem/arena.py says so
    in as many words), and this box is not quiet: the same 500 queries are run
    against a plain fp32 matmul in every cycle so each arm's p50 can be read as a
    multiple of the floor at that moment rather than as an absolute.
    """
    rec = {"corpus": corpus, "ok": False}
    try:
        d = os.path.join(CACHE, corpus)
        D = np.load(os.path.join(d, "docs.npy"))
        Q = np.load(os.path.join(d, "queries.npy"))
        D16 = D.astype(np.float16).astype(np.float32)   # what the vault holds
        lat = []
        for i in range(min(WARMUP_QUERIES, Q.shape[0])):
            D16 @ Q[i]
        for i in range(Q.shape[0]):
            t = time.perf_counter()
            sc = D16 @ Q[i]
            np.argpartition(-sc, TOP_K)[:TOP_K]
            lat.append((time.perf_counter() - t) * 1000.0)
        rec.update({"p50": round(pct(lat, 50), 4), "p95": round(pct(lat, 95), 4),
                    "n": len(lat), "load_avg": loadavg(), "ok": True})
    except Exception:
        rec["error"] = traceback.format_exc()
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    return 0 if rec["ok"] else 1


def duel(cycles: int = 5) -> None:
    """Re-time the serve phase of each arm, CYCLED, against a per-cycle floor."""
    tmp = os.path.join(WORK, "_json")
    os.makedirs(tmp, exist_ok=True)
    arms = ["committed_baseline_loop", "reserved_view_loop", "reserved_view_hint"]
    out_doc = {"what": "the same vaults the build phase left on disk, re-opened and "
                       "re-timed CYCLE BY CYCLE, arms always in the same order, with a "
                       "bare numpy matvec measured in every cycle. The machine was NOT "
                       "quiet (see load_avg): read the ratio column, not the "
                       "milliseconds.",
               "cycles": cycles, "corpora": {}}
    for corpus in CORPORA:
        per_arm = {a: [] for a in arms}
        floors, loads = [], []
        for c in range(cycles):
            o = os.path.join(tmp, "floor.%s.%d.json" % (corpus, c))
            spawn(["--floor-worker", "--corpus", corpus, "--out", o])
            fl = json.load(open(o))
            floors.append(fl["p50"])
            loads.append(fl["load_avg"])
            for a in arms:
                o = os.path.join(tmp, "duel.%s.%s.%d.json" % (a, corpus, c))
                spawn(["--worker", "--arm", a, "--corpus", corpus,
                       "--phase", "serve", "--out", o])
                r = json.load(open(o))
                per_arm[a].append(r["latency_ms"]["p50"] if r.get("ok") else None)
            print("[duel] %s cycle %d floor=%.4f %s" % (
                corpus, c, fl["p50"],
                " ".join("%s=%.4f" % (a, per_arm[a][-1]) for a in arms)), flush=True)
        med = {a: round(float(np.median([x for x in per_arm[a] if x])), 4) for a in arms}
        fmed = round(float(np.median(floors)), 4)
        out_doc["corpora"][corpus] = {
            "per_cycle_p50_ms": per_arm,
            "per_cycle_numpy_floor_ms": floors,
            "load_avg_per_cycle": loads,
            "median_p50_ms": med,
            "median_numpy_floor_ms": fmed,
            "p50_as_multiple_of_floor": {a: round(med[a] / fmed, 3) for a in arms},
            "paired_ratio_new_over_baseline": round(
                float(np.median([n / b for n, b in zip(
                    per_arm["reserved_view_loop"], per_arm["committed_baseline_loop"])
                    if n and b])), 4),
        }
    doc = json.load(open(RESULTS))
    doc["latency_duel"] = out_doc
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)
    print("\nlatency_duel written into %s" % RESULTS)
    for corpus in CORPORA:
        r = out_doc["corpora"][corpus]
        print("  %s median p50: %s  floor=%s  paired new/baseline=%s"
              % (corpus, r["median_p50_ms"], r["median_numpy_floor_ms"],
                 r["paired_ratio_new_over_baseline"]))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _code_sha(path: str) -> str:
    """SHA-256 of a module's EXECUTABLE code, with docstrings removed.

    A docstring is not behaviour, and this file is full of measured numbers that
    end up quoted in docstrings; hashing the raw bytes would make the
    fingerprint change every time a number is written down, which is exactly
    backwards. Stripping them means the hash changes when, and only when, what
    ran changes.
    """
    import ast
    import hashlib
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.unparse(ast.fix_missing_locations(tree))
                          .encode("utf-8")).hexdigest()[:16]


def package_fingerprint() -> dict:
    """What actually ran, hashed, so these numbers can be tied to it."""
    out = {"what": "sha256[:16] of each module's executable code with docstrings "
                   "stripped -- see _code_sha. Writing a measured number into a "
                   "docstring does not move these; changing a line of code does."}
    for label, d in (("new", NEW_DIR), ("baseline_%s" % BASE_COMMIT, BASE_DIR)):
        pkg = os.path.join(d, "nanomem")
        out[label] = {"dir": pkg,
                      "code_sha256_16": {f: _code_sha(os.path.join(pkg, f))
                                         for f in sorted(os.listdir(pkg))
                                         if f.endswith(".py")}}
    out["harness"] = _code_sha(os.path.abspath(__file__))
    return out


def verify() -> None:
    """Re-fingerprint and re-measure two arms, checking they land in range.

    Run after the report is written: it proves the code that is on disk now
    still measures what this file says, and records the attempt either way.
    """
    tmp = os.path.join(WORK, "_json")
    doc = json.load(open(RESULTS))
    doc["package_fingerprint"] = package_fingerprint()
    checks = []
    for arm in ("committed_baseline_loop", "reserved_view_loop"):
        o = os.path.join(tmp, "verify.%s.n71433.build.json" % arm)
        print("[verify] %s" % arm, flush=True)
        spawn(["--worker", "--arm", arm, "--corpus", "n71433",
               "--phase", "build", "--out", o])
        r = json.load(open(o))
        got = r["memory"]["peak_delta_mb"] if r.get("ok") else None
        rng = doc["replicate"]["corpora"]["n71433"][arm]["range_mb"]
        checks.append({"arm": arm, "loader_peak_rss_mb": got,
                       "recorded_range_mb": rng,
                       "inside_recorded_range": bool(got and rng[0] <= got <= rng[1]),
                       "recall_at_4_pct": r.get("evidence_recall_at_4_pct"),
                       "load_avg": r.get("load_avg_start")})
        print("   %s -> %s MB, recorded range %s" % (arm, got, rng))
    doc["post_report_verification"] = {
        "what": "one more build of each end of the comparison, run AFTER the "
                "report and the docstrings were written, against the ranges "
                "recorded in `replicate`.",
        "checks": checks}
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)


def reference_points() -> dict:
    """The competitors' numbers, READ from their own results file, never retyped."""
    src = os.path.join(REFOUND, "competitors_standard_results.json")
    out = {"source": src,
           "read_from": "winners.<corpus>.<metric>.all -- every arm that file "
                        "reports, not a selection"}
    try:
        w = json.load(open(src))["winners"]
    except Exception as exc:
        out["error"] = repr(exc)
        return out
    for corpus in CORPORA:
        if corpus not in w:
            continue
        for metric in ("peak_rss_delta_mb", "p50_query_ms",
                       "evidence_recall_at_4_pct", "index_bytes"):
            if metric in w[corpus]:
                out["%s_%s" % (corpus, metric)] = dict(w[corpus][metric]["all"])
    return out


def recall_block(topk: np.ndarray, gold) -> dict:
    nq = len(gold)
    out = {}
    for k in RECALL_KS:
        allg, perdoc = [], []
        for i in range(nq):
            got = set(int(x) for x in topk[i, :k] if x >= 0)
            g = set(gold[i])
            allg.append(1.0 if g <= got else 0.0)
            perdoc.append(len(g & got) / max(1, len(g)))
        out["evidence_recall_at_%d_pct" % k] = round(100 * float(np.mean(allg)), 2)
        out["per_doc_recall_at_%d_pct" % k] = round(100 * float(np.mean(perdoc)), 2)
    out["n_questions"] = nq
    return out


def spawn(args_list: List[str]) -> None:
    subprocess.run([sys.executable, os.path.abspath(__file__)] + args_list, check=False)


# --------------------------------------------------------------------------
# SAME-HARNESS COMPETITORS
#
# Added 2026-09-16 after a verifier found the claim "nanomem is now the LOWEST
# of every arm measured at 10,000 documents" to be CROSS-HARNESS: nanomem's
# 42.7 MB came from this file, while FAISS's 50.3 and sqlite-vec's 53.5 were
# read out of competitors_standard_results.json, produced by a different
# harness that reads about 4.3% higher (this file's own baseline arm: 103.7 MB
# against that file's published 108.4 for the same configuration).
#
# These arms re-measure FAISS and sqlite-vec under EXACTLY this harness: same
# Ctx (same cache, same rng(0) insertion order, same queries), same
# ru_maxrss-delta-over-build+reopen+500-queries metric, one subprocess per
# (arm, corpus). They need faiss / sqlite_vec, which live only in venv_bench,
# so the worker is spawned with that interpreter -- never imported here and
# never anywhere near nanomem/.
# --------------------------------------------------------------------------
VENV_PY = os.path.join(REPO, "venv_bench", "bin", "python")

COMPETITOR_ARMS = ("faiss_flat_ip", "sqlitevec_bruteforce")


def run_competitor_worker(arm: str, corpus: str, out: str) -> int:
    """Build + reopen + query one competitor, charged exactly like a nanomem arm."""
    work = os.path.join(WORK, "competitor_" + arm, corpus)
    if os.path.exists(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    rec = {"arm": arm, "corpus": corpus, "phase": "build", "ok": False,
           "load_avg_start": loadavg(), "harness": "bench_ingest_ram.py"}
    try:
        ctx = Ctx(corpus)
        gc.collect()
        base_peak, base_cur = maxrss_bytes(), rss_bytes()
        t0 = time.perf_counter()
        if arm == "faiss_flat_ip":
            import faiss
            rec["lib_version"] = faiss.__version__
            path = os.path.join(work, "flat.faissindex")
            tb = time.perf_counter()
            index = faiss.IndexFlatIP(DIM)
            index.add(ctx.Dp)
            faiss.write_index(index, path)
            rec["ingest_s"] = round(time.perf_counter() - tb, 3)
            rec["ingest_peak_mb"] = round(maxrss_bytes() / 2**20, 1)
            del index
            gc.collect()
            t_ro = time.perf_counter()
            idx = faiss.read_index(path)
            rec["reopen_s"] = round(time.perf_counter() - t_ro, 4)
            order = ctx.order

            def q(v_, k):
                _d, i = idx.search(v_.reshape(1, -1), k)
                return [int(order[int(r)]) for r in i[0] if r >= 0]

            res = measure_queries(q, ctx.queries, TOP_K)
            rec["index_bytes"] = os.path.getsize(path)
            rec["index_stores_text"] = False
        elif arm == "sqlitevec_bruteforce":
            import sqlite3
            import sqlite_vec
            path = os.path.join(work, "vec.db")

            def connect():
                db = sqlite3.connect(path)
                db.enable_load_extension(True)
                sqlite_vec.load(db)
                db.enable_load_extension(False)
                return db

            db = connect()
            rec["lib_version"] = db.execute("select vec_version()").fetchone()[0]
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA cache_size=-262144")
            db.execute("CREATE VIRTUAL TABLE v USING vec0(idx integer primary key, "
                       "embedding float[%d] distance_metric=cosine)" % DIM)
            db.execute("CREATE TABLE docs(idx integer primary key, text text)")
            tb = time.perf_counter()
            db.execute("BEGIN")
            db.executemany("INSERT INTO v(idx, embedding) VALUES (?, ?)",
                           ((ctx.ids[j], ctx.Dp[j].tobytes()) for j in range(ctx.n)))
            db.executemany("INSERT INTO docs(idx, text) VALUES (?, ?)",
                           ((ctx.ids[j], ctx.Tp[j]) for j in range(ctx.n)))
            db.execute("COMMIT")
            rec["ingest_s"] = round(time.perf_counter() - tb, 3)
            rec["ingest_peak_mb"] = round(maxrss_bytes() / 2**20, 1)
            db.close()
            del db
            gc.collect()
            t_ro = time.perf_counter()
            db2 = connect()
            db2.execute("PRAGMA cache_size=-262144")
            rec["reopen_s"] = round(time.perf_counter() - t_ro, 4)

            def q(v_, k):
                rows = db2.execute(
                    "SELECT idx FROM v WHERE embedding MATCH ? AND k = ? "
                    "ORDER BY distance", (v_.tobytes(), int(k))).fetchall()
                return [int(r[0]) for r in rows]

            res = measure_queries(q, ctx.queries, TOP_K)
            rec["index_bytes"] = sum(
                os.path.getsize(os.path.join(work, f))
                for f in os.listdir(work) if f.startswith("vec.db"))
            rec["index_stores_text"] = True
        else:
            raise SystemExit("unknown competitor arm %r" % arm)
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        topk = res.pop("topk")
        np.save(out.replace(".json", ".topk.npy"), topk)
        rec.update(res)
        end_peak, end_cur = maxrss_bytes(), rss_bytes()
        rec["memory"] = {
            "method": "identical to the nanomem arms in this file: ru_maxrss "
                      "delta over build + reopen + queries, baseline taken "
                      "AFTER the shared vectors/texts are resident",
            "baseline_peak_mb": round(base_peak / 2**20, 1),
            "baseline_rss_mb": round((base_cur or 0) / 2**20, 1),
            "end_peak_mb": round(end_peak / 2**20, 1),
            "peak_rss_delta_mb": round((end_peak - base_peak) / 2**20, 1),
            "rss_delta_mb": round(((end_cur or 0) - (base_cur or 0)) / 2**20, 1),
        }
        rec["ok"] = True
    except Exception:
        rec["error"] = traceback.format_exc()
    rec["load_avg_end"] = loadavg()
    json.dump(rec, open(out, "w"), indent=1, default=float)
    shutil.rmtree(work, ignore_errors=True)
    return 0 if rec["ok"] else 1


def competitors_same_harness(repeats: int = 2) -> dict:
    """Run the competitor arms under THIS harness and report them next to the
    cross-harness figures they are replacing."""
    if not os.path.exists(VENV_PY):
        raise SystemExit("venv_bench interpreter not found at %s" % VENV_PY)
    tmpd = os.path.join(WORK, "_competitor_out")
    os.makedirs(tmpd, exist_ok=True)
    runs = {}
    for corpus in CORPORA:
        for arm in COMPETITOR_ARMS:
            vals = []
            for r in range(repeats):
                out = os.path.join(tmpd, "%s_%s_%d.json" % (arm, corpus, r))
                cp = subprocess.run(
                    [VENV_PY, os.path.abspath(__file__), "--competitor-worker",
                     "--arm", arm, "--corpus", corpus, "--out", out],
                    capture_output=True, text=True)
                if not os.path.exists(out):
                    raise SystemExit("competitor worker produced nothing: %s\n%s"
                                     % (cp.stderr[-2000:], cp.stdout[-800:]))
                rec = json.load(open(out))
                if not rec.get("ok"):
                    raise SystemExit("competitor arm failed:\n" + rec.get("error", "?"))
                vals.append(rec)
            peaks = sorted(v["memory"]["peak_rss_delta_mb"] for v in vals)
            runs["%s/%s" % (arm, corpus)] = dict(
                arm=arm, corpus=corpus, repeats=repeats,
                lib_version=vals[0].get("lib_version"),
                peak_rss_delta_mb_runs=peaks,
                peak_rss_delta_mb_median=round(
                    float(np.median(peaks)), 1),
                rss_delta_mb=[v["memory"]["rss_delta_mb"] for v in vals],
                p50_ms=[v["latency_ms"]["p50"] for v in vals],
                ingest_s=[v["ingest_s"] for v in vals],
                index_bytes=vals[0].get("index_bytes"),
                index_stores_text=vals[0].get("index_stores_text"))
            print("   %-22s %-8s peak_rss_delta %s MB" % (
                arm, corpus, peaks), flush=True)
    doc = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    published = (doc.get("headline", {}) or {}).get(
        "competitors_from_competitors_standard_results", {})
    block = dict(
        why=("A verifier found the 10,000-document claim 'nanomem is now the "
             "LOWEST of every arm measured' to be a cross-harness comparison: "
             "nanomem's figure came from this file, the competitors' from "
             "competitors_standard_results.json, and this harness reads about "
             "4.3% lower on the identical configuration. These arms remove the "
             "comparison from the argument by re-running the competitors HERE."),
        metric=("ru_maxrss delta over build + reopen + 500 queries, baseline "
                "after the shared vectors are resident -- the same metric, the "
                "same Ctx and the same insertion order as every nanomem arm in "
                "this file"),
        runs=runs,
        cross_harness_figures_being_replaced=published,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        loadavg=loadavg())
    doc["competitors_same_harness"] = block
    json.dump(doc, open(RESULTS, "w"), indent=1, default=float)
    print("wrote competitors_same_harness -> %s" % os.path.basename(RESULTS))
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--policy-worker", action="store_true")
    ap.add_argument("--floor-worker", action="store_true")
    ap.add_argument("--exact-worker", action="store_true")
    ap.add_argument("--exactness", action="store_true")
    ap.add_argument("--replicate", action="store_true")
    ap.add_argument("--headline", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--chunked-worker", action="store_true")
    ap.add_argument("--chunked", action="store_true")
    ap.add_argument("--competitor-worker", action="store_true")
    ap.add_argument("--competitors", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--duel", action="store_true")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--arm")
    ap.add_argument("--corpus")
    ap.add_argument("--phase")
    ap.add_argument("--policy")
    ap.add_argument("--rows", type=int, default=71433)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.competitor_worker:
        sys.exit(run_competitor_worker(a.arm, a.corpus, a.out))
    if a.competitors:
        competitors_same_harness(a.repeats)
        return
    if a.worker:
        sys.exit(run_worker(a.arm, a.corpus, a.phase, a.out))
    if a.policy_worker:
        sys.exit(run_policy_worker(a.policy, a.rows, a.out))
    if a.floor_worker:
        sys.exit(run_floor_worker(a.corpus, a.out))
    if a.exact_worker:
        sys.exit(run_exact_worker(a.arm, a.corpus, a.out))
    if a.exactness:
        exactness()
        return
    if a.replicate:
        replicate(a.cycles if a.cycles != 5 else 3)
        return
    if a.headline:
        headline()
        return
    if a.verify:
        verify()
        return
    if a.chunked_worker:
        sys.exit(run_chunked_worker(a.corpus, a.out))
    if a.chunked:
        chunked_probe()
        return
    if a.duel:
        duel(a.cycles)
        return

    if not os.path.isdir(BASE_DIR):
        raise SystemExit("baseline package missing: %s\n"
                         "git archive %s nanomem_standalone/nanomem | tar -x -C %s"
                         % (BASE_DIR, BASE_COMMIT, os.path.dirname(BASE_DIR)))
    os.makedirs(WORK, exist_ok=True)
    tmp = os.path.join(WORK, "_json")
    os.makedirs(tmp, exist_ok=True)

    arms_out, failures = {}, []
    for corpus in CORPORA:
        arms_out[corpus] = {}
        for arm in ARMS:
            arms_out[corpus][arm] = {}
            for phase in ("build", "serve"):
                out = os.path.join(tmp, "%s.%s.%s.json" % (arm, corpus, phase))
                print("[run] %-20s %-8s %s" % (arm, corpus, phase), flush=True)
                spawn(["--worker", "--arm", arm, "--corpus", corpus,
                       "--phase", phase, "--out", out])
                try:
                    rec = json.load(open(out))
                except Exception as exc:
                    failures.append({"arm": arm, "corpus": corpus, "phase": phase,
                                     "error": repr(exc)})
                    continue
                if not rec.get("ok"):
                    failures.append({"arm": arm, "corpus": corpus, "phase": phase,
                                     "error": rec.get("error")})
                    continue
                tk = np.load(out.replace(".json", ".topk.npy"))
                rec["topk_file"] = out.replace(".json", ".topk.npy")
                arms_out[corpus][arm][phase] = rec
                arms_out[corpus][arm][phase]["_topk"] = tk

    # recall + exactness, computed here, never by an arm
    corpora_meta = {}
    for corpus in CORPORA:
        gold = json.load(open(os.path.join(CACHE, corpus, "gold.json")))
        corpora_meta[corpus] = json.load(open(os.path.join(CACHE, corpus, "meta.json")))
        ref = arms_out[corpus].get("committed_baseline_loop", {}).get("build", {}).get("_topk")
        for arm in arms_out[corpus]:
            for phase in list(arms_out[corpus][arm]):
                rec = arms_out[corpus][arm][phase]
                tk = rec.pop("_topk")
                rec.update(recall_block(tk, gold))
                if ref is not None and tk.shape == ref.shape:
                    rec["top10_rows_changed_vs_committed_baseline"] = int(
                        (tk != ref).any(axis=1).sum())
                    rec["top4_rows_changed_vs_committed_baseline"] = int(
                        (tk[:, :4] != ref[:, :4]).any(axis=1).sum())
                    rec["top10_identical_vs_committed_baseline"] = bool((tk == ref).all())

    # allocator policies, one subprocess each
    policies = {}
    for policy in ("double", "exact_fit", "reserved_view", "reserve"):
        out = os.path.join(tmp, "policy.%s.json" % policy)
        print("[run] growth policy %s" % policy, flush=True)
        spawn(["--policy-worker", "--policy", policy, "--rows", "71433", "--out", out])
        try:
            policies[policy] = json.load(open(out))
        except Exception as exc:
            failures.append({"policy": policy, "error": repr(exc)})

    summary = {}
    for corpus in CORPORA:
        summary[corpus] = {}
        for arm in ARMS:
            b = arms_out[corpus].get(arm, {}).get("build")
            s = arms_out[corpus].get(arm, {}).get("serve")
            if not b:
                continue
            summary[corpus][arm] = {
                "loader_peak_rss_mb": b["memory"]["peak_delta_mb"],
                "ingest_only_peak_mb": round(
                    b["ingest_peak_mb"] - b["memory"]["baseline_peak_bytes"] / 2**20, 1),
                "server_rss_mb": s["memory"]["rss_delta_mb"] if s else None,
                "server_peak_rss_mb": s["memory"]["peak_delta_mb"] if s else None,
                "ingest_s": b["ingest_s"],
                "reopen_s": b["reopen_s"],
                "p50_ms_build_phase": b["latency_ms"]["p50"],
                "p50_ms_server_phase": s["latency_ms"]["p50"] if s else None,
                "p95_ms_server_phase": s["latency_ms"]["p95"] if s else None,
                "evidence_recall_at_4_pct": b["evidence_recall_at_4_pct"],
                "per_doc_recall_at_10_pct": b["per_doc_recall_at_10_pct"],
                "top10_rows_changed_vs_committed_baseline": b.get(
                    "top10_rows_changed_vs_committed_baseline"),
                "top10_identical_vs_committed_baseline": b.get(
                    "top10_identical_vs_committed_baseline"),
                "index_bytes": b.get("index_bytes"),
                "arena_reservation_mb": round(
                    (b.get("engine_stats_after_ingest", {}) or {}).get(
                        "arena_reservation_bytes", 0) / 2**20, 1),
                "arena_growth_copies": (b.get("engine_stats_after_ingest", {}) or {}).get(
                    "arena_growth_copies"),
            }

    doc = {
        "title": "nanomem v3 -- ingest-path peak RSS before and after the reserved arena",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": os.path.abspath(__file__),
        "machine": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "load_avg_at_end": loadavg(),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "baseline_commit": BASE_COMMIT, "baseline_dir": BASE_DIR,
                     "new_dir": NEW_DIR},
        "protocol": {
            "cache": CACHE,
            "cache_provenance": "built by scratch/refound/bench_competitors.py --prep; "
                                "the SAME docs.npy / queries.npy / gold.json / rng(0) "
                                "insertion order used by competitors_standard_results.json",
            "top_k_queried": TOP_K,
            "phases": "build = ingest, flush, close, reopen, query (the shape "
                      "bench_competitors.py charges as peak_rss_delta_mb). serve = a "
                      "fresh process opening the file the build left behind.",
            "isolation": "one subprocess per (arm, corpus, phase) so ru_maxrss is clean",
            "recall": "computed in the driver from each arm's returned id matrix; an "
                      "arm cannot score itself",
            "exactness": "top-10 compared row by row against committed_baseline_loop, the "
                         "engine exactly as committed at git %s" % BASE_COMMIT,
            "contention": "the machine was NOT quiesced; each arm records the load "
                          "average at its start and end",
        },
        "arms_described": {k: {"package": v[0], "feed": v[1], "hint": v[2],
                               "what": v[3]} for k, v in ARMS.items()},
        "corpora": corpora_meta,
        "arms": arms_out,
        "summary": summary,
        "growth_policy": {
            "what": "the allocator question with the engine taken out: a 768-d fp32 "
                    "arena is grown to 71,433 rows in 50-row steps under each policy, "
                    "one subprocess per policy so ru_maxrss is clean. Directly "
                    "comparable to memory_results.json -> growth_policy.",
            "rows": policies,
        },
        "directory_estimate": directory_estimate(),
        "reference_points": reference_points(),
        "package_fingerprint": package_fingerprint(),
        "failures": failures,
    }
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)
    print("\nwrote %s" % RESULTS)
    for corpus in CORPORA:
        print("\n== %s ==" % corpus)
        for arm, row in summary[corpus].items():
            print("  %-20s loader_peak=%7.1f MB  server=%7s MB  p50=%6s ms  "
                  "recall@4=%5.1f  top10_changed=%s"
                  % (arm, row["loader_peak_rss_mb"], row["server_rss_mb"],
                     row["p50_ms_server_phase"], row["evidence_recall_at_4_pct"],
                     row["top10_rows_changed_vs_committed_baseline"]))


if __name__ == "__main__":
    main()
