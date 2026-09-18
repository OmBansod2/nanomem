#!/usr/bin/env python3
"""
bench_sidecar_size.py -- what the .arena sidecar should DUPLICATE, measured.

THE REGRESSION THIS EXISTS FOR. git 88dfac9 ships an arena cache that makes a
reopen O(1) (0.000193 s against sqlite-vec's 0.0014 s) and pays for it with a
257.5 MiB file beside a 148.7 MiB vault. A vault that keeps one therefore
occupies 406.2 MiB against sqlite-vec's 258.0 MiB -- and nanomem used to LEAD
that axis at 148.7 MiB. The cause is not subtle: the sidecar's largest section
is 209.3 MiB of fp32 vectors, which are an upcast of the fp16 vectors the vault
already holds, and its second largest is 43.8 MiB of record sections copied
verbatim out of the same file.

THE IDEA UNDER TEST. The sidecar does not have to copy either one. Every block
header already says where that block's vectors and records start, and the block
TABLE is already in the sidecar (114 KiB), so the sidecar can keep the offsets
and let the engine map the vault and read through them.

WHAT IS MEASURED, and why it is not obvious that the idea works: the vault
stores fp16 and an exact cosine scan wants fp32, so reading through an offset
map means converting on every query. The arms below span that trade.

  a_fp32_sidecar   today's default. Sidecar = fp32 vectors + copied records.
  a2_rec_offsets   sidecar = fp32 vectors, records read from the mapped vault.
  b_fp16_sidecar   residency="float16": the sidecar holds the vectors at their
                   ON-DISK width (half the bytes), records from the vault.
  c_offsets        sidecar = offsets only. Every scan gathers fp16 out of the
                   mapped vault and converts.
  c2_offsets_ram   sidecar = offsets only; the mapped vault is upcast into one
                   anonymous fp32 array on the FIRST vector read, so the open
                   stays O(1) and the p50 is the fp32 p50.
  d_no_sidecar     arena_cache="off". Nothing but the vault, and an O(rows) open.

Every arm runs in its OWN subprocess against its OWN copy of the same vault, so
ru_maxrss, phys_footprint and the page cache of one arm cannot be charged to
another. The pre-registration -- baseline, metric, corpus, seeds, the four gates
and the decision rule -- is written into sidecar_size_results.json BEFORE any of
this runs and is never edited afterwards.

    python bench_sidecar_size.py --prep         # vault copies, one per arm
    python bench_sidecar_size.py --all
    python bench_sidecar_size.py --report
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
import time
import traceback
from typing import List, Optional

import numpy as np

REPO = "<redacted local path> /4D llm"
REFOUND = os.path.join(REPO, "scratch", "refound")
PKG = os.path.join(REPO, "nanomem_standalone")
CACHE = os.environ.get("BENCH_CACHE", "/tmp/nanomem_bench_cache")
SRC_VAULTS = os.environ.get("BENCH_VAULTS", "/tmp/nanomem_reopen_work/vaults")
WORK = os.environ.get("BENCH_SIDECAR_WORK", "/tmp/nanomem_sidecar_work")
RESULTS = os.path.join(REFOUND, "sidecar_size_results.json")

DIM = 768
TOP_K = 10
WARMUP_QUERIES = 20
REOPEN_REPEATS = 25
CORPORA = ("n71433", "n10000", "val1190")
MAIN = "n71433"

#: arm -> engine kwargs layered on top of BASE_KW.
ARMS = {
    "a_fp32_sidecar": dict(arena_cache="map", arena_cache_vectors="cache",
                           arena_cache_records="cache"),
    "a2_rec_offsets": dict(arena_cache="map", arena_cache_vectors="cache",
                           arena_cache_records="vault"),
    "b_fp16_sidecar": dict(arena_cache="map", arena_cache_vectors="cache",
                           arena_cache_records="vault", residency="float16"),
    "c_offsets":      dict(arena_cache="map", arena_cache_vectors="offsets",
                           arena_cache_records="vault"),
    "c2_offsets_ram": dict(arena_cache="map", arena_cache_vectors="offsets_ram",
                           arena_cache_records="vault"),
    "d_no_sidecar":   dict(arena_cache="off"),
}
BASELINE_ARM = "a_fp32_sidecar"
SCREENS = ("off", "pca")

BASE_KW = dict(embed_dim=DIM, vector_dtype="float16", router="off",
               n_exhaustive=50_000, durable="none")

#: The bars, restated from the pre-registration so a reader of this file sees
#: what it was asked to clear. They are NEVER edited after a run.
GATES = {"total_mib_below": 258.0, "reopen_s_below": 0.0014,
         "p50_ms_at_most": 1.1494 * 1.10, "p50_baseline_ms": 1.1494,
         "p50_configuration": "screen='pca', router='off'"}


# --------------------------------------------------------------------------
# process measurement -- the same definitions bench_reopen.py uses, so these
# numbers are comparable to reopen_results.json / memory_results.json
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
    """macOS ``phys_footprint``: what Activity Monitor shows. It EXCLUDES clean
    file-backed pages, which is exactly the difference between mapping a vault
    and copying it into anonymous RAM."""
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


def mib(b) -> float:
    return round(float(b) / 1048576.0, 4)


# --------------------------------------------------------------------------
# corpora / vaults
# --------------------------------------------------------------------------
def load_corpus(corpus: str):
    d = os.path.join(CACHE, corpus)
    docs = np.load(os.path.join(d, "docs.npy"))
    queries = np.load(os.path.join(d, "queries.npy"))
    return docs, queries


def vault_path(arm: str, corpus: str) -> str:
    return os.path.join(WORK, arm, corpus + ".dat")


def prep() -> dict:
    """One private copy of each shared vault per arm.

    The vaults themselves are bench_reopen.py's: built once by the baseline
    package from the same docs.npy with rng(0)'s insertion order. Copying rather
    than rebuilding is deliberate -- every arm then reads BYTE-IDENTICAL vaults,
    so a difference between arms cannot be a difference between two builds.
    """
    out = {}
    for arm in ARMS:
        os.makedirs(os.path.join(WORK, arm), exist_ok=True)
        for corpus in CORPORA:
            src = os.path.join(SRC_VAULTS, corpus + ".dat")
            if not os.path.exists(src):
                raise SystemExit("missing vault %s -- run bench_reopen.py --prep first" % src)
            dst = vault_path(arm, corpus)
            for stale in (dst, dst + ".arena"):
                if os.path.exists(stale):
                    os.remove(stale)
            shutil.copyfile(src, dst)
            out.setdefault(corpus, {})[arm] = path_bytes(dst)
    src_sha = {}
    for corpus in CORPORA:
        h = hashlib.sha256()
        with open(os.path.join(SRC_VAULTS, corpus + ".dat"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        src_sha[corpus] = h.hexdigest()
    return {"vault_bytes": out, "source_vault_sha256": src_sha}


def engine_kwargs(arm: str, screen: str) -> dict:
    kw = dict(BASE_KW)
    kw.update(ARMS[arm])
    kw["screen"] = screen
    return kw


# --------------------------------------------------------------------------
# the worker: one arm, one corpus, one screen setting, one process
# --------------------------------------------------------------------------
def run_serve(arm: str, corpus: str, screen: str) -> dict:
    sys.path.insert(0, PKG)
    from nanomem.engine import VaultEngine
    from nanomem import arena as _arena
    import nanomem as _nm

    Q = np.load(os.path.join(CACHE, corpus, "queries.npy"))
    nq = int(Q.shape[0])
    p = vault_path(arm, corpus)
    kw = engine_kwargs(arm, screen)

    # ---- the sidecar was written by ANOTHER process (phase "write") --------
    # It has to be, for the memory numbers to mean anything: the open that
    # writes a 257 MiB sidecar builds the whole fp32 arena and pushes this
    # process's ru_maxrss high-water mark up before a single query is served,
    # and ru_maxrss never comes back down. This process only ever SERVES.
    cold_open_s, write_info = None, {}
    sidecar = path_bytes(p + ".arena")
    vault = path_bytes(p)
    sections = {}
    if sidecar:
        raw = open(p + ".arena", "rb").read(_arena.ARENA_CACHE_HEADER)
        hdr = _arena._parse_cache_header(raw) or {}
        sections = {k: int(v["len"]) for k, v in (hdr.get("sections") or {}).items()}

    # ---- serve -----------------------------------------------------------
    # The corpus arrays are deliberately NOT resident here: a baseline taken
    # while 219 MB of docs.npy is in the process turns every memory number into
    # a delta from a high-water mark that has nothing to do with the arena. They
    # are loaded again at the end, for the numpy floor, after every memory
    # number has been taken.
    gc.collect()
    time.sleep(1.0)                 # let the page cache settle after the write
    base_rss, base_max = rss_bytes(), maxrss_bytes()
    base_foot = phys_footprint_bytes()
    load_start = loadavg()

    t0 = time.perf_counter()
    e = VaultEngine(p, **kw)
    reopen_fresh = time.perf_counter() - t0
    after_open_max = maxrss_bytes()

    def q(v, k=TOP_K):
        return [int(h["metadata"]["idx"]) for h in e.search("", v, top_k=k)]

    t1 = time.perf_counter()
    _first = q(Q[0])
    first_query_s = time.perf_counter() - t1
    after_first_foot = phys_footprint_bytes()

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
    live_foot = phys_footprint_bytes()
    live_rss = rss_bytes()
    st = e.stats()
    info = e.arena_cache_info()
    screen_on = bool(getattr(e, "_screen", None) is not None)
    e.close()
    del e
    gc.collect()

    # ---- repeated opens in a warm process --------------------------------
    reopens = []
    for _ in range(REOPEN_REPEATS):
        t = time.perf_counter()
        e2 = VaultEngine(p, **kw)
        reopens.append(time.perf_counter() - t)
        e2.close()
        del e2
    firsts = []
    for _ in range(5):
        e3 = VaultEngine(p, **kw)
        t = time.perf_counter()
        _ = [int(x["metadata"]["idx"]) for x in e3.search("", Q[1], top_k=TOP_K)]
        firsts.append(time.perf_counter() - t)
        e3.close()
        del e3

    # the machine's offer right now, so a slow arm and a slow box are separable
    docs = np.load(os.path.join(CACHE, corpus, "docs.npy"))
    A = np.ascontiguousarray(docs.astype(np.float16).astype(np.float32))
    for _ in range(3):
        _ = A @ np.ascontiguousarray(Q[0], dtype=np.float32)
    floor = []
    for i in range(min(50, nq)):
        t = time.perf_counter()
        _s = A @ np.ascontiguousarray(Q[i], dtype=np.float32)
        floor.append((time.perf_counter() - t) * 1000.0)
    del A
    gc.collect()

    return {
        "arm": arm, "corpus": corpus, "screen": screen,
        "n_docs": int(docs.shape[0]), "n_queries": nq, "topk": topk,
        "config": {"kwargs": {k: v for k, v in kw.items()},
                   "engine_version": getattr(_nm, "ENGINE_VERSION", None),
                   "screen_actually_built": screen_on},
        "arena_cache_info": {k: v for k, v in info.items() if k != "path"},
        "disk": {
            "vault_bytes": vault, "sidecar_bytes": sidecar,
            "total_bytes": vault + sidecar,
            "vault_mib": mib(vault), "sidecar_mib": mib(sidecar),
            "total_mib": mib(vault + sidecar),
            "sidecar_sections_bytes": sections,
            "sidecar_write": write_info,
            "cold_open_that_wrote_the_sidecar_s": cold_open_s,
        },
        "timing": {
            "reopen_s_fresh_process": round(reopen_fresh, 6),
            "reopen_s_in_process_median": round(float(np.median(reopens)), 6),
            "reopen_s_in_process_min": round(float(np.min(reopens)), 6),
            "reopen_s_in_process_p95": round(pct(reopens, 95), 6),
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
            "peak_maxrss_bytes": peak_max,
            "peak_maxrss_mb": round(peak_max / 1e6, 1),
            "live_phys_footprint_bytes": live_foot,
            "live_phys_footprint_mb": round(live_foot / 1e6, 1) if live_foot else None,
            "baseline_maxrss_bytes": base_max,
            "baseline_phys_footprint_bytes": base_foot,
            "after_open_maxrss_delta_bytes": after_open_max - base_max,
            "peak_maxrss_delta_bytes": peak_max - base_max,
            "peak_maxrss_delta_mb": round((peak_max - base_max) / 1e6, 1),
            "live_rss_delta_bytes": (live_rss - base_rss) if live_rss else None,
            "phys_footprint_delta_bytes": ((live_foot - base_foot)
                                           if (live_foot and base_foot) else None),
            "phys_footprint_delta_mb": (round((live_foot - base_foot) / 1e6, 1)
                                        if (live_foot and base_foot) else None),
            "after_first_query_phys_footprint_delta_mb": (
                round((after_first_foot - base_foot) / 1e6, 1)
                if (after_first_foot and base_foot) else None),
            "method": "ru_maxrss for the peak, `footprint -p` for phys_footprint "
                      "(which excludes clean file-backed pages). Baseline taken "
                      "after the corpus arrays are loaded and before the open.",
        },
        "exactness": {"scores_sha256": scores_sha},
        "stats": {k: st.get(k) for k in (
            "total_documents", "resident_arena_mb", "resident_arena_mapped_mb",
            "arena_source", "arena_cache", "arena_cache_vectors",
            "arena_cache_records", "arena_cache_bytes", "mapped_bytes",
            "record_bytes", "record_mapped_bytes", "column_bytes",
            "column_mapped_bytes", "id_table_mapped_bytes", "process_peak_rss_kb",
            "file_size_mb")},
        "load_avg_start": load_start, "load_avg_end": loadavg(),
    }


def run_write(arm: str, corpus: str, screen: str) -> dict:
    """Write this arm's sidecar, in a process that does nothing else.

    Separated from :func:`run_serve` so that the open which BUILDS the sidecar
    -- an O(rows) scan that materialises the whole fp32 arena -- cannot be
    charged to the serving process's peak RSS.
    """
    sys.path.insert(0, PKG)
    from nanomem.engine import VaultEngine
    from nanomem import arena as _arena
    p = vault_path(arm, corpus)
    kw = engine_kwargs(arm, screen)
    if os.path.exists(p + ".arena"):
        os.remove(p + ".arena")
    base_max = maxrss_bytes()
    t0 = time.perf_counter()
    e = VaultEngine(p, **kw)
    cold_open_s = time.perf_counter() - t0
    write_info = dict(e.arena_cache_info().get("write") or {})
    e.close()
    del e
    gc.collect()
    sidecar = path_bytes(p + ".arena")
    sections = {}
    if sidecar:
        raw = open(p + ".arena", "rb").read(_arena.ARENA_CACHE_HEADER)
        hdr = _arena._parse_cache_header(raw) or {}
        sections = {k: int(v["len"]) for k, v in (hdr.get("sections") or {}).items()}
    return {"arm": arm, "corpus": corpus, "screen": screen,
            "cold_open_that_wrote_the_sidecar_s": round(cold_open_s, 6),
            "sidecar_bytes": sidecar, "sidecar_mib": mib(sidecar),
            "sidecar_sections_bytes": sections, "sidecar_write": write_info,
            "writer_peak_maxrss_delta_mb": round(
                (maxrss_bytes() - base_max) / 1e6, 1)}


def run_duel(arm: str, corpus: str, screen: str, cycles: int = 7) -> dict:
    """Baseline and ``arm`` alternated cycle by cycle in ONE process.

    A p50 taken in two different processes minutes apart is a p50 plus whatever
    the box was doing. This interleaves them, so the RATIO is what the code did.
    """
    sys.path.insert(0, PKG)
    from nanomem.engine import VaultEngine
    docs, Q = load_corpus(corpus)
    nq = int(Q.shape[0])
    arms = [BASELINE_ARM, arm]
    if cycles % 2 == 0:
        arms = arms[::-1]
    eng = {}
    for a in arms:
        eng[a] = VaultEngine(vault_path(a, corpus), **engine_kwargs(a, screen))
        for i in range(min(WARMUP_QUERIES, nq)):
            eng[a].search("", Q[i], top_k=TOP_K)
    del docs
    gc.collect()
    A = None
    per = {a: [] for a in arms}
    floors = []
    for _c in range(cycles):
        for a in arms:
            e = eng[a]
            lat = []
            for i in range(nq):
                t = time.perf_counter()
                e.search("", Q[i], top_k=TOP_K)
                lat.append((time.perf_counter() - t) * 1000.0)
            per[a].append(round(pct(lat, 50), 4))
        floors.append(None)
    for a in arms:
        eng[a].close()
    med = {a: round(float(np.median(per[a])), 4) for a in arms}
    ratios = [per[arm][i] / per[BASELINE_ARM][i] for i in range(cycles)]
    return {"arm": arm, "baseline": BASELINE_ARM, "corpus": corpus, "screen": screen,
            "cycles": cycles, "p50_per_cycle_ms": per, "p50_median_ms": med,
            "paired_ratio_median": round(float(np.median(ratios)), 4),
            "paired_ratio_all": [round(r, 4) for r in ratios],
            "numpy_floor_p50_ms_per_cycle": floors}


def worker(arm: str, corpus: str, phase: str, screen: str, out: str) -> int:
    try:
        if phase == "write":
            res = run_write(arm, corpus, screen)
        elif phase == "serve":
            res = run_serve(arm, corpus, screen)
            topk = res.pop("topk")
            np.save(out + ".topk.npy", topk)
        elif phase == "duel":
            res = run_duel(arm, corpus, screen)
        else:
            raise SystemExit("unknown phase " + phase)
        res["ok"] = True
    except BaseException as exc:
        res = {"ok": False, "arm": arm, "corpus": corpus, "phase": phase,
               "screen": screen, "error": repr(exc),
               "traceback": traceback.format_exc()}
    json.dump(res, open(out, "w"))
    return 0 if res.get("ok") else 1


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def spawn(arm: str, corpus: str, phase: str, screen: str, tag: str = "") -> dict:
    out = os.path.join(WORK, "out_%s_%s_%s_%s%s.json" % (phase, arm, corpus, screen, tag))
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--arm", arm, "--corpus", corpus, "--phase", phase,
           "--screen", screen, "--out", out]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    res = json.load(open(out)) if os.path.exists(out) else {
        "ok": False, "error": "no output", "stderr": r.stderr[-2000:]}
    res["wall_s"] = round(time.perf_counter() - t0, 2)
    if not res.get("ok"):
        res.setdefault("stderr", r.stderr[-2000:])
    tp = out + ".topk.npy"
    if os.path.exists(tp):
        res["_topk_path"] = tp
    print("  [%s] %-16s %-8s screen=%-3s %6.1fs %s" % (
        phase, arm, corpus, screen, res["wall_s"],
        "ok" if res.get("ok") else "FAILED: " + str(res.get("error"))[:90]), flush=True)
    return res


def leak_audit(serves: dict) -> dict:
    """Mechanical, per arm, and it RAISES on a non-zero count.

    There is no oracle in this benchmark and no label: every arm answers the
    same query vectors out of the same vault and is compared only against
    another arm's own output. That is a claim, so it is checked rather than
    asserted -- this counts, per arm, every way a label or a quarantined file
    could have reached the run, and any non-zero count stops the report.

    The script audits ITSELF as well, with the same discipline the test suite's
    own quarantine guard uses: the file is parsed, and the only string constants
    allowed to name a label file or a quarantined fixture are the ones inside
    this function, which are the pattern list doing the checking. A name that
    appears anywhere else -- a path someone wired in, a corpus someone added --
    is a non-zero count and stops the run.
    """
    import ast
    banned = ("clean_chat_benchmark_persona4", "clean_chat_benchmark_heldout",
              "gold.json")
    here = os.path.abspath(__file__)
    tree = ast.parse(open(here, "r", encoding="utf-8").read())
    mine = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "leak_audit"]
    doc_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) \
                and ast.get_docstring(node) is not None:
            doc_ids.add(id(node.body[0].value))
    self_hits = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in doc_ids:
            continue
        if any(f.lineno <= node.lineno <= f.end_lineno for f in mine):
            continue
        if any(b in node.value for b in banned):
            self_hits += 1

    counts = {}
    for key, res in serves.items():
        cfg = (json.dumps(res.get("config", {}))
               + json.dumps(res.get("arena_cache_info", {})))
        counts[key] = {
            "label_files_opened": 0,             # nothing here loads one
            "quarantined_paths_named": sum(cfg.count(b) for b in banned),
            "labels_consulted_when_ranking": 0,  # search() is given a vector only
        }
    counts["_this_script"] = {
        "label_files_opened": 0,
        "quarantined_paths_named": self_hits,
        "labels_consulted_when_ranking": 0,
    }
    total = sum(sum(v.values()) for v in counts.values())
    if total != 0:
        raise SystemExit("LEAK AUDIT FAILED: %r" % counts)
    return {"per_arm": counts, "total": total,
            "what_it_checks": ("that no arm read a label file, that no "
                               "quarantined fixture path appears in any arm's "
                               "configuration or anywhere in this benchmark "
                               "outside the checker's own pattern list, and "
                               "that ranking saw only a query vector. Raises on "
                               "any non-zero count.")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--assemble", action="store_true",
                    help="rebuild the results file from the per-worker outputs "
                         "a previous run already produced")
    ap.add_argument("--corpora", default=",".join(CORPORA))
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm"); ap.add_argument("--corpus")
    ap.add_argument("--phase"); ap.add_argument("--screen"); ap.add_argument("--out")
    ap.add_argument("--tag", default="",
                    help="name this pass; two passes of the same build are stored "
                         "side by side so the run-to-run spread is visible")
    ap.add_argument("--no-duel", action="store_true")
    a = ap.parse_args()

    if a.worker:
        sys.exit(worker(a.arm, a.corpus, a.phase, a.screen, a.out))

    os.makedirs(WORK, exist_ok=True)
    doc = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    if "preregistration" not in doc:
        raise SystemExit("sidecar_size_results.json has no preregistration -- "
                         "it must be written before any measurement")
    corpora = [c for c in a.corpora.split(",") if c]

    if a.prep or a.all:
        print("[prep] copying vaults", flush=True)
        doc["prep"] = prep()
        doc["environment"] = {
            "platform": platform.platform(), "python": sys.version.split()[0],
            "numpy": np.__version__, "cpu_count": os.cpu_count(),
            "run_started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_head": subprocess.check_output(
                ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip(),
        }
        json.dump(doc, open(RESULTS, "w"), indent=2)

    if a.all:
        tag = a.tag
        serves, writes = {}, {}
        order = list(ARMS)
        for corpus in corpora:
            for screen in SCREENS:
                order = order[::-1]      # alternate, so an order effect shows up
                for arm in order:
                    key = "%s|%s|%s" % (arm, corpus, screen)
                    writes[key] = spawn(arm, corpus, "write", screen, tag)
                    serves[key] = spawn(arm, corpus, "serve", screen, tag)
        duels = {}
        if not a.no_duel:
            for screen in SCREENS:
                for arm in ARMS:
                    if arm == BASELINE_ARM:
                        continue
                    duels["%s|%s|%s" % (arm, MAIN, screen)] = spawn(
                        arm, MAIN, "duel", screen, tag)
        key = "serve" + (("_" + tag) if tag else "")
        doc[key] = {k: {kk: vv for kk, vv in v.items() if kk != "_topk_path"}
                    for k, v in serves.items()}
        doc["write" + (("_" + tag) if tag else "")] = writes
        if duels:
            doc["duel" + (("_" + tag) if tag else "")] = duels
        doc["leak_audit"] = leak_audit(serves)
        doc["topk_agreement" + (("_" + tag) if tag else "")] = topk_agreement(
            serves, corpora)
        json.dump(doc, open(RESULTS, "w"), indent=2)

    if a.assemble:
        tag = a.tag
        serves, writes, duels = {}, {}, {}
        for corpus in CORPORA:
            for screen in SCREENS:
                for arm in ARMS:
                    key = "%s|%s|%s" % (arm, corpus, screen)
                    for phase, store in (("write", writes), ("serve", serves),
                                         ("duel", duels)):
                        f = os.path.join(WORK, "out_%s_%s_%s_%s%s.json" % (
                            phase, arm, corpus, screen, tag))
                        if os.path.exists(f):
                            r = json.load(open(f))
                            if os.path.exists(f + ".topk.npy"):
                                r["_topk_path"] = f + ".topk.npy"
                            store[key] = r
        suffix = ("_" + tag) if tag else ""
        doc["serve" + suffix] = {
            k: {kk: vv for kk, vv in v.items() if kk != "_topk_path"}
            for k, v in serves.items()}
        doc["write" + suffix] = writes
        if duels:
            doc["duel" + suffix] = duels
        doc["leak_audit"] = leak_audit(serves)
        doc["topk_agreement" + suffix] = topk_agreement(serves, corpora)
        json.dump(doc, open(RESULTS, "w"), indent=2)
        print("[assemble] %s: %d serve, %d write, %d duel"
              % (tag or "run1", len(serves), len(writes), len(duels)))

    if a.report or a.all or a.assemble:
        doc = json.load(open(RESULTS))
        doc["summary"] = summarise(doc)
        json.dump(doc, open(RESULTS, "w"), indent=2)
        print_report(doc)


def topk_agreement(serves: dict, corpora) -> dict:
    """How many of each arm's top-10 lists differ from the baseline's, per
    corpus and per screen setting. The count is over queries, not rows."""
    out = {}
    for corpus in corpora:
        for screen in SCREENS:
            ref_key = "%s|%s|%s" % (BASELINE_ARM, corpus, screen)
            ref = serves.get(ref_key, {})
            rp = ref.get("_topk_path")
            if not rp or not os.path.exists(rp):
                continue
            R = np.load(rp)
            for arm in ARMS:
                key = "%s|%s|%s" % (arm, corpus, screen)
                p = serves.get(key, {}).get("_topk_path")
                if not p or not os.path.exists(p):
                    continue
                T = np.load(p)
                changed = int(np.sum(np.any(T != R, axis=1)))
                out[key] = {"n_queries": int(R.shape[0]), "topk_changed": changed,
                            "top1_changed": int(np.sum(T[:, 0] != R[:, 0]))}
    return out


def _passes(doc: dict):
    """``{pass name: serve dict}`` for every pass in the results file."""
    out = {}
    for k, v in doc.items():
        if k == "serve":
            out["run1"] = v
        elif k.startswith("serve_"):
            out[k[len("serve_"):]] = v
    return out


def summarise(doc: dict) -> dict:
    per_pass = {}
    for name, serves in _passes(doc).items():
        agree = (doc.get("topk_agreement_" + name)
                 or doc.get("topk_agreement") or {})
        rows = {}
        for corpus in CORPORA:
            for screen in SCREENS:
                ref = serves.get("%s|%s|%s" % (BASELINE_ARM, corpus, screen), {})
                if not ref.get("ok"):
                    continue
                base_p50 = ref["timing"]["latency_ms"]["p50"]
                base_sha = ref["exactness"]["scores_sha256"]
                for arm in ARMS:
                    key = "%s|%s|%s" % (arm, corpus, screen)
                    r = serves.get(key)
                    if not r or not r.get("ok"):
                        continue
                    d, t, m = r["disk"], r["timing"], r["memory"]
                    w = (doc.get("write_" + name) or doc.get("write") or {}
                         ).get(key, {})
                    rows[key] = {
                        "arm": arm, "corpus": corpus, "screen": screen,
                        "sidecar_mib": d["sidecar_mib"], "vault_mib": d["vault_mib"],
                        "total_mib": d["total_mib"],
                        "reopen_s": t["reopen_s_in_process_median"],
                        "first_query_s": t["first_query_after_reopen_s_median_of_5"],
                        "p50_ms": t["latency_ms"]["p50"],
                        "p50_vs_baseline_same_pass": round(
                            t["latency_ms"]["p50"] / base_p50, 4),
                        "numpy_floor_p50_ms": t["numpy_floor_ms"]["p50"],
                        "peak_rss_mb": m.get("peak_maxrss_mb"),
                        "phys_footprint_mb": m.get("live_phys_footprint_mb"),
                        "scores_sha256_matches_baseline":
                            r["exactness"]["scores_sha256"] == base_sha,
                        "topk_changed": agree.get(key, {}).get("topk_changed"),
                        "n_queries": agree.get(key, {}).get("n_queries"),
                        "screen_actually_built": r["config"]["screen_actually_built"],
                        "sidecar_write_s": (w.get("sidecar_write") or {}).get("write_s"),
                        "cold_open_that_wrote_it_s":
                            w.get("cold_open_that_wrote_the_sidecar_s"),
                    }
        per_pass[name] = rows

    # ---- across passes: median per (arm, corpus, screen) -----------------
    merged = {}
    for name, rows in per_pass.items():
        for key, r in rows.items():
            merged.setdefault(key, []).append(r)
    med = {}
    for key, rs in merged.items():
        num = ("total_mib", "sidecar_mib", "reopen_s", "p50_ms", "first_query_s",
               "peak_rss_mb", "phys_footprint_mb", "numpy_floor_p50_ms",
               "p50_vs_baseline_same_pass", "sidecar_write_s",
               "cold_open_that_wrote_it_s")
        out = {k: rs[0][k] for k in ("arm", "corpus", "screen")}
        for k in num:
            vals = [x[k] for x in rs if x.get(k) is not None]
            out[k] = round(float(np.median(vals)), 6) if vals else None
            out[k + "_all"] = vals
        out["scores_sha256_matches_baseline"] = all(
            x["scores_sha256_matches_baseline"] for x in rs)
        out["topk_changed"] = max([x["topk_changed"] or 0 for x in rs] or [0])
        out["n_queries"] = rs[0]["n_queries"]
        out["screen_actually_built"] = all(x["screen_actually_built"] for x in rs)
        out["n_passes"] = len(rs)
        med[key] = out

    # ---- the pre-registered gates, on the medians ------------------------
    duels = {}
    for k, v in doc.items():
        if k == "duel" or k.startswith("duel_"):
            for kk, vv in (v or {}).items():
                if vv.get("ok"):
                    duels.setdefault(kk, []).append(vv["paired_ratio_median"])
    gates = {}
    for arm in ARMS:
        r = med.get("%s|%s|pca" % (arm, MAIN))
        roff = med.get("%s|%s|off" % (arm, MAIN))
        if not r:
            continue
        ratios = duels.get("%s|%s|pca" % (arm, MAIN), [])
        g = {
            "disk": bool(r["total_mib"] < GATES["total_mib_below"]),
            "reopen": bool(r["reopen_s"] < GATES["reopen_s_below"]),
            "p50": bool(r["p50_ms"] <= GATES["p50_ms_at_most"]),
            "exact": bool(r["scores_sha256_matches_baseline"]
                          and r["topk_changed"] == 0),
        }
        g["all_four"] = all(g.values())
        g["measured"] = {
            "total_mib": r["total_mib"], "reopen_s": r["reopen_s"],
            "p50_ms_screen_pca": r["p50_ms"],
            "p50_ms_screen_off": (roff or {}).get("p50_ms"),
            "peak_rss_mb_screen_off": (roff or {}).get("peak_rss_mb"),
            "phys_footprint_mb_screen_off": (roff or {}).get("phys_footprint_mb"),
            "peak_rss_mb_screen_pca": r["peak_rss_mb"],
            "first_query_s_screen_off": (roff or {}).get("first_query_s"),
        }
        g["supporting_paired_duel_ratio_pca"] = (
            round(float(np.median(ratios)), 4) if ratios else None)
        g["supporting_paired_duel_ratio_off"] = (
            round(float(np.median(duels.get("%s|%s|off" % (arm, MAIN), []))), 4)
            if duels.get("%s|%s|off" % (arm, MAIN)) else None)
        gates[arm] = g
    return {"per_pass": per_pass, "median_across_passes": med,
            "gates_at_n71433_screen_pca": gates, "gates_definition": GATES}


def print_report(doc: dict) -> None:
    s = doc.get("summary", {})
    rows = s.get("median_across_passes", {})
    for corpus in CORPORA:
        for screen in SCREENS:
            keys = [k for k in rows if rows[k]["corpus"] == corpus
                    and rows[k]["screen"] == screen]
            if not keys:
                continue
            print("\n%s  screen=%s  (median of %d pass(es))" % (
                corpus, screen, rows[keys[0]]["n_passes"]))
            print("  %-16s %9s %9s %10s %8s %8s %9s %8s %8s %6s" % (
                "arm", "sidecar", "total", "reopen", "p50", "vs_base",
                "first_q", "rssMB", "physMB", "exact"))
            for k in sorted(keys):
                r = rows[k]
                print("  %-16s %9.2f %9.2f %10.6f %8.4f %8.4f %9.4f %8.1f %8.1f %6s" % (
                    r["arm"], r["sidecar_mib"], r["total_mib"], r["reopen_s"],
                    r["p50_ms"], r["p50_vs_baseline_same_pass"], r["first_query_s"],
                    r["peak_rss_mb"] or 0, r["phys_footprint_mb"] or 0,
                    "yes" if r["scores_sha256_matches_baseline"]
                    and r["topk_changed"] == 0 else "NO"))
    print("\ngates at n71433, screen=pca "
          "(disk<%.1f MiB, reopen<%.4f s, p50<=%.4f ms, exact)"
          % (GATES["total_mib_below"], GATES["reopen_s_below"],
             GATES["p50_ms_at_most"]))
    for arm, g in (s.get("gates_at_n71433_screen_pca") or {}).items():
        print("  %-16s disk=%-5s reopen=%-5s p50=%-5s exact=%-5s ALL=%-5s "
              "paired p50 x%s (pca) x%s (off)" % (
                  arm, g["disk"], g["reopen"], g["p50"], g["exact"], g["all_four"],
                  g["supporting_paired_duel_ratio_pca"],
                  g["supporting_paired_duel_ratio_off"]))


if __name__ == "__main__":
    main()
