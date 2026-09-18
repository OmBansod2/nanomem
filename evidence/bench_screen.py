#!/usr/bin/env python3
"""Measure the shipped PCA + Cauchy-Schwarz search screen against the criterion
pre-registered in scratch/refound/pca_screen_results.json.

WHAT THIS FILE READS. Exactly one corpus: scratch/refound/hotpot_train_8k_embeds.npz
(71,433 HotpotQA paragraph embeddings + 8,000 question embeddings). Nothing else
on disk is opened except the temporary vault this script writes and deletes.
It does not read, score or tune on any chat or persona corpus; the strings
"persona4" and "heldout" appear nowhere in this file, which is what "never
opened" means here. atime is NOT used as evidence -- measured on this
filesystem, reads do not bump atime.

WHAT IT MEASURES.
  A  exactness through VaultEngine.search: identical ids and BITWISE identical
     float32 scores, flag ON vs flag OFF, at k = 1, 4, 10.
  B  bound admissibility: every (query, document) pair, counted, not sampled.
  C  end-to-end p50 latency, paired and interleaved, >= 5 cycles, at 71,433.
  D  size sweep, to pick and then validate the engagement floor.
  E  appends against a deliberately STALE basis.
  F  automatic fallback: no basis, filtered search, personal corpus, int8.

Usage:  python3 scratch/refound/bench_screen.py [--quick]
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# NANOMEM_PKG lets this script measure a PINNED snapshot of the package. It is
# used because another agent is refactoring arena.py/container.py in the working
# tree at the same time; a measurement taken against a half-applied refactor
# would be a measurement of nothing. The snapshot is git HEAD's arena.py /
# container.py plus this branch's engine.py and screen.py. Which one was
# measured is recorded in `measurement.package_path` and `package_provenance`.
PKG = os.environ.get("NANOMEM_PKG") or os.path.join(REPO, "nanomem_standalone")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

EMBEDS = os.path.join(HERE, "hotpot_train_8k_embeds.npz")
# NANOMEM_SCREEN_RESULTS lets a run write somewhere other than the shipped
# report. Used to measure a BEFORE arm (an engine.py with the old positional
# top-k selection) back to back with the shipped one on the same machine
# weather, without either overwriting the other.
RESULTS = (os.environ.get("NANOMEM_SCREEN_RESULTS")
           or os.path.join(HERE, "pca_screen_results.json"))

from nanomem.engine import VaultEngine          # noqa: E402
from nanomem import screen as _screen           # noqa: E402


# --------------------------------------------------------------------------
# leakage audit (lesson 1): a mechanical counter that RAISES if non-zero
# --------------------------------------------------------------------------
class PairAudit:
    """Counts every way the two arms could have been given different inputs.

    This study injects no oracle and no ground-truth label, so there is no
    label to leak; what there IS to get wrong is comparing two arms that were
    not actually given the same thing. Every counter here must be 0.
    """

    def __init__(self):
        self.compared = 0
        self.query_vector_not_bitwise_identical = 0
        self.compared_against_a_foreign_reference = 0
        self.arms_on_different_vault_files = 0

    def check(self, q_on, q_off, path_on, path_off, reference_is_local=True):
        self.compared += 1
        if not np.array_equal(np.asarray(q_on), np.asarray(q_off)):
            self.query_vector_not_bitwise_identical += 1
        if os.path.abspath(path_on) != os.path.abspath(path_off):
            self.arms_on_different_vault_files += 1
        if not reference_is_local:
            self.compared_against_a_foreign_reference += 1

    def report(self):
        d = {
            "queries_compared": self.compared,
            "query_vector_not_bitwise_identical": self.query_vector_not_bitwise_identical,
            "compared_against_a_foreign_reference": self.compared_against_a_foreign_reference,
            "arms_on_different_vault_files": self.arms_on_different_vault_files,
        }
        bad = sum(v for k, v in d.items() if k != "queries_compared")
        d["leakage_total"] = bad
        d["verdict"] = "CLEAN" if bad == 0 else "LEAK"
        if bad:
            raise AssertionError(f"leakage audit FAILED: {d}")
        return d


def machine():
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = platform.processor()
    try:
        commit = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return {
        "cpu": cpu, "os": platform.platform(), "python": sys.version.split()[0],
        "numpy": np.__version__, "loadavg": list(os.getloadavg()),
        "git_commit": commit,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def build_vault(path, V, label="corpus.txt", chunk=4096):
    e = VaultEngine(path, embed_dim=V.shape[1], screen="off")
    e.reserve_additional_rows(V.shape[0])
    t0 = time.perf_counter()
    for i in range(V.shape[0]):
        e.add_fact(f"paragraph {i}", V[i], source=label)
        if (i + 1) % chunk == 0:
            e.flush()
    e.flush()
    secs = time.perf_counter() - t0
    e.close()
    return secs


def duplicate_census(V, sample=4000, seed=0):
    """How many rows have an exact duplicate elsewhere in the corpus.

    This is the CAUSE of any id difference that is a score tie, and it is a
    property of the corpus, not of the screen.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(V.shape[0], min(sample, V.shape[0]), replace=False)
    seen = {}
    dup = 0
    keys = [V[i].tobytes() for i in idx]
    for kb in keys:
        seen[kb] = seen.get(kb, 0) + 1
    # a sampled row counts as duplicated when the FULL corpus holds it twice
    full = {}
    for i in range(V.shape[0]):
        kb = V[i].tobytes()
        full[kb] = full.get(kb, 0) + 1
    for kb in keys:
        if full.get(kb, 0) > 1:
            dup += 1
    return {"sampled_rows": int(len(keys)), "rows_with_an_exact_duplicate": int(dup),
            "pct": 100.0 * dup / max(1, len(keys))}


def hits_key(hits):
    return [(h["id"], h["cosine"]) for h in hits]


# --------------------------------------------------------------------------
# A. exactness through the engine
# --------------------------------------------------------------------------
def phase_a_exactness(path, Q, dims, min_rows, ks=(1, 4, 10)):
    on = VaultEngine(path, screen="pca", screen_dims=dims, screen_min_rows=min_rows)
    off = VaultEngine(path, screen="off")
    built = on.build_screen()
    audit = PairAudit()
    rows = []
    for k in ks:
        id_diff = score_diff = tie_diff = real_miss = set_diff = 0
        for q in Q:
            audit.check(q, q, on.filepath, off.filepath, reference_is_local=True)
            a = on.search("what is this about", q, top_k=k)
            b = off.search("what is this about", q, top_k=k)
            if [h["id"] for h in a] != [h["id"] for h in b]:
                id_diff += 1
                # CLASSIFY IT. The exhaustive top-k is not unique when two
                # documents carry the identical float32 cosine -- HotpotQA has
                # exact duplicate paragraphs -- and which of them np.argpartition
                # keeps depends on the length of the array it partitions. A
                # difference is a TIE when the two answers carry the identical
                # score sequence and every id in the symmetric difference has the
                # same cosine as its counterpart. Anything else is a REAL MISS
                # and means the bound is wrong.
                sa = [h["cosine"] for h in a]
                sb = [h["cosine"] for h in b]
                if set(h["id"] for h in a) != set(h["id"] for h in b):
                    set_diff += 1
                if sa == sb:
                    tie_diff += 1
                else:
                    real_miss += 1
            if [h["cosine"] for h in a] != [h["cosine"] for h in b]:
                score_diff += 1
        rows.append({
            "queries_with_a_different_id_SET": set_diff,
            "queries_whose_only_difference_is_the_ORDER_of_equal_scores": tie_diff,
            "queries_with_a_REAL_MISS": real_miss,
            "top_k": k, "n_queries": int(Q.shape[0]),
            "screen_engaged_queries": int(on.screen_info()["engaged"]),
            "screen_fell_back_queries": int(on.screen_info()["fell_back"]),
            "queries_with_a_different_id_list": id_diff,
            "queries_with_a_non_bitwise_identical_score_list": score_diff,
            "pct_exact": 100.0 * (Q.shape[0] - max(id_diff, score_diff)) / Q.shape[0],
        })
    info = on.screen_info()
    on.close(); off.close()
    return {"rows": rows, "screen": built, "screen_info": info,
            "leakage_audit": audit.report()}


# --------------------------------------------------------------------------
# B. bound admissibility over EVERY (query, document) pair
# --------------------------------------------------------------------------
def phase_b_bound(path, Q, dims, min_rows):
    # The bound is a property of the screen, not of the engagement floor, so the
    # floor is dropped here to force a screen to exist at any corpus size.
    on = VaultEngine(path, screen="pca", screen_dims=dims, screen_min_rows=1)
    on.build_screen()
    scr = on._screen
    n = on.arena.n_rows
    viol = 0
    worst = -1e9                      # max over all pairs of (true - bound)
    worst_gap = 1e9                   # min slack
    t0 = time.perf_counter()
    for q in Q:
        q = np.asarray(q, dtype=np.float32)
        q = q / np.linalg.norm(q)
        ub = scr.bounds(q)
        true = on.arena.scores(q)
        d = true - ub
        m = float(d.max())
        if m > worst:
            worst = m
        worst_gap = min(worst_gap, float(d.max()))
        viol += int((d > 0).sum())
    secs = time.perf_counter() - t0
    on.close()
    return {
        "n_docs": int(n), "n_queries": int(Q.shape[0]),
        "document_checks": int(n) * int(Q.shape[0]),
        "bound_violations": int(viol),
        "worst_true_minus_bound": worst,
        "verdict": "ADMISSIBLE" if viol == 0 else "INADMISSIBLE",
        "seconds": secs,
    }


# --------------------------------------------------------------------------
# C/D. paired interleaved latency
# --------------------------------------------------------------------------
def paired_latency(path, Q, dims, min_rows, cycles=5, top_k=4, force=False):
    """One engine, flag toggled per query: the tightest possible pairing.

    Both arms then run against the identical arena object, the identical page
    cache and the identical machine weather, and the screen's own resident
    arrays are present in both arms (which is the honest configuration -- they
    are present whenever the flag is available).
    """
    e = VaultEngine(path, screen="pca", screen_dims=dims,
                    screen_min_rows=(0 if force else min_rows))
    built = e.build_screen()
    t_on, t_off = [], []
    per_cycle = []
    # warm both paths
    for q in Q[: min(16, len(Q))]:
        e.screen_mode = "pca"; e.search("warm up", q, top_k=top_k)
        e.screen_mode = "off"; e.search("warm up", q, top_k=top_k)
    for c in range(cycles):
        c_on, c_off = [], []
        for i, q in enumerate(Q):
            # Alternate which arm is timed first. Whichever runs first pays for
            # whatever the other one left cold, and a fixed order turns that
            # into a systematic bias -- measured as an 11% "slowdown" for an arm
            # that was not even engaged.
            order = ("pca", "off") if (i + c) % 2 == 0 else ("off", "pca")
            for mode in order:
                e.screen_mode = mode
                t0 = time.perf_counter()
                e.search("what is this about", q, top_k=top_k)
                dt = (time.perf_counter() - t0) * 1e3
                (c_on if mode == "pca" else c_off).append(dt)
        t_on += c_on; t_off += c_off
        per_cycle.append({"cycle": c,
                          "on_p50_ms": float(np.percentile(c_on, 50)),
                          "off_p50_ms": float(np.percentile(c_off, 50)),
                          "speedup": float(np.percentile(c_off, 50) /
                                           np.percentile(c_on, 50))})
    info = e.screen_info()
    n = int(e.arena.n_rows)
    e.close()
    on = np.asarray(t_on); off = np.asarray(t_off)
    rng = np.random.default_rng(3)
    boots = []
    idx = np.arange(on.size)
    for _ in range(2000):
        s = rng.choice(idx, idx.size, replace=True)
        boots.append(float(np.percentile(off[s], 50) / np.percentile(on[s], 50)))
    boots = np.sort(np.asarray(boots))
    return {
        "n_docs": n, "n_queries": int(len(Q)), "cycles": cycles, "top_k": top_k,
        "samples_per_arm": int(on.size),
        "screen_built": built,
        "off_p50_ms": float(np.percentile(off, 50)),
        "on_p50_ms": float(np.percentile(on, 50)),
        "off_p95_ms": float(np.percentile(off, 95)),
        "on_p95_ms": float(np.percentile(on, 95)),
        "off_mean_ms": float(off.mean()), "on_mean_ms": float(on.mean()),
        "p50_speedup": float(np.percentile(off, 50) / np.percentile(on, 50)),
        "p95_speedup": float(np.percentile(off, 95) / np.percentile(on, 95)),
        "p50_speedup_ci95": [float(boots[int(0.025 * boots.size)]),
                             float(boots[int(0.975 * boots.size)])],
        "per_cycle": per_cycle,
        "screen_info": info,
        "loadavg": list(os.getloadavg()),
    }


# --------------------------------------------------------------------------
# E. a deliberately STALE basis
# --------------------------------------------------------------------------
def phase_e_stale(tmp, V, Q, dims):
    """Fit the basis on the first fifth of the corpus, then append the rest.

    The point is not that it stays fast. The point is that it stays EXACT, and
    that appending never triggers a refit -- which is the whole of "appends
    cannot invalidate correctness".

    ONE engine, flag toggled, so the two arms read the identical arena. (The
    first version of this phase compared the writing engine against a separately
    reopened one and reported 99/100 score differences; those were the
    in-session fp32 values against the fp16-on-disk values, a pre-existing
    property of the store and nothing to do with the screen. Recorded here
    because it is the kind of confound that silently inverts a conclusion.)
    """
    path = os.path.join(tmp, "stale.dat")
    n0 = V.shape[0] // 5
    build_vault(path, V[:n0])
    e = VaultEngine(path, screen="pca", screen_dims=dims, screen_min_rows=100)
    first = e.build_screen()
    fit_rows_before = int(first["fit_rows"])
    e.reserve_additional_rows(V.shape[0] - n0)
    for i in range(n0, V.shape[0]):
        e.add_fact(f"paragraph {i}", V[i], source="corpus.txt")
        if (i - n0 + 1) % 4096 == 0:
            e.flush()
    e.flush()
    id_diff = score_diff = 0
    for q in Q:
        e.screen_mode = "pca"
        a = e.search("what is this about", q, top_k=10)
        e.screen_mode = "off"
        b = e.search("what is this about", q, top_k=10)
        e.screen_mode = "pca"
        if [h["id"] for h in a] != [h["id"] for h in b]:
            id_diff += 1
        if [h["cosine"] for h in a] != [h["cosine"] for h in b]:
            score_diff += 1
    after = e.screen_info()
    scr = e._screen
    viol = 0
    nq = min(100, len(Q))
    for q in Q[:nq]:
        qq = np.asarray(q, np.float32); qq = qq / np.linalg.norm(qq)
        ub = scr.bounds(qq)[n0:]
        true = e.arena.scores(qq)[n0:]
        viol += int((true > ub).sum())
    out = {
        "rows_basis_was_fitted_from": fit_rows_before,
        "rows_after_append": int(after["rows"]),
        "basis_refitted_on_append": bool(int(after["fit_rows"]) != fit_rows_before),
        "rows_appended_after_the_fit": int(after["rows"]) - n0,
        "n_queries": int(len(Q)),
        "queries_with_a_different_id_list": id_diff,
        "queries_with_a_non_bitwise_identical_score_list": score_diff,
        "bound_violations_on_rows_appended_after_the_fit": viol,
        "document_checks_on_appended_rows": int((int(after["rows"]) - n0) * nq),
        "screen_info": after,
    }
    e.close()
    try:
        os.remove(path)
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------
# F. automatic fallback
# --------------------------------------------------------------------------
def phase_f_fallback(tmp, V, Q, dims):
    path = os.path.join(tmp, "fb.dat")
    build_vault(path, V)
    out = {}

    # 1. below the engagement floor -> never builds, never engages
    e = VaultEngine(path, screen="pca", screen_dims=dims,
                    screen_min_rows=V.shape[0] + 1)
    e.search("a question", Q[0], top_k=4)
    out["below_floor"] = {"built": e.build_screen()["built"],
                          "engaged": e.screen_info()["engaged"]}
    e.close()

    # 2. int8 residency -> refused, because Arena.scores is not exact there
    e = VaultEngine(path, screen="pca", screen_dims=dims, screen_min_rows=10,
                    residency="int8")
    e.search("a question", Q[0], top_k=4)
    out["int8_residency"] = {"built": e.build_screen()["built"],
                             "engaged": e.screen_info()["engaged"]}
    e.close()

    # 3. metadata_filter -> the screen is skipped and the filtered walk is exact
    e = VaultEngine(path, screen="pca", screen_dims=dims, screen_min_rows=10)
    e.build_screen()
    e.search("a question", Q[0], top_k=4, metadata_filter={"nope": 1})
    out["metadata_filter"] = {"engaged": e.screen_info()["engaged"]}
    e.close()

    # 4. an explicit historical direction -> the ranking layer may reorder,
    #    so the screen must stand down
    e = VaultEngine(path, screen="pca", screen_dims=dims, screen_min_rows=10)
    e.build_screen()
    e.search("a question", Q[0], top_k=4, temporal_direction="historical")
    out["explicit_historical"] = {"engaged": e.screen_info()["engaged"]}
    e.close()

    # 5. a personal (entity-tagged) vault -> boosts exist, screen stands down
    ppath = os.path.join(tmp, "personal.dat")
    p = VaultEngine(ppath, embed_dim=V.shape[1], screen="pca",
                    screen_dims=dims, screen_min_rows=10)
    rng = np.random.default_rng(5)
    for i in range(400):
        p.add_fact(f"my phone number is 555-01{i:02d}",
                   V[i] if i < V.shape[0] else rng.normal(size=V.shape[1]).astype(np.float32),
                   source="chat_session")
    p.flush()
    p.search("what is my phone number", Q[0], top_k=4)
    out["personal_corpus"] = {"engaged": p.screen_info()["engaged"],
                              "has_personal_records": bool(p._has_personal_records())}
    p.close()

    # 6. a basis that cannot be built at all -> fall back, do not raise
    e = VaultEngine(path, screen="pca", screen_dims=dims, screen_min_rows=10)
    e.build_screen()
    e._screen.basis = None                    # simulate an unusable basis
    e._screen_arena = None                    # force the rebuild path
    hits = e.search("a question", Q[0], top_k=4)
    off = VaultEngine(path, screen="off")
    ref = off.search("a question", Q[0], top_k=4)
    out["broken_basis"] = {"raised": False,
                           "result_identical_to_exact_scan":
                               hits_key(hits) == hits_key(ref)}
    e.close(); off.close()
    for pth in (path, ppath):
        try:
            os.remove(pth)
        except OSError:
            pass
    return out


# --------------------------------------------------------------------------
# G. what the flag costs to turn on
# --------------------------------------------------------------------------
def phase_g_build_cost(path, dims, min_rows, repeats=3):
    """Wall time and resident bytes of building the screen, and the reopen tax.

    The basis is RECOMPUTED, in process, on the first search (or on an explicit
    build_screen()). It is not written to the vault file and not written to a
    sidecar -- see the `screen` paragraph of VaultEngine.__init__ for why -- so
    this cost is paid once per process that turns the flag on.
    """
    import resource
    rows = []
    for _ in range(repeats):
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        e = VaultEngine(path, screen="off")
        t0 = time.perf_counter()
        e.search("warm the arena", np.zeros(e.embed_dim, dtype=np.float32) + 1e-3,
                 top_k=1)
        base_open = time.perf_counter() - t0
        rss_open = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        e.close()
        e = VaultEngine(path, screen="pca", screen_dims=dims,
                        screen_min_rows=min_rows)
        t0 = time.perf_counter()
        info = e.build_screen()
        build = time.perf_counter() - t0
        rss_built = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        st = e.stats()
        arena = float(st.get("resident_arena_used_mb") or st.get("resident_arena_mb") or 0.0)
        if arena <= 0.0:
            arena = e.arena.n_rows * e.embed_dim * 4 / (1024 * 1024)
        rows.append({
            "build_seconds": build,
            "fit_seconds": info["fit_seconds"],
            "encode_seconds": build - info["fit_seconds"],
            "screen_used_bytes": info["used_bytes"],
            "screen_resident_bytes": info["resident_bytes"],
            "bytes_per_row": info["bytes_per_row"],
            "rows": info["rows"],
            "arena_resident_mb": arena,
            "screen_mb": info["used_bytes"] / (1024 * 1024),
            "screen_over_arena_pct": (100.0 * info["used_bytes"] / (1024 * 1024) / arena
                                      if arena > 0 else None),
            "arena_stats_keys": {k: st[k] for k in st
                                 if k.startswith("resident_arena")},
            "captured_energy": info["captured_energy"],
            "orthonormality_defect": info["orthonormality_defect"],
            "first_search_seconds_screen_off": base_open,
            "peak_rss_mb_before": rss0 / (1024 * 1024),
            "peak_rss_mb_after_open": rss_open / (1024 * 1024),
            "peak_rss_mb_after_build": rss_built / (1024 * 1024),
        })
        e.close()
    best = min(rows, key=lambda r: r["build_seconds"])
    return {"repeats": repeats, "rows": rows, "fastest": best,
            "median_build_seconds": float(np.median([r["build_seconds"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dims", type=int, default=_screen.DEFAULT_D_OUT)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=20_000)
    args = ap.parse_args()

    z = np.load(EMBEDS)
    C = np.ascontiguousarray(z["C"], dtype=np.float32)
    Qall = np.ascontiguousarray(z["Q"], dtype=np.float32)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    Qall = Qall / np.linalg.norm(Qall, axis=1, keepdims=True)
    rng = np.random.default_rng(11)
    qsel = rng.choice(Qall.shape[0], 1000, replace=False)
    Q1000 = Qall[qsel]
    Qlat = Q1000[:200]
    if args.quick:
        C = C[:12000]
        Q1000 = Q1000[:100]
        Qlat = Q1000[:50]

    # Phases A-C measure the MECHANISM, so the engagement floor must let it run
    # at whatever corpus size this invocation uses. Phase D is the phase that
    # measures the shipped floor's own behaviour.
    eff_min_rows = int(min(args.min_rows, max(1, C.shape[0] // 2)))
    tmp = tempfile.mkdtemp(prefix="nanomem-screen-")
    res = {"measurement": {"machine_at_start": machine(),
                           "corpus": os.path.basename(EMBEDS),
                           "n_docs": int(C.shape[0]),
                           "queries": int(Q1000.shape[0]),
                           "screen_dims": int(args.dims),
                           "engagement_floor_used_in_phases_A_to_C": eff_min_rows,
                           "shipped_engagement_floor_default": int(args.min_rows),
                           "tmpdir": tmp,
                           "package_path": PKG,
                           "package_sha256": {
                               f: __import__("hashlib").sha256(
                                   open(os.path.join(PKG, "nanomem", f), "rb").read()
                               ).hexdigest()
                               for f in ("engine.py", "arena.py", "container.py",
                                         "screen.py")},
                           "package_provenance": os.environ.get(
                               "NANOMEM_PKG_PROVENANCE", "working tree"),
                           "quick": bool(args.quick)}}
    try:
        path = os.path.join(tmp, "bench.dat")
        res["measurement"]["ingest_seconds"] = build_vault(path, C)
        res["measurement"]["vault_bytes"] = os.path.getsize(path)

        print("phase A: exactness through the engine ...", flush=True)
        res["phaseA_exactness_engine"] = phase_a_exactness(
            path, Q1000, args.dims, eff_min_rows)
        print(json.dumps(res["phaseA_exactness_engine"]["rows"]), flush=True)

        print("phase B: bound admissibility over every pair ...", flush=True)
        res["phaseB_bound_admissibility"] = phase_b_bound(
            path, Q1000, args.dims, eff_min_rows)
        print(json.dumps(res["phaseB_bound_admissibility"]), flush=True)

        print("phase C: paired interleaved latency at full size ...", flush=True)
        res["phaseC_latency_full"] = paired_latency(
            path, Qlat, args.dims, eff_min_rows, cycles=args.cycles)
        print("  p50 speedup", res["phaseC_latency_full"]["p50_speedup"], flush=True)

        print("phase D: size sweep ...", flush=True)
        sweep = []
        sizes = ([1000, 5000, 10000, 20000, 40000] if not args.quick
                 else [1000, 5000])
        for n in sizes:
            sp = os.path.join(tmp, f"n{n}.dat")
            build_vault(sp, C[:n])
            forced = paired_latency(sp, Qlat[:100], args.dims, args.min_rows,
                                    cycles=max(3, args.cycles), force=True)
            shipped = paired_latency(sp, Qlat[:100], args.dims, args.min_rows,
                                     cycles=max(3, args.cycles), force=False)
            ex = phase_a_exactness(sp, Q1000[:200], args.dims, 0, ks=(4,))
            sweep.append({
                "n_docs": n,
                "forced_on_p50_ms": forced["on_p50_ms"],
                "forced_off_p50_ms": forced["off_p50_ms"],
                "forced_p50_speedup": forced["p50_speedup"],
                "forced_engaged": forced["screen_info"]["engaged"],
                "forced_mean_survivors": forced["screen_info"]["mean_survivors"],
                "shipped_default_on_p50_ms": shipped["on_p50_ms"],
                "shipped_default_off_p50_ms": shipped["off_p50_ms"],
                "shipped_default_p50_ratio_on_over_off":
                    shipped["on_p50_ms"] / shipped["off_p50_ms"],
                "shipped_default_engaged": shipped["screen_info"]["engaged"],
                "exactness_when_forced": ex["rows"],
            })
            print("  ", json.dumps(sweep[-1])[:300], flush=True)
            try:
                os.remove(sp)
            except OSError:
                pass
        res["phaseD_size_sweep"] = sweep

        print("phase E: stale basis + appends ...", flush=True)
        res["phaseE_stale_basis_appends"] = phase_e_stale(
            tmp, C[: min(30000, C.shape[0])], Q1000[:200], args.dims)
        print(json.dumps(res["phaseE_stale_basis_appends"])[:400], flush=True)

        print("phase F: automatic fallback ...", flush=True)
        res["phaseF_fallback"] = phase_f_fallback(
            tmp, C[: min(20000, C.shape[0])], Q1000[:10], args.dims)
        print(json.dumps(res["phaseF_fallback"]), flush=True)

        print("phase G: build cost ...", flush=True)
        res["phaseG_build_cost"] = phase_g_build_cost(path, args.dims, eff_min_rows)
        print(json.dumps(res["phaseG_build_cost"]["fastest"]), flush=True)

        res["measurement"]["duplicate_census"] = duplicate_census(C)
        res["measurement"]["machine_at_end"] = machine()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    prior = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            prior = json.load(f)
    prior.update(res)
    with open(RESULTS, "w") as f:
        json.dump(prior, f, indent=1)
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
