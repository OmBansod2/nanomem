#!/usr/bin/env python3
"""Where a real agent turn actually spends its time -- and what that means for
whether more store optimisation is worth anything.

THE QUESTION
------------
nanomem's search is 1.1494 ms at 71,433 documents with the PCA screen on
(``competitors_standard_results.json`` / ``pca_screen_results.json``).  The
embedding call in FRONT of it has been assumed to be "roughly 9 ms" since
DECISIONS #2 and has never been measured in this project.  If the embedder
dominates, every millisecond of index work is invisible to a user and the
project's priorities should move.  Nobody has measured an end-to-end turn here,
so this file does.

WHAT IT MEASURES
----------------
A turn is decomposed into the stages a caller actually pays for:

  (a) embed_query      one real Ollama ``nomic-embed-text`` call on the real
                       question text (NOT a cached vector);
  (b) search_scan      ``VaultEngine._candidates`` -- the exact cosine scan, or
                       the PCA screen + gather when the screen is engaged;
  (c) rank             everything ``search()`` does between the scan and the
                       hits: ``_columns``, the entity boosts, ``_resolve_revisions``
                       and ``_select_top_k``;
  (c') materialise     ``_row_record`` per returned hit -- the block read and
                       text decode that turns row ids into dicts;
  (d) write path       ``WriteClassifier.classify`` (both the cheap caller-vector
                       form and the naive form that dials the embedder itself),
                       the storage embedding, ``add_fact``, and ``flush``.

and then the SHAPES an agent really produces: 1 lookup per turn vs 4, a cold
embedder vs a warm one, 4 embeddings one at a time vs one batched call, and a
read-only turn vs a turn that also writes.

HEADROOM, which is the actual deliverable: per-turn paired arithmetic for
"search is instant", "screen off -> on", "arena cache off -> on", "the embedder
gets 2x faster", and an embedding LRU, ranked by the milliseconds each removes
from the median turn.

METHOD RULES THIS FILE FOLLOWS
------------------------------
1. PRE-REGISTRATION.  ``prereg()`` is written to the results JSON before a
   single number is measured, and it names the baseline, the metric, the corpora,
   the splits, the seeds and the exact decision bar.  ``--prereg`` writes it and
   exits; every later phase refuses to run if the file's prereg block does not
   match the one compiled into this module (``_assert_prereg_intact``).
2. THIS IS A LATENCY DELIVERABLE, NOT A RECALL ONE.  No arm here can change an
   answer, so a recall CI would be the wrong instrument and is not used.  The
   pre-registered criterion is a SHARE-OF-TURN threshold with a paired bootstrap
   CI on the p50, fixed up front in ``prereg()``.  Not one bar in this file was
   written after seeing a number.
3. EVERY ABLATION SHIPS A MECHANICAL LEAK AUDIT.  The screen-on / screen-off arms
   must return byte-identical top-k id lists (``audit_screen_answer_identity``),
   the "instant search" arm may only subtract components measured on the SAME
   turn (``audit_pairing``), and the embedding-cache analysis may only match a
   query against a STRICTLY earlier query in the SAME session
   (``audit_cache_no_self_match``).  Each counts per arm and RAISES on non-zero.
4. A CLEAN NEGATIVE IS A DELIVERABLE.  Every arm is reported, including the ones
   that turn out to buy nothing.

QUARANTINE
----------
This module reads ``chat_bench_v2.json``, ``temporal_bench.json``,
``hotpot_train_8k.json``, ``hotpot_train_8k_embeds.npz``, ``embeds_val.npz`` and
``../hotpotqa_scaled_1k.json``.  The held-out fixtures are not named anywhere in
this file, not opened, and not globbed: ``DATA_FILES`` below is the complete,
literal list of inputs and ``audit_inputs_are_declared`` re-checks at run time
that nothing outside it was opened, by construction rather than by atime (atime
is unreliable on this filesystem).

Usage
-----
    bench_turn_latency.py --prereg          # write the pre-registration, exit
    bench_turn_latency.py --phase A         # one phase (A..G)
    bench_turn_latency.py                   # prereg (if absent) + every phase
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import http.client
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = "<redacted local path> /4D llm"
REFOUND = os.path.join(REPO, "scratch", "refound")
# WHICH nanomem is under the clock.  The repository working tree is shared with
# other work in this session, so the package being measured is pinned by env var
# to a tree extracted from a named commit, and the exact bytes are hashed into
# the results (`engine_provenance`).  A benchmark that cannot say which build it
# timed is a benchmark of nothing.
NANOMEM_DIR = os.environ.get("TURN_BENCH_NANOMEM_DIR",
                             os.path.join(REPO, "nanomem_standalone"))
RESULTS = os.path.join(REFOUND, "turn_latency_results.json")
WORK = os.environ.get(
    "TURN_BENCH_WORK",
    "/private/tmp/claude-501/-Users-om-Desktop-Om--4D-llm/"
    "7dcdd445-bc6a-41e5-8566-b87ad8aff67b/scratchpad/turnbench")

sys.path.insert(0, NANOMEM_DIR)

# The COMPLETE list of inputs this file reads.  Nothing else is opened; the
# quarantined fixtures are absent by construction, not by filtering.
DATA_FILES = {
    "hotpot_train_8k": os.path.join(REFOUND, "hotpot_train_8k.json"),
    "hotpot_train_8k_embeds": os.path.join(REFOUND, "hotpot_train_8k_embeds.npz"),
    "hotpotqa_scaled_1k": os.path.join(REPO, "scratch", "hotpotqa_scaled_1k.json"),
    "embeds_val": os.path.join(REFOUND, "embeds_val.npz"),
    "chat_bench_v2": os.path.join(REFOUND, "chat_bench_v2.json"),
    "temporal_bench": os.path.join(REFOUND, "temporal_bench.json"),
}

DIM = 768
TOP_K = 4                    # the project's headline metric is recall@4
SEED = 0
N_TURNS = 240                # >= 200 timed turns per configuration
N_WARMUP = 20
EMBED_MODEL = "nomic-embed-text"
OLLAMA_HOST, OLLAMA_PORT = "localhost", 11434
CORPORA = ("val1190", "n10000", "n71433")
BOOT_ITERS = 2000


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def pct(a: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def summ(a: Sequence[float], nd: int = 4) -> Dict[str, Any]:
    x = np.asarray(a, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"p50": round(float(np.percentile(x, 50)), nd),
            "p95": round(float(np.percentile(x, 95)), nd),
            "p99": round(float(np.percentile(x, 99)), nd),
            "mean": round(float(x.mean()), nd),
            "min": round(float(x.min()), nd),
            "max": round(float(x.max()), nd),
            "n": int(x.size)}


def l2norm(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return np.ascontiguousarray(a / n, dtype=np.float32)


def loadavg() -> Dict[str, float]:
    one, five, fifteen = os.getloadavg()
    return {"1min": round(one, 2), "5min": round(five, 2), "15min": round(fifteen, 2)}


def maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    b = int(r) if sys.platform == "darwin" else int(r) * 1024
    return round(b / 1048576.0, 1)


def boot_p50_ci(x: Sequence[float], seed: int = SEED, iters: int = BOOT_ITERS) -> List[float]:
    """Bootstrap CI on the p50 of one sample."""
    a = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(iters, a.size))
    med = np.median(a[idx], axis=1)
    return [round(float(np.percentile(med, 2.5)), 4), round(float(np.percentile(med, 97.5)), 4)]


def paired_boot_delta_ci(a: Sequence[float], b: Sequence[float], seed: int = SEED,
                         iters: int = BOOT_ITERS) -> Dict[str, Any]:
    """PAIRED bootstrap on (p50(a) - p50(b)) and p50(a)/p50(b).

    ``a`` and ``b`` are two arms measured on the SAME turns in the same order, so
    the resample index is shared -- which is what makes the ratio robust to the
    machine's weather drifting under the run.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    assert x.size == y.size and x.size > 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(iters, x.size))
    mx = np.median(x[idx], axis=1)
    my = np.median(y[idx], axis=1)
    d = mx - my
    r = mx / np.maximum(my, 1e-12)
    return {
        "p50_a": round(float(np.median(x)), 4),
        "p50_b": round(float(np.median(y)), 4),
        "delta_p50": round(float(np.median(x) - np.median(y)), 4),
        "delta_ci95": [round(float(np.percentile(d, 2.5)), 4),
                       round(float(np.percentile(d, 97.5)), 4)],
        "ratio_p50": round(float(np.median(x) / max(np.median(y), 1e-12)), 4),
        "ratio_ci95": [round(float(np.percentile(r, 2.5)), 4),
                       round(float(np.percentile(r, 97.5)), 4)],
        "n_pairs": int(x.size),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(100.0 * max(0.0, c - h), 2), round(100.0 * min(1.0, c + h), 2)]


def path_bytes(p: str) -> int:
    if os.path.isfile(p):
        return os.path.getsize(p)
    t = 0
    for root, _d, files in os.walk(p):
        for f in files:
            try:
                t += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return t


def load_results() -> Dict[str, Any]:
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            return json.load(fh)
    return {}


def save_results(d: Dict[str, Any]) -> None:
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=2, sort_keys=False)
    os.replace(tmp, RESULTS)


# ---------------------------------------------------------------------------
# the embedder under test
# ---------------------------------------------------------------------------
class OllamaEmbedder:
    """The real thing, with the two dialling strategies kept separate.

    ``nanomem.embed.EmbeddingProvider`` opens a NEW HTTP connection per call
    (``urllib.request.urlopen``).  That is the shipped path and is what
    ``mode="urllib"`` measures.  ``mode="keepalive"`` reuses one
    ``http.client.HTTPConnection`` and is a lever, not the baseline: it is a
    four-line change to ``nanomem/embed.py`` that this file does not make.
    """

    def __init__(self, mode: str = "urllib", model: str = EMBED_MODEL):
        self.mode = mode
        self.model = model
        self.dim = DIM
        self._conn: Optional[http.client.HTTPConnection] = None
        self.calls = 0

    def _keepalive_conn(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=30)
        return self._conn

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        self.calls += 1
        if self.mode == "keepalive":
            for attempt in (0, 1):
                try:
                    c = self._keepalive_conn()
                    c.request("POST", "/api/embed", body=body,
                              headers={"Content-Type": "application/json"})
                    data = json.loads(c.getresponse().read().decode("utf-8"))
                    break
                except Exception:
                    try:
                        if self._conn is not None:
                            self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    if attempt:
                        raise
        else:
            import urllib.request
            req = urllib.request.Request(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embed", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        vecs = np.asarray(data["embeddings"], dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise RuntimeError(f"embedder returned {vecs.shape}, expected (*, {self.dim})")
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        return vecs / n

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def assert_real_embedder(emb: OllamaEmbedder) -> Dict[str, Any]:
    """Refuse to measure anything if Ollama is not actually answering.

    ``EmbeddingProvider`` swallows a dead daemon and returns md5 n-gram vectors
    at 0.05 ms, which would make this whole file read as "embedding is free".
    This checks the daemon by hand and cross-checks the shipped provider against
    it, so a silent fallback cannot be mistaken for a fast embedder.
    """
    from nanomem.embed import EmbeddingProvider
    probe = "nanomem turn latency embedder provenance probe"
    v = emb.embed(probe)
    p = EmbeddingProvider(model=EMBED_MODEL)
    off = p._offline_encode_batch([probe])[0]
    shipped = np.asarray(p.embed(probe), dtype=np.float32)
    cos_off = float(np.dot(off, v))
    cos_shipped = float(np.dot(shipped, v))
    if cos_off > 0.999:
        raise RuntimeError("the embedder is nanomem's md5 OFFLINE FALLBACK, not Ollama; "
                           "start the daemon before measuring turn latency")
    if cos_shipped < 0.999:
        raise RuntimeError(f"this file's embedder disagrees with nanomem.embed "
                           f"(cos={cos_shipped:.5f}); the measurement would not be "
                           f"about the shipped path")
    try:
        import urllib.request
        with urllib.request.urlopen(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=5) as r:
            tags = json.loads(r.read().decode("utf-8"))
        info = next((m for m in tags.get("models", [])
                     if m.get("name", "").startswith(EMBED_MODEL)), {})
    except Exception:
        info = {}
    return {"model": EMBED_MODEL, "dim": int(v.size),
            "cos_vs_nanomem_embed_provider": round(cos_shipped, 6),
            "cos_vs_offline_md5_fallback": round(cos_off, 6),
            "is_real_neural_embedder": True,
            "ollama_model_digest": info.get("digest", "")[:12],
            "ollama_model_size_bytes": info.get("size"),
            "quantization": (info.get("details") or {}).get("quantization_level"),
            "parameter_size": (info.get("details") or {}).get("parameter_size")}


# ---------------------------------------------------------------------------
# PRE-REGISTRATION -- written before a single number is measured
# ---------------------------------------------------------------------------
def prereg() -> Dict[str, Any]:
    return {
        "written_before_any_measurement": True,
        "question": (
            "Where does an agent turn spend its time, and is further store "
            "optimisation worth anything to a user?"),
        "baseline": {
            "engine": "nanomem VaultEngine at git 88dfac9, nanomem 0.4.0",
            "shipped_defaults": {"screen": "off", "arena_cache": "map",
                                 "router": "off", "vector_dtype": "float16",
                                 "residency": "float32", "durable": "fsync",
                                 "block_capacity": 50},
            "embedder": "Ollama nomic-embed-text, one real HTTP call per query, "
                        "dialled exactly as nanomem.embed.EmbeddingProvider does "
                        "it (a NEW urllib connection per call)",
            "turn_definition": (
                "PRIMARY SHAPE: a read turn -- embed the question, search(top_k=4), "
                "rank, materialise the hits. SECONDARY SHAPES: 4 lookups per turn; "
                "a turn that also writes (classify + storage embedding + add_fact "
                "+ flush); batched vs one-at-a-time embedding; a cold embedder."),
        },
        "metric": {
            "primary": "median (p50) wall-clock milliseconds per stage and per turn, "
                       "and each stage's SHARE of the median turn",
            "dispersion": "p95 and p99 reported for every stage",
            "uncertainty": "paired bootstrap (shared resample index, 2000 iters) on "
                           "the p50 difference and ratio between two arms measured "
                           "on the SAME turns; Wilson interval for the cache hit rate",
            "why_not_recall": (
                "No arm in this file can change an answer -- the screen is exact by "
                "proof and the other levers do not touch ranking -- so a recall CI "
                "would measure nothing. Per method rule 2 the criterion is fixed "
                "here, before the run, as a share-of-turn threshold."),
        },
        "corpora": {
            "val1190": "1,190 HotpotQA paragraphs (scratch/hotpotqa_scaled_1k.json), "
                       "vectors from embeds_val.npz",
            "n10000": "first 10,000 paragraphs of hotpot_train_8k.json",
            "n71433": "all 71,433 paragraphs of hotpot_train_8k.json",
            "insertion_order": "rng(0).permutation(N), materialised before the clock "
                               "starts -- the same rule as bench_competitors.py, so "
                               "the vaults are the ones the published p50s came from",
            "source_tag": "wiki (NOT in entities.PERSONAL_SOURCES) for the document "
                          "corpora, so the entity/temporal layer is inert -- the "
                          "personal-vault shape is measured separately in phase D",
        },
        "splits": {
            "dev": "the first 200 eligible questions of each corpus, used to settle "
                   "harness parameters (warmup count, turn count, interleave order) "
                   "and never reported as a headline number",
            "test": "the NEXT 240 eligible questions, disjoint from dev, scored ONCE; "
                    "val1190 has only 120 questions in total, so its 240 turns are "
                    "2 passes over the same 120 texts and the file says so",
            "chat_cache_analysis": "dev personas p01-p04 of chat_bench_v2 for the "
                                   "sanity pass; test personas p05-p14 scored once. "
                                   "The near-duplicate threshold is NOT tuned: cos > "
                                   "0.98 is fixed by the task statement.",
        },
        "seeds": {"numpy": SEED, "bootstrap": SEED, "iters": BOOT_ITERS},
        "decision_bar": {
            "BAR_1_primary": (
                "Configuration: 71,433 documents, shipped defaults, warm embedder, "
                "the PRIMARY read-turn shape (1 lookup), which is the shape most "
                "FAVOURABLE to the store (no second embedding call, no write work). "
                "Statistic: search_instant_headroom_pct = 100 * (p50 turn - p50 turn "
                "with the scan+rank+materialise stages removed) / p50 turn, computed "
                "by per-turn paired subtraction. "
                "IF search_instant_headroom_pct < 10.0, ACCEPT 'further store "
                "optimisation has near-zero user-visible payoff'. "
                "IF >= 10.0, REJECT it and report how much is really there."),
            "BAR_2_lever_ranking": (
                "Levers are ranked by the milliseconds each removes from the p50 "
                "turn in the primary configuration. A lever outranks another only "
                "if the paired bootstrap CI on its p50 reduction excludes the "
                "other's point estimate; otherwise they are reported as tied."),
            "BAR_3_cache": (
                "The embedding LRU is worth more than every index optimisation "
                "shipped so far IF its expected p50 turn reduction (session hit rate "
                "x measured embed p50) exceeds the measured p50 turn reduction of "
                "the PCA screen at 71,433 documents. Hit rate is scored on the TEST "
                "personas once, with a Wilson interval."),
            "written_when": "before any measurement; not restated afterwards",
        },
        "leak_audits": {
            "audit_screen_answer_identity": "the screen-on and screen-off arms must "
                "return identical ordered top-k id lists on every turn; counts "
                "mismatches per corpus and RAISES if non-zero",
            "audit_pairing": "the 'instant search' arm subtracts only components "
                "measured on the same turn index in the same arm; the harness "
                "re-checks that every per-turn stage vector has identical length and "
                "that stage sums reconcile with the measured turn total to within "
                "the recorded residual, and RAISES if the vectors are misaligned",
            "audit_cache_no_self_match": "a query may only match a STRICTLY earlier "
                "query in the SAME session; counts self-matches and cross-session "
                "matches and RAISES if non-zero",
            "audit_inputs_are_declared": "every file opened by this module is in "
                "DATA_FILES; the quarantined fixtures are not named in this file at "
                "all, so non-access is by construction, not by atime",
        },
        "instrumentation_honesty": (
            "Stage decomposition inside search() is done by SUBCLASSING VaultEngine "
            "in this file (TimedEngine) -- nanomem/ is not edited. The subclass adds "
            "two time.perf_counter pairs per search, and the harness measures the "
            "same queries through an UNINSTRUMENTED engine as well and reports the "
            "overhead so the decomposition can be discounted."),
        "clean_negative_policy": "every arm is reported, including the ones that buy "
                                 "nothing; no arm is dropped after the fact",
    }


PREREG_HASH = hashlib.sha256(
    json.dumps(prereg(), sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _assert_prereg_intact(res: Dict[str, Any]) -> None:
    got = res.get("prereg_sha256_16")
    if got and got != PREREG_HASH:
        raise RuntimeError(
            f"the pre-registration in {RESULTS} ({got}) is not the one compiled into "
            f"this module ({PREREG_HASH}). The bar may not move mid-run; delete the "
            f"results file and re-register if the design genuinely changed.")


def engine_provenance() -> Dict[str, Any]:
    """sha256 of every module of the nanomem package actually imported."""
    import nanomem
    pkg = os.path.dirname(os.path.abspath(nanomem.__file__))
    files = {}
    for f in sorted(os.listdir(pkg)):
        if f.endswith(".py"):
            with open(os.path.join(pkg, f), "rb") as fh:
                files[f] = hashlib.sha256(fh.read()).hexdigest()[:12]
    repo_pkg = os.path.join(REPO, "nanomem_standalone", "nanomem")
    differs = []
    for f, h in files.items():
        rp = os.path.join(repo_pkg, f)
        try:
            with open(rp, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest()[:12] != h:
                    differs.append(f)
        except OSError:
            differs.append(f + " (absent from the working tree)")
    return {
        "package_dir": pkg,
        "version": nanomem.__version__,
        "engine_version": getattr(nanomem, "ENGINE_VERSION", None),
        "module_sha256_12": files,
        "modules_differing_from_the_repo_working_tree": sorted(differs),
        "note": ("the working tree at scratch time also carried another author's "
                 "in-progress edits to arena.py / engine.py / container.py; this run "
                 "imports the pinned tree above, and the list names every module "
                 "where the two differ"),
    }


def env_block() -> Dict[str, Any]:
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        brand = platform.processor()
    try:
        ncpu = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip())
    except Exception:
        ncpu = os.cpu_count() or 0
    return {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "cpu": brand, "ncpu": ncpu,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "load_average": loadavg(),
        "git_head": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
    }


# ---------------------------------------------------------------------------
# corpora and vaults
# ---------------------------------------------------------------------------
class Corpus:
    """Vectors, texts and question TEXT for one corpus size.

    Document vectors are the cached nomic vectors (identical to the ones every
    published number in this project came from).  Question vectors are NOT
    loaded: the whole point is to pay the real embedding cost per turn.
    """

    def __init__(self, name: str):
        self.name = name
        if name == "val1190":
            d = json.load(open(DATA_FILES["hotpotqa_scaled_1k"]))
            self.texts = [c["text"] for c in d["corpus"]]
            z = np.load(DATA_FILES["embeds_val"])
            D = l2norm(z["D"])
            qtexts = [q["question"] for q in d["questions"]]
            self.dev_q = qtexts[:0]            # only 120 questions exist in total
            self.test_q = qtexts               # 120 texts, run for 2 passes
            self.passes = 2
            self.note = ("only 120 questions exist for this corpus, so the 240 timed "
                         "turns are 2 passes over the same 120 question texts; each "
                         "pass still pays a real embedding call")
        else:
            N = 10000 if name == "n10000" else None
            z = np.load(DATA_FILES["hotpot_train_8k_embeds"])
            C, G = z["C"], z["G"]
            src = json.load(open(DATA_FILES["hotpot_train_8k"]))
            all_texts = [c["text"] for c in src["corpus"]]
            N = int(C.shape[0]) if N is None else N
            self.texts = all_texts[:N]
            D = l2norm(C[:N])
            qsel = [qi for qi in range(G.shape[0]) if int(G[qi].max()) < N]
            qtexts = [src["questions"][qi]["question"] for qi in qsel]
            self.dev_q = qtexts[:200]
            self.test_q = qtexts[200:200 + N_TURNS]
            self.passes = 1
            self.note = ("questions restricted to those whose gold paragraphs are "
                         "inside the corpus -- the same rule as bench_competitors.py; "
                         "dev = first 200, test = the next 240, disjoint")
        self.n = int(D.shape[0])
        order = np.random.default_rng(SEED).permutation(self.n)
        self.Dp = np.ascontiguousarray(D[order])
        self.Tp = [self.texts[int(i)] for i in order]
        self.ids = [int(i) for i in order]
        del D
        gc.collect()

    def turns(self) -> List[str]:
        out: List[str] = []
        for _ in range(self.passes):
            out.extend(self.test_q)
        return out[:N_TURNS] if len(out) >= N_TURNS else out


def vault_path(corpus: str, tag: str = "read") -> str:
    return os.path.join(WORK, f"{corpus}_{tag}.dat")


def build_vault(c: Corpus, tag: str = "read", source: str = "wiki") -> Dict[str, Any]:
    """Ingest the corpus once; reuse the file on later runs."""
    from nanomem.engine import VaultEngine
    os.makedirs(WORK, exist_ok=True)
    path = vault_path(c.name, tag)
    stamp = path + ".built.json"
    if os.path.exists(path) and os.path.exists(stamp):
        with open(stamp) as fh:
            prev = json.load(fh)
        if prev.get("n") == c.n and prev.get("source") == source:
            return prev
    for p in (path, path + ".arena", stamp):
        if os.path.exists(p):
            os.remove(p)
    t0 = time.perf_counter()
    e = VaultEngine(path, embed_dim=DIM, vector_dtype="float16", router="off")
    e.reserve_rows(c.n)
    for j in range(c.n):
        e.add_fact(c.Tp[j], c.Dp[j], source=source, metadata={"idx": c.ids[j]})
    e.flush()
    ingest_s = time.perf_counter() - t0
    st = e.stats()
    e.close()
    gc.collect()
    info = {"n": c.n, "source": source, "path": path,
            "ingest_s": round(ingest_s, 3),
            "vault_bytes": path_bytes(path),
            "arena_sidecar_bytes": path_bytes(path + ".arena"),
            "engine_version": st.get("engine_version")}
    with open(stamp, "w") as fh:
        json.dump(info, fh, indent=2)
    return info


# ---------------------------------------------------------------------------
# stage instrumentation -- a SUBCLASS, so nanomem/ is untouched
# ---------------------------------------------------------------------------
def timed_engine_class():
    from nanomem.engine import VaultEngine

    class TimedEngine(VaultEngine):
        """VaultEngine with two stopwatches and no behaviour change.

        ``_candidates`` is the scan (or the PCA screen plus its gather);
        ``_row_record`` is the per-hit block read and text decode.  Everything
        else search() does -- ``_columns``, the boosts, ``_resolve_revisions``,
        ``_select_top_k``, the dict building and the lock -- falls out as
        ``total - scan - materialise`` and is reported as ``rank``.
        """

        def __init__(self, *a, **k):
            self.reset_timers()
            super().__init__(*a, **k)

        def reset_timers(self) -> None:
            self.t_scan = 0.0
            self.t_rowrec = 0.0
            self.t_columns = 0.0
            self.t_revisions = 0.0
            self.t_intent = 0.0
            self.n_rowrec = 0

        def _candidates(self, *a, **k):
            t = time.perf_counter()
            r = super()._candidates(*a, **k)
            self.t_scan += time.perf_counter() - t
            return r

        def _row_record(self, row):
            t = time.perf_counter()
            r = super()._row_record(row)
            self.t_rowrec += time.perf_counter() - t
            self.n_rowrec += 1
            return r

        # -- the finer breakdown of the rank stage -----------------------
        def _columns(self, rows):
            t = time.perf_counter()
            r = super()._columns(rows)
            self.t_columns += time.perf_counter() - t
            return r

        def _resolve_revisions(self, *a, **k):
            t = time.perf_counter()
            r = super()._resolve_revisions(*a, **k)
            self.t_revisions += time.perf_counter() - t
            return r

        def _resolve_intent(self, *a, **k):
            t = time.perf_counter()
            r = super()._resolve_intent(*a, **k)
            self.t_intent += time.perf_counter() - t
            return r

    return TimedEngine


def boot_headroom_pct(turn: Sequence[float], reduced: Sequence[float],
                      seed: int = SEED, iters: int = BOOT_ITERS) -> Dict[str, Any]:
    """100 * (p50(turn) - p50(reduced)) / p50(turn), paired.

    ``reduced[i]`` must be the SAME turn i with one stage removed, so the
    resample index is shared between the two vectors.
    """
    a = np.asarray(turn, dtype=np.float64)
    b = np.asarray(reduced, dtype=np.float64)
    assert a.size == b.size and a.size > 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(iters, a.size))
    ma = np.median(a[idx], axis=1)
    mb = np.median(b[idx], axis=1)
    h = 100.0 * (ma - mb) / np.maximum(ma, 1e-12)
    p50a, p50b = float(np.median(a)), float(np.median(b))
    return {"p50_turn_ms": round(p50a, 4), "p50_turn_without_ms": round(p50b, 4),
            "removed_ms": round(p50a - p50b, 4),
            "headroom_pct": round(100.0 * (p50a - p50b) / max(p50a, 1e-12), 2),
            "headroom_pct_ci95": [round(float(np.percentile(h, 2.5)), 2),
                                  round(float(np.percentile(h, 97.5)), 2)],
            "n_pairs": int(a.size)}


# ---------------------------------------------------------------------------
# PHASE A -- the stage breakdown of a read turn
# ---------------------------------------------------------------------------
def phase_A(corpus_name: str) -> Dict[str, Any]:
    from nanomem.engine import VaultEngine
    TE = timed_engine_class()

    c = Corpus(corpus_name)
    vinfo = build_vault(c, tag="read", source="wiki")
    path = vinfo["path"]

    emb = OllamaEmbedder("urllib")
    prov = assert_real_embedder(emb)

    eng_on_t = TE(path, embed_dim=DIM, screen="pca")
    eng_off_t = TE(path, embed_dim=DIM, screen="off")
    eng_on_p = VaultEngine(path, embed_dim=DIM, screen="pca")
    eng_off_p = VaultEngine(path, embed_dim=DIM, screen="off")
    built_t = eng_on_t.build_screen()
    built_p = eng_on_p.build_screen()

    warm_src = c.dev_q if c.dev_q else c.test_q
    load_start = loadavg()
    for i in range(min(N_WARMUP, len(warm_src))):
        v = emb.embed(warm_src[i])
        for e in (eng_on_t, eng_off_t, eng_on_p, eng_off_p):
            e.search("", v, top_k=TOP_K)

    turns = c.turns()
    arms = ("on_t", "off_t", "on_p", "off_p")
    engines = {"on_t": eng_on_t, "off_t": eng_off_t, "on_p": eng_on_p, "off_p": eng_off_p}
    per: Dict[str, List[float]] = {f"{a}_total": [] for a in arms}
    per.update({"embed": [], "on_t_scan": [], "on_t_rowrec": [], "on_t_rank": [],
                "off_t_scan": [], "off_t_rowrec": [], "off_t_rank": [],
                "on_t_columns": [], "on_t_revisions": [], "on_t_rankother": [],
                "off_t_columns": [], "off_t_revisions": [], "off_t_rankother": []})
    ids: Dict[str, List[Tuple[str, ...]]] = {a: [] for a in arms}

    for i, qt in enumerate(turns):
        t0 = time.perf_counter()
        v = emb.embed(qt)
        per["embed"].append((time.perf_counter() - t0) * 1000.0)
        rot = i % len(arms)
        order = arms[rot:] + arms[:rot]
        for a in order:
            e = engines[a]
            timed = a.endswith("_t")
            if timed:
                e.reset_timers()
            t1 = time.perf_counter()
            hits = e.search("", v, top_k=TOP_K)
            tot = (time.perf_counter() - t1) * 1000.0
            per[f"{a}_total"].append(tot)
            ids[a].append(tuple(h["id"] for h in hits))
            if timed:
                scan = e.t_scan * 1000.0
                rowrec = e.t_rowrec * 1000.0
                cols = e.t_columns * 1000.0
                revs = e.t_revisions * 1000.0
                per[f"{a}_scan"].append(scan)
                per[f"{a}_rowrec"].append(rowrec)
                per[f"{a}_rank"].append(tot - scan - rowrec)
                per[f"{a}_columns"].append(cols)
                per[f"{a}_revisions"].append(revs)
                per[f"{a}_rankother"].append(tot - scan - rowrec - cols - revs)
    load_end = loadavg()

    # ---- leak audit: the screen may not change an answer ------------------
    audit = audit_screen_answer_identity(corpus_name, ids)

    # ---- turn arithmetic (paired, per turn) -------------------------------
    E = np.asarray(per["embed"])
    out: Dict[str, Any] = {
        "corpus": corpus_name, "n_docs": c.n, "n_turns": len(turns),
        "corpus_note": c.note,
        "vault": {k: vinfo[k] for k in
                  ("ingest_s", "vault_bytes", "arena_sidecar_bytes", "engine_version")},
        "embedder_provenance": prov,
        "load_average_start": load_start, "load_average_end": load_end,
        "screen_built": {"timed_engine": built_t, "plain_engine": built_p},
        "screen_info_after_run": {
            "on_t": eng_on_t.screen_info(), "on_p": eng_on_p.screen_info()},
        "stages_ms": {
            "embed_query": summ(per["embed"]),
            "search_total_screen_off": summ(per["off_p_total"]),
            "search_total_screen_on": summ(per["on_p_total"]),
            "search_total_screen_off_instrumented": summ(per["off_t_total"]),
            "search_total_screen_on_instrumented": summ(per["on_t_total"]),
            "scan_screen_off": summ(per["off_t_scan"]),
            "rank_screen_off": summ(per["off_t_rank"]),
            "materialise_screen_off": summ(per["off_t_rowrec"]),
            "scan_screen_on": summ(per["on_t_scan"]),
            "rank_screen_on": summ(per["on_t_rank"]),
            "materialise_screen_on": summ(per["on_t_rowrec"]),
        },
        "rank_breakdown_ms": {
            "screen_off": {
                "columns_entity_revision_timestamp": summ(per["off_t_columns"]),
                "resolve_revisions": summ(per["off_t_revisions"]),
                "boosts_select_and_dicts": summ(per["off_t_rankother"])},
            "screen_on": {
                "columns_entity_revision_timestamp": summ(per["on_t_columns"]),
                "resolve_revisions": summ(per["on_t_revisions"]),
                "boosts_select_and_dicts": summ(per["on_t_rankother"])},
            "note": "on a DOCUMENT corpus the entity/temporal layer is inert -- "
                    "_has_personal_records() is false, _max_boost() is 0.0 and "
                    "_resolve_revisions returns its input untouched -- so whatever "
                    "this costs is the price of the GATE, paid over every candidate",
        },
        "instrumentation_overhead_ms": {
            "screen_off": round(pct(per["off_t_total"], 50) - pct(per["off_p_total"], 50), 4),
            "screen_on": round(pct(per["on_t_total"], 50) - pct(per["on_p_total"], 50), 4),
            "note": "the decomposition is measured on the *_t engines and the turn "
                    "totals on the *_p (uninstrumented) ones; this is the price of "
                    "the two perf_counter pairs and it is not charged to any turn",
        },
        "leak_audit": audit,
        "hits_materialised_per_search": TOP_K,
    }

    for tag, key in (("shipped_default_screen_off", "off_p_total"),
                     ("screen_on", "on_p_total")):
        S = np.asarray(per[key])
        turn = E + S
        out[f"turn_{tag}"] = {
            "turn_ms": summ(turn),
            "turn_p50_ci95": boot_p50_ci(turn),
            "share_of_p50_turn_pct": {
                "embed_query": round(100.0 * pct(E, 50) / pct(turn, 50), 2),
                "search_all_stages": round(100.0 * pct(S, 50) / pct(turn, 50), 2),
            },
            "search_instant_headroom": boot_headroom_pct(turn, E),
        }
    # stage-level shares using the instrumented decomposition, scaled so the
    # decomposition sums to the UNINSTRUMENTED search total (the overhead is not
    # charged to any stage).
    for tag, tkey, pkey in (("shipped_default_screen_off", "off_t", "off_p"),
                            ("screen_on", "on_t", "on_p")):
        s50 = pct(per[f"{tkey}_scan"], 50)
        r50 = pct(per[f"{tkey}_rank"], 50)
        m50 = pct(per[f"{tkey}_rowrec"], 50)
        tot_t = s50 + r50 + m50
        tot_p = pct(per[f"{pkey}_total"], 50)
        scale = tot_p / max(tot_t, 1e-12)
        turn50 = pct(E, 50) + tot_p
        out[f"turn_{tag}"]["stage_share_of_p50_turn_pct"] = {
            "embed_query": round(100.0 * pct(E, 50) / turn50, 2),
            "search_scan": round(100.0 * s50 * scale / turn50, 2),
            "rank_entity_temporal": round(100.0 * r50 * scale / turn50, 2),
            "materialise_hits": round(100.0 * m50 * scale / turn50, 2),
            "_decomposition_rescaled_by": round(scale, 4),
        }

    out["_per_turn"] = {k: [round(x, 5) for x in v] for k, v in per.items()}
    for e in (eng_on_t, eng_off_t, eng_on_p, eng_off_p):
        e.close()
    emb.close()
    gc.collect()
    return out


def audit_screen_answer_identity(corpus: str, ids: Dict[str, List[Tuple[str, ...]]]
                                 ) -> Dict[str, Any]:
    """The screen is exact by proof; this checks it mechanically, per arm.

    Any mismatch would mean a LATENCY arm had changed an ANSWER, which would make
    every share-of-turn number in this file a comparison between two different
    products.  Counts per arm and raises on non-zero.
    """
    n = len(ids["on_p"])
    counts = {
        "screen_on_vs_off_plain": sum(1 for i in range(n) if ids["on_p"][i] != ids["off_p"][i]),
        "screen_on_vs_off_timed": sum(1 for i in range(n) if ids["on_t"][i] != ids["off_t"][i]),
        "timed_vs_plain_screen_off": sum(1 for i in range(n) if ids["off_t"][i] != ids["off_p"][i]),
        "timed_vs_plain_screen_on": sum(1 for i in range(n) if ids["on_t"][i] != ids["on_p"][i]),
    }
    bad = {k: v for k, v in counts.items() if v}
    if bad:
        raise AssertionError(
            f"[{corpus}] leak audit FAILED: a latency arm changed an answer {bad}")
    return {"n_turns_checked": n, "mismatches_per_arm": counts, "raised": False,
            "meaning": "every arm returned the identical ordered top-4 id list on "
                       "every turn, so the latency arms are comparing one product"}


# ---------------------------------------------------------------------------
# PHASE B -- the write path
# ---------------------------------------------------------------------------
def chat_turns(split: str = "dev") -> List[Dict[str, Any]]:
    d = json.load(open(DATA_FILES["chat_bench_v2"]))
    out = []
    for s in d["sets"]:
        if s["split"] != split:
            continue
        for t in s["turns"]:
            out.append({"persona": s["persona"], "text": t["text"],
                        "gold_should_store": bool(t["should_store"])})
    return out


def phase_B(corpus_name: str, n_turns: int = 600) -> Dict[str, Any]:
    """What a turn pays to WRITE, on a vault of the given size.

    Runs on a COPY of the read vault so the read arms in phase A keep a corpus
    with no personal records in it (adding chat records flips
    ``_has_personal_records`` and with it the ranking gate -- which is phase D's
    subject, not this one's).
    """
    from nanomem.engine import VaultEngine
    from nanomem.classifier import WriteClassifier

    c = Corpus(corpus_name)
    vinfo = build_vault(c, tag="read", source="wiki")
    src = vinfo["path"]
    dst = vault_path(corpus_name, "write")
    for suffix in ("", ".arena"):
        if os.path.exists(dst + suffix):
            os.remove(dst + suffix)
    shutil.copy2(src, dst)
    if os.path.exists(src + ".arena"):
        shutil.copy2(src + ".arena", dst + ".arena")

    emb = OllamaEmbedder("urllib")
    assert_real_embedder(emb)
    clf = WriteClassifier()
    e = VaultEngine(dst, embed_dim=DIM, screen="off")

    turns = chat_turns("dev")[:n_turns]
    load_start = loadavg()
    # warm the classifier's weights, the embedder and the daemon
    for t in turns[:N_WARMUP]:
        clf.classify(t["text"], embedding=emb.embed(t["text"]))

    per: Dict[str, List[float]] = {"embed_utterance": [], "classify_caller_vec": [],
                                   "classify_own_embed": [], "add_fact": [],
                                   "flush": [], "write_turn_total": []}
    stored = 0
    own_embed_dialled = 0
    rule_short_circuit = 0
    for t in turns:
        txt = t["text"]
        t0 = time.perf_counter()
        v = emb.embed(txt)
        per["embed_utterance"].append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        dec = clf.classify(txt, embedding=v)
        per["classify_caller_vec"].append((time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        dec2 = clf.classify(txt)
        per["classify_own_embed"].append((time.perf_counter() - t2) * 1000.0)
        if dec2.get("embedder") == "provider":
            own_embed_dialled += 1
        if dec2.get("reason") != "learned_head":
            rule_short_circuit += 1

        add_ms = 0.0
        flush_ms = 0.0
        if dec["should_store"]:
            stored += 1
            t3 = time.perf_counter()
            e.add_fact(txt, v, source="chat", metadata={"user_id": t["persona"]})
            add_ms = (time.perf_counter() - t3) * 1000.0
            per["add_fact"].append(add_ms)
            t4 = time.perf_counter()
            e.flush()
            flush_ms = (time.perf_counter() - t4) * 1000.0
            per["flush"].append(flush_ms)
        per["write_turn_total"].append(
            per["embed_utterance"][-1] + per["classify_caller_vec"][-1] + add_ms + flush_ms)
    load_end = loadavg()

    st = e.stats()
    e.close()
    emb.close()
    for suffix in ("", ".arena"):
        if os.path.exists(dst + suffix):
            os.remove(dst + suffix)
    gc.collect()

    return {
        "corpus": corpus_name, "n_docs": c.n, "n_turns": len(turns),
        "write_corpus": "chat_bench_v2 dev personas p01-p04, turns in order",
        "load_average_start": load_start, "load_average_end": load_end,
        "stages_ms": {k: summ(v) for k, v in per.items()},
        "store_rate_pct": round(100.0 * stored / max(1, len(turns)), 2),
        "classifier": {
            "own_embed_dialled_pct": round(100.0 * own_embed_dialled / max(1, len(turns)), 2),
            "rule_layer_short_circuit_pct": round(100.0 * rule_short_circuit / max(1, len(turns)), 2),
            "note": "classify(text, embedding=v) reuses the vector the turn already "
                    "has and only pays the md5 provenance check; classify(text) "
                    "dials the embedder itself -- a SECOND network call in the turn",
        },
        "engine_stats_after": {k: st[k] for k in sorted(st)
                               if not isinstance(st[k], (list, dict))},
        "_per_turn": {k: [round(x, 5) for x in v] for k, v in per.items()},
    }


# ---------------------------------------------------------------------------
# PHASE C -- the shapes an agent actually produces
# ---------------------------------------------------------------------------
def ollama_unload(model: str = EMBED_MODEL) -> bool:
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=60, check=False)
    except Exception:
        return False
    try:
        out = subprocess.check_output(["ollama", "ps"], text=True, timeout=30)
        return model.split(":")[0] not in out
    except Exception:
        return False


def phase_C(corpus_name: str = "n71433", n_cold: int = 12,
            n_groups: int = 200) -> Dict[str, Any]:
    """1 lookup vs 4; batched vs one-at-a-time; cold vs warm; and the HTTP dial."""
    from nanomem.engine import VaultEngine

    c = Corpus(corpus_name)
    vinfo = build_vault(c, tag="read", source="wiki")
    path = vinfo["path"]
    # ONE engine only. An idle second engine on the same 71,433-row vault keeps a
    # second 220 MB arena mapped and measurably perturbs the first one's scans,
    # which is how the first version of this phase reported a 4-lookup search cost
    # 1.7x the 1-lookup one for identical work.
    eng = VaultEngine(path, embed_dim=DIM, screen="off")

    emb_u = OllamaEmbedder("urllib")
    emb_k = OllamaEmbedder("keepalive")
    assert_real_embedder(emb_u)
    assert_real_embedder(emb_k)

    qs = c.turns()
    # >= n_groups groups of four DISTINCT questions.  The 240 test questions are
    # cycled with a stride so a group never repeats a text and every group still
    # pays four real embedding calls; the cycle is recorded in the output.
    quad = [[qs[(4 * g + j + (g // (len(qs) // 4))) % len(qs)] for j in range(4)]
            for g in range(n_groups)]
    load_start = loadavg()

    for q in qs[:N_WARMUP]:
        eng.search("", emb_u.embed(q), top_k=TOP_K)

    per: Dict[str, List[float]] = {
        "turn_1_lookup": [], "turn_4_lookups_serial": [], "turn_4_lookups_batched": [],
        "embed_1": [], "embed_4_serial": [], "embed_4_batched": [],
        "search_1": [], "search_4": [], "search_4_batched_arm": [],
        "embed_urllib": [], "embed_keepalive": [],
        "search_in_turn_after_embed": [], "search_back_to_back": [],
    }
    for i, group in enumerate(quad):
        # (i) one lookup
        t0 = time.perf_counter()
        v = emb_u.embed(group[0])
        te = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        eng.search("", v, top_k=TOP_K)
        ts = (time.perf_counter() - t1) * 1000.0
        # The SAME search again, immediately. The first one is what a turn pays --
        # it follows ~10 ms of blocking on the embedder, so it starts on a cold
        # core and a cold cache. The second is what a tight benchmark loop
        # measures. Identical work, paired, same query.
        t1b = time.perf_counter()
        eng.search("", v, top_k=TOP_K)
        ts_b2b = (time.perf_counter() - t1b) * 1000.0
        per["embed_1"].append(te)
        per["search_1"].append(ts)
        per["search_in_turn_after_embed"].append(ts)
        per["search_back_to_back"].append(ts_b2b)
        per["turn_1_lookup"].append(te + ts)

        # (ii) four lookups, embedded one at a time (the shipped path)
        t0 = time.perf_counter()
        vs = [emb_u.embed(g) for g in group]
        te4 = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        for vv in vs:
            eng.search("", vv, top_k=TOP_K)
        ts4 = (time.perf_counter() - t1) * 1000.0
        per["embed_4_serial"].append(te4)
        per["search_4"].append(ts4)
        per["turn_4_lookups_serial"].append(te4 + ts4)

        # (iii) four lookups, ONE batched embedding call
        t0 = time.perf_counter()
        V = emb_u.embed_batch(group)
        te4b = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        for j in range(V.shape[0]):
            eng.search("", V[j], top_k=TOP_K)
        ts4b = (time.perf_counter() - t1) * 1000.0
        per["embed_4_batched"].append(te4b)
        per["search_4_batched_arm"].append(ts4b)
        per["turn_4_lookups_batched"].append(te4b + ts4b)

        # (iv) the HTTP dial: a fresh connection per call vs one reused
        if i % 2 == 0:
            t0 = time.perf_counter(); emb_u.embed(group[0])
            per["embed_urllib"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter(); emb_k.embed(group[0])
            per["embed_keepalive"].append((time.perf_counter() - t0) * 1000.0)
        else:
            t0 = time.perf_counter(); emb_k.embed(group[0])
            per["embed_keepalive"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter(); emb_u.embed(group[0])
            per["embed_urllib"].append((time.perf_counter() - t0) * 1000.0)

    # ---- cold embedder ----------------------------------------------------
    cold: List[float] = []
    second: List[float] = []
    unload_ok = True
    for i in range(n_cold):
        ok = ollama_unload()
        unload_ok = unload_ok and ok
        t0 = time.perf_counter()
        emb_u.embed(qs[i % len(qs)])
        cold.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        emb_u.embed(qs[(i + 1) % len(qs)])
        second.append((time.perf_counter() - t0) * 1000.0)
    # leave the daemon warm for whatever runs next
    emb_u.embed(qs[0])
    load_end = loadavg()

    W = np.asarray(per["embed_1"])
    out = {
        "corpus": corpus_name, "n_docs": c.n, "n_groups": len(quad),
        "n_distinct_question_texts": len(set(qs)),
        "group_note": ("each group is four distinct questions; the %d-question test "
                       "set is cycled with a stride to reach %d groups, and every "
                       "group pays four real embedding calls"
                       % (len(set(qs)), len(quad))),
        "load_average_start": load_start, "load_average_end": load_end,
        "shapes_ms": {k: summ(v) for k, v in per.items()},
        "cold_embedder": {
            "ms": summ(cold), "n": len(cold),
            "next_call_after_cold_ms": summ(second),
            "warm_p50_ms": round(pct(W, 50), 4),
            "cold_minus_warm_p50_ms": round(pct(cold, 50) - pct(W, 50), 4),
            "cold_over_warm_ratio": round(pct(cold, 50) / max(pct(W, 50), 1e-12), 2),
            "unload_verified_by_ollama_ps": unload_ok,
            "method": "`ollama stop nomic-embed-text` before every sample, verified "
                      "with `ollama ps`; n is small (each sample costs a model "
                      "reload) and is reported as such, not padded",
        },
        "batched_vs_serial": paired_boot_delta_ci(per["embed_4_batched"], per["embed_4_serial"]),
        "four_vs_one_lookup": paired_boot_delta_ci(per["turn_4_lookups_serial"],
                                                   per["turn_1_lookup"]),
        "keepalive_vs_urllib": paired_boot_delta_ci(per["embed_keepalive"], per["embed_urllib"]),
        "search_share_pct": {
            "one_lookup": round(100.0 * pct(per["search_1"], 50) /
                                pct(per["turn_1_lookup"], 50), 2),
            "four_lookups_serial": round(100.0 * pct(per["search_4"], 50) /
                                         pct(per["turn_4_lookups_serial"], 50), 2),
            "four_lookups_batched": round(100.0 * pct(per["search_4_batched_arm"], 50) /
                                          pct(per["turn_4_lookups_batched"], 50), 2),
        },
        "search_after_idle_vs_back_to_back": dict(
            paired_boot_delta_ci(per["search_in_turn_after_embed"],
                                 per["search_back_to_back"]),
            meaning=("a: the search a TURN pays, immediately after blocking ~10 ms on "
                     "the embedder. b: the same search run again straight away, which "
                     "is what a tight benchmark loop measures. Identical work on the "
                     "same query and the same engine; the difference is a cold core "
                     "and a cold cache. Every published search p50 in this project is "
                     "the b shape.")),
        "_per_turn": {k: [round(x, 5) for x in v] for k, v in per.items()},
    }
    eng.close(); emb_u.close(); emb_k.close()
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# PHASE D -- a PERSONAL vault, where the ranking layer is actually alive
# ---------------------------------------------------------------------------
def temporal_user(uid: str = "u01") -> Dict[str, Any]:
    d = json.load(open(DATA_FILES["temporal_bench"]))
    for u in d["users"]:
        if u["user_id"] == uid:
            return u
    raise KeyError(uid)


def phase_D(uid: str = "u01") -> Dict[str, Any]:
    """Stage (c) -- entity/temporal ranking -- measured where it is NOT inert.

    On a document corpus ``_has_personal_records()`` is false, ``_max_boost()``
    is exactly 0.0 and ``_resolve_revisions`` returns its input untouched, so the
    ranking stage in phase A is the cost of the gate and nothing else.  This
    phase puts real personal records in the vault so the layer runs, at two
    sizes: a personal-only store (the deployment shape) and the same 71,433
    document corpus with one user's records added (an agent that remembers you
    AND has a library).
    """
    from nanomem.engine import VaultEngine
    TE = timed_engine_class()

    u = temporal_user(uid)
    docs = u["docs"]
    qs = u["questions"]
    emb = OllamaEmbedder("urllib")
    assert_real_embedder(emb)
    DV = emb.embed_batch([d["text"] for d in docs])

    results: Dict[str, Any] = {"user": uid, "n_personal_docs": len(docs),
                               "n_questions": len(qs), "load_average_start": loadavg()}

    shapes = (("personal_only", None), ("personal_in_1190", "val1190"),
              ("personal_in_10000", "n10000"), ("personal_in_71433", "n71433"))
    for shape, base_corpus in shapes:
        path = os.path.join(WORK, f"phaseD_{shape}.dat")
        for suffix in ("", ".arena"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
        if base_corpus is not None:
            c = Corpus(base_corpus)
            vinfo = build_vault(c, tag="read", source="wiki")
            shutil.copy2(vinfo["path"], path)
            if os.path.exists(vinfo["path"] + ".arena"):
                shutil.copy2(vinfo["path"] + ".arena", path + ".arena")
            n_docs = c.n + len(docs)
            del c
            gc.collect()
        else:
            n_docs = len(docs)
        w = VaultEngine(path, embed_dim=DIM, screen="off")
        for j, d in enumerate(docs):
            w.add_fact(d["text"], DV[j], source="chat",
                       metadata={"user_id": uid, "ts": d["ts"]}, timestamp=d["ts"])
        w.flush()
        w.close()

        eng_t = TE(path, embed_dim=DIM, screen="off")
        eng_p = VaultEngine(path, embed_dim=DIM, screen="off")
        eng_screen = VaultEngine(path, embed_dim=DIM, screen="pca")
        built = eng_screen.build_screen()

        texts = [q["text"] for q in qs]
        dirs = [q.get("direction", "current") for q in qs]
        reps = int(np.ceil(N_TURNS / max(1, len(texts))))
        per = {"embed": [], "search_total": [], "scan": [], "rank": [], "materialise": [],
               "search_total_screen_on": [], "rank_columns": [], "rank_revisions": [],
               "rank_intent": [], "rank_other": []}
        for i in range(min(N_WARMUP, len(texts))):
            v = emb.embed(texts[i])
            eng_t.search(texts[i], v, top_k=TOP_K)
            eng_p.search(texts[i], v, top_k=TOP_K)
            eng_screen.search(texts[i], v, top_k=TOP_K)
        n_boosted = 0
        for r in range(reps):
            for i, qt in enumerate(texts):
                t0 = time.perf_counter()
                v = emb.embed(qt)
                per["embed"].append((time.perf_counter() - t0) * 1000.0)
                eng_t.reset_timers()
                t1 = time.perf_counter()
                hits = eng_t.search(qt, v, top_k=TOP_K, temporal_direction=dirs[i])
                tot_t = (time.perf_counter() - t1) * 1000.0
                t2 = time.perf_counter()
                eng_p.search(qt, v, top_k=TOP_K, temporal_direction=dirs[i])
                per["search_total"].append((time.perf_counter() - t2) * 1000.0)
                t3 = time.perf_counter()
                eng_screen.search(qt, v, top_k=TOP_K, temporal_direction=dirs[i])
                per["search_total_screen_on"].append((time.perf_counter() - t3) * 1000.0)
                scan = eng_t.t_scan * 1000.0
                mat = eng_t.t_rowrec * 1000.0
                cols = eng_t.t_columns * 1000.0
                revs = eng_t.t_revisions * 1000.0
                intent = eng_t.t_intent * 1000.0
                per["scan"].append(scan)
                per["materialise"].append(mat)
                per["rank"].append(tot_t - scan - mat)
                per["rank_columns"].append(cols)
                per["rank_revisions"].append(revs)
                per["rank_intent"].append(intent)
                per["rank_other"].append(tot_t - scan - mat - cols - revs - intent)
                if hits and abs(hits[0]["score"] - hits[0]["cosine"]) > 1e-9:
                    n_boosted += 1
        E = np.asarray(per["embed"])
        S = np.asarray(per["search_total"])
        turn = E + S
        results[shape] = {
            "n_docs": n_docs,
            "n_turns": len(per["embed"]),
            "ranking_layer_live": {
                "max_boost": float(eng_p.stats().get("max_boost", 0.0)),
                "turns_whose_rank1_was_boosted_pct":
                    round(100.0 * n_boosted / max(1, len(per["embed"])), 2),
            },
            "screen": {
                "built": built,
                "engaged_queries": int(eng_screen.screen_info()["engaged"]),
                "fell_back_queries": int(eng_screen.screen_info()["fell_back"]),
                "search_total_screen_on_ms": summ(per["search_total_screen_on"]),
                "note": "the screen may only engage where the ranking is provably "
                        "inert (_ranking_is_inert); a vault holding personal records "
                        "is never in that region",
            },
            "stages_ms": {"embed_query": summ(E), "search_total": summ(S),
                          "scan": summ(per["scan"]), "rank": summ(per["rank"]),
                          "materialise": summ(per["materialise"])},
            "rank_breakdown_ms": {
                "resolve_intent": summ(per["rank_intent"]),
                "columns_entity_revision_timestamp": summ(per["rank_columns"]),
                "resolve_revisions": summ(per["rank_revisions"]),
                "boosts_select_and_dicts": summ(per["rank_other"]),
                "note": "measured on the instrumented engine; `boosts_select_and_dicts` "
                        "is the remainder of search() after the scan, the three named "
                        "calls and the per-hit materialisation"},
            "turn_ms": summ(turn),
            "share_of_p50_turn_pct": {
                "embed_query": round(100.0 * pct(E, 50) / pct(turn, 50), 2),
                "search_all_stages": round(100.0 * pct(S, 50) / pct(turn, 50), 2),
                "rank_only": round(100.0 * pct(per["rank"], 50) / pct(turn, 50), 2),
            },
            "search_instant_headroom": boot_headroom_pct(turn, E),
            "_per_turn": {k: [round(x, 5) for x in v] for k, v in per.items()},
        }
        eng_t.close(); eng_p.close(); eng_screen.close()
        for suffix in ("", ".arena"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
        gc.collect()
    results["load_average_end"] = loadavg()
    emb.close()
    return results


# ---------------------------------------------------------------------------
# PHASE E -- the arena cache, the reopen, and the cold process
# ---------------------------------------------------------------------------
_COLD_CHILD = r'''
import json, os, sys, time
t_import0 = time.perf_counter()
sys.path.insert(0, %(nanomem_dir)r)
import numpy as np
from nanomem.engine import VaultEngine
t_import = (time.perf_counter() - t_import0) * 1000.0
path, mode, query = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.perf_counter()
e = VaultEngine(path, embed_dim=768, screen="off", arena_cache=mode)
t_open = (time.perf_counter() - t0) * 1000.0
import http.client
body = json.dumps({"model": %(model)r, "input": [query]}).encode()
t1 = time.perf_counter()
c = http.client.HTTPConnection("localhost", 11434, timeout=30)
c.request("POST", "/api/embed", body=body, headers={"Content-Type": "application/json"})
v = np.asarray(json.loads(c.getresponse().read())["embeddings"], dtype=np.float32)[0]
t_embed = (time.perf_counter() - t1) * 1000.0
v = v / np.linalg.norm(v)
t2 = time.perf_counter()
hits = e.search("", v, top_k=4)
t_first_search = (time.perf_counter() - t2) * 1000.0
t3 = time.perf_counter()
e.search("", v, top_k=4)
t_second_search = (time.perf_counter() - t3) * 1000.0
info = e.arena_cache_info()
print(json.dumps({"import_ms": t_import, "open_ms": t_open, "embed_ms": t_embed,
                  "first_search_ms": t_first_search, "second_search_ms": t_second_search,
                  "arena_source": info.get("source"), "n_hits": len(hits)}))
'''


def phase_E(corpus_name: str = "n71433", n_cold_processes: int = 8) -> Dict[str, Any]:
    from nanomem.engine import VaultEngine

    c = Corpus(corpus_name)
    vinfo = build_vault(c, tag="read", source="wiki")
    path = vinfo["path"]
    emb = OllamaEmbedder("urllib")
    assert_real_embedder(emb)
    qs = c.turns()
    load_start = loadavg()

    # ---- per-search cost of arena_cache map vs off ------------------------
    t0 = time.perf_counter()
    e_map = VaultEngine(path, embed_dim=DIM, screen="off", arena_cache="map")
    open_map_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    e_off = VaultEngine(path, embed_dim=DIM, screen="off", arena_cache="off")
    open_off_ms = (time.perf_counter() - t0) * 1000.0

    vecs = [emb.embed(q) for q in qs[:N_TURNS]]
    for v in vecs[:N_WARMUP]:
        e_map.search("", v, top_k=TOP_K)
        e_off.search("", v, top_k=TOP_K)
    per = {"search_arena_map": [], "search_arena_off": []}
    mism = 0
    for i, v in enumerate(vecs):
        pair = ("search_arena_map", "search_arena_off") if i % 2 == 0 else \
               ("search_arena_off", "search_arena_map")
        got = {}
        for key in pair:
            e = e_map if key.endswith("map") else e_off
            t = time.perf_counter()
            h = e.search("", v, top_k=TOP_K)
            per[key].append((time.perf_counter() - t) * 1000.0)
            got[key] = tuple(x["id"] for x in h)
        if got["search_arena_map"] != got["search_arena_off"]:
            mism += 1
    if mism:
        raise AssertionError(f"arena_cache changed an answer on {mism} turns")

    cache_info = {"map": e_map.arena_cache_info(), "off": e_off.arena_cache_info()}
    e_map.close()
    e_off.close()
    gc.collect()

    # ---- reopen, repeated, in-process -------------------------------------
    reopen = {"map": [], "off": []}
    for i in range(10):
        for mode in ("map", "off") if i % 2 == 0 else ("off", "map"):
            t = time.perf_counter()
            e = VaultEngine(path, embed_dim=DIM, screen="off", arena_cache=mode)
            reopen[mode].append((time.perf_counter() - t) * 1000.0)
            e.close()
            del e
            gc.collect()

    # ---- a COLD PROCESS turn ---------------------------------------------
    child = os.path.join(WORK, "_cold_child.py")
    with open(child, "w") as fh:
        fh.write(_COLD_CHILD % {"nanomem_dir": NANOMEM_DIR, "model": EMBED_MODEL})
    cold: Dict[str, List[Dict[str, float]]] = {"map": [], "off": []}
    for i in range(n_cold_processes):
        for mode in ("map", "off") if i % 2 == 0 else ("off", "map"):
            t = time.perf_counter()
            out = subprocess.run([sys.executable, child, path, mode, qs[i % len(qs)]],
                                 capture_output=True, text=True, timeout=300)
            wall = (time.perf_counter() - t) * 1000.0
            if out.returncode != 0:
                raise RuntimeError(out.stderr[-2000:])
            d = json.loads(out.stdout.strip().splitlines()[-1])
            d["process_wall_ms"] = wall
            cold[mode].append(d)
    load_end = loadavg()
    emb.close()

    def coldsumm(rows: List[Dict[str, float]]) -> Dict[str, Any]:
        keys = ("process_wall_ms", "import_ms", "open_ms", "embed_ms",
                "first_search_ms", "second_search_ms")
        return {k: summ([r[k] for r in rows]) for k in keys} | {
            "arena_source": sorted({r["arena_source"] for r in rows})}

    return {
        "corpus": corpus_name, "n_docs": c.n,
        "load_average_start": load_start, "load_average_end": load_end,
        "per_search_ms": {k: summ(v) for k, v in per.items()},
        "per_search_caveat": (
            "BOTH engines are open at once so the arms can be interleaved on the "
            "identical queries, which means this process holds two full 71,433-row "
            "arenas (one mapped, one on the heap). The absolute p50 here is therefore "
            "HIGHER than phase A's single-engine p50 and should not be quoted as the "
            "search latency; the paired RATIO is the measurement."),
        "arena_map_vs_off_paired": paired_boot_delta_ci(per["search_arena_map"],
                                                        per["search_arena_off"]),
        "answers_identical": True,
        "first_open_ms": {"map": round(open_map_ms, 3), "off": round(open_off_ms, 3)},
        "reopen_ms": {k: summ(v) for k, v in reopen.items()},
        "arena_cache_info": {k: {kk: vv for kk, vv in v.items() if kk != "path"}
                             for k, v in cache_info.items()},
        "disk_bytes": {"vault": path_bytes(path),
                       "arena_sidecar": path_bytes(path + ".arena"),
                       "total": path_bytes(path) + path_bytes(path + ".arena")},
        "cold_process_turn_ms": {k: coldsumm(v) for k, v in cold.items()},
        "cold_process_note": (
            "a fresh interpreter that imports nanomem, opens the vault, embeds one "
            "question and searches -- the shape of a CLI agent turn, where the "
            "import and the open are paid once per turn instead of once per session"),
        "_per_turn": {k: [round(x, 5) for x in v] for k, v in per.items()},
    }


# ---------------------------------------------------------------------------
# PHASE F -- how much of the embedding bill an LRU would erase
# ---------------------------------------------------------------------------
def _norm_text(s: str) -> str:
    return " ".join(str(s).lower().split())


def build_sessions() -> Dict[str, List[Dict[str, Any]]]:
    """The embed streams a deployment really produces, one list per session.

    Three shapes, all built from the two conversational benchmarks:

    ``queries_as_issued``   the verification queries of one persona / user, in
                            order -- one phrasing each, which is the benchmark
                            exactly as it is scored elsewhere in this project.
    ``queries_multi_phrasing``  the same questions, but every distinct phrasing
                            the fixture carries is a separate turn: the shape of
                            a user who asks the same thing again in other words.
    ``full_embed_stream``   every string the deployment sends to the embedder in
                            that session: each conversational turn (the write
                            classifier embeds them all) followed by the queries.
    """
    out: Dict[str, List[Dict[str, Any]]] = {
        "queries_as_issued": [], "queries_multi_phrasing": [], "full_embed_stream": []}
    cb = json.load(open(DATA_FILES["chat_bench_v2"]))
    for s in cb["sets"]:
        vq = s["verification_queries"]
        issued = [q["query"] for q in vq]
        multi: List[str] = []
        for q in vq:
            seen = []
            for key in ("query", "query_templated", "query_paraphrase", "query_heady"):
                t = q.get(key)
                if t and _norm_text(t) not in {_norm_text(x) for x in seen}:
                    seen.append(t)
            multi.extend(seen)
        turns = [t["text"] for t in s["turns"]]
        base = {"corpus": "chat_bench_v2", "session": s["persona"], "split": s["split"]}
        out["queries_as_issued"].append(dict(base, texts=issued))
        out["queries_multi_phrasing"].append(dict(base, texts=multi))
        out["full_embed_stream"].append(dict(base, texts=turns + issued))
    tb = json.load(open(DATA_FILES["temporal_bench"]))
    dev_users = {"u01", "u02"}
    for u in tb["users"]:
        split = "dev" if u["user_id"] in dev_users else "test"
        qs = [q["text"] for q in u["questions"]]
        base = {"corpus": "temporal_bench", "session": u["user_id"], "split": split}
        out["queries_as_issued"].append(dict(base, texts=qs))
        out["queries_multi_phrasing"].append(dict(base, texts=qs))
        out["full_embed_stream"].append(dict(base, texts=[d["text"] for d in u["docs"]] + qs))
    return out


def phase_F(near_dup_cos: float = 0.98, batch: int = 32) -> Dict[str, Any]:
    """Exact repeats and near-duplicates within a session, and what an LRU saves.

    The threshold is the one the task fixes (cos > 0.98); nothing here is swept.
    """
    emb = OllamaEmbedder("keepalive")
    assert_real_embedder(emb)
    sessions = build_sessions()

    uniq: List[str] = []
    seen: Dict[str, int] = {}
    for shape in sessions.values():
        for s in shape:
            for t in s["texts"]:
                if t not in seen:
                    seen[t] = len(uniq)
                    uniq.append(t)
    t0 = time.perf_counter()
    V = np.zeros((len(uniq), DIM), dtype=np.float32)
    for i in range(0, len(uniq), batch):
        V[i:i + batch] = emb.embed_batch(uniq[i:i + batch])
    embed_all_s = time.perf_counter() - t0
    emb.close()

    audits = {"self_matches": 0, "backward_index_violations": 0, "cross_session_matches": 0}
    out: Dict[str, Any] = {
        "near_dup_cos_threshold": near_dup_cos,
        "threshold_provenance": "fixed by the task statement, not swept here",
        "n_unique_texts_embedded": len(uniq),
        "embed_all_unique_s": round(embed_all_s, 2),
        "shapes": {},
    }
    for shape, rows in sessions.items():
        by_split: Dict[str, Dict[str, int]] = {}
        per_session = []
        maxcos_by_split: Dict[str, List[float]] = {}
        sweep_by_split: Dict[str, Dict[str, int]] = {}
        for s in rows:
            idx = [seen[t] for t in s["texts"]]
            n = len(idx)
            exact_hit = 0
            near_hit = 0
            near_hit_distinct = 0
            n_distinct_positions = 0
            first_seen: Dict[str, int] = {}
            for i in range(n):
                t = _norm_text(s["texts"][i])
                is_exact = t in first_seen
                if is_exact:
                    if first_seen[t] >= i:
                        audits["backward_index_violations"] += 1
                    exact_hit += 1
                else:
                    first_seen[t] = i
                if i == 0:
                    if not is_exact:
                        n_distinct_positions += 1
                    continue
                prev = V[idx[:i]]
                cos = prev @ V[idx[i]]
                mx = float(cos.max()) if cos.size else -1.0
                sp0 = s["split"]
                maxcos_by_split.setdefault(sp0, []).append(mx)
                sw = sweep_by_split.setdefault(sp0, {})
                for thr in (0.999, 0.99, 0.98, 0.95, 0.90, 0.85):
                    sw[str(thr)] = sw.get(str(thr), 0) + int(mx > thr)
                    sw["_n"] = sw.get("_n", 0) + (1 if thr == 0.999 else 0)
                if not is_exact:
                    n_distinct_positions += 1
                    # the semantic rate with the generator's verbatim redraws
                    # taken out: a hit here is a NEW string close to an old one
                    prev_distinct = [seen[x] for x in
                                     dict.fromkeys(s["texts"][:i])]
                    cd = V[prev_distinct] @ V[idx[i]]
                    if cd.size and float(cd.max()) > near_dup_cos:
                        near_hit_distinct += 1
                if cos.size and float(cos.max()) > near_dup_cos:
                    near_hit += 1
                    j = int(cos.argmax())
                    if j >= i:
                        audits["backward_index_violations"] += 1
                    if j == i or idx[j] == idx[i] and j == i:
                        audits["self_matches"] += 1
                    # Catches the plausible implementation slip `prev = V[:i]`
                    # instead of `V[idx[:i]]`, which would silently match against
                    # another session's text.
                    if idx[j] not in set(idx[:i]):
                        audits["cross_session_matches"] += 1
            sp = s["split"]
            d = by_split.setdefault(sp, {"n": 0, "exact": 0, "near": 0, "sessions": 0})
            d["n"] += n
            d["exact"] += exact_hit
            d["near"] += near_hit
            d["sessions"] += 1
            d["near_distinct"] = d.get("near_distinct", 0) + near_hit_distinct
            d["n_distinct"] = d.get("n_distinct", 0) + n_distinct_positions
            per_session.append({"corpus": s["corpus"], "session": s["session"],
                                "split": sp, "n_texts": n,
                                "n_distinct_texts": n_distinct_positions,
                                "exact_repeats": exact_hit, "near_dups": near_hit,
                                "near_dups_among_distinct_texts": near_hit_distinct})
        out["shapes"][shape] = {
            "per_split": {
                sp: {"n_sessions": d["sessions"], "n_embed_calls": d["n"],
                     "exact_repeat_pct": round(100.0 * d["exact"] / max(1, d["n"]), 2),
                     "near_dup_pct": round(100.0 * d["near"] / max(1, d["n"]), 2),
                     "near_dup_ci95": wilson_ci(d["near"], d["n"]),
                     "exact_repeat_ci95": wilson_ci(d["exact"], d["n"]),
                     "semantic_near_dup_pct_excluding_verbatim_redraws":
                         round(100.0 * d["near_distinct"] / max(1, d["n_distinct"]), 2),
                     "semantic_near_dup_ci95":
                         wilson_ci(d["near_distinct"], d["n_distinct"]),
                     "n_distinct_text_positions": d["n_distinct"]}
                for sp, d in sorted(by_split.items())},
            "max_cos_to_any_earlier_text_in_session": {
                sp: {"p50": round(float(np.percentile(v, 50)), 4),
                     "p90": round(float(np.percentile(v, 90)), 4),
                     "p95": round(float(np.percentile(v, 95)), 4),
                     "p99": round(float(np.percentile(v, 99)), 4),
                     "max": round(float(np.max(v)), 4), "n": len(v)}
                for sp, v in sorted(maxcos_by_split.items())},
            "threshold_sweep_hit_pct": {
                sp: {thr: round(100.0 * cnt / max(1, sw.get("_n", 1)), 2)
                     for thr, cnt in sorted(sw.items()) if thr != "_n"}
                for sp, sw in sorted(sweep_by_split.items())},
            "threshold_sweep_note": "DESCRIPTIVE ONLY. The pre-registered threshold is "
                                    "0.98, fixed by the task; nothing here is tuned on "
                                    "it and no headline number uses another value.",
            "per_session": per_session,
        }
    # ---- how close ARE two phrasings of one question? ---------------------
    # The interesting failure mode for a semantic cache is a user who re-asks the
    # same thing in other words. chat_bench_v2 ships up to four phrasings per
    # question, so that distance can be measured rather than assumed.
    cb = json.load(open(DATA_FILES["chat_bench_v2"]))
    pair_cos: Dict[str, List[float]] = {"dev": [], "test": []}
    for st in cb["sets"]:
        for q in st["verification_queries"]:
            ph = []
            for key in ("query", "query_templated", "query_paraphrase", "query_heady"):
                t = q.get(key)
                if t and t in seen and _norm_text(t) not in {_norm_text(x) for x in ph}:
                    ph.append(t)
            for a in range(len(ph)):
                for b in range(a + 1, len(ph)):
                    pair_cos[st["split"]].append(float(V[seen[ph[a]]] @ V[seen[ph[b]]]))
    out["paraphrase_cosine_same_question"] = {
        sp: {"p50": round(float(np.percentile(v, 50)), 4),
             "p90": round(float(np.percentile(v, 90)), 4),
             "p95": round(float(np.percentile(v, 95)), 4),
             "max": round(float(np.max(v)), 4),
             "pct_above_0_98": round(100.0 * float(np.mean(np.asarray(v) > 0.98)), 2),
             "n_pairs": len(v)}
        for sp, v in pair_cos.items() if v}
    out["paraphrase_note"] = (
        "two phrasings of the SAME question, embedded by nomic-embed-text. If these "
        "do not clear 0.98, a semantic embedding cache cannot fire on a re-ask, "
        "however often a user re-asks.")

    # ---- the fixture artefact, stated plainly -----------------------------
    dup_pool = {}
    for st in cb["sets"]:
        texts = [t["text"] for t in st["turns"]]
        dup_pool[st["persona"]] = {
            "turns": len(texts), "distinct": len(set(texts)),
            "verbatim_redraw_pct": round(100.0 * (1 - len(set(texts)) / len(texts)), 2)}
    out["fixture_artefact_warning"] = {
        "what": "chat_bench_v2's noise turns are drawn from a small pool WITH "
                "replacement, so a persona's conversational stream repeats whole "
                "sentences verbatim. Those repeats are a property of the generator, "
                "not of human conversation, and they are the entire source of the "
                "exact-repeat rate in `full_embed_stream`.",
        "per_persona": dup_pool,
        "consequence": "`full_embed_stream.exact_repeat_pct` is reported but must NOT "
                       "be read as an LRU hit rate for a deployment. The de-artefacted "
                       "figure is `semantic_near_dup_pct_excluding_verbatim_redraws`, "
                       "and the query streams -- the ones a retrieval-side cache would "
                       "actually serve -- are the honest measurement.",
    }

    if any(audits.values()):
        raise AssertionError(f"cache-analysis leak audit FAILED: {audits}")
    out["leak_audit"] = dict(
        audits, raised=False,
        meaning="every match is against a STRICTLY earlier text in the SAME session; "
                "sessions are never pooled and a text is never matched against itself")
    return out


# ---------------------------------------------------------------------------
# PHASE H -- the primary configuration, measured a SECOND time
# ---------------------------------------------------------------------------
def phase_H() -> Dict[str, Any]:
    """A straight replication of the primary configuration.

    Every absolute millisecond in this file was measured on a machine carrying
    other work (the 1-minute load average is recorded per phase and ran between
    6 and 17 on 12 cores).  A share of the turn is only worth reporting if it
    survives that, so the primary configuration is simply measured again, later,
    under whatever load the machine happens to be under then, and the two runs
    are printed side by side.
    """
    return phase_A("n71433")


# ---------------------------------------------------------------------------
# input audit -- non-access proved by construction, at run time
# ---------------------------------------------------------------------------
class OpenAudit:
    """Record every path this process opens, so non-access is a FACT, not an atime.

    ``atime`` is unreliable on this filesystem, so instead of reading timestamps
    off the quarantined fixtures (which would itself be an access) this patches
    ``builtins.open`` for the whole run and keeps the set of paths.  Anything
    under ``scratch/refound`` that is not in ``DATA_FILES`` (or this file's own
    outputs) fails the audit.  ``np.load`` and ``json.load`` both go through
    ``builtins.open``, so the record is complete for file reads.
    """

    def __init__(self):
        self.paths: set = set()
        self._real = None

    def __enter__(self):
        import builtins
        self._real = builtins.open
        real = self._real
        paths = self.paths

        def patched(file, *a, **k):
            try:
                paths.add(os.path.abspath(os.fspath(file)))
            except Exception:
                pass
            return real(file, *a, **k)

        builtins.open = patched
        return self

    def __exit__(self, *exc):
        import builtins
        builtins.open = self._real
        return False

    def report(self) -> Dict[str, Any]:
        allowed = {os.path.abspath(p) for p in DATA_FILES.values()}
        allowed |= {os.path.abspath(RESULTS), os.path.abspath(RESULTS + ".tmp")}
        touched = sorted(p for p in self.paths
                         if p.startswith(os.path.abspath(REFOUND) + os.sep))
        undeclared = [p for p in touched if p not in allowed]
        if undeclared:
            raise AssertionError(
                "input audit FAILED: this run opened files under scratch/refound "
                f"that it never declared: {undeclared}")
        return {
            "files_opened_under_scratch_refound": [os.path.basename(p) for p in touched],
            "declared_inputs": sorted(os.path.basename(p) for p in DATA_FILES.values()),
            "undeclared_opens": undeclared,
            "raised": False,
            "method": "builtins.open patched for the whole run; the set of paths is "
                      "recorded and checked against DATA_FILES. No timestamp on any "
                      "file was read, and no path outside DATA_FILES is named "
                      "anywhere in this module.",
        }


def audit_pairing(res: Dict[str, Any]) -> Dict[str, Any]:
    """Every per-turn ablation must subtract components measured on the SAME turn.

    Counts, per phase and per corpus, the stage vectors whose length differs from
    the turn vector's -- the failure that would silently pair turn i's embedding
    with turn j's search -- and the turns where the reconstructed total does not
    equal the measured one.  Raises on non-zero.
    """
    bad_len: Dict[str, Any] = {}
    bad_recon: Dict[str, int] = {}
    for corpus, block in (res.get("phase_A", {}) or {}).items():
        per = block.get("_per_turn", {})
        if not per:
            continue
        lens = {k: len(v) for k, v in per.items()}
        n = lens.get("embed", 0)
        odd = {k: v for k, v in lens.items() if v != n}
        if odd:
            bad_len[f"phase_A/{corpus}"] = odd
        for tkey in ("off_t", "on_t"):
            s = np.asarray(per[f"{tkey}_scan"])
            r = np.asarray(per[f"{tkey}_rank"])
            m = np.asarray(per[f"{tkey}_rowrec"])
            t = np.asarray(per[f"{tkey}_total"])
            # `_per_turn` is serialised rounded to 5 decimal places of a
            # millisecond, so three rounded terms can miss their rounded total by
            # up to 1.5e-5 ms. The tolerance is that rounding and nothing else: a
            # real misalignment moves a whole stage, not 15 nanoseconds.
            miss = int(np.sum(np.abs((s + r + m) - t) > 1e-4))
            if miss:
                bad_recon[f"phase_A/{corpus}/{tkey}"] = miss
    for shape, block in (res.get("phase_D", {}) or {}).items():
        if not isinstance(block, dict) or "_per_turn" not in block:
            continue
        per = block["_per_turn"]
        lens = {k: len(v) for k, v in per.items()}
        n = lens.get("embed", 0)
        odd = {k: v for k, v in lens.items() if v != n}
        if odd:
            bad_len[f"phase_D/{shape}"] = odd
    if bad_len or bad_recon:
        raise AssertionError(f"pairing audit FAILED: lengths={bad_len} recon={bad_recon}")
    return {"misaligned_stage_vectors": bad_len,
            "turns_where_stages_do_not_sum_to_the_measured_total": bad_recon,
            "raised": False,
            "meaning": "every headroom number in this file is a per-turn subtraction "
                       "of stages measured on that same turn"}


# ---------------------------------------------------------------------------
# PHASE G -- the ranked levers and the verdict
# ---------------------------------------------------------------------------
def _replication_block(res: Dict[str, Any]) -> Dict[str, Any]:
    """Run 1 vs run 2 of the primary configuration, under different machine load."""
    H = res.get("phase_H")
    A = res["phase_A"]["n71433"]
    if not H:
        return {"available": False,
                "reason": "phase H was not run in this pass"}
    def row(b):
        t = b["turn_shipped_default_screen_off"]
        return {"load_1min_at_start": b["load_average_start"]["1min"],
                "embed_p50_ms": b["stages_ms"]["embed_query"]["p50"],
                "search_p50_ms": b["stages_ms"]["search_total_screen_off"]["p50"],
                "turn_p50_ms": t["turn_ms"]["p50"],
                "search_share_pct": t["share_of_p50_turn_pct"]["search_all_stages"],
                "search_instant_headroom_pct": t["search_instant_headroom"]["headroom_pct"],
                "headroom_ci95": t["search_instant_headroom"]["headroom_pct_ci95"]}
    r1, r2 = row(A), row(H)
    return {"available": True, "run_1": r1, "run_2": r2,
            "headroom_pct_difference": round(
                r2["search_instant_headroom_pct"] - r1["search_instant_headroom_pct"], 2),
            "reading": "if the two headroom intervals overlap, the share of the turn "
                       "is a property of the system rather than of the machine's "
                       "weather on the afternoon it was measured"}


def phase_G(res: Dict[str, Any]) -> Dict[str, Any]:
    A = res["phase_A"]["n71433"]
    perA = A["_per_turn"]
    E = np.asarray(perA["embed"])
    S_off = np.asarray(perA["off_p_total"])
    S_on = np.asarray(perA["on_p_total"])
    turn = E + S_off
    p50_turn = pct(turn, 50)

    levers: List[Dict[str, Any]] = []

    def add(name: str, removed_ms: float, baseline_ms: float, baseline: str,
            evidence: str, ci: Optional[List[float]] = None, **extra):
        row = {"lever": name,
               "removes_ms_from_p50_turn": round(removed_ms, 4),
               "baseline_p50_turn_ms": round(baseline_ms, 4),
               "pct_of_that_turn": round(100.0 * removed_ms / max(baseline_ms, 1e-12), 2),
               "baseline_shape": baseline, "evidence": evidence}
        if ci:
            row["removed_ms_ci95"] = ci
        row.update(extra)
        levers.append(row)

    PRIMARY = ("read turn, 1 lookup, 71,433 docs, shipped defaults "
               "(screen off, arena_cache map, router off), warm embedder")

    h_instant = boot_headroom_pct(turn, E)
    add("search is INSTANT (scan + rank + materialise -> 0 ms)",
        h_instant["removed_ms"], p50_turn, PRIMARY,
        "phase_A.n71433.turn_shipped_default_screen_off.search_instant_headroom",
        headroom_pct=h_instant["headroom_pct"],
        headroom_pct_ci95=h_instant["headroom_pct_ci95"])

    h_screen = boot_headroom_pct(turn, E + S_on)
    add("PCA screen off -> on (exact, opt-in today)",
        h_screen["removed_ms"], p50_turn, PRIMARY,
        "phase_A.n71433, paired per-turn, screen-on and screen-off arms interleaved",
        headroom_pct=h_screen["headroom_pct"],
        headroom_pct_ci95=h_screen["headroom_pct_ci95"])

    h_2x = boot_headroom_pct(turn, turn - E / 2.0)
    add("embedder 2x faster (hypothetical)",
        h_2x["removed_ms"], p50_turn, PRIMARY,
        "phase_A.n71433, per-turn embed time halved",
        headroom_pct=h_2x["headroom_pct"], headroom_pct_ci95=h_2x["headroom_pct_ci95"])

    Eb = res["phase_E"]["n71433"]
    d_arena = (Eb["per_search_ms"]["search_arena_off"]["p50"]
               - Eb["per_search_ms"]["search_arena_map"]["p50"])
    add("arena cache off -> on (map), per search",
        d_arena, p50_turn, PRIMARY,
        "phase_E.n71433.arena_map_vs_off_paired (interleaved, identical answers)",
        also="its real win is process START, not the turn: "
             f"reopen p50 {Eb['reopen_ms']['off']['p50']} ms -> "
             f"{Eb['reopen_ms']['map']['p50']} ms",
        reopen_ms_saved=round(Eb["reopen_ms"]["off"]["p50"] - Eb["reopen_ms"]["map"]["p50"], 3))

    C = res["phase_C"]
    d_ka = (C["shapes_ms"]["embed_urllib"]["p50"] - C["shapes_ms"]["embed_keepalive"]["p50"])
    add("reuse the HTTP connection to the embedder (keep-alive)",
        d_ka, p50_turn, PRIMARY,
        "phase_C.keepalive_vs_urllib (interleaved, order alternated)",
        ci=[-C["keepalive_vs_urllib"]["delta_ci95"][1],
            -C["keepalive_vs_urllib"]["delta_ci95"][0]],
        note="nanomem.embed opens a new urllib connection per call; this is a "
             "four-line change this file does NOT make")

    F = res["phase_F"]
    embed_p50 = pct(E, 50)
    cache_rows = {}
    for shape in ("queries_as_issued", "queries_multi_phrasing", "full_embed_stream"):
        sp = F["shapes"][shape]["per_split"]
        test = sp.get("test", {})
        cache_rows[shape] = {
            "test_near_dup_pct": test.get("near_dup_pct"),
            "test_near_dup_ci95": test.get("near_dup_ci95"),
            "test_exact_repeat_pct": test.get("exact_repeat_pct"),
            "test_semantic_near_dup_pct_excluding_verbatim_redraws":
                test.get("semantic_near_dup_pct_excluding_verbatim_redraws"),
            "expected_ms_removed_from_p50_turn":
                round((test.get("near_dup_pct", 0.0) / 100.0) * embed_p50, 4),
        }
    # The task asks about QUERIES in a session, so the headline lever is the
    # query stream. `full_embed_stream` is reported beside it and is NOT used to
    # rank anything: its repeat rate is the generator redrawing noise turns
    # verbatim (phase_F.fixture_artefact_warning), not a deployment property.
    head_shape = "queries_as_issued"
    add("embedding LRU on the QUERY stream (exact repeat or cos > 0.98 near-dup)",
        cache_rows[head_shape]["expected_ms_removed_from_p50_turn"], p50_turn, PRIMARY,
        "phase_F.shapes.queries_as_issued, test split scored once; expected value = "
        "hit rate x measured embed p50",
        hit_rate_pct=cache_rows[head_shape]["test_near_dup_pct"],
        hit_rate_ci95=cache_rows[head_shape]["test_near_dup_ci95"],
        all_stream_shapes=cache_rows,
        paraphrase_cosine=F.get("paraphrase_cosine_same_question"),
        caveat="an EXPECTED value over a session, not a per-turn measurement; and it "
               "is an upper bound from two benchmarks that ask each question once, "
               "so it measures what THESE streams offer a cache, not how often a "
               "real user re-asks")

    levers.sort(key=lambda r: -r["removes_ms_from_p50_turn"])
    for i, r in enumerate(levers, 1):
        r["rank"] = i

    # ---- levers that only exist in another shape --------------------------
    other: List[Dict[str, Any]] = []
    t4s = C["shapes_ms"]["turn_4_lookups_serial"]["p50"]
    t4b = C["shapes_ms"]["turn_4_lookups_batched"]["p50"]
    other.append({
        "lever": "batch the 4 embeddings of a 4-lookup turn into one call",
        "baseline_shape": "read turn, 4 lookups, 71,433 docs",
        "baseline_p50_turn_ms": t4s,
        "removes_ms_from_p50_turn": round(t4s - t4b, 4),
        "pct_of_that_turn": round(100.0 * (t4s - t4b) / max(t4s, 1e-12), 2),
        "evidence": "phase_C.batched_vs_serial",
        "ci": C["batched_vs_serial"]["delta_ci95"]})
    D = res["phase_D"]
    pd = D["personal_in_71433"]
    other.append({
        "lever": "make the entity/temporal RANK stage cost O(top_k) instead of "
                 "O(all rows) on a vault that holds personal records",
        "baseline_shape": "read turn, 1 lookup, 71,478 rows of which 45 are personal "
                          "(the agent-memory shape), shipped defaults",
        "baseline_p50_turn_ms": pd["turn_ms"]["p50"],
        "removes_ms_from_p50_turn": pd["stages_ms"]["rank"]["p50"],
        "pct_of_that_turn": round(100.0 * pd["stages_ms"]["rank"]["p50"] /
                                  max(pd["turn_ms"]["p50"], 1e-12), 2),
        "evidence": "phase_D.personal_in_71433.rank_breakdown_ms",
        "of_which_resolve_revisions_ms":
            pd["rank_breakdown_ms"]["resolve_revisions"]["p50"],
        "note": "this is the ONE store lever in this file bigger than the embedder's "
                "noise floor. 45 personal records among 71,433 documents take search "
                "from %s ms to %s ms, and the PCA screen engages on 0 of %d queries "
                "because _ranking_is_inert() is false for the whole vault."
                % (res["phase_A"]["n71433"]["stages_ms"]["search_total_screen_off"]["p50"],
                   pd["stages_ms"]["search_total"]["p50"], pd["n_turns"])})
    B = res["phase_B"]["n71433"]
    own = B["stages_ms"]["classify_own_embed"]["p50"]
    caller = B["stages_ms"]["classify_caller_vec"]["p50"]
    wt = B["stages_ms"]["write_turn_total"]["p50"]
    other.append({
        "lever": "write classifier reuses the turn's embedding instead of dialling "
                 "the embedder itself",
        "baseline_shape": "write turn (embed utterance + classify + store), 71,433 docs",
        "baseline_p50_turn_ms": round(wt + own - caller, 4),
        "removes_ms_from_p50_turn": round(own - caller, 4),
        "pct_of_that_turn": round(100.0 * (own - caller) / max(wt + own - caller, 1e-12), 2),
        "evidence": "phase_B.n71433.stages_ms",
        "note": "this is what nanomem already does when the caller passes a vector; "
                "Vault.should_store(text) does NOT, and pays a second network call"})
    flush = B["stages_ms"]["flush"]["p50"]
    other.append({
        "lever": "stop calling flush() after every stored fact (durability trade)",
        "baseline_shape": "write turn that stores, 71,433 docs",
        "baseline_p50_turn_ms": B["stages_ms"]["write_turn_total"]["p50"],
        "removes_ms_from_p50_turn": round(flush, 4),
        "pct_of_that_turn": round(100.0 * flush /
                                  max(B["stages_ms"]["write_turn_total"]["p50"], 1e-12), 2),
        "evidence": "phase_B.n71433.stages_ms.flush",
        "note": "NOT free: it trades an fsync for a window in which a crash loses the "
                "fact. Listed because it is the only write-path cost of the size of a "
                "search, not because it is recommended"})

    # ---- the pre-registered bars -----------------------------------------
    bar1_stat = A["turn_shipped_default_screen_off"]["search_instant_headroom"]
    bar1_pass = bar1_stat["headroom_pct"] < 10.0
    screen_removed = h_screen["removed_ms"]
    lru_removed = cache_rows[head_shape]["expected_ms_removed_from_p50_turn"]

    verdict = {
        "BAR_1_primary": {
            "bar": "search_instant_headroom_pct < 10.0 at 71,433 docs, shipped "
                   "defaults, 1-lookup read turn (the shape most favourable to the "
                   "store) => ACCEPT 'further store optimisation has near-zero "
                   "user-visible payoff'",
            "measured_headroom_pct": bar1_stat["headroom_pct"],
            "ci95": bar1_stat["headroom_pct_ci95"],
            "p50_turn_ms": bar1_stat["p50_turn_ms"],
            "p50_turn_without_search_ms": bar1_stat["p50_turn_without_ms"],
            "result": "ACCEPTED" if bar1_pass else "REJECTED",
            "reading": ("removing the store ENTIRELY -- an infinitely fast, zero-cost "
                        "index -- moves the median turn by "
                        f"{bar1_stat['removed_ms']} ms, "
                        f"{bar1_stat['headroom_pct']}% of it"),
        },
        "BAR_3_cache_vs_screen": {
            "bar": "the embedding LRU beats every index optimisation shipped so far "
                   "iff its expected p50 turn reduction exceeds the PCA screen's "
                   "measured one at 71,433 docs",
            "lru_stream": head_shape,
            "lru_hit_rate_pct": cache_rows[head_shape]["test_near_dup_pct"],
            "lru_hit_rate_ci95": cache_rows[head_shape]["test_near_dup_ci95"],
            "lru_expected_ms": round(lru_removed, 4),
            "screen_measured_ms": round(screen_removed, 4),
            "result": "LRU WINS" if lru_removed > screen_removed else "SCREEN WINS",
        },
        "replication_of_the_primary_configuration": _replication_block(res),
        "search_share_of_turn_pct_by_corpus": {
            k: res["phase_A"][k]["turn_shipped_default_screen_off"][
                "share_of_p50_turn_pct"] for k in res["phase_A"]},
    }
    # ---- the composite turn a deployed agent really runs ------------------
    # One embedding of the user's message, reused for BOTH the lookup and the
    # write gate; the search; the classifier; and the store work amortised at the
    # classifier's own measured store rate. Built from measured p50s, and
    # reported for the document vault and for the agent-memory vault side by side.
    Bn = res["phase_B"]["n71433"]
    store_rate = float(Bn["store_rate_pct"]) / 100.0
    store_ms = (Bn["stages_ms"]["add_fact"]["p50"] + Bn["stages_ms"]["flush"]["p50"]) * store_rate
    clf_ms = Bn["stages_ms"]["classify_caller_vec"]["p50"]
    composite = {}
    for tag, emb_ms, srch_ms, rows in (
            ("document_vault_71433", pct(E, 50), pct(S_off, 50), 71433),
            ("agent_memory_vault_71478_45_personal",
             res["phase_D"]["personal_in_71433"]["stages_ms"]["embed_query"]["p50"],
             res["phase_D"]["personal_in_71433"]["stages_ms"]["search_total"]["p50"], 71478)):
        tot = emb_ms + srch_ms + clf_ms + store_ms
        composite[tag] = {
            "rows": rows,
            "stages_p50_ms": {"embed_user_message_once": round(emb_ms, 4),
                              "search": round(srch_ms, 4),
                              "write_classifier_reusing_that_vector": round(clf_ms, 4),
                              "store_work_amortised_at_%.1f_pct_store_rate"
                              % (100 * store_rate): round(store_ms, 4)},
            "turn_p50_ms": round(tot, 4),
            "share_pct": {"embed": round(100 * emb_ms / tot, 2),
                          "search": round(100 * srch_ms / tot, 2),
                          "classify": round(100 * clf_ms / tot, 2),
                          "store": round(100 * store_ms / tot, 2)},
        }
    composite["note"] = ("arithmetic over measured p50s from phases A, B and D, not a "
                         "separately timed loop; the components were measured on "
                         "different turn sets so this is a composition, and it is "
                         "labelled as one rather than quoted as an observation")

    return {"primary_configuration": PRIMARY,
            "p50_turn_ms": round(p50_turn, 4),
            "composite_agent_turn": composite,
            "ranked_levers": levers,
            "levers_in_other_shapes": other,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def write_prereg() -> Dict[str, Any]:
    res = load_results()
    if "prereg" not in res:
        res = {"prereg": prereg(), "prereg_sha256_16": PREREG_HASH,
               "prereg_written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "environment": env_block()}
        save_results(res)
        print(f"[prereg] written to {RESULTS} (sha {PREREG_HASH})")
    else:
        _assert_prereg_intact(res)
        print(f"[prereg] already registered (sha {res.get('prereg_sha256_16')})")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true", help="write the pre-registration and exit")
    ap.add_argument("--phase", action="append", default=None,
                    help="A|B|C|D|E|F|G (repeatable); default: all")
    ap.add_argument("--corpus", action="append", default=None,
                    help="restrict phases A/B to these corpora")
    args = ap.parse_args(argv)

    os.makedirs(WORK, exist_ok=True)
    res = write_prereg()
    if args.prereg:
        return 0
    _assert_prereg_intact(res)

    phases = [p.upper() for p in (args.phase or list("ABCDEFH") + ["G"])]
    corpora = tuple(args.corpus) if args.corpus else CORPORA

    with OpenAudit() as oa:
        if "A" in phases:
            res.setdefault("phase_A", {})
            for cn in corpora:
                print(f"[A] stage breakdown :: {cn}", flush=True)
                res["phase_A"][cn] = phase_A(cn)
                save_results(res)
        if "B" in phases:
            res.setdefault("phase_B", {})
            for cn in corpora:
                print(f"[B] write path :: {cn}", flush=True)
                res["phase_B"][cn] = phase_B(cn)
                save_results(res)
        if "C" in phases:
            print("[C] turn shapes :: n71433", flush=True)
            res["phase_C"] = phase_C("n71433")
            save_results(res)
        if "D" in phases:
            print("[D] personal vault, ranking live", flush=True)
            res["phase_D"] = phase_D("u01")
            save_results(res)
        if "E" in phases:
            res.setdefault("phase_E", {})
            print("[E] arena cache / reopen / cold process :: n71433", flush=True)
            res["phase_E"]["n71433"] = phase_E("n71433")
            save_results(res)
        if "F" in phases:
            print("[F] embedding cache potential", flush=True)
            res["phase_F"] = phase_F()
            save_results(res)
        if "H" in phases:
            print("[H] replication of the primary configuration", flush=True)
            res["phase_H"] = phase_H()
            save_results(res)
        if "G" in phases:
            print("[G] ranked levers + verdict", flush=True)
            res["audits"] = {"pairing": audit_pairing(res)}
            res["phase_G"] = phase_G(res)
            save_results(res)
        res.setdefault("audits", {})["inputs"] = oa.report()

    res["engine_provenance"] = engine_provenance()
    res["environment_at_end"] = env_block()
    res["peak_rss_mb_driver"] = maxrss_mb()
    save_results(res)
    print(f"[done] {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
