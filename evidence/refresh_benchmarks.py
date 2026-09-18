"""Re-measure everything BENCHMARKS.md cites, stamp it, and stage it.

`release_preflight.py` refuses to ship when a results file was measured on an
engine other than the one being released. This is the other half of that deal:
without a single command, "re-run the benchmarks" is a chore that gets skipped,
and the guard becomes something to work around rather than satisfy.

    python3 refresh_benchmarks.py            # the pre-release refresh, ~16 min
    python3 refresh_benchmarks.py --deep     # + 60 fuzz seeds and the head-to-head
    python3 refresh_benchmarks.py --list     # what produces what, with real timings

The default is sized to be RUN, not skipped. The first version budgeted 16 min
and took 53, because the fuzzer estimate was written before `merge` and `export`
joined its operation pool -- each builds a side vault or writes and reopens a
full one, and 60 seeds of that is 46 minutes on its own. A pre-release gate that
costs an hour is one people route around, which is the exact failure the
preflight guard exists to prevent. So the default runs 12 fuzz seeds (both
defects it has ever found surfaced within 8) and `--deep` runs 60.

Times below are MEASURED, not estimated. That distinction is the whole reason
this file exists.

Every number on the evidence page comes from one of these. If a measurement has
no script here, it cannot be re-measured -- only retyped -- and that is how
numbers drift away from the code they describe.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quarantine_guard                        # noqa: E402
quarantine_guard.enforce()

import argparse, collections, json, shutil, subprocess, time   # noqa: E402
PKG = os.path.abspath(os.path.join(HERE, "..", "..", "nanomem_standalone"))
sys.path.insert(0, PKG)
from nanomem import __version__, ENGINE_VERSION                # noqa: E402

BENCH_PY = os.path.join(os.path.dirname(PKG), "venv_bench", "bin", "python")

FUZZ_SEEDS_DEFAULT, FUZZ_SEEDS_DEEP = "12", "60"

#: results file -> (argv, measured seconds, note)
JOBS = collections.OrderedDict([
    ("edge_cases_results.json",
     ([sys.executable, "exp_edge_cases.py"], 1, None)),
    ("untested_configs_results.json",
     ([sys.executable, "exp_untested_configs.py"], 8, None)),
    ("unexplored_surface_results.json",
     ([sys.executable, "exp_unexplored_surface.py"], 10, None)),
    ("clean_chat_results_floorfix.json",
     ([sys.executable, "benchmark_clean_chat_v3r2.py", PKG, "floorfix"], 24, None)),
    ("latency_split_results.json",
     ([sys.executable, "exp_latency_split.py"], 99, "needs a live embedder")),
    ("query_truncation_results.json",
     ([sys.executable, "exp_query_truncation.py"], 2, "needs a live embedder")),
    ("entity_declaration_results.json",
     ([sys.executable, "exp_entity_declaration.py"], 3, "needs a live embedder")),
    ("decompose_order_results.json",
     ([sys.executable, "exp_decompose_order.py"], 3, "needs a live embedder")),
    ("new_usecases_results.json",
     ([sys.executable, "exp_new_usecases.py"], 243, "needs a live embedder")),
    ("fuzz_ops_results.json",
     ([sys.executable, "exp_fuzz_ops.py", FUZZ_SEEDS_DEFAULT], 553,
      "~46 s per seed; --deep raises 12 seeds to 60")),
    ("competitors_standard_results.json",
     ([BENCH_PY, "bench_competitors.py"], 720,
      "DEEP ONLY: builds four stores over 71,433 documents; needs venv_bench "
      "and a cache built once with --prep")),
])
DEEP_ONLY = "competitors_standard_results.json"


def stamp(path):
    obj = json.load(open(path), object_pairs_hook=collections.OrderedDict)
    if not isinstance(obj, dict):
        obj = collections.OrderedDict([("results", obj)])
    out = collections.OrderedDict()
    out["measured_on"] = {"nanomem": __version__, "engine": ENGINE_VERSION,
                          "note": "results are stale once `engine` differs from "
                                  "the shipped ENGINE_VERSION; "
                                  "release_preflight.py enforces this"}
    for k, v in obj.items():
        if k != "measured_on":
            out[k] = v
    json.dump(out, open(path, "w"), indent=1)


def redact(path):
    """Strip absolute paths into the private tree, leaving every number alone."""
    raw = open(path).read()
    before = json.loads(raw)
    root = os.path.dirname(os.path.dirname(HERE))
    raw = raw.replace(root + "/scratch/refound/", "<repo>/benchmarks/")
    raw = raw.replace(root + "/scratch/", "<repo>/corpora/")
    raw = raw.replace(root + "/nanomem_standalone/", "<repo>/")
    raw = raw.replace(root + "/", "<repo>/")
    import re
    raw = re.sub(r'"<redacted local path>"]*"', '"<redacted local path>"', raw)
    after = json.loads(raw)

    def nums(o, acc=None):
        acc = [] if acc is None else acc
        if isinstance(o, dict):
            [nums(v, acc) for v in o.values()]
        elif isinstance(o, list):
            [nums(v, acc) for v in o]
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            acc.append(o)
        return acc
    assert nums(before) == nums(after), f"redaction changed a number in {path}"
    open(path, "w").write(raw)


def rewrite_doc():
    """Rewrite the generated blocks in BENCHMARKS.md from the results files.

    The page used to hand-quote these to two decimals. Wall-clock medians move a
    few percent between runs, so every refresh left the prose disagreeing with
    its own cited file -- the exact drift the preflight guard was added to catch,
    reintroduced by hand one refresh later. Typing is the bug; this removes it.
    """
    doc = os.path.join(PKG, "BENCHMARKS.md")
    if not os.path.exists(doc):
        return
    src = os.path.join(PKG, "benchmarks", "latency_split_results.json")
    if not os.path.exists(src):
        return
    d = json.load(open(src))
    m = d["measurements_ms_median"]
    s = d["store_side_filtered_search"]
    blocks = {
        "latency": (
            "```\n"
            f"store only, query already embedded  {m['store_only_query_pre_embedded']:8.2f} ms"
            "   <- the part nanomem owns\n"
            f"embedding round-trip (local Ollama) {m['embed_query_only_http']:8.2f} ms"
            "   <- your embedder, not the store\n"
            f"Vault.search(decompose=False)       {m['vault_search_decompose_false']:8.2f} ms\n"
            f"Vault.search(...)  the DEFAULT      {m['vault_search_decompose_true_DEFAULT']:8.2f} ms"
            f"   <- decomposition adds {m['decomposition_overhead_ms']} ms\n"
            "```"),
        "storeside": (
            "```\n"
            f"unfiltered              {s['unfiltered_ms']:7.2f} ms  "
            f"{s['unfiltered_record_decodes']:>6,} record decodes\n"
            f"filtered, before 0.7.9  {s['filtered_before_0_7_9_ms']:7.2f} ms  "
            f"{s['filtered_before_0_7_9_record_decodes']:>6,} record decodes"
            "   (the whole corpus)\n"
            f"filtered, now           {s['filtered_ms']:7.2f} ms  "
            f"{s['filtered_record_decodes']:>6,} record decodes\n"
            "```"),
    }
    text = open(doc).read()
    import re as _re
    changed = 0
    for name, body in blocks.items():
        pat = _re.compile(
            r"(<!-- GENERATED: " + name + r" [^>]*-->\n).*?(\n<!-- /GENERATED -->)",
            _re.S)
        new, n = pat.subn(lambda mo: mo.group(1) + body + mo.group(2), text)
        if n:
            changed += n
            text = new
        else:
            print(f"  !! BENCHMARKS.md has no GENERATED block named {name!r}")
    open(doc, "w").write(text)
    print(f"  rewrote {changed} generated block(s) in BENCHMARKS.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true",
                    help="60 fuzz seeds and the competitor head-to-head")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", metavar="SUBSTR",
                    help="re-run just the jobs whose results file matches. For "
                         "adding ONE measurement without paying for the sweep; "
                         "safe only while ENGINE_VERSION is unchanged, since "
                         "that is what makes the other staged files still current")
    a = ap.parse_args()
    if a.list:
        for out, (argv, secs, why) in JOBS.items():
            tag = "  [deep only]" if out == DEEP_ONLY else ""
            print(f"  {out:<40} {secs // 60:>2}m{secs % 60:02d}s  "
                  f"{os.path.basename(argv[1])}{tag}"
                  f"{'   -- ' + why if why else ''}")
        return 0
    jobs = collections.OrderedDict(
        (k, v) for k, v in JOBS.items() if a.deep or k != DEEP_ONLY)
    if a.only:
        jobs = collections.OrderedDict(
            (k, v) for k, v in jobs.items() if a.only in k)
        if not jobs:
            print(f"--only {a.only!r} matched no job; --list shows them all")
            return 2
    if a.deep:
        argv, secs, why = jobs["fuzz_ops_results.json"]
        jobs["fuzz_ops_results.json"] = (argv[:-1] + [FUZZ_SEEDS_DEEP], 2765, why)
    total = sum(v[1] for v in jobs.values())
    print(f"nanomem {__version__} / engine {ENGINE_VERSION} -- "
          f"{len(jobs)} measurements, about {total // 60} min "
          f"({'deep' if a.deep else 'default; --deep for the full sweep'})\n")
    failed = []
    for out, (argv, _m, _w) in jobs.items():
        t = time.time()
        r = subprocess.run(argv, cwd=HERE, capture_output=True, text=True)
        p = os.path.join(HERE, out)
        if r.returncode != 0 or not os.path.exists(p):
            failed.append(out)
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
            print(f"  XX {out:<40} {tail[0][:80]}")
            continue
        stamp(p); redact(p)
        print(f"  ok {out:<40} ({time.time()-t:.0f}s)")
    dest = os.path.join(PKG, "benchmarks")
    os.makedirs(dest, exist_ok=True)
    staged = 0
    for out in jobs:
        p = os.path.join(HERE, out)
        if os.path.exists(p) and out not in failed:
            shutil.copy(p, os.path.join(dest, out)); staged += 1
    print(f"\nstaged {staged} file(s) into nanomem_standalone/benchmarks/")
    if not a.only:
        rewrite_doc()
    if failed:
        print(f"{len(failed)} FAILED: {', '.join(failed)}")
        print("The evidence page still cites them, so either fix the run or "
              "drop the claim -- preflight will refuse the release either way.")
        return 1
    print("now run:  python3 release_preflight.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
