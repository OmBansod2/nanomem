"""Latency and write-cost smoke tests. Seeded vectors only; no network."""

import os
import sys
import time

import numpy as np
import pytest

from conftest import D, unit_rows
from nanomem.engine import VaultEngine


def _percentile(xs, q):
    return float(np.percentile(np.asarray(xs), q))


@pytest.mark.parametrize("n", [10_000])
def test_search_latency_at_10k(tmp_path, n):
    """Search cost is the matmul plus bounded post-processing.

    The absolute floor here is the dense ``(n, 768) @ (768,)`` product, so the
    assertion is stated relative to that product measured in the same process --
    an absolute millisecond budget would only be measuring the machine's current
    load. The absolute number is printed either way; on an idle machine this is
    ~0.40 ms p50 at 10k documents.
    """
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(n, seed=1)
    for i in range(n):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    assert e.count() == n
    arena = e.arena.vec[:n]
    Q = unit_rows(220, seed=2)
    for i in range(20):
        e.search("", Q[i], top_k=4)
        _ = arena @ Q[i]
    lat, base = [], []
    for i in range(20, 220):
        t = time.perf_counter()
        e.search("", Q[i], top_k=4)
        lat.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        idx = np.argpartition(-(arena @ Q[i]), 4)[:4]
        base.append((time.perf_counter() - t) * 1000)
    p50, p95 = _percentile(lat, 50), _percentile(lat, 95)
    b50 = _percentile(base, 50)
    print(f"\n10k docs: engine p50 {p50:.3f} ms  p95 {p95:.3f} ms | "
          f"raw matmul p50 {b50:.3f} ms | load {os.getloadavg()[0]:.2f}")
    assert p50 < max(0.5, 2.5 * b50 + 0.1), f"p50 {p50:.3f} ms vs matmul {b50:.3f} ms"
    assert p50 < 2.0, f"p50 {p50:.3f} ms"
    e.close()


def test_search_is_exact_below_n_exhaustive(tmp_path):
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(5000, seed=3)
    for i in range(5000):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    arena = e.arena.vec[:e.arena.n_rows]
    ids = e.arena.ids
    Q = unit_rows(50, seed=4)
    for q in Q:
        got = [h["id"] for h in e.search("", q, top_k=4, min_score=-1.0)]
        want = [ids[i] for i in np.argsort(-(arena @ q))[:4]]
        assert got == want
    assert e.stats()["routing_mode"] == "exhaustive"
    e.close()


def test_write_cost_is_flat(tmp_path):
    """Write cost does not grow with the number of rows already written.

    v2 measured 9.4x more per record at 10,000 rows than at 500; this asserts the
    growth term is gone. THE ESTIMATOR IS THE MEASURED PART, not just the bound.

    The assertion used to be ``mean(last 500 wall-clock) < 1.5 * mean(first
    500)``, and it FAILED at 1.74x on a loaded box during this session while
    passing 5/5 in isolation on the same code at 0.58-0.96x. A wall-clock MEAN
    over 500 samples is dominated by whichever window happened to catch a
    scheduler stall: measured on the real samples, TWO injected 5 ms stalls
    anywhere in the last-500 window are enough to push that ratio past 1.5x,
    while the same window's MEDIAN needs 243 of them
    (``scratch/refound/write_cost_flatness.json``, ``stall_sensitivity``). So the
    old test flipped on the machine's load, not on the library's behaviour.

    What is asserted now, in order of what it measures. All figures are over 15
    repetitions of this exact loop in three conditions -- ``quiet``,
    ``busy_phase_a_running`` and ``concurrent_with_pytest`` (load 1.9-4.3) --
    in ``scratch/refound/write_cost_flatness.json``:

      * ``time.process_time`` medians. CPU time does not advance while the
        process is descheduled, so a stall adds nothing to it -- it measures the
        WORK, which is the claim. Measured 0.95x-1.05x, 19-20 us per add_fact at
        both ends of the run, with no trend across the three loads.
      * the wall-clock median as a cross-check at a deliberately looser 2.0x:
        measured 0.97x-1.02x across all three conditions.

    The estimator that was dropped, on the same runs: the wall-clock MEAN ranged
    0.68x-1.18x, a 1.7x swing on a build whose CPU cost did not move, and one
    repetition needed a single 5 ms stall to break the 1.5x bound.

    A 9.4x growth term would fail both surviving assertions by a wide margin.
    """
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D, durable="none")
    n = 10_000
    V = unit_rows(n, seed=5)
    wall = np.empty(n)
    cpu = np.empty(n)
    for i in range(n):
        t = time.perf_counter()
        c = time.process_time()
        e.add_fact(f"my phone number is 555-0{i:04d}", V[i], source="chat_session",
                   metadata={"user_id": f"u{i % 50}"})
        wall[i] = time.perf_counter() - t
        cpu[i] = time.process_time() - c
    cpu_first = float(np.median(cpu[:500]))
    cpu_last = float(np.median(cpu[-500:]))
    wall_first = float(np.median(wall[:500]))
    wall_last = float(np.median(wall[-500:]))
    print(f"\nadd cost, first 500 -> last 500: cpu median {cpu_first * 1e6:.1f} -> "
          f"{cpu_last * 1e6:.1f} us ({cpu_last / cpu_first:.2f}x), wall median "
          f"{wall_first * 1e6:.1f} -> {wall_last * 1e6:.1f} us "
          f"({wall_last / wall_first:.2f}x), wall mean "
          f"{float(wall[:500].mean()) * 1e6:.1f} -> {float(wall[-500:].mean()) * 1e6:.1f} us "
          f"| load {os.getloadavg()[0]:.2f}")
    assert cpu_last < 1.5 * cpu_first, (
        f"cpu median {cpu_last / cpu_first:.2f}x (wall median "
        f"{wall_last / wall_first:.2f}x)")
    assert wall_last < 2.0 * wall_first, (
        f"wall median {wall_last / wall_first:.2f}x (cpu median "
        f"{cpu_last / cpu_first:.2f}x -- if this one is flat the machine is busy, "
        f"not the library)")
    assert cpu_last < 200e-6
    e.flush()
    assert e.count() == n
    e.close()


def test_add_never_reads_a_block(tmp_path, monkeypatch):
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(300, seed=6)
    for i in range(120):
        e.add_fact(f"my email address is user{i}@example.org", V[i],
                   source="chat_session", metadata={"user_id": "u"})
    e.flush()
    reads = {"n": 0}
    real = e.arena.record
    monkeypatch.setattr(e.arena, "record",
                        lambda r: (reads.__setitem__("n", reads["n"] + 1), real(r))[1])
    for i in range(120, 300):
        e.add_fact(f"my email address is user{i}@example.org", V[i],
                   source="chat_session", metadata={"user_id": "u"})
    assert reads["n"] == 0
    e.close()


def test_stats_scale_with_n(tmp_path):
    """RAM per document, measured on the USED slice.

    ``active_heap_ram_kb`` counts allocated capacity, which grows by doubling and
    is therefore lumpy; the marginal cost per document is the honest per-doc
    figure and it is what this asserts.
    """
    sizes, used = {}, {}
    for n in (0, 1000, 5000):
        e = VaultEngine(filepath=str(tmp_path / f"v{n}.dat"), embed_dim=D)
        V = unit_rows(max(n, 1), seed=7)
        for i in range(n):
            e.add_fact(f"row {i}", V[i])
        e.flush()
        s = e.stats()
        sizes[n] = s["active_heap_ram_kb"]
        used[n] = (s["arena_used_bytes"] + s["column_bytes"] + s["record_bytes"]
                   + s["index_bytes_estimated"]) / 1024.0
        assert s["arena_bytes"] >= s["arena_used_bytes"]
        e.close()
    assert sizes[0] < sizes[1000] < sizes[5000]
    per_doc = (used[5000] - used[1000]) / 4000
    print(f"\nresident RAM per document: {per_doc:.2f} KB "
          f"(allocated heap at 5,000 docs {sizes[5000]:.0f} KB)")
    assert 2.8 < per_doc < 5.0                                 # ~3.1 KB per document


def test_index_size_vs_raw_text(tmp_path):
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(1190, seed=8)
    raw = 0
    for i in range(1190):
        text = f"paragraph {i} " + "lorem ipsum dolor sit amet " * 15
        raw += len(text.encode("utf-8"))
        e.add_fact(text, V[i])
    e.flush()
    on_disk = os.path.getsize(e.filepath)
    budget = raw + 1190 * D * 4
    print(f"\nindex {on_disk / 1e6:.2f} MB vs raw+fp32 {budget / 1e6:.2f} MB "
          f"({on_disk / budget:.2f}x)")
    assert on_disk <= 0.85 * budget                            # fp16 default
    e.close()


#: What a nanomem process may never pull in. The three benchmark engines are on
#: this list for a reason that is not hypothetical: `faiss` IS importable from
#: the interpreter that runs this suite (another project in the same repository
#: installed it), so "it is not installed" is not what keeps it out of nanomem --
#: this test is.
FORBIDDEN = {"faiss", "chromadb", "sqlite_vec", "torch", "cryptography",
             "sklearn", "mlx", "scipy", "pandas", "transformers", "sentence_transformers"}


def test_no_forbidden_third_party_imports():
    """numpy is the only third-party runtime dependency.

    Run in a subprocess so that re-importing the package cannot replace the
    exception classes the rest of this session already holds. EVERY module in the
    package is imported, not a hand-picked few, so a new module cannot quietly
    add a dependency.
    """
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(root, "nanomem")
    mods = sorted(f[:-3] for f in os.listdir(pkg)
                  if f.endswith(".py") and not f.startswith("_"))
    code = (
        "import sys, importlib; sys.path.insert(0, %r);"
        "import nanomem;"
        "[importlib.import_module('nanomem.' + m) for m in %r];"
        "print(sorted(%r & set(sys.modules)))" % (root, mods, FORBIDDEN)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_the_package_imports_nothing_but_numpy_and_the_stdlib():
    """The static half of the same rule: read the imports, do not just run them.

    The subprocess test above can only see what an import actually executes; a
    dependency hidden inside a function body, a lazy import in a rarely-taken
    branch, or a module nobody imports at start-up would slip past it. This walks
    the AST of every file in the package instead and collects every top-level
    module name any import statement mentions, wherever it sits.
    """
    import ast
    pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "nanomem")
    stdlib = set(sys.stdlib_module_names)
    local = {f[:-3] for f in os.listdir(pkg) if f.endswith(".py")}
    found = {}
    for fn in sorted(os.listdir(pkg)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(pkg, fn)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    found.setdefault(a.name.split(".")[0], set()).add(fn)
            elif isinstance(node, ast.ImportFrom):
                if node.level:                      # relative: same package
                    continue
                if node.module:
                    found.setdefault(node.module.split(".")[0], set()).add(fn)
    third_party = {m: sorted(v) for m, v in found.items()
                   if m not in stdlib and m not in local}
    assert set(third_party) == {"numpy"}, third_party
    assert not (FORBIDDEN & set(found)), sorted(FORBIDDEN & set(found))
