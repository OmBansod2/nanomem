# nanomem: competitive position

**nanomem 0.5.0 / engine 3.2.0**, measured
2026-09-16/17 on Mac16,11,
12 logical CPUs, macOS-26.5-arm64-arm-64bit.
Every number below comes out of a results JSON; the file it came from is named
in each section. Where a target is missed, the measured value is given.

**What moved in this revision.** **Axis 3 is the headline and it is a
retraction: the disk loss 0.4.0 booked is gone.** The `.arena` sidecar stopped
duplicating the vault, so a cached 71,433-document vault occupies
**153.05 MiB instead of 406.15**, below sqlite-vec's 258.0 — nanomem holds disk
*and* reopen at the same time, which 0.4.0 could not. Axis 2's reopen column
and Axis 3 were re-measured on the final tree by `bench_sidecar_size.py` and
independently by `scratch/refound/quality_summary.json`, which also re-derives
Axis 1's recall from the same vault. **Axis 6 gains the largest single
improvement in this document's history and it is not an engine change**: the
write gate's decision point moved, worth **+13.3 pt of end-to-end top-1**,
because 32.1% of that benchmark's questions were being destroyed at write time.
A new section, "Where a turn's milliseconds actually go", answers a question
none of the six axes could: with a real embedder in the loop the whole store is
**17.3% of a read turn at 71,433 documents and 0.8% at 1,190**, which reprices
every latency figure in Axis 2. Axes 4 and 5 are carried over. **Four more
mechanisms were measured and not shipped, one of which beat the exact-cosine
ceiling** — see Axis 1, Axis 6 and "Where nanomem loses" item 5.

---

## The one-paragraph verdict

The differentiator held up on the benchmark built to test it, and did not hold
up on the benchmark that looks like the product. On the temporal-supersession
set, nanomem at **shipped defaults with no arguments** scores
**90.5% top-1**, ahead of the strongest competitor
configuration -- Chroma with a hand-written three-way recency heuristic whose
parameters were swept to their optimum *on the test set* -- at
**86.2%**, a paired margin of
**+4.3 pts, 95% CI
[+0.9, +7.7]**,
which survives a user-level cluster bootstrap
([+1.5,
+7.4]). Against the
twenty-line two-way recency heuristic that a competent engineer writes in an
afternoon, the margin is **+17.8 pts
[+12.9, +22.7]**.
That reverses the previous measurement of this build's predecessor, which scored
44.2% and lost to the same heuristic. **But** on
the personal-memory chat benchmark the same engine scores
**63.9%** gold-store top-1 on a three-persona set that had
never been scored by anything before this session. That used to be reported
against an **80% target**; **the target is withdrawn as unreachable**, because
43.3% of the questions never name the attribute they ask about, and the
measured achievable ceiling from query text is
**66.4% [61.8, 70.8]**. So the honest reading is
**2.5 points left, not 16.1** — and four pre-registered mechanisms aimed at that
ceiling all failed, two of them significantly. On the previously-burned held-out
pair the work moved **75.0% -> 70.8%**, i.e. one question worse, not better. On
the standard retrieval task nanomem neither wins nor can win on recall: it ties
every exact engine at the ceiling.

**0.4.0 won reopen and paid for it on disk. 0.5.0 stops paying.** The sidecar
keeps *offsets* into the vault's own fp16 vectors and record sections instead of
a second fp32 copy of them, so a cached vault occupies
**153.05 MiB against 0.4.0's 406.15** — 4.4 MiB above a vault with no sidecar at
all, **below sqlite-vec's 258.0**, and still reopening in
**0.000228 s** against sqlite-vec's 0.0014 s. Scores are bitwise identical in
36 of 36 comparisons and 0 of 500 top-10 lists move. The cost is real and is
**not** disk: the upcast that used to be a mapped file is now anonymous memory,
so a serving process's `phys_footprint` reads 270.5 MB where 0.4.0 read 43.5 MB,
and a vault truncated out-of-band under a live engine now raises
`VaultShrankError` where 0.4.0 could not notice. Resident memory remains this
engine's worst axis — 298.4 MB over a 71,433-document load against FAISS's
230.2 MB, 1.296x — still a loss at scale, the **lowest** of every arm at 10,000
documents, and stated in full below.

**The largest number in this revision is not an engine number.** On the
personal-memory pipeline the shipped write gate was refusing 35.4% of real facts
and making **135 of 420 questions unanswerable before retrieval ran**. Moving
one constant takes the end-to-end top-1 of a 300-question held-out split from
**35.7% to 49.0%, +13.3 pt [+8.7, +18.3]**. Every retrieval mechanism this
project has shipped, measured end to end on the same fixture, is worth less than
that one threshold.

---

## How to read this

* **Two disjoint code paths.** The standard retrieval task stores HotpotQA
  paragraphs with `source="wiki"`, which is not in
  `nanomem.entities.PERSONAL_SOURCES`, so the entity/temporal layer is inert
  there by design -- proved by the engine returning FAISS's top-10 on 499 of 500
  queries. The temporal and chat tasks exercise that layer and nothing else.
  Neither task says anything about the other.
* **Every competitor got its strongest reasonable configuration**, recorded per
  arm in `scratch/refound/competitors_standard_results.json` and
  `scratch/refound/temporal_bench_results.json`: Chroma at ef_construction 512 /
  ef_search 512 / max_neighbors 64, Chroma's deprecated SegmentAPI brute-force
  path (which needs an undeclared wheel), sqlite-vec at float32 with WAL and a
  256 MB page cache, FAISS with 12 OMP threads, and on the temporal set the
  competitor's recency `delta` and pool size **swept on the test set and kept at
  the competitor's own optimum**. nanomem was given no tuning on either set.
* **Exactness is a ceiling, not an edge.** Six arms tie at the top of the recall
  table because all six compute the same cosines. nanomem cannot beat them by
  computing cosines better, and does not claim to. **One mechanism that beats
  the ceiling by not being a cosine at all was measured this round** — BM25
  fusion, +6.2 pt at recall@10 — and it is not in the build; Axis 1 says why,
  and the reasons are its own registered bars, not a preference for exactness.
* **The 0.5.0 rows were re-measured; the competitor rows were not.** FAISS,
  sqlite-vec and Chroma figures are carried from
  `competitors_standard_results.json`, measured on another day at another machine
  load. Only nanomem's side of each axis was re-run
  (`scratch/refound/quality_summary.json`, machine load 1.4-1.9).

---

## Axis 1 -- recall

Source: `scratch/refound/competitors_standard_results.json`. Evidence recall@4
is 1.0 only if *every* gold paragraph for the question is in the top 4. The CI
is a paired bootstrap over questions, 10,000 resamples, against nanomem.

### 71,433 documents / 500 questions

| engine | evidence recall@4 % | per-doc recall@10 % | delta vs nanomem (pts) |
|---|---|---|---|
| **nanomem (exact, shipped)** | 60.6 | 84.6 | -- |
| FAISS IndexFlatIP | 60.6 | 84.6 | +0.00 [+0.00, +0.00] tied |
| sqlite-vec | 60.6 | 84.6 | +0.00 [+0.00, +0.00] tied |
| Chroma default HNSW | 55.2 | 79.5 | -5.40 [-7.60, -3.40] **significant** |
| Chroma tuned (ef 512) | 60.6 | 84.6 | +0.00 [+0.00, +0.00] tied |
| Chroma "brute force" (Rust) | 54.4 | 79.1 | -6.20 [-8.40, -4.20] **significant** |
| Chroma legacy brute force | 60.6 | 84.6 | +0.00 [+0.00, +0.00] tied |
| nanomem router=auto | 60.8 | 84.5 | +0.20 [+0.00, +0.60] tied |

### 10,000 documents / 500 questions

| engine | evidence recall@4 % | per-doc recall@10 % | delta vs nanomem (pts) |
|---|---|---|---|
| **nanomem (exact, shipped)** | 70.4 | 92.0 | -- |
| FAISS IndexFlatIP | 70.4 | 92.0 | +0.00 [+0.00, +0.00] tied |
| sqlite-vec | 70.4 | 92.0 | +0.00 [+0.00, +0.00] tied |
| Chroma default HNSW | 69.6 | 91.2 | -0.80 [-1.60, -0.20] **significant** |
| Chroma tuned (ef 512) | 70.4 | 92.0 | +0.00 [+0.00, +0.00] tied |
| Chroma "brute force" (Rust) | 69.6 | 91.3 | -0.80 [-1.60, -0.20] **significant** |
| Chroma legacy brute force | 70.4 | 92.0 | +0.00 [+0.00, +0.00] tied |
| nanomem router=auto | 69.6 | 91.5 | -0.80 [-1.80, +0.00] tied |

At 1,190 documents every arm ties at 68.3% / 96.2%; that corpus cannot separate
the engines and is not reproduced here.

**Who wins:** nobody. A six-way tie at the exact ceiling -- nanomem plus
Chroma tuned (ef 512), Chroma legacy brute force, FAISS IndexFlatIP, sqlite-vec,
with nanomem's own router arm tied alongside.
The only real result is negative for Chroma: its **default** configuration and
its Rust "brute force" path both lose significantly, and Chroma 1.5.9's Rust
core exposes no exact index a stock user can reach -- `CreateCollectionConfiguration`
offers only `hnsw` and `spann`.

**Re-derived on the final 0.5.0 tree** (`scratch/refound/quality_summary.json`,
same vault, same 500 queries, shipped defaults): **60.6% evidence recall@4** and
**71.2% evidence recall@10**, identical with `arena_cache="off"`, **0 of 500**
top-10 lists differing. The recall row is the control for every mechanism in
this release and it did not move.

**Read the two @10 columns carefully.** The table's `per-doc recall@10` (84.6%)
counts a gold paragraph found; `evidence recall@10` (71.2%) requires *both* gold
paragraphs in the top 10. They are different metrics and have been confused
across files in this project. Where this document says "recall@10" from here on,
it means **evidence** recall@10 and the baseline is 71.2%.

### The one mechanism that beat this ceiling — and why it is not in the build

Source: `scratch/refound/hybrid_results.json`
(`prime_4d_unified_engine_2026_09_13/hybrid_retrieval.py`). A BM25 inverted
index written in numpy + stdlib — term-major CSR, Lucene's always-positive IDF,
no stemmer, no stopword list, no external dependency — fused with the dense
scores over the union of both candidate lists.

| at 71,433 documents, 500 questions | evidence recall@4 | evidence recall@10 |
|---|---|---|
| dense, shipped | 60.6% | 71.2% |
| BM25 alone | 35.4% (-25.2) | 52.8% (-18.4) |
| best @10 arm (min-max fusion, exact backfill) | 63.4% (+2.8 [+0.2, +5.4]) | **77.4% (+6.2 [+4.0, +8.6])** |
| the **pre-committed** arm | 63.2% (+2.6 **[-0.20, +5.20]**) | 77.2% (+6.0 [+3.8, +8.4]) |

**This is the first mechanism in this project to beat exact cosine on recall at
all**, and the reason is parameter-free: at depth 200 the dense pool contains
all gold for 92.4% of questions, BM25's for 83.0%, and the **union for 95.4%** —
3.0 points of gold the cosine never returns at any depth. Every one of five
fusion arms gains +5.0 to +6.2 pt at @10, with Bonferroni-adjusted lower bounds
still +1.6 to +3.2.

**It is not shipped, for four reasons, and none of them is a bar that moved.**

1. **It failed the bar it registered.** The registered primary was recall@**4**.
   The pre-committed arm scores +2.60 with a CI lower bound of **-0.20**, and
   under Bonferroni over seven families no arm's @4 lower bound stays above zero.
   At @4 the best arm fixes 30 questions and breaks 16; at @10 it fixes 35 and
   breaks 4. Fusion fills a top-10 reliably and reorders a head unreliably.
2. **It failed the registered latency gate**: **+1.0611 ms** added to a query
   (BM25 lookup 0.8832, fusion merge 0.1779) against a +1.0 ms limit. That was
   measured at machine load 6.6 and was deliberately **not** re-measured, which
   is correct — re-measuring until a bar is crossed is the failure mode this
   project has a rule against.
3. **Its disk claim no longer holds.** The index is 50.39 MiB: 12.4% of the
   406.2 MiB vault it registered 25% against, and **32.9%** of the 153.05 MiB
   vault this release writes. The denominator moved underneath it in the same
   session.
4. **The pipeline does not read that far down.** The win is at k=10. `chat.py`
   retrieves at `top_k=1` and `top_k=3`; `server.py` defaults to 3 and 5. At the
   window nanomem actually consumes, the mechanism has a point estimate of
   +2.6 pt and an interval that includes zero.

**Verdict: not integrated, not behind a flag, and it should not be the default.**
What would change that, in order: a caller that genuinely consumes a top-10 (a
re-ranker or an LLM context stuffer — then this is worth ~+6 pt for ~50 MiB and
~1 ms); a single-query fusion implementation re-measured on a quiet machine
against the same +1.0 ms bar; and a disk criterion re-registered against
153.05 MiB. What would **not** change it is another fusion sweep. Also
unmeasured: the mechanism was never run against the conversational benchmarks
nanomem ships for, where 43.3% of questions never name the attribute they ask
about — exactly where a lexical arm could fail differently.

---

## Axis 2 -- query latency, ingest, reopen

Sources: `scratch/refound/competitors_standard_results.json` for every arm, and
`scratch/refound/quality_summary.json` (0.5.0, re-measured on the final tree at
machine load 1.4-1.9, median of 9 in-process opens) for the two nanomem reopen
cells marked †; `reopen_results.json` holds the 0.4.0 readings the before/after
table cites. Every arm is built,
closed, **reopened from disk**, warmed with 20 queries, then timed on 500 unique
queries, one at a time, no batching. Reopen is the median of repeated opens in
one warm interpreter — the same protocol in both files, which is why the two
can be read in one column, and the agreement is checked rather than assumed:
`reopen_results.json` measures a cache-less nanomem open at
**0.1634 s / 0.0226 s / 0.0027 s** against the competitor harness's
**0.1609 / 0.0218 / 0.0027** at 71,433 / 10,000 / 1,190 documents.

### 71,433 documents

| engine | p50 ms | p95 ms | ingest s | reopen s |
|---|---|---|---|---|
| **nanomem (exact, shipped)** | 1.774 | 2.013 | 1.208 | **0.000228** † |
| nanomem `arena_cache="off"` | 1.774 | 2.013 | 1.208 | 0.1601 † |
| FAISS IndexFlatIP | 1.066 | 1.134 | 0.033 | 0.0121 |
| sqlite-vec | 53.353 | 55.258 | 1.354 | 0.0014 |
| Chroma default HNSW | 1.173 | 1.474 | 30.805 | 0.0017 |
| Chroma tuned (ef 512) | 8.527 | 11.093 | 75.617 | 0.0017 |
| Chroma "brute force" (Rust) | 1.258 | 1.572 | 43.575 | 0.0017 |
| Chroma legacy brute force | 279.148 | 282.531 | 21.577 | 0.0019 |
| nanomem router=auto | 3.302 | 3.521 | 13.056 | 7.0127 |

### 10,000 documents

| engine | p50 ms | p95 ms | ingest s | reopen s |
|---|---|---|---|---|
| **nanomem (exact, shipped)** | 0.364 | 0.426 | 0.164 | **0.000206** † |
| nanomem `arena_cache="off"` | 0.364 | 0.426 | 0.164 | 0.0212 † |
| FAISS IndexFlatIP | 0.392 | 0.403 | 0.005 | 0.0017 |
| sqlite-vec | 6.814 | 7.427 | 0.195 | 0.0010 |
| Chroma default HNSW | 0.886 | 1.031 | 3.212 | 0.0015 |
| Chroma tuned (ef 512) | 3.671 | 4.340 | 5.731 | 0.0016 |
| Chroma "brute force" (Rust) | 0.894 | 1.056 | 3.339 | 0.0016 |
| Chroma legacy brute force | 61.678 | 62.972 | 2.480 | 0.0017 |
| nanomem router=auto | 0.444 | 0.496 | 0.735 | 0.2670 |

**Who wins p50 at 71,433:** FAISS, at
1.066 ms
against nanomem's 1.774 ms --
nanomem is **1.67x slower**.
Chroma's default and Rust-brute-force arms also beat nanomem on p95, but those
are the two arms that gave up ~6 recall points. Restricted to arms *proven exact
against FAISS*, nanomem is second of five, and the gaps are large: 4.8x faster
than Chroma tuned, 30x faster than sqlite-vec,
157x faster than Chroma's only
real brute-force path.

**0.5.0 did not move p50 and was not supposed to.** The paired duel against the
0.4.0 layout reads x0.9976 with the screen on and x1.0037 with it off, and an
independent re-measurement on the final tree reads 1.1920 ms (screen `pca`) and
1.8032 ms (screen off) at 71,433 documents, against this table's cross-harness
1.774 ms — the same number to within the harness difference
(`quality_summary.json`).

**At 10,000 documents nanomem wins p50 outright**
(0.364 ms
against FAISS's
0.392 ms).
Together with reopen below, that is the whole of what nanomem wins on speed.

**Ingest:** FAISS wins at
0.033 s; nanomem is second at
1.208 s,
37x slower, but unlike
FAISS it is also writing the document text. nanomem is
26x faster than Chroma's
default ingest and 63x faster than Chroma tuned.

**Reopen: nanomem wins this axis, having lost it 115x.** At 71,433 documents the
0.5.0 default opens in **0.000228 s** (two independent runs, four cells:
0.000219-0.000246 s) against sqlite-vec's **0.0014 s** —
**6.1x faster**, and 7.5x faster than Chroma's 0.0017 s. At 10,000 documents it
is 0.000206 s against sqlite-vec's 0.0010 s. Against nanomem's own cache-less
behaviour the same open goes **0.1601 s → 0.000228 s, 702x**
(`quality_summary.json`; 0.4.0's offsets-free layout read 0.000193 s and 846x in
`reopen_results.json`, and the 0.4.0 → 0.5.0 difference of ~35 µs is the offset
binding, not a regression worth a decimal place — **0.5.0 buys 253 MiB of disk
for it**).

The mechanism is `arena_cache="map"`, new in 0.4.0 and **on by default**: the
resident arena is kept in a `<vault>.arena` sidecar laid out so an open is a
stat, a 4 KiB header read, a bind and an `mmap`, instead of a replay of every
block. In 0.5.0 that sidecar no longer contains the vectors or the record text —
only offsets into the vault's own regions — which is Axis 3. It is exactness-preserving and that is checked, not asserted — same
build, cache on against cache off, 500 queries scored against every row at all
three corpus sizes: the SHA-256 of the concatenated fp32 score vectors is
identical and **0 of 500** top-10 lists change, for `map`, `copy` and `verify`
alike. Steady-state p50 does not move either (paired duel, arms alternated
cycle by cycle, mapped/scanned = **0.9983**).

**Three things that win does not say, all measured.** First, on *time to first
answer* — reopen plus the first query, which is what a mapped arena actually
shifts rather than removes — the margin at 71,433 is
**15.4x** (0.1687 s → 0.0109 s),
not 846x, because the page faults the open skipped are paid by the first query
(0.0053 s → 0.0107 s). That still beats sqlite-vec's 0.0545 s and FAISS's
0.0153 s on the same arithmetic, and still loses to Chroma's 0.0030 s, which
pages its storage lazily and never builds a resident arena at all. Second, a
**fresh process** — a CLI invocation, not a server — sees
26.7x at 71,433 and 4.8x at
10,000, but a **loss at 1,190 documents**: 0.00764 s with the cache against
0.00321 s without it. Third, in 0.4.0 it cost 257.5 MiB of disk; **in 0.5.0 it
costs 4.4 MiB of disk and 227 MB of `phys_footprint` instead**, which is Axis 3.

The open that *writes* the sidecar pays for it once —
**0.3051 s against a 0.1659 s plain open** at 71,433 documents, of which
0.1315 s is the write (0.4.0's 257.5 MiB sidecar; 0.5.0 writes 4.4 MiB and the
fp32 upcast lands on the first query instead) — after which every open is the
0.000228 s one. Ingest above is unaffected and is still a loss.

### `screen="pca"` — 1.718x on p50, opt-in, and it changes no answer

Source: `scratch/refound/pca_screen_results.json`, re-run against this build
(read the `phase*` blocks; that file's `verdict` block is a carried-over copy of
the previous run, see Known gaps).

A second 0.4.0 mechanism, **default OFF**. It bounds every document's cosine
from above in a 256-dimensional subspace, skips the rows whose bound provably
cannot reach the top-k, and scores the survivors with the same fp32 kernel the
full scan uses. It is **not an ANN index**: no recall knob, no accuracy
trade-off, and its recall delta is **identically 0 by construction**.

| at 71,433 documents, 200 questions x 5 paired interleaved cycles | p50 | p95 | mean |
|---|---|---|---|
| `screen="off"` | 1.9495 ms | 2.0898 | 1.9328 |
| `screen="pca"` | **1.1350 ms** | 1.7318 | 1.1829 |
| speedup | **1.718x** [1.695, 1.736] | 1.207x | 1.634x |

Against the competitor table that puts nanomem within
**1.06x of FAISS's 1.066 ms** on p50 — but the two figures come from different
harnesses on different days, so it is not claimed as a win over FAISS, only as
1.718x over nanomem's own scan, measured with the arms interleaved query by
query inside one engine so both pay the same machine weather.

Pre-registered and all three clauses hold on this re-run: **exactness** — 0 of
3,000 searches with a different id list or a non-bitwise-identical float32
score list, 0 bound violations in **71,433,000** document checks, and 0 again
over 2,400,000 checks against rows appended after the basis was fitted;
**speed** — 1.718x against a 1.25x bar; **no regression below 10,000** — the
screen is never built below `screen_min_rows=20,000` and the shipped-default
p50 ratio at 1,000 / 5,000 / 10,000 documents is 1.0005 / 1.0002 / 0.9949.
Forced on below the floor it genuinely loses (0.478x at 1,000, 0.919x at 5,000,
0.977x at 10,000), which is why the floor exists.

**What it costs:** 1032 bytes per document resident — 71.6 MiB at 71,433
against a 209.3 MiB arena, +34.2% — plus a one-off 0.191 s basis build per
process. The basis is **recomputed, never written to disk**: the bound is
admissible for any basis, so persisting it could buy latency and could not buy
correctness. It stands down automatically, changing no result, below the row
floor, under `residency="int8"`, with a `metadata_filter`, on entity-tagged
vaults, for an explicit `temporal_direction="historical"`, when survivors would
exceed 35% of the corpus, and if the basis cannot be built.

**The one thing it charges to everyone:** reaching "identical top-k" required
breaking ties on the row id, which costs **16 us per search on the default
exact path** (0.8% of its p50) whether or not the flag is ever turned on, and
which is what reorders 8 of 500 top-10 lists against 0.3.2. Both are in the
before/after table.

---

## Where a turn's milliseconds actually go

Source: `scratch/refound/turn_latency_results.json`
(`scratch/refound/bench_turn_latency.py`), pre-registration hashed before any
measurement, engine pinned to a pristine `88dfac9` checkout with every imported
module hashed. 240 timed turns per corpus with the **real** Ollama
`nomic-embed-text` embedder — verified against nanomem's own provider (cos
1.000000) and against its md5 offline fallback (cos 0.011) so a silent fallback
could not be mistaken for a fast embedder.

**Every latency number in Axis 2 is a component figure. This is the whole turn.**

| documents | embed p50 | search p50 | turn p50 | search share | headroom if search were free |
|---|---|---|---|---|---|
| 1,190 | 11.211 ms | 0.088 | 11.312 | 0.78% | 0.89% [0.63, 1.60] |
| 10,000 | 11.893 ms | 0.416 | 12.255 | 3.39% | 2.95% [2.77, 4.39] |
| 71,433 | 11.910 ms | 2.494 | 14.404 | **17.31%** | **17.32% [16.60, 18.63]** |

The pre-registered bar said: if removing the store entirely buys less than 10%
of a turn, conclude that store optimisation has near-zero payoff. **The bar was
REJECTED at 17.32%**, replicated at 15.91% [14.68, 16.68] under different load.
So the suspicion that the store is a rounding error is right in direction and
wrong in size — but the *shape* of the answer is what matters:

* **Below ~10,000 documents the entire store is 0.8-3.4% of a turn.** Every
  millisecond Axis 2 argues about there is invisible to a user.
* **At 71,433 documents, scan optimisation specifically is spent.** The opt-in
  PCA screen already takes search 2.494 → 1.459 ms, and a search that cost
  *nothing at all* would buy 17% of the turn. There is no second index worth
  building on that budget.
* **80.3% of a realistic read turn is one embedding call**, and it is not
  nanomem's to optimise — but three things about it are the caller's:

| lever | saves | share of turn | where it lives |
|---|---|---|---|
| keep the embedding model resident (cold = 282.6 ms vs 11.4 warm, **24.7x**) | 271 ms | — | deployment |
| stop the write classifier dialling its own embedder (it does on 95.8% of turns) | 11.519 ms | **49.1% of a write turn** | nanomem's caller API |
| batch a 4-lookup turn's embeddings (4 serial = 57.5 ms, batched = 51.4) | 6.116 ms | 10.6% | caller |
| reuse the HTTP connection to the daemon | 0.365 ms | 2.5% | caller |
| `screen="pca"`, exact and opt-in | 0.872 ms | 6.1% | nanomem |
| search made instantaneous (hypothetical floor) | 2.494 ms | 17.3% | nanomem |
| arena cache off → on, **per search** | 0.0003 ms | 0.00% | nanomem |

The arena cache row is not a typo. Its entire value is reopen — 196.885 ms →
0.582 ms, and a cold process 353.0 → 160.9 ms. It buys a process start, not a
query.

### The finding that matters: the shape nanomem is actually for

Put **45 personal records into the same 71,433-document vault** — an agent that
remembers you while it also holds a corpus — and search goes **2.494 → 6.857 ms,
36.9% of the turn**. 5.564 ms of that is *ranking*, and 4.649 ms of the ranking
is `_resolve_revisions` alone, which walks all 71,478 candidates. And the PCA
screen engages on **0 of 256 queries** at every personal size, because
`_ranking_is_inert()` is false for the whole vault once any row is personal.

| personal rows in a 71,433-document vault | search share of turn |
|---|---|
| 45 | 1.02% |
| 1,235 | 1.80% |
| 10,045 | 7.80% |
| 71,478 | 36.91% |

**The cost scales with total rows, not with the rows the layer can affect.**

### What this implies for future work

1. **Make the entity/temporal layer cost O(top_k) instead of O(all rows).** It
   removes 5.564 ms — 29.9% of that turn — *and* re-enables the PCA screen for
   the one shape this engine exists for. This is the only store-side lever left
   that a user would feel.
2. **Do not build another index.** The router already failed on this argument
   (it ships off), the BM25 index fails its own cost gate above, and a free scan
   buys 17% of a turn at 71k and 3% at 10k.
3. **An embedding cache is dead on arrival**, and this was measured rather than
   assumed: 0.00% hit rate on the queries as issued, CI [0, 0.68] over 561
   queries; 0.00% [0, 0.33] over a 1,161-query multi-phrasing stream. Two
   phrasings of the *same* question reach cos p50 0.697 and **max 0.9649**, so a
   0.98 near-duplicate cache cannot fire on a re-ask with this embedder.
4. **The write path is where a cheap win is left.** A storing turn is 11.958 ms
   of which the store's own write work is 0.70 ms (5.9%); the classifier dials
   the embedder *itself* on 95.8% of turns, duplicating a vector the caller
   already has. Handing it that vector is a 11.5 ms saving in the caller's hands
   today (`classify(text, embedding=...)`) and an API default that should change.
5. **The 0.5.0 write-gate change costs 2.8x fewer rows than the alternative, and
   at these vault sizes it costs nothing a user can perceive**: p50 0.313 →
   0.377 ms on per-persona vaults inside a turn whose embedding call is 11.9 ms.
   nanomem's scan is linear, so that must **not** be read as a claim about 71k
   rows; it is a claim about the vault a personal deployment actually has.

**Caveat on every absolute millisecond here.** These were measured at 1-minute
load average 4.0-5.3 on 12 cores with a sibling process running; a quiet probe
read embed p50 9.0 ms, so the absolutes are inflated roughly 15-30%. The
*shares* are stable across replications (17.32% vs 15.91%) and every ratio is
paired and interleaved. Quote the shares, not the milliseconds.

---

## Axis 3 -- bytes on disk

**This axis moved the wrong way in 0.4.0 and moved back in 0.5.0. nanomem holds
disk and reopen at the same time, which no revision has managed before.**

Sources: `scratch/refound/competitors_standard_results.json`, `index_bytes`, as
each engine left it (no VACUUM, no forced compaction); the nanomem rows are
`scratch/refound/sidecar_size_results.json` and `quality_summary.json`, both of
which stat the files on disk rather than estimating them.

| engine | index MiB @71,433 | stores the text? |
|---|---|---|
| **nanomem 0.5.0 (shipped default)** | **153.05** = 148.66 vault + 4.39 sidecar | yes |
| nanomem `arena_cache="off"` | 148.66 | yes |
| FAISS IndexFlatIP | 209.3 | **no** |
| sqlite-vec | 258.0 | yes |
| nanomem 0.4.0 (shipped default) | 406.15 = 148.7 vault + 257.5 sidecar | yes |
| Chroma default HNSW | 488.4 | yes |
| Chroma tuned (ef 512) | 514.4 | yes |
| Chroma "brute force" (Rust) | 560.5 | yes |
| Chroma legacy brute force | 564.0 | yes |

**Who wins: nanomem, and by 1.69x over the nearest store that keeps the text.**
153.05 MiB against sqlite-vec's 258.0 and Chroma's 488.4-564.0 (3.2-3.7x). It is
also below FAISS's 209.3 MiB, which holds **vectors only, no text**, and is not a
like-for-like figure either way. At 10,000 documents the shipped default is
**21.60 MiB** (20.96 vault + 0.64 sidecar) where 0.4.0 needed 57.2.

**What changed.** 0.4.0's sidecar *was* the resident fp32 arena, so it duplicated
the vault: 209.28 MiB of vectors plus 43.81 MiB of record text beside a 148.7 MiB
file that already held both. It now stores neither. The block table the sidecar
already carried is enough to derive, for every block, where that block's fp16
vector run and its record section start **inside the vault** — so the offsets
cost zero new bytes, and what remains is 4.38 MiB of columns, id index and block
table.

| at 71,433 rows | sidecar | total | reopen | p50 (screen `pca`) | phys_footprint |
|---|---|---|---|---|---|
| 0.4.0 layout (`cache` / `cache`) | 257.48 MiB | 406.15 | 0.000187 s | baseline | 43.5 MB |
| **0.5.0 default** (`offsets_ram` / `vault`) | **4.38 MiB** | **153.05** | 0.000232 s | x0.9976 paired | 270.5 MB |
| `offsets` (no upcast) | 4.38 MiB | 153.05 | 0.000220 s | **x1.7325** | 57.1 MB |
| fp16 sidecar | 109.02 MiB | 257.69 | 0.000229 s | x0.9829 | 57.1 MB |
| `arena_cache="off"` | 0 | 148.66 | 0.1748 s | x0.9913 | 337.1 MB |

**It is a three-cornered trade and the corner nanomem chose is not free.** The
fp32 upcast still happens — it moved from a file the kernel maps to anonymous
memory this process dirties, so a serving process's `phys_footprint` reads
**270.5 MB against 0.4.0's 43.5 MB**. That cost was **not** priced by the
pre-registered decision rule that chose this default, which named `ru_maxrss`
only; it is stated here rather than quietly amended into the rule.
`arena_cache_vectors="cache"` restores 0.4.0 exactly, and `"offsets"` is the
cheapest arm on *both* disk and memory (57.1 MB phys) at a **x1.73 / x2.98** p50
penalty, because an exact cosine wants fp32 and the vault stores fp16.

**Exactness is measured, not asserted.** 36 of 36 arm x corpus x screen
comparisons share one SHA-256 over the concatenated fp32 score vectors of 500
queries against every row, and **0 of 500** top-10 lists differ, at all three
corpus sizes with the screen on and off. Independently re-derived on the final
tree: same digest behaviour, and evidence recall@4/@10 identical between the
shipped default and `arena_cache="off"` (`quality_summary.json`).

**The sidecar now reads the vault, so it can be hurt by the vault.** A vault
truncated **out of band, under a live engine** used to be survivable under the
copying layout and is a `SIGBUS` — an uncatchable process kill — when the arena
reads through a mapping. Every read through the mapping now `fstat`s the file
against the last byte its block table can address and raises `VaultShrankError`
instead, at 0.486 us per call (~0.4% of a query that materialises ten records).
It is a guard, not a guarantee: a truncation landing between the check and the
page touch still faults, and `arena_cache_vectors="cache"` is the layout with no
window at all. nanomem's own torn-tail recovery does not trip it.

---

## Axis 4 -- resident memory

**This axis moved in 0.3.2, moved again in 0.4.0, and moved BACK in 0.5.0.**
0.3.2 cut the *loader* peak by 0.35x. 0.4.0 did not cut the peak much further;
it changed what the resident bytes *are*, from dirty anonymous RAM to clean
file-backed pages the kernel may evict. **0.5.0 undoes that second change**: the
sidecar no longer holds fp32 vectors to map, so the upcast is anonymous memory
again and a serving process's `phys_footprint` reads **270.5 MB where 0.4.0 read
43.5 MB** — the price of Axis 3's 253 MiB of disk, paid in RAM, and it was not in
the decision rule that chose the default. `ru_maxrss` moves much less (331.0 MB
against 0.4.0's 318.4 in the same harness). Everything below is 0.3.2/0.4.0
measurement and is **not** re-run for 0.5.0; read it with that sentence in
front of it.

Sources: `scratch/refound/ingest_ram_results.json` (competitor-comparable
harness, four runs per arm plus two post-report checks; **not** re-run for
0.4.0),
`scratch/refound/memory_results.json` and `scratch/refound/reopen_results.json`
(both re-run against engine 3.1.0 in this session), and
`scratch/refound/competitors_standard_results.json` (the published competitor
column). Metric throughout: `ru_maxrss` high-water mark minus a baseline taken
after the shared vectors are resident.

### The loader shape at 71,433 documents -- one process ingests, then serves

| engine | peak RSS delta MB | vs nanomem |
|---|---|---|
| FAISS IndexFlatIP | 230.6 | **0.77x** -- wins |
| Chroma default HNSW | 290.5 | 0.97x -- level |
| **nanomem 0.3.2 (loader shape, not re-run for 0.4.0)** | **298.4** | -- |
| Chroma tuned (ef 512) | 316.5 | 1.06x |
| sqlite-vec | 335.9 | 1.13x |
| Chroma legacy brute force | 667.0 | 2.24x |
| nanomem 0.3.1 (the previous build) | 826.4 | 2.77x |
| Chroma "brute force" (Rust) | 885.3 | 2.97x |

### The loader shape at 10,000 documents

| engine | peak RSS delta MB | vs nanomem |
|---|---|---|
| **nanomem 0.3.2 (loader shape, not re-run for 0.4.0)** | **42.7** | -- |
| FAISS IndexFlatIP | 50.3 | 1.18x |
| sqlite-vec | 53.5 | 1.25x |
| nanomem 0.3.1 (the previous build) | 108.4 | 2.54x |
| Chroma default HNSW | 126.7 | 2.97x |
| Chroma tuned (ef 512) | 140.9 | 3.30x |
| Chroma "brute force" (Rust) | 167.1 | 3.91x |
| Chroma legacy brute force | 199.4 | 4.67x |

**Who wins:** FAISS at 71,433, at 230.6 MB against nanomem's 298.4 --
**1.296x**, still nanomem's loss. At 10,000 documents **nanomem wins**, at
42.7 MB against FAISS's 50.3.

### The before/after is a controlled A/B, not two different runs

The earlier version of this section compared a nanomem number to competitor
numbers from another harness. That is fixed twice over.

*The nanomem column* is one harness (`bench_memory.py`, arm `fp32_reserved`, the
shipped default) on one machine within the same hour, against a checkout whose
`arena.py` is byte-identical to `git show 6aa6923:` -- the committed 0.3.1 build:

| corpus | 0.3.1 | 0.3.2 | ratio |
|---|---|---|---|
| 71,433 documents | 819.1, 818.6 MB | 286.4, 286.9 MB | **0.35x** |
| 10,000 documents | 102.0, 103.0 MB | 41.2, 41.1 MB | **0.40x** |

Two runs per cell, both printed. The same file's driver pass reads 305.1 MB at
71,433, so today's honest range for the new build at that size is
**286.4-305.1 MB**; the other harness puts it at **287.9-310.0 MB** over six
runs, median 298.4, which is the figure quoted in the tables above (the higher,
less flattering of the two medians). **Quote a range, not a point.**

*The competitor column* was re-run inside the ingest harness rather than read
across from another one, because the 10,000-document margin was close enough to
matter (`ingest_ram_results.json`, `competitors_same_harness`, 3 runs each):
FAISS 50.3 MB at 10,000 and 230.2 MB at 71,433; sqlite-vec 53.0 and 335.8. Those
land within 0.5 MB of the published figures, so the cross-harness offset does
not reach the competitors and the ordering above stands. The Chroma rows are
still cross-harness, deliberately: nanomem is 0.21x-0.34x of them at 10,000 and
0.34x-0.45x at 71,433, margins far beyond any plausible harness offset.

### Nothing an answer depends on changed

The SHA-256 of the concatenated fp32 score vectors of all 500 queries against
all rows is **identical** between the two builds at both sizes
(`ingest_ram_results.json`, `exactness`) -- a check that does not depend on the
machine. **0 of 500** top-10 lists changed in all 20 (arm, corpus, phase) cells.
Evidence recall@4 is 70.4% / 60.6%, unchanged. p50 is
**1.7495 ms** at 71,433 against a 0.9605 ms bare-numpy floor measured in the
same process; a 7-cycle paired duel reads the change as **0.9966x**.

### Where the remaining ~290 MB is

It is no longer the allocator. On a reopened vault the arena is exactly the rows
-- `arena_bytes == arena_used_bytes == arena_reservation_bytes` = 219,442,176
and `arena_growth_copies` = 0 (that reading is 0.3.2's, i.e. `arena_cache="off"`;
under 0.4.0's default the same rows are mapped and `arena_bytes` is 0 — see
below) -- so the loader now peaks at the **server floor**,
i.e. what a process that only reads the file pays. Of it, 209.3 MiB is the fp32
vectors (the same 209.3 MiB FAISS holds), 43.8 MiB is the document text nanomem
returns and FAISS does not store at all, and 8.9 MiB is the columns and id
index. Getting under FAISS from here means attacking resident text or the id
index -- a format decision, not an allocator one.

### What 0.4.0 changed: the server's bytes became evictable

A query server that opens a vault another process wrote and serves from it now
maps its arena instead of rebuilding it, and that shows up twice
(`reopen_results.json`, `headline.n71433`, and `memory_results.json`,
`reopen_only`):

| at 71,433 documents | scanned arena | mapped arena |
|---|---|---|
| peak `ru_maxrss` | 287.5 MB | **258.2 MB** |
| `phys_footprint` | 288 MB | **9 MB** |
| p50 | 1.8127 ms | 1.7806 ms |

The `ru_maxrss` saving is 10%. The `phys_footprint` saving is **32x**, and it
is the one that matters for a machine under pressure: the 219 MiB of vectors
are now clean pages backed by the sidecar, so the OS can drop and re-fault them
instead of swapping them. `arena_cache="copy"` spends 219 MB of dirty anonymous
RAM to make the first query as fast as the second; it is one keyword and it is
measured in the same table (`headline.n71433.cache_copy`: 467.5 MB peak).

**A `stats()` key changed meaning, and anything trending it will see a step.**
`arena_bytes` counts ANONYMOUS memory only — deliberately, because counting
clean evictable pages as RAM is the over-reporting 0.3.2 removed. On a cached
open it is therefore **0**, and the 219,442,176 bytes of vectors appear under
`arena_cache_mapped_bytes` / `arena_mapped_bytes` instead, with
`arena_used_bytes` unchanged at 219,442,176. `arena_from_cache` and
`arena_vectors_mapped` say which shape you have.

### The loader shape did not move materially, and was not supposed to

Build 71,433 documents, close, reopen, serve: `fp32_reserved` reads
**290.0 MB** on this build against **286.4, 286.9 MB** for 0.3.2 in the same
harness, and **42.4 MB** against 41.2, 41.1 at 10,000
(`memory_results.json`, re-run). The ingest peak dominates that shape and the
sidecar write adds a few MB to it; at one run per cell against a documented
286.4–305.1 MB spread, **~3 MB is not resolvable** and no claim is made in
either direction. The competitor-comparable harness that produced the 298.4 MB
figure in the tables above was **not** re-run, so those tables are 0.3.2's
loader numbers and are labelled as such.

### The server shape did not move in 0.3.2, and was not supposed to

Opening a vault another process wrote and serving from it: **286.0 MB ->
285.2 MB** at 71,433 (`memory_results.json`, `reopen_only`). 0.3.1 already fixed
that shape; 0.3.2 fixed the loader shape, which was the one still losing; 0.4.0
then made the server shape's bytes evictable, as above.
`residency="float16"` reaches 193.2 MB -- below FAISS -- but costs
**2.33x** the p50 over 5 interleaved
cycles, so it stays one keyword away rather than becoming the default.

**Caveat on the reservation.** `arena_reservation_bytes` is address space, not
RAM: an untouched page of an anonymous mapping is not resident. Unhinted, a
209 MB arena sits behind 552 MB of reserved VM. It is reported separately and
must never be added to `arena_bytes`.

**A harness defect found while re-running this.** `bench_memory.py`'s
`legacy_303_fp32_doubling` arm builds its baseline by disabling
`Arena.reserve`, which was the entire difference under 3.0.4. Under 3.0.5 the
vectors size themselves exactly whether or not anyone calls `reserve`, so that
arm no longer reconstructs anything -- its server figure moved 643.6 MB ->
285.7 MB between builds while `fp32_reserved` stayed at 286.0 -> 285.2. The
harness cannot self-detect it, because its own assertion is on `_next_capacity`,
which still doubles for the column arrays. **Do not read that arm as a 0.3.1
baseline**; the before-column above comes from a real checkout instead. A test in
`tests/test_arena_residency.py` now pins this so it cannot be misread.

---

## Axis 5 -- temporal supersession (top-1 by question type)

Source: `scratch/refound/temporal_bench_results.json` (10 synthetic users, 456
documents, 326 questions, vocabulary audited clean against every persona fixture
in the tree). All arms rank identical vectors; Chroma and sqlite-vec return the
same top-3 as exhaustive cosine on 326/326, so no competitor is losing recall to
approximation. The competitor's `delta`/pool were swept on the test set and kept
at the competitor's optimum
(kpool=3,
delta=0.08).

| arm | overall top-1 | current | historical | previous | adjacent sibling | third party | excl. previous | 8-user holdout |
|---|---|---|---|---|---|---|---|---|
| plain cosine (floor) | 47.9 | 43.0 | 22.0 | 37.5 | 100.0 | 100.0 | 50.0 | 48.7 |
| Chroma, no temporal logic | 47.9 | 43.0 | 22.0 | 37.5 | 100.0 | 100.0 | 50.0 | 48.7 |
| Chroma + recency heuristic (swept) | 72.7 | 94.0 | 73.0 | 16.1 | 77.5 | 100.0 | 84.4 | 73.2 |
| sqlite-vec + recency, as SQL | 72.7 | 94.0 | 73.0 | 16.1 | 77.5 | 100.0 | 84.4 | 73.2 |
| Chroma + 3-way recency (strongest competitor) | 86.2 | 94.0 | 73.0 | 94.6 | 77.5 | 100.0 | 84.4 | 86.6 |
| Chroma + PERFECT extractor (oracle) | 82.5 | 100.0 | 99.0 | 0.0 | 100.0 | 100.0 | 99.6 | 82.8 |
| **nanomem, shipped defaults, no arguments** | 90.5 | 100.0 | 75.0 | 98.2 | 90.0 | 96.7 | 88.9 | 90.4 |
| nanomem with temporal_direction passed | 77.3 | 100.0 | 75.0 | 21.4 | 90.0 | 96.7 | 88.9 | 77.4 |
| nanomem + documented metadata['entity'] | 59.2 | 89.0 | 64.0 | 23.2 | 17.5 | 66.7 | 66.7 | 59.0 |

Paired bootstrap, 10,000 resamples, question-level (source:
`scratch/refound/final_scorecard.json` -> `temporal.extra_paired_tests`, whose
per-question digits are embedded so any interval can be recomputed):

| comparison | delta | 95% CI | verdict |
|---|---|---|---|
| nanomem_default - plain cosine floor | +42.6 | [+36.8, +48.5] | significant |
| nanomem_default - Chroma + recency | +17.8 | [+12.9, +22.7] | significant |
| nanomem_default - Chroma + 3-way recency | +4.3 | [+0.9, +7.7] | significant |
| nanomem_default - Chroma + 3-way recency, **user-level cluster bootstrap** | +4.3 | [+1.5, +7.4] | significant |
| nanomem_default - Chroma + PERFECT extractor (oracle) | +8.0 | [+2.8, +13.5] | significant |
| **excluding previous_value** -- nanomem_default - Chroma oracle | -10.7 | [-14.4, -7.0] | **nanomem loses** |

The single cleanest number on this axis: on "what is my *current* X?" questions a
plain vector store returns a **superseded** value
57.0% of
the time; the Chroma recency heuristic
5.0%;
nanomem at shipped defaults
0.0%.

---

## Axis 6 -- personal-memory chat (the product shape)

Source: `scratch/refound/clean_chat_results_persona4_v3r5.json`,
`clean_chat_results_heldout_v3r5.json`, `clean_chat_results_r5_3p.json`,
`clean_chat_results_v3r4_engine*.json`. Gold-store top-1: the labelled turns are
written, the vault is **closed and reopened**, the question is asked at top_k=3.

| fixture | status | before | after |
|---|---|---|---|
| clean_chat_benchmark.json (3 personas) | tuned on during development | 91.7% | **97.2%** |
| clean_chat_benchmark_heldout.json (2 personas) | measured repeatedly -- an upper bound | 75.0% | **70.8%** |
| clean_chat_benchmark_persona4.json (3 personas) | **never scored before; scored once, here** | -- | **63.9%** |

The clean estimate is **63.9%** (23 of 36). It used to be reported here as
"against an 80% target, missed by 16.1 points". **That target is withdrawn**:
see "The 80% target was not reachable" below, which replaces it with a measured
ceiling of **66.4% [61.8, 70.8]**. Against *that*, 63.9% is 2.5 points short of
the ceiling rather than 16.1 points short of a number nothing on this task can
reach. Top-3 on the same set is
86.1%,
per-persona 7/12, 7/12, 9/12.

Two further arms on the same set, both declared before either was scored
(`final_scorecard.json` -> `chat_personal_memory`):

* **58.3%
  if no `temporal_direction` argument is passed at all** -- which is what
  `chat.py` and `cli.py` actually do. The 63.9% headline is
  the arm that passes it, because that is the arm the
  75.0% figure was produced by and the
  comparison has to be like-for-like. The shipped code path is the lower number.

End-to-end, where the write-time classifier decides what to store instead of the
fixture's labels, persona4 scores
55.6%
(`clean_chat_results_persona4_v3r5.json`). The classifier is not part of the
engine work, but it is what a user would experience — and it turned out to be
the **largest single loss on this task**, larger than anything retrieval was
doing. See "The largest loss on this task was at WRITE time" below; 0.5.0 moves
it and this 55.6% figure is a 0.4.0 measurement that was not re-run.

Is the gap between the burned set's 75.0%
headline (70.8% on this build) and persona4's
63.9% contamination, or is the new set simply scored more
strictly? It is mostly strictness. persona4
has **0 of 36** queries whose expected substring appears in more than one stored
turn; the tuned set has
10 of 36
and the burned set
4 of 24,
so a top-1 hit on the older sets can be credited for landing on the wrong turn.
Re-scored under persona4's own rule -- unique-substring queries only, no persona4
question inspected:

| fixture, strict rule | top-1 |
|---|---|
| clean_chat_benchmark.json (tuned on) | 26/26 = 100.0% |
| clean_chat_benchmark_heldout.json (burned) | 13/20 = 65.0% |
| clean_chat_benchmark_persona4.json (clean, already strict) | 23/36 = 63.9% |

So the honest clean level for this task is about **64%**, not 75%. The 100% on
the tuned-on set is not a capability claim; it is what a set that was optimised
against looks like.

### The 80% target was not reachable, and the number that replaces it

Source: `scratch/refound/chat_target_decision.json`, `restated_target`.

**43.3%** of the questions on the chat benchmark never name the attribute they
are asking about — "what was it again?", "and the other one?" — so no amount
of ranking or tagging can identify which stored attribute is wanted, because
the identity is not in the query. Driving **both** ends of the entity chain to
perfect — perfect document tags *and* perfect query tags — reaches
**66.4% [61.8, 70.8]**, with the whole interval **13.6 points below the 80%
target**. The 89.5% figure that does clear 80% is an *annotation* ceiling: it is
reached only by handing the engine a label the query does not carry.

So the target was not missed, it was mis-specified. Three consequences, all
stated rather than implied:

* **The defensible level for this task is at most 66.4%**, not 80%, unless the
  question distribution itself changes.
* **When the question does name the attribute, the shipped build already scores
  84.3% [80.5, 87.5]**. The product lever is therefore the *task shape* —
  canonicalise the question, or ask the user which attribute they mean — not
  the retriever.
* An earlier "+32.6 pt from perfect entity tagging" rung, and the 86.7%
  "achievable ceiling" it rested on, are **withdrawn**: that oracle keyed on
  normalised query text and leaked labels through paraphrases shared across
  personas, tagging 411 of 420 questions including 173 of the 182 it existed to
  withhold from. Do not resurrect them.

### Four mechanisms that tried to break that ceiling. All negative.

Source: `scratch/refound/context_lever_results.json` (91 audited arms,
`audit_summary.all_clean = true`), pre-registered before measurement
(`preregistration_sha256` 07864f187f066e11), scored on the 300-question **test**
split of `chat_bench_v2.json`, which is split **by persona** so dev and test
share no user. Bar: **+5.0 pt with the CI lower bound above 0**.

| mechanism | test top-1 | delta vs 52.3% baseline |
|---|---|---|
| (b) per-user attribute priors | 53.3% | +1.0 [-0.3, +2.7] — best arm, **not significant** |
| (c) multi-intent retrieval | 53.0% | +0.7 [-0.7, +2.0] |
| (e) combined / override | 53.0% | +0.7 [-1.3, +2.7] |
| (d) session recency | 50.7% | -1.7 [-4.0, +0.7] |
| (a) conversation context | 49.0% | **-3.3 [-6.0, -0.7] — significantly negative** |

**Nothing cleared the bar; the best arm reached one fifth of it.** On the
ceiling stack the same mechanisms moved 66.7% → **65.0%, -1.7 [-3.3, -0.3]** —
the ceiling was not broken, it was *defended*. And on the 130 head-dropped test
questions each mechanism exists for, the two best arms moved the number by
**nothing at all**: 28.5% → 28.5%.

Two facts explain it. Conversation context cannot help because the needed
attribute is in the prior window only **7.0%** of the time at a 5-turn window
and **16.7%** at 12; it reaches 97.0% only at the whole 118-turn session, where
the "window" holds ~22 candidate attributes and is no longer a window. And the
most recently discussed attribute is the one being asked about **4.3%** of the
time, which kills session recency outright.

**But the ceiling is *selection*-bounded, not information-bounded, and that is
the useful result.** A diagnostic oracle that supplies no value, no document and
no direction — only *which* of the attributes the session already names is the
right one — reaches **80.8% on dev, +15.0 pt [9.2, 21.7]**. The answer is in
the vault. What no shipped mechanism can do is pick it: on head-dropped
questions the best implementable selector names the right attribute 75.9% of
the time *where the retriever was already right* (n=29) and **3.9%** where it
was wrong (n=76). It recovers the attribute only where it was not needed.

**What this means for the product.** Query-text-only retrieval is close to done
on this task at ~66%, and the remaining ~14 points need a *closed-set selector*
— multiple choice over the vault's own entity table — which is a different
problem from the open-vocabulary tagging whose ceiling was withdrawn. Embedding
similarity does not solve it (23.8% on head-dropped, errors correlated with the
retriever's own). Anything that would — an LLM, or a trained cross-encoder over
(question, attribute name) — sits **outside nanomem's numpy-only runtime** and
belongs in the caller. That is an architectural decision, not an incremental
one, and it is left open here rather than pre-judged.

**Do not retry** (measured negative): short context windows, session recency,
unthresholded priors, and any arm that overrides a confident tagger.

### The largest loss on this task was at WRITE time, and 0.5.0 fixes it

Source: `scratch/refound/write_policy_results.json`
(`prime_4d_unified_engine_2026_09_13/write_policy.py`), pre-registered and
sha256-verified before any number, swept on a 120-question dev split, scored
**once** on a disjoint 300-question test split, with a mechanical leak audit
that counts every oracle read per arm and raises on a non-zero count.
Reproduced end to end through the shipped classifier in
`scratch/refound/quality_summary.json`.

Everything above this line is about *retrieval*. The biggest number on this task
was never there. The shipped write gate ran at **100.0% precision and 64.6%
recall**: it admitted zero of 1,064 noise turns and refused **208 of 588 real
facts**, so **135 of 420 questions (32.1%) were unanswerable before retrieval
ran at all**. 10.6% of everything the gate dropped was later asked about.

| on the 300-question test split | top-1 | delta | docs | MiB | questions unanswerable |
|---|---|---|---|---|---|
| shipped gate (threshold 0.60) | 35.7% | — | 271 | 0.492 | 100 |
| **0.5.0: threshold 0.05** | **49.0%** | **+13.3 [+8.7, +18.3]** | 425 | 0.764 | 16 |
| store everything | 49.7% | +14.0 [+8.3, +19.7] | 1,180 | 2.014 | 0 |
| perfect gate **[ORACLE]** | 52.3% | +16.6 [+11.7, +22.0] | 420 | 0.760 | 0 |

Both readings of the pre-registered bar are cleared, including the strict one
(CI lower bound itself ≥ +5.0), and so is the persona-clustered interval
([+9.7, +16.7]). **Store-everything captures 96% of what a perfect gate recovers
and costs 5.3x more bytes per point of accuracy**, so the threshold move is the
efficient frontier point, not the extreme one: 0.0205 MiB per point against
0.1087.

The decomposition is worth keeping. Of the 100 test questions the old gate made
unanswerable, store-everything answers 51 and the threshold move answers 44 —
but on the 200 questions the gate did *not* break, store-everything falls
53.5% → 49.0%. That −4.5 pt is the measured cost of noise, and the threshold arm
pays only −2.0 pt of it. Noise is not free; it is just much cheaper than a
permanently missing fact.

**Diagnosis, because the number is less useful than the reason.** The threshold
was chosen by leave-one-persona-out accuracy/F1, where a false positive and a
false negative cost the same. In deployment they do not: a false negative
destroys a question permanently, a false positive costs ~1.9 kB and no
measurable time. The gate's own F1 is better at 0.20 (0.945) than at the shipped
0.60 (0.784), so 0.60 was not optimal even on the objective it was picked for.
**Any threshold in a library that was tuned on a symmetric metric is suspect for
the same reason.**

**A second, independent write-time loss is measured and NOT fixed.** With the
learned head fully off (threshold 0.00) the classifier's *rule layers* still drop
52 of 1,180 test turns, 14 of them answers to questions, capping gate recall at
95.0% and costing ~3.4 pt. Moving the threshold does not touch it. It needs its
own pass over the forget-directive and too-short rules, and that is the next
thing to measure on this task.

**Two arms that also passed a bar and are NOT shipped.** Writing the gate's
confidence into metadata and letting retrieval down-weight low-confidence rows is
dead: its dev argmax is weight 0, i.e. the mechanism switched off. A two-tier
confidence shard passes the bar (+8.0 [+3.7, +12.3]) and is Pareto-dominated on
accuracy, bytes *and* p50 — and it produced a result worth carrying forward:
splitting one vault into two costs **−7.5 pt on dev even when both shards are
always searched and merged by raw score**, because nanomem's entity boosts are
per-vault statistics. **A confidence shard is not a free index split, and
neither is any other sharding scheme in this engine.**

### Selection without an LLM: the bar was cleared, and it still does not ship

Source: `scratch/refound/selection_results.json`
(`prime_4d_unified_engine_2026_09_13/selection.py`). Six mechanisms replacing
exactly one function — `entities.query_intents`, the engine's own selection hook
— each seeing only the vault's interned entity names and record texts, never a
gold label, with a gold-scramble audit that permutes every label and requires
byte-identical predictions.

The section above says the remaining ~14 points need a *closed-set selector* and
that embedding similarity does not solve it (23.8% on head-dropped questions).
**That conclusion was drawn from one signal, and that signal is the worst of the
six**: question-vs-attribute-*name* cosine is not merely weak, it is −6.3 pt on
test, actively harmful. Two things it missed: pooling the **same** embeddings
over an attribute's whole record *group* is +6.7 pt untrained, and a dev-fitted
bilinear map over the **same** vectors is +10.0 pt [+2.3, +17.7] on exactly the
head-dropped questions.

| test, 300 questions | overall top-1 | head-dropped |
|---|---|---|
| shipped selection (baseline) | 52.3% | 28.5% |
| ensemble (A7) | **63.3% (+11.0 [+6.7, +15.7])** | **39.2% (+10.7 [+3.8, +18.5])** |
| attribute-agnostic (A8) | 57.7% (+5.3 [+1.7, +9.3]) | 26.9% (−1.6 [−6.2, +3.1]) |
| name cosine (the prior study's one signal) | 46.0% (−6.3) | 23.1% (−5.4) |
| oracle selection | 77.3% | 65.4% |

**The ensemble clears its pre-registered bar and is still not integrated**, and
the reason is the study's own leave-one-ATTRIBUTE-out check, declared as a
confound *before* test was scored: the benchmark draws one question template per
(attribute, type, head) cell and dev and test share that table. Held out by
attribute, the two components carrying the head-dropped gain collapse to **3.7%
and 1.9%** selection accuracy and the ensemble falls to 57.0%, *below* untrained
group cosine at 71.0%. So the gain is per-attribute supervision transferring
through shared wording, not language understanding. For a fixed attribute
vocabulary with a few labelled questions each, it is real. For an attribute a
user invents tomorrow, it is nothing.

The arm that *does* survive that check, A8, clears the overall bar and does
**nothing on the head-dropped questions the study exists for** (−1.6
[−6.2, +3.1]) — its whole gain is on questions that already name their
attribute, where group-pooled cosine simply resolves the attribute better than
the engine's rule table. That is a different win from the stated problem.

**Three things must be measured before any of this enters the engine**, and none
of them was: (1) the temporal set, whose 90.5% depends on the very hook A8
replaces and which the study never ran; (2) the end-to-end rung with the real
write gate, rather than the gold-store rung the pre-registration named; (3) an
API for handing the engine a learned selector plus somewhere to keep per-vault
parameters, neither of which exists. Until then this is a measured result, not a
feature.

**One diagnostic from it is worth more than the arms.** Answer-type matching —
the idea this document previously rated most promising — is a clean negative as
a selector (13.7% test, −38.6 pt), and the reason is isolated: an attribute's
value-shape signature narrows ~24 candidates to 2.73 and an **oracle** answer
type plus a group-cosine tiebreak selects at 87.9% overall / 72.3% head-dropped.
Value shape separates the attributes well. **Reading the answer type off the
question is the step that fails** — the hand map puts the gold attribute in its
predicted tier only 57.0% of the time — and that step is precisely what an LLM
or a cross-encoder is for. The architectural conclusion above stands, with a
sharper boundary around it.

---

## Where nanomem loses

Not softened. Each item cites the file it comes from.

1. **Peak RSS -- still lost at scale, but no longer badly, and no longer
   last.** At 71,433 documents nanomem is
   298.4 MB against FAISS's
   230.2 MB in the same
   harness: **1.296x**, and
   FAISS wins. That is the loss. What changed is its size: the previous build
   was 826.4 MB, i.e.
   **3.58x** FAISS, and
   second-worst of every shipped arm. nanomem is now **level with Chroma's
   default HNSW** (290.5 MB;
   1.027x, inside that arm's
   own 7% run-to-run spread, so call it level) and **below** Chroma tuned
   (316.5), sqlite-vec (335.9), Chroma legacy brute force (667.0) and Chroma's
   Rust brute-force path (885.3). At **10,000 documents nanomem is the lowest
   arm measured** -- 42.7 MB against FAISS's 50.3 and sqlite-vec's 53.5 -- so
   the loss is specific to scale.
   **What is left is not fixable in the allocator.** The loader now peaks at the
   server floor: the arena on a reopened vault is exactly the rows, allocated
   once, copied zero times. Of the remaining ~290 MB, 209.3 MiB is the fp32
   vectors FAISS also holds, 43.8 MiB is document text FAISS does not store at
   all, and 8.9 MiB is columns and the id index. Beating FAISS from here is a
   storage-format decision.
   **0.4.0 does not change this loss**, and the loader figure above is 0.3.2's,
   because the competitor-comparable harness was not re-run; the same shape in
   `bench_memory.py` reads 290.0 MB on this build against 286.4/286.9 for
   0.3.2, which is inside the documented 286.4-305.1 MB spread and is not
   resolvable at one run per cell. What 0.4.0 *does* change is the shape of a
   SERVING process's bytes -- `phys_footprint` 288 MB -> 9 MB at 71,433, because
   the vectors become clean file-backed pages -- and `ru_maxrss` 287.5 -> 258.2.
   Neither is the loader number in the table above, and they must not be pasted
   into it. (`ingest_ram_results.json`,
   `competitors_standard_results.json`, `memory_results.json`,
   `reopen_results.json`.)
2. **p50 and p95 at scale.** 1.67x
   FAISS on p50 and
   1.77x
   on p95, at the shipped default, which 0.4.0 does not move (paired duel
   0.9983). The opt-in `screen="pca"` cuts nanomem's own p50 by
   **1.718x**, which would put it
   within 1.06x of FAISS's figure — but that is two harnesses on two days, it
   ships OFF, and no win over FAISS is claimed from it.
   (`competitors_standard_results.json`, `pca_screen_results.json`.)
3. **A serving process's `phys_footprint` is 6.2x what 0.4.0's was -- this is
   what bought the disk axis back.** 270.5 MB against 43.5 MB at 71,433
   documents. The fp32 upcast did not go away; it moved out of a file the kernel
   maps and into anonymous memory this process dirties, which the kernel cannot
   evict. **This cost was not priced by the pre-registered rule that chose the
   default**, which named `ru_maxrss` only (331.0 MB, inside its 388.2 MB cap),
   and it is stated rather than amended into the rule after the fact.
   `arena_cache_vectors="cache"` restores 0.4.0's 43.5 MB and its 406 MiB;
   `"offsets"` gets the disk *and* 57.1 MB of phys at x1.73 on p50.
   (`sidecar_size_results.json`.)
   **Bytes on disk and reopen are both off this list**: 153.05 MiB against
   sqlite-vec's 258.0 (1.69x ahead) and 0.000228 s against its 0.0014 s (6.1x
   ahead). Axis 3 has the numbers and the trade.
   **New failure mode, introduced here.** The arena now reads through a mapping
   of the vault, so a vault truncated **out of band under a live engine** was an
   uncatchable `SIGBUS` -- exit 138, no Python exception -- where the copying
   layout survived it. It now raises `VaultShrankError` at 0.486 us per read,
   but that is a guard with a window, not a guarantee: a truncation landing
   between the `fstat` and the page touch still faults. `"cache"` has no window
   at all. Untested and pre-existing: truncating the `.arena` file itself under
   a live engine. (`tests/test_arena_offsets.py`.)
4. **Ingest.** 37x FAISS.
   (`competitors_standard_results.json`.)
5. **Recall: nanomem ties every exact engine and does not beat one.** 60.6%
   evidence recall@4 at 71,433 documents, identical for FAISS, sqlite-vec,
   Chroma tuned and Chroma legacy brute force, because all five compute the same
   cosine. Re-derived on the 0.5.0 tree (`quality_summary.json`).
   **One mechanism beat that ceiling this round and was not shipped**: BM25
   fusion reaches 77.4% evidence recall@10 against 71.2% (+6.2 [+4.0, +8.6],
   Bonferroni-positive), because BM25's top-200 holds 3.0 pt of gold the cosine
   never returns at any depth. It failed the recall@**4** bar it registered
   (+2.60, CI lower bound **-0.20**), failed its latency gate (+1.0611 ms
   against +1.0 ms), its index is 32.9% of the vault against a registered 25%
   limit, and the pipeline reads a top-1 to top-3 window, not a top-10. So the
   honest statement is not "nanomem cannot beat exact cosine" -- it is
   **"nanomem has now measured the thing that does, and it does not pay at the
   depth this engine is read at."** (`hybrid_results.json`, Axis 1.)
6. **Sibling attributes -- still worse than doing nothing.** On
   adjacent-but-different attributes ("backup email" vs "primary email") nanomem
   scores 90.0%
   against a plain cosine scan's
   100.0%,
   with a 10.0%
   sibling-confusion rate where plain cosine has 0.0%. `USER_MANUAL_DEVELOPER.md`
   still claims parity with plain cosine here; on this fixture-free set that is
   false. (`temporal_bench_results.json`.)
7. **Third-party facts -- still worse than doing nothing.**
   96.7% against
   plain cosine's 100.0%
   and the Chroma heuristic's
   100.0%; the
   remaining failure is the inner-possessive form ("my colleague X's desk
   phone"), an own-fact leak rate of
   3.3%.
   (`temporal_bench_results.json`.)
8. **Given the same gift, the competitor is still better on the types both
   handle.** Excluding `previous_value` (which both oracles score 0.0% on),
   nanomem is 88.9%
   against Chroma-plus-a-perfect-write-time-extractor at
   99.6%:
   -10.7 pts,
   CI [-14.4,
   -7.0].
   nanomem's overall win is partly the competitor oracle's total failure on one
   question type. (`final_scorecard.json`.)
9. **The documented `metadata["entity"]` path is now a regression.** Writing
   facts with the tag `USER_MANUAL_DEVELOPER.md` documents scores
   59.2% -- far *below* the
   90.5% you get by doing nothing,
   and below the 74.2% the
   previous build scored on the same arm. Following the manual makes the engine
   worse. (`temporal_bench_results.json`.)
10. **The documented `temporal_direction` argument is now a footgun.** Passing
    the two-way direction costs
    13.2 points
    (90.5% ->
    77.3%) because it overrides
    the engine's own three-way reading of the question.
    (`temporal_bench_results.json`.)
11. **The chat task is near its RETRIEVAL ceiling and the ceiling is low** --
    but the deployment number was never at that ceiling, because the write gate
    was destroying a third of the questions before retrieval ran. With
    gold stores the clean set scores **63.9%** against a measured achievable
    ceiling of **66.4% [61.8, 70.8]**, so there are about 2.5 points left in
    retrieval, not 16. End to end, with the gate deciding, 0.4.0 scored
    **35.7%** on a 300-question held-out split; 0.5.0's threshold takes that to
    **49.0%**, and a perfect gate would reach 52.3%. **The gap between the
    engine's ceiling and the product's number was a write-time constant, not a
    ranking problem.** (`write_policy_results.json`.) The previously published "80% target, missed by 16.1 points" is
    **withdrawn as mis-specified**, not met: 43.3% of the questions never name
    the attribute they ask about, so 80% is unreachable from query text however
    good the engine gets. The work also did not improve the burned held-out set
    (75.0% -> 70.8%,
    one question, n=24).
    (`chat_target_decision.json`, `clean_chat_results_persona4_v3r5.json`,
    `clean_chat_results_heldout_v3r5.json`.)
12. **Four mechanisms aimed at that ceiling all failed, two of them
    significantly.** Conversation context, per-user attribute priors,
    multi-intent retrieval and session recency were pre-registered against a
    +5.0 pt bar and scored on a held-out persona split: best arm +1.0 pt
    [-0.3, +2.7] (not significant), conversation context **-3.3 pt
    [-6.0, -0.7]**, and the ceiling stack moved **66.7% -> 65.0%,
    -1.7 [-3.3, -0.3]**. On the 130 questions they exist for, the two best arms
    moved the number by 0.0. Nothing from that study ships.
    (`context_lever_results.json`.)
13. **A plaintext vault has no tamper resistance, with or without a sidecar.**
    A plaintext block trailer is an unkeyed SHA-256 that anyone holding this
    library can recompute; a forged row is served by a full
    `arena_cache="off"` scan at cosine **0.999997**. `arena_cache="verify"`
    checks the cache's own digest *and* re-reads every vault block, but that
    digest is unkeyed and sits in the header of the file it describes, so a
    ~10-line forgery updates both and `verify` serves the planted row at cosine
    **1.0**. All three are passing tests, not caveats. `verify` buys corruption
    detection over two files -- at 0.1478 s against 0.1656 s for a plain rescan,
    so it costs a scan and is not a security control. Set a password if you need
    one. (`reopen_results.json` `cache_integrity`, `tests/test_arena_cache.py`,
    `nanomem/crypto.py` `THREAT_MODEL`.)
14. **`arena_cache="map"` and `"copy"` re-read zero vault blocks.** That is the
    default. A full scan recomputes every block trailer before a row is
    believed; a cached open checks the cache's binding to the vault -- uuid,
    header digest, row and block counts, dtype, last covered trailer, and the
    vault's (size, mtime) as of the scan it was built from -- and then trusts
    the bytes. `arena_cache_info()["vault_blocks_checked"]` answers "all" /
    "appended tail only" / "none" so a caller can tell. The one case that was
    *closed* rather than described: a vault edited in place at the same length
    with a stale trailer used to be refused by `off` and served silently by
    `map`; all three modes now raise, at +4.08 us per open. `os.utime` still
    defeats that binding, which is also a passing test.
    (`reopen_results.json` `cache_integrity`, `binding_cost`.)
15. **A fresh process opening a SMALL vault is slower with the cache than
    without it.** At 1,190 documents: **0.00764 s against 0.00321 s**. The
    crossover is bracketed only by 1,190 (loss) and 10,000 (4.8x win), and
    `ARENA_CACHE_MIN_ROWS` was deliberately left at 256 rather than re-tuned
    inside a bracket after seeing the result. Small vault, short-lived
    processes: pass `arena_cache="off"`. (`reopen_results.json`,
    `headline.val1190`.)
16. **The router loses to nanomem's own exact scan** on every axis at every size
    measured, so it ships off. Its recall gate passes
    (13 of 30 swept configs clear CI lower bound >= -1.0 pt; cheapest is cell_target=12 beam_frac=0.10 at 77.00% recall@4 vs 77.30% exhaustive, delta -0.30 pt CI [-0.55, -0.1]), but it is
    slower (0.562x
    the exact scan over five replicates at the shipped budget), adds
    970--1058 MB
    of peak RSS and 13.8--25.6 s
    to every open, and never breaks even
    (`break_even_queries_per_open` is null for both budgets).
    (`scratch/refound/router_gate_results.json`.)
17. **On the shape nanomem is FOR, ranking is O(all rows) and it switches the
    screen off.** Put 45 personal records into a 71,433-document vault and
    search goes 2.494 -> 6.857 ms, 36.9% of the turn; 5.564 ms of that is
    ranking and 4.649 ms is `_resolve_revisions` alone, walking all 71,478
    candidates to serve 45. The PCA screen engages on **0 of 256 queries** at
    every personal size, because `_ranking_is_inert()` gates on the whole vault.
    The cost scales with total rows, not with the rows the layer can affect.
    This is the one store-side lever left that a user would feel, and it is not
    done. (`turn_latency_results.json`.)
18. **The write gate still throws away 5% of real facts in a layer the
    threshold cannot reach.** With the learned head fully off, the classifier's
    rule layers drop 52 of 1,180 turns, 14 of them answers to questions --
    ~3.4 pt, gate recall capped at 95.0%. Measured, not attributed to a specific
    rule, and not fixed. (`write_policy_results.json`.)
19. **The no-embedder write path was not measured and did not move.** The study
    that moved the threshold handed a real embedding to every turn, so it
    measured the full head only; the surface fallback head still decides at
    0.60. A deployment with no embedding daemon gets 0.4.0's gate behaviour and
    no evidence either way. (`nanomem/classifier.py`.)
20. **The classifier dials its own embedder on 95.8% of turns**, duplicating a
    vector the caller usually already has: 11.5 ms, **49.1% of a write turn**.
    `classify(text, embedding=...)` avoids it and nothing in the shipped call
    path passes it. That is an API default, and it is the cheapest unfixed
    millisecond in this document. (`turn_latency_results.json`.)
21. **The write-gate and selection results rest on ONE synthetic fixture.**
    14 personas, 1,652 turns, 420 questions, ~64% designed noise. The
    classifier's own held-out generalisation set is quarantined and was not
    opened, so what threshold 0.05 does to its published 90.5% held-out accuracy
    is **unmeasured**. The noise tax that justifies not storing everything
    (-4.5 pt) is the least transferable number of the set: real conversational
    chaff may be more or less confusable with facts than generated banter.

---

## What is and is not defensible about the temporal layer

**Defensible.**

* The problem is real, not manufactured: a plain vector store answers
  current-value questions with a superseded value
  57.0%
  of the time.
* nanomem's layer removes it: superseded-answer rate
  0.0%,
  current-value top-1 100.0%
  against the floor's 43.0%.
* **It beats the recency heuristic, which it did not before.** The direct answer
  to "did a simple recency heuristic on top of Chroma match it?": on the previous
  build, yes -- the two-way heuristic scored
  72.7% against nanomem's best
  non-oracle arm at 64.4%. On
  this build, **no**: nanomem
  90.5% vs
  72.7% for the two-way heuristic and
  86.2% for the three-way one,
  both significant, and significant again under a user-level cluster bootstrap.
* It needs **no parameter**. At the competitor's chosen pool size the heuristic
  swings from 47.9% at delta=0 to 72.7% at its swept
  optimum and back to 54.9% at delta=0.30, and across the whole
  56-point grid it ranges 19.6%--72.7%
  (`temporal_bench_results.json` -> `sweeps.chroma_recency.grid`); nanomem has no
  such knob to get wrong, and it reads the question's direction from wording with
  no classifier for the application to write.
* The gain is not tuning: 90.5% over
  all 326 questions and 90.4%
  on the 8 users whose data set no competitor parameter was chosen on -- the two
  agree to 0.1 pt.

**Not defensible.**

* **It is not defensible as a general improvement over plain cosine.** It is
  still net-negative on sibling attributes and on third-party facts (items 6 and
  7 above). "Better than doing nothing" is true on average and false on two of
  five question types.
* **It is not defensible on the product shape.** The chat benchmark is what the
  library is for, and the clean estimate there is 63.9%,
  below target, with no improvement on the burned set. A
  46.3-point
  gain on a synthetic supersession set did not transfer.
* **The documented API for it is wrong** (items 9 and 10). Both `metadata
  ["entity"]` and `temporal_direction` now make results worse, and the manual
  still recommends them.
* **The margin over the strongest competitor is small.** +4.3 pts on 326
  questions from 10 synthetic users. The lower bound of the user-level interval
  is +1.5 pts. It is a real win and a narrow one.
* **The benchmark is synthetic and small** (~46 documents per user). Nothing here
  measures the layer at 10k+ rows per user or across tenants.

---

## The exotic-mathematics study

Nine mechanisms drawn from outside ordinary vector search were pre-registered,
implemented and measured against nanomem's two decision points -- **routing**
(which documents get scored at all) and **ranking** (which of several documents
about the same fact wins). **Nothing from this study shipped.** This section
exists so the next round does not pay for the same answers twice.

Sources: `scratch/refound/exotic_routing_results.json`,
`scratch/refound/exotic_ranking_results.json`, and for the kernel result
`prime_4d_unified_engine_2026_09_13/prime_routing_results.json`. Both studies
wrote a binding decision rule -- win = 95% CI lower bound above zero on a paired
bootstrap, plus a regression guard on the chat set -- before any number existed.

### The one finding every future round must inherit

**Pointwise nonlinear kernels -- including prime exponents -- cannot change a
ranking. This is settled by proof and by measurement, and no future round should
retry them.**

The proof is one line. Ranking sorts a list of scores. Applying the same
strictly increasing function to every score separately cannot change their
order, so raising a cosine to a power -- 13, 12.5, any prime, any exponent --
returns the same ranking it was given. The transform is *rank-inert* by
construction. It is not that it was tried and did not help; it cannot help.

The measurement agrees exactly, which is how we know the implementations were
right and not accidentally doing something else. Over **142,800 document pairs**:
**0** argsort positions differ between the plain cosine and the p=13 kernel, **0**
top-10 positions differ, the largest cosine gap between any pair the transform
swaps is **0.0**, and the antisymmetry error is **0.0** at p = 12, 12.5, 13 and
15 alike. A second transform in the same audit moved 8 of the 142,800 positions,
with a maximum gap of 2.98e-08 -- exactly one float32 ulp at that cosine, i.e.
rounding, not reordering. (`prime_routing_results.json`, `kernel_autopsy`.)

**The same argument has a second form, established this round, and it closes a
whole family of proposals.** Grouping documents by single linkage is invariant
under *any* strictly increasing transform of the distance matrix. So re-embedding
a neighbourhood into a "better" space -- hyperbolic or otherwise -- and then
clustering it cannot change the grouping either, as long as the new distances
order the same way the old ones did. Measured: at the tagged-floor decision the
Poincare-ball partition was identical to the plain cosine partition on **152 of
152** calls on one seed and **150 of 152** on the other two, with rank
correlation (Spearman, hyperbolic distance vs cosine) of **1.000** at the
median; and the arm's output was rank-identical to the plain Euclidean arm on
all 261 test questions. The handful of calls that do differ are embedding
distortion, not information. "Embed it in a richer geometry" is the clustering version of
"raise it to a prime power", and it is inert for the same reason.

A transform only earns a measurement if it **combines two or more numbers** --
pooling across landmarks, mixing a similarity with a time gap. Those were the
arms actually worth running, and they are below.

### What was tried, and what happened

| # | Mechanism | Plain-language claim | Measured verdict |
|---|---|---|---|
| 1 | **Structured projection + Cauchy-Schwarz certificate** | Score a cheap 256-dimensional summary first, and use a bound to *prove* which documents cannot reach the top 4 | **Works, but not shipped.** Exact by proof: 0 bound violations in 14.3 M checks, 0 of 1000 queries with a different top-4 score vector; 1.529x faster, with 0.314% of the corpus surviving the screen. It does not clear the rule as written (see below). |
| 2 | *the structured half of (1)* | Use a fast Hadamard/random projection so the summary is cheap to build | **Negative.** A random subspace captures only 0.24-0.35 of the energy, so almost nothing can be pruned: those screens certify 98.8-99.9% of the corpus and run **0.19x** -- five times *slower* than just scanning. Only the data-fitted PCA basis (0.934 energy) compacts enough to pay. Structure is not what makes it work; energy compaction is. |
| 3 | **Power-mean pooling** | Instead of taking the best-matching landmark per page, blend them with a tunable exponent | **Negative.** On the shipped layout plain max is at or above the optimum: at a 25% budget over 71,433 documents the whole curve is p=4 74.75%, p=8 76.35, p=16 76.85, p=32 77.00, p=64 77.05, max 77.00 -- a best gain of **+0.05 pt** with a CI lower bound of **+0.00**, and everything at or below p=4 loses **2.25 to 24.60 pt**. The largest gain *anywhere* in the sweep is +1.875 pt with a CI lower bound of +0.208, on the 1,190-document corpus, at one budget, on one seed of three, where the metric is top-4 agreement rather than gold recall -- it does not replicate and the study declines it as a multiplicity artifact. |
| 4 | **Low-discrepancy seeding** (Halton, Kronecker, van der Corput) | Seed the clustering with evenly-spread points instead of random ones | **Negative.** Worse by k-means' own objective on the 71,433-document corpus, 3-seed means (0.8245 for Halton and Kronecker against 0.8256 for k-means++), and **30% worse** cell balance -- spread/mean 0.971 against 0.746, largest cell 257 rows against 160. Recall is indistinguishable. These vectors sit on a thin curved manifold; a low-discrepancy sequence covers a *box*, and the box-to-data map is where the guarantee is spent. |
| 5 | **CRT product quantisation** | Split the index into co-prime shards so a cell id can be decoded arithmetically, with no stored map | **Negative.** Loses recall at matched cost in every configuration and at every budget -- each 95% interval lies *entirely* below zero, from [-1.20, -0.15] pt for the best shard layout at a 25% budget to [-9.70, -7.10] for the coarsest at 15%. It does cut auxiliary memory **12.2x** (8.94 MiB -> 0.73 MiB; centroids alone 51x), but that memory is **4.3%** of the 209.3 MiB arena that actually drives peak RSS. It also turned out to be unnecessary: after `compact(recluster=True)` nanomem already derives a cell from a row id with no stored map. The content-blind variant, which shards on the id itself, scores like a random slice -- CI [-65.6, -61.0] pt -- which is what it is. |
| 6 | **Tropical / min-plus geometry** | Bound each cluster by an angular radius and use min-plus algebra to skip clusters | **Negative.** The bound is valid (0 violations in 571,600 checks) and useless: a strict loss at every budget and it prunes **0.06%** of the corpus. Mean cell radius is 38.9 degrees, so the bound saturates near 1 and all ordering information is destroyed. |
| 7 | **Persistent homology** | Cut the group at the largest gap in the topological barcode, so no threshold has to be chosen | **Negative.** -8.43 pt [-12.64, -4.21]. The threshold-free cut is worse than the constant it replaced. |
| 8 | **Hyperbolic (Poincare) embedding** | Revision chains are trees, and trees embed naturally in hyperbolic space | **Negative, and provably so** -- see the inertness argument above. It also costs **117-140x** the shipped latency, which disqualifies it outright. |
| 9 | **Multi-variable logistic rule** | Decide "same fact?" from four signals at once rather than one threshold | **The only family with content, and still not shipped.** +4.21 pt [+1.92, +6.90] and the only arm that fixes sibling attributes (87.5% -> 100.0%). But it is no better than a one-line threshold nanomem already ships turned off, and it is fitted on the benchmark's own users -- a fixture-fitted rule, not a mechanism. |

### The one mechanism that works has since SHIPPED, off by default

**This subsection used to say the projection screen was not shipped. It now is**
-- as `screen="pca"`, default OFF -- and the three reasons it was held back are
worth keeping because two of them were answered by measurement and one was
answered by fixing the rule.

The projection screen (row 1) is real: it is exact by a *proof* rather than by a
lucky sample, and an independent reimplementation reproduced it (0.9337 captured
energy against 0.9335, 199/200 exact top-4 id sets with the remaining one a
genuine tie, 1.458x against 1.529x). What was outstanding, and what happened:

1. **It did not clear the bar as written, and still does not.** The
   pre-registered win rule was a confidence interval on a *recall delta*. An
   exactness-preserving screen has a recall delta of exactly zero, so its
   interval is [0.00, 0.00] and the bar is unreachable **by construction**, not
   merely unmet. That rule was not quietly reinterpreted: a **new** criterion
   was written down before the mechanism was built into the engine -- exactness,
   a p50 speed-up of at least 1.25x at 71,433 documents, and no regression below
   10,000 -- and this is stated as what it is, a criterion chosen because the
   original could not express the benefit. The count for the *exotic study*
   therefore remains **zero wins and eight negatives**; the screen ships against
   a different, later, pre-registered rule and claims **no recall improvement of
   any kind**.
2. **The speed-up was measured at the wrong level. It has now been measured at
   the right one.** The 1.529x was an arena-level kernel ratio, and
   `DECISIONS.md` #3 records the router looking good at exactly that level and
   then losing at the engine level. Measured end to end through
   `VaultEngine.search`, the screen survives that step: **1.718x p50**
   [1.695, 1.736], five per-cycle ratios inside 1.709-1.731. See Axis 2.
3. **The basis is data-dependent and its decay was never measured. It now is.**
   Correctness never decays -- the bound is admissible for *any* basis, using no
   property of it -- and the speed does not collapse either: a basis fitted from
   6,000 rows with 24,000 rows appended afterwards and never refitted gave 0
   different id lists, 0 non-bitwise score lists and 0 bound violations over
   2,400,000 document checks.

### The knob this study found, which is a documented tension rather than a fix

Six ranking arms **did** clear the literal win rule on the temporal test split.
All six were declined, and the reason is worth publishing because it is a real
property of the product rather than a detail of the study.

Three of those six are *rank-identical, on all 261 test questions*, to
`group_floor_sim` -- a one-line threshold nanomem already ships set to `0.0`.
Turning it on is worth **+4.6 pts [+2.30, +7.28]** on temporal supersession
(historical questions 75.0% -> 90.0%) and **-13.9 pts [-25.00, -2.78]** on the
three-persona chat set (97.2% -> 83.3%). Those are the same knob pulled in
opposite directions by two benchmarks, and the pre-registered regression guard
declines it on that basis alone.

No local geometry resolves it, and the study measured why: document-to-document
cosine separates "this is a revision of that" from "these are two different
facts" at **AUC 0.676**, while the query-relative gap the shipped floor already
uses separates them at **AUC 0.936**. Every exotic arm on the menu moved the
decision onto the *less* informative variable. Clustering cannot recover
information that the distance matrix it was handed does not contain.

**The shipped default (`group_floor_sim = 0.0`) is correct as it stands**, and
its cost is now published rather than only its existence.

---

## Before and after

Both harnesses were re-run against the improved build in this session; the
pre-improvement values are embedded in `scratch/refound/final_scorecard.json`
(the harnesses write their results in place).

| measure | before | after |
|---|---|---|
| evidence recall@4 @71,433 | 60.6% | 60.6% |
| p50 @71,433 | 1.821 ms | 1.774 ms |
| peak RSS @71,433 (loader) | 825.5 MB | 826.4 MB |
| index bytes @71,433 | 148.7 MiB | 148.7 MiB |
| router arm peak RSS @71,433 | 3363.6 MB | 1941.6 MB |
| temporal top-1, shipped defaults | 44.2% | 90.5% |
| temporal, historical questions | 9.0% | 75.0% |
| temporal, previous-value questions | 1.8% | 98.2% |
| temporal, sibling attributes | 57.5% | 90.0% |
| temporal, documented entity tag | 74.2% | 59.2% |
| chat, burned held-out | 75.0% | 70.8% |

**0.3.1 -> 0.3.2 (engine 3.0.4 -> 3.0.5).** A controlled A/B: one harness, one
machine, one hour, the "before" column a checkout byte-identical to committed
`6aa6923`. Source `memory_results.json` and `ingest_ram_results.json`.

| measure | before (0.3.1) | after (0.3.2) |
|---|---|---|
| peak RSS @71,433 (loader) | 819.1, 818.6 MB | **286.4, 286.9 MB** |
| peak RSS @10,000 (loader) | 102.0, 103.0 MB | **41.2, 41.1 MB** |
| peak RSS @71,433 (server) | 286.0 MB | 285.2 MB |
| evidence recall@4 @71,433 | 60.6% | 60.6% |
| evidence recall@4 @10,000 | 70.4% | 70.4% |
| p50 @71,433 | 1.8702 ms | 1.7495 ms |
| full score vectors, all 500 queries | — | **bitwise identical (SHA-256)** |
| top-10 lists changed | — | **0 of 500**, all 20 cells |
| temporal top-1, shipped defaults | 90.5% | 90.5% (all 19 arms to the digit) |
| `stats()["arena_bytes"]` @71,433 | 402,653,184 | 219,442,176 (the rows) |
| arena growth copies, 71,433-row build | 12 growths | 2 |

The retrieval and temporal rows are the control: nothing in the arena work was
supposed to move them, and nothing did.

**0.3.2 -> 0.4.0 (engine 3.0.5 -> 3.1.0).** Two mechanisms integrated, both
exactness-preserving, one on by default and one off. Sources
`reopen_results.json`, `pca_screen_results.json`, `memory_results.json`, all
re-run against the integrated build in one session at machine load 1.3-2.2.

| measure | before (0.3.2) | after (0.4.0) |
|---|---|---|
| reopen @71,433, in-process median | 0.1634 s | **0.000193 s** (846x) |
| reopen vs sqlite-vec's 0.0014 s | 115x behind | **7.3x ahead** |
| time to first answer @71,433 | 0.1687 s | **0.0109 s** (15.4x) |
| fresh-process open @71,433 | 0.1696 s | **0.00635 s** (26.7x) |
| fresh-process open @1,190 | 0.00321 s | **0.00764 s** — a loss |
| bytes on disk @71,433 | **148.7 MiB** | 406.2 MiB — a loss |
| serving `phys_footprint` @71,433 | 288 MB | **9 MB** (32x) |
| serving peak RSS @71,433 | 287.5 MB | **258.2 MB** |
| loader peak RSS @71,433 | 286.4, 286.9 MB | 290.0 MB (one run; not resolvable) |
| p50 @71,433, paired duel | — | **0.9983x** (unchanged) |
| p50 @71,433 with `screen="pca"` | — | **1.1350 ms** vs 1.9495 (1.718x) |
| full score vectors, cache on vs off | — | **bitwise identical (SHA-256)**, all 3 sizes |
| top-10 lists changed, cache on vs off | — | **0 of 500**, all 3 sizes, all 3 modes |
| top-10 lists changed vs the 0.3.2 package | — | **8 of 500** @71,433, every one a bitwise tie |
| evidence recall@4 @71,433 / @10,000 | 60.6% / 70.4% | 60.6% / 70.4% |
| tests | 322 | 408 |

The recall row is the control. The **8 of 500** row is not a control and is not
a defect: `_select_top_k` now breaks ties on the row id instead of inheriting
`np.argpartition`'s unspecified order, so lists containing two documents with
the identical float32 cosine can come back in a different order than 0.3.2 gave.
Every changed position was scored and the maximum score gap is **exactly 0.0**
(`reopen_results.json`, `tie_forensics`, which fails loudly if one is not). The
isolating control — same build, cache on against cache off — is 0 of 500.

The standard-retrieval row is the control: nothing in this work was supposed to
move it, and nothing did.

**0.4.0 -> 0.5.0 (engine 3.1.0 -> 3.2.0).** Two mechanisms integrated: one
changes the sidecar's layout, one changes a write-time constant. Four more were
measured and not shipped. Sources `sidecar_size_results.json`,
`write_policy_results.json` and `quality_summary.json` (the last re-measured on
the final tree at machine load 1.4-1.9).

| measure | before (0.4.0) | after (0.5.0) |
|---|---|---|
| bytes on disk @71,433 | 406.15 MiB | **153.05 MiB** (2.65x) |
| the `.arena` sidecar itself | 257.48 MiB | **4.38 MiB** (58.8x) |
| bytes on disk @10,000 | 57.2 MiB | **21.60 MiB** |
| disk against sqlite-vec's 258.0 MiB | 1.57x behind | **1.69x ahead** |
| reopen @71,433, in-process median | 0.000193 s | 0.000228 s (2 runs, 4 cells: 0.000219-0.000246) |
| p50 @71,433, paired duel against the 0.4.0 layout | — | **0.9976x** (screen `pca`), 1.0037x (off) |
| serving `phys_footprint` @71,433 | 43.5 MB | **270.5 MB — a loss** |
| serving peak `ru_maxrss` @71,433 | 318.4 MB | 331.0 MB |
| vault truncated out of band under a live engine | `SIGBUS`, exit 138 | `VaultShrankError` |
| full score vectors, every layout | — | **bitwise identical, 36 of 36** |
| top-10 lists changed vs the exact scan | — | **0 of 500**, both corpus sizes |
| evidence recall@4 / @10 @71,433 | 60.6% / 71.2% | 60.6% / 71.2% |
| evidence recall@4 @10,000 | 70.4% | 70.4% |
| chat end-to-end top-1, 300-question test split | 35.7% | **49.0%** (+13.3 [+8.7, +18.3]) |
| chat end-to-end top-1, all 420 questions | 38.1% | **51.0%** |
| write gate recall / precision | 64.5% / 100.0% | **94.0% / 92.9%** |
| questions made unanswerable at write time (of 300) | 100 | **16** |
| documents stored per 10 personas | 271 | 425 (+57%), 0.492 -> 0.764 MiB |
| tests | 408 | **443** |

The recall rows are the control, and this time they are a control over a
*layout* change that reads vectors from a different file entirely: they did not
move, bitwise. The `phys_footprint` row is the one regression and it is the
price of the disk row directly above it.

---

## Known gaps

* THE p50 GATE THAT CHOSE THE 0.5.0 SIDECAR DEFAULT WAS FIRST REPORTED AGAINST A SUBSTITUTED CRITERION. The pre-registration required search p50 within 1.10x of 1.1494 ms (<= 1.2643 ms) and it failed for all six arms INCLUDING the unmodified baseline, which read 1.2577 / 1.3882 ms on a loaded box. The rule's own failing branch said to report the negative; instead an eligibility definition based on the paired ratio was substituted and a new default was shipped on it. That was then repaired: the gate was re-measured ONCE on a quiet machine under a rule fixed in advance whose failing branch would have reverted the default, the baseline read 1.2055 ms, the gate discriminated (only `"offsets"` fails, at 2.0857 ms), and the same default stands on the bar as written. Both readings, the restatement and the superseded verdict are in `sidecar_size_results.json`. A reader who holds that any re-measure after a failed bar is disqualifying should read the registered fallback: no arm eligible, default reverted. Reaching the same answer does not retro-justify the substitution.
* THE 0.5.0 p50 MARGIN IS THIN IN BOTH DIRECTIONS. The baseline clears its own bar by 4.7% and the shipped arm by 4.0%. That gate separates `"offsets"` from everything else and resolves nothing finer; for finer differences the instrument is the paired duel, not the absolute number.
* `phys_footprint` WAS NOT IN THE 0.5.0 DECISION RULE and is the new default's real cost (43.5 -> 270.5 MB). It is reported rather than added to the rule after the fact.
* THE 0.5.0 RE-MEASUREMENT IS NARROWER THAN THE RUN IT REPAIRS: one pass of 500 queries per arm, `screen="pca"` only. Every disk, `ru_maxrss`, `phys_footprint`, first-query and `screen="off"` figure quoted for 0.5.0 comes from the earlier run and was not re-taken.
* THE WRITE-GATE RESULT IS ONE FIXTURE, AND ITS CONFIRMATION IS FORBIDDEN HERE. 14 synthetic personas, 1,652 turns, 420 questions. The classifier's own held-out generalisation set (2 unseen personas, 231 turns, published at 90.5% accuracy) is QUARANTINED and was not opened, so the effect of threshold 0.05 on that number is unmeasured; `train_write_classifier.py --heldout` at the new threshold is the missing confirmation and it is exactly the test this work was not allowed to run. The threshold itself was chosen on 4 dev personas where 0.05 and 0.02 tied, and both curves are flat from 0.10 to 0.02, so 0.05 is a region rather than a tuned constant.
* THE WRITE-GATE LATENCY AND BYTES ARE SMALL-VAULT NUMBERS. 73-1,180 rows per persona, 1,902 bytes/doc against 2,183 in the published 71,433-document figure, p50 measured with the query embedding served from a cache so it excludes the 11.9-23.6 ms embedding call. nanomem's scan is linear: 2.8x more rows costs about 2.8x the scan at 71k, and the sub-linear growth measured on per-persona vaults is fixed-overhead dominated. Do not read it as a claim about 71k.
* THE SELECTION STUDY'S TEST NUMBER IS AN IN-VOCABULARY NUMBER. Dev and test share one question template per (attribute, type, head) cell, which the study declared before scoring and quantified with leave-one-attribute-out (79.4% -> 57.0% selection accuracy). No fixture exists here that is both open-vocabulary and not quarantined. Its ensemble weights are also five hardcoded literals whose originating search was never written down; a from-scratch coordinate ascent does not reproduce them, although it scores HIGHER on test (64.3% / +12.0 against the frozen 63.3% / +11.0), so the conclusion does not depend on them.
* THE BM25 STUDY IS ONE CORPUS (HotpotQA), ONE EMBEDDER (nomic-embed-text), AND WAS NEVER RUN ON THE CONVERSATIONAL BENCHMARKS NANOMEM SHIPS FOR. Its latency figure was taken at machine load 6.6 and deliberately not re-measured on the now-quiet box, because re-measuring until a bar is crossed is the failure mode this project has a rule against. That measurement is owed by whoever re-registers the criterion first.
* FUSION IS NOT MONOTONE: the best @4 arm breaks 18 of 500 questions the dense arm got right (3.6%). For a memory engine a regression on queries that used to work may matter more than the average gain, and which questions those are was not characterised.
* THE TURN-LATENCY NUMBERS WERE TAKEN UNDER LOAD. 1-minute load average 4.0-5.3 on 12 cores with a sibling process running; a quiet probe read embed p50 9.0 ms, so absolutes are inflated ~15-30%. The shares replicate (17.32% [16.60, 18.63] against 15.91% [14.68, 16.68]) and every ratio is paired and interleaved. Quote the shares, not the milliseconds. Its personal-vault sweep also varies the DOCUMENT count against a fixed 45 personal records from one user; a pure personal vault at 10k-70k rows is unmeasured.
* SEVERAL OF THESE STUDIES RAN AGAINST A WORKING TREE ANOTHER AGENT WAS EDITING, and all of them disclose it. The disclosures check out where they could be checked (the selection harness still reproduces its 52.3%/28.5% baseline or aborts; the write-policy harness reproduces 38.1%/53.3% exactly; the turn-latency run pinned a pristine checkout and hashed every imported module). One concrete cost is visible anyway: the BM25 study's disk criterion was computed against a 406.2 MiB vault while a sibling was in the middle of making it 153.05 MiB, which turned a "passes" into a "fails".
* THE PRE-REGISTERED EXACTNESS GATE FOR THE ARENA CACHE DID NOT PASS AS WRITTEN. It required 0 changed top-10 lists against the committed package and measured 8 of 500 at 71,433 and 1 of 500 at 10,000. Every one is mechanically proved to be a bitwise tie (max |score gap| 0.0) owned by a different change in the same tree -- the PCA screen's deterministic tie-break -- and the isolating control that the gate could not express, same build with the cache on against off, is 0 of 500 at all three sizes. Reported as a failure with both measurements rather than restated as "exact up to ties". The 8 is also insertion-order dependent: an independent re-derivation with a different insertion order measured 7.
* NO COLD-PAGE-CACHE NUMBER EXISTS ANYWHERE. `purge` needs root and was refused, so every reopen, latency and footprint figure in this document is warm -- the same condition the competitor file was measured under, so the comparison is like-for-like, but the cost of faulting the mapping in from cold disk is unmeasured and would land on first-query latency, which is exactly where the mapped arena is already weakest. 0.5.0 makes this gap LARGER rather than smaller: the offset layout faults from the VAULT rather than from a sidecar, and its one-shot fp32 upcast already reads 0.0227 s on the first query against 0.0117 s for the fp32 sidecar, warm.
* THE COMPETITOR ARMS WERE NOT RE-RUN FOR 0.4.0. sqlite-vec 0.0014 s, Chroma 0.0017 s, the disk column and the Axis 4 loader table are all cited from `competitors_standard_results.json` and `ingest_ram_results.json`, measured on another day at another machine load. Only the nanomem side was re-measured. The cross-harness offset was checked on the one axis where it could be: a cache-less nanomem open reads 0.1634 / 0.0226 / 0.0027 s here against 0.1609 / 0.0218 / 0.0027 in the competitor file.
* `bench_memory.py` DOES NOT PIN `arena_cache=`, so its residency arms are now confounded: whichever arm runs first for a given dtype scans the vault and writes the sidecar, and the next arm with that dtype maps it (measured: `fp32_reserved` from cache, `legacy_303_fp32_doubling` not). Its reopen_only RSS column is therefore no longer a clean residency comparison. `tests/test_arena_residency.py` was amended to assert its claim through keys valid on both open paths and to pin the confound; repairing the harness is not work this revision did.
* THE `verdict` BLOCK IN `pca_screen_results.json` IS STALE. `bench_screen.py` merges a new run over the old file with `prior.update(res)`, so keys no phase regenerates are carried over. Its `C2_speed` holds the previous run's 1.705x; this document cites the freshly measured phase blocks (1.718x). Every PASS/FAIL agrees; the third decimal does not.
* SINGLE RUN PER (arm, corpus). One insertion order (rng(0)), one embedding model (nomic-embed-text 768-d), one machine. There are no confidence intervals on latency, ingest, disk or RSS anywhere in this scorecard -- only on recall (paired bootstrap, 10,000 resamples) and on the temporal top-1 deltas.
* THE BEFORE COLUMN IS NOT A CLEAN A/B. bench_competitors.py and bench_temporal.py write their results in place, so the pre-improvement values embedded here were produced by a run 3 hours earlier on a machine under different load (competitor before-run load average is in that block). Recall and top-1 are deterministic and reproduce exactly; the latency and RSS 'before' figures should be read as a different run, not a controlled baseline.
* NO BEFORE NUMBER EXISTS FOR persona4. The pre-improvement tree is not recoverable from git (HEAD 441aaca already contains the new entities.py), and scoring a second build would burn the set. So the 63.9% is an absolute level, not a delta: this scorecard cannot say whether the ranking work helped or hurt on a clean chat set. What it can say is that on the BURNED chat set the same work moved 75.0% -> 70.8%.
* n=24 AND n=36 ON THE CHAT SETS. The 75.0 -> 70.8 change on the held-out set is ONE question. No significance is claimed for it in either direction, and none should be read into it.
* THE TEMPORAL SET IS 10 SYNTHETIC USERS, ~46 documents each. The question-level bootstrap is optimistic because questions within a user are not independent; a user-level cluster bootstrap is reported alongside every headline pair and the +4.3 pt win survives it (CI [+1.5, +7.4]). Cross-user leakage in a shared multi-tenant vault is not measured, and nothing here says how the cosine-window grouping behaves at 10k+ rows per user.
* THE COMPETITOR'S TEMPORAL PARAMETERS WERE SWEPT ON THE TEST SET and kept at the competitor's optimum (kpool=3, delta=0.08). That gift is deliberate and stays. nanomem got no tuning of any kind on that set. The dev-tuned competitor variants and the 8-user holdout are reported and give the same ordering.
* THE THREE-WAY COMPETITOR ARM IS NOT A LIBRARY FEATURE EITHER. chroma_recency_3way needs a hand-written three-way direction classifier on top of Chroma; nanomem_default needs nothing. That asymmetry favours nanomem in deployment terms and is not captured by the top-1 number.
* THE STANDARD BENCHMARK DOES NOT EXERCISE THE TEMPORAL LAYER AT ALL. The corpus is HotpotQA paragraphs stored with source='wiki', which is not in nanomem.entities.PERSONAL_SOURCES, so the layer is inert by design -- proven empirically by the engine matching FAISS on 499 of 500 queries. The two tasks in this scorecard measure two disjoint code paths.
* PEAK RSS IS THE LOADER SHAPE. The competitor harness ingests and serves in one process, so nanomem's figure includes the ingest high-water mark. As of 0.3.2 that figure is 298.4 MB (range 286.4-310.0 over six runs across two harnesses), down from 826.4. bench_memory.py measures a SERVER shape of 285.2 MB for the same corpus, but no server-shape figure exists for FAISS, Chroma or sqlite-vec, so there is still no like-for-like server comparison in this work. Do not pair the 285.2 MB with the competitors' numbers. FAISS and sqlite-vec WERE re-run inside the ingest harness (competitors_same_harness) so the loader-shape comparison is like-for-like; the Chroma rows remain cross-harness and are labelled as such.
* THE PEAK-RSS IMPROVEMENT IS TWO RUNS PER CELL, NOT A DISTRIBUTION. The 0.3.1 -> 0.3.2 A/B is two runs of each arm at each size, plus one driver pass and the four-run medians of a second harness. There is no confidence interval on any RSS figure anywhere in this scorecard. The new arm is also bimodal at 71,433 rows -- readings cluster near 288 and near 305-310 MB, a 7% spread that the previous build's hinted arm shows too, so it is an ingest transient rather than the reservation. It was not chased.
* THE PCA SCREEN NOW HAS ENGINE-LEVEL CONFIRMATION, AND IT SHIPS OFF. The 1.529x this gap used to flag as an arena-level kernel ratio has been re-measured end to end through `VaultEngine.search` at 1.718x p50 [1.695, 1.736], so it survived the step DECISIONS #3 records the router failing. It is in the library as `screen="pca"`, default OFF, against a criterion written AFTER the exotic study's recall rule was found unreachable for it by construction -- that substitution is disclosed in the section above rather than buried, and no recall claim is made. Remaining unknowns: one machine, one corpus, one embedding distribution; the ~20,000-row knee and the 256-dim width are properties of that distribution, and a corpus with higher intrinsic dimension would prune fewer rows. The gather-size calibration is a 400-trial sample rather than a proof, though the engine declines to engage rather than returning a different score if it ever fails.
* FAISS's 209.3 MiB INDEX STORES NO TEXT. nanomem's 148.7 MiB contains the corpus text as well, so the disk axis is only like-for-like against sqlite-vec and Chroma. No engine was VACUUMed or force-compacted.
* recall@4 IS THE FIRST FOUR OF A k=10 QUERY, which is identical to k=4 for the exact arms and a small gift to the HNSW arms. Chroma's deficits are if anything understated.
* NO THREAD PINNING. FAISS had 12 OMP threads, Chroma built on ~4.5 cores, nanomem used numpy/Accelerate defaults. The latency column is an out-of-the-box comparison, not a single-core one.
* THE findings BLOCK INSIDE temporal_bench_results.json NOW CONTRADICTS ITS OWN NUMBERS. Its `claim` strings were written for the pre-improvement build and are not regenerated ('shipped_default_has_no_net_benefit' now carries 90.5 against a floor of 47.9). That file is not owned by this agent. Read its numbers, not its prose.
* THE ROUTER ARM IN THE STANDARD TABLE IS NOT A SHIPPED CONFIGURATION: n_exhaustive was lowered from 50,000 to 1,000 so routing engages at every size. At shipped defaults it is byte-identical to the exact arm below 50k documents.
* NOTHING HERE MEASURES END-TO-END ANSWER QUALITY WITH AN LLM IN THE LOOP, encryption on, multi-tenant filtering at scale, or any corpus that is not HotpotQA or a synthetic personal log.

* **DISCLOSURE ON THE CLEAN SET.** persona4 was scored once, on the improved
  build, and nothing was tuned afterwards. Its metadata blocks
  (`scoring_rule`, `provenance`, `difficulty_comparability`) were read to write
  the caveats above; no persona4 question, answer or failure was inspected, and
  the strictness-matched control was run on the two older fixtures only. The
  0-of-36 strictness figure for persona4 is quoted from the fixture's own
  metadata, not recomputed from its questions.

---

*Axes 1-6 and "Before and after" through 0.4.0 are generated from
`scratch/refound/final_scorecard.json`, itself assembled by script from the
results files named above. The 0.5.0 rows, the turn-latency section and the two
"not shipped" subsections are written against
`scratch/refound/quality_summary.json`, `sidecar_size_results.json`,
`write_policy_results.json`, `hybrid_results.json`, `selection_results.json` and
`turn_latency_results.json`. **If a number here disagrees with the file it
cites, the file is right.**

Two integration decisions are recorded in `quality_summary.json` rather than
here, and a reader checking this document's honesty should start with them: the
sidecar default was shipped after its p50 gate first failed as written and was
answered with a substituted criterion (disclosed, then repaired by one
re-measurement under a rule fixed in advance), and the BM25 study's disk
criterion passed against a vault size that this very release made obsolete.*
