# Temporal API — promoting the revision layer from ranking to primitives

Status: spec, pre-registered 2026-09-17, BEFORE any measurement.
Binding order: DECISIONS.md > this file.

## Premise (verified, not assumed)

`VaultEngine._columns()` (engine.py:2555) already returns three RESIDENT per-row
arrays on every search: `arena.entity_id`, `arena.rev`, `arena.ts`.
`_resolve_revisions()` (engine.py:2121) already computes `group` — the set of
candidate rows that are competing statements of ONE fact — uses it to reorder,
and then DISCARDS it.

Measured value of that machinery, already in the tree:
90.5% top-1 on the 326-question temporal benchmark vs 47.9% for a plain cosine
scan (`scratch/refound/ranking_r5_temporal_layer.json`), with no LLM in the loop.

So the temporal capability is already paid for. It is simply not reachable
through the public API, which is `add_fact / search / get / iter_records` —
a vector-store API.

## What ships

Three primitives, all riding on columns that already load.

### 1. `history(query_text, query_vec, max_len=None)`
The revision chain for the fact a query names: every value it ever had,
oldest first, each with `id, text, timestamp, revision, cosine, superseded`.
Implementation: extract group formation out of `_resolve_revisions` into
`_revision_group()`, call it from both. `history()` orders the group by
`entities.temporal_order(..., historical=True, historical_mode="oldest")`.

### 2. `search(..., as_of=<unix ts>)`
What the vault believed at time T. Mask `tsv <= as_of` over candidates.
CORRECTNESS CONSTRAINT: a post-hoc filter over a screened/routed candidate set
is WRONG (newer rows can crowd older ones out of selection before the mask is
applied). So `as_of is not None` forces exhaustive candidates and disables the
PCA screen, exactly as `metadata_filter` already does. Exhaustive returns
`arange(N)` (engine.py:2538), so the mask is then exactly correct by construction.

### 3. `changes(since, until=None, limit=None)`
Rows asserted in a time window, grouped into facts where the grouping is
knowable (shared revision-group key). Answers "what did I learn / change my mind
about between T1 and T2" without a query vector.

## Pre-registered gates

G1. **EXACTNESS.** With `as_of=None`, `search()` returns BITWISE-IDENTICAL
    results to the current build — same ids, same order, same float scores —
    on the full existing suite. The `_revision_group` extraction is a pure
    refactor; any diff at all fails this gate and the refactor is reverted.
    Metric: 443/443 existing tests pass unchanged, plus a dedicated
    before/after score-vector comparison on >=500 queries.

G2. **as_of SOUNDNESS.** For every `t`, every hit returned by
    `search(q, as_of=t)` has `timestamp <= t`, and the result equals the result
    of running the current engine against a vault containing ONLY the rows with
    `ts <= t`. Metric: mechanical equivalence check over >=200 (query, t) pairs,
    built by physically constructing the truncated vault. 0 mismatches required.

G3. **NO NEW VOCABULARY.** No benchmark/persona fixture words enter library
    files. Enforced by the existing suite-wide guard.

G4. **COST.** `as_of=None` search p50 must not regress by more than 1.0%
    against the 0.5.0 build on the same machine, same corpus. If it does, the
    added branch moves behind a flag.

FAILURE RULE (this project's standing rule, restated so it cannot be
quietly dropped): if a gate fails, the negative is reported and the mechanism
does not ship. The bar is not restated after the fact.

## Explicitly NOT in this change

- `why()` / score attribution — second milestone, needs the boost path threaded.
- The `_resolve_revisions` O(top_k) rewrite. It is a real 4.649 ms win but it
  changes WHICH rows form a group, so it carries accuracy risk and needs its own
  pre-registered accuracy bar on the temporal benchmark. Doing it in the same
  change as the primitives would make G1 unprovable. Sequenced after.
