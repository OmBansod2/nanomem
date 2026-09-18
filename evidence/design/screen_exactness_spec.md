# What "exact" can mean across two float32 matmuls

Status: PRE-REGISTERED 2026-09-17, before the measurement was written and before
any number was seen. Binding order: DECISIONS.md > this file.

## What CI found

`test_screen.py` asserts that the screened path returns BITWISE the same scores
as the exhaustive scan. It passes here and fails on every CI runner:

    assert [0.21058768033981323] == [0.21058765053749084]      # 2.98e-8, 2 ulps

This machine's numpy is built against Apple **Accelerate**; the runners use
**OpenBLAS**. Float addition is not associative and BLAS picks its blocking by
matrix shape, so a gathered sub-scan of m rows and a full scan of n rows reduce
in different orders and land a bit or two apart. Varying thread counts locally
does not reproduce it; only a different BLAS does.

This project has already established this once. The G2 gate asserted bitwise
equality between vaults of different row counts, was unsatisfiable for the same
reason, and was replaced by a claim that is exact (`gate_as_written`,
`process_note`). The same unprovable assertion was sitting in a second file.

## The question this measures

Not "do the bits match" -- they provably need not. The product claim is in
README: search "returns what an exhaustive fp32 cosine scan returns". So:

**How often can an ulp-scale perturbation change WHICH documents come back, or
in what order?**

## Method

Cross-BLAS cannot be measured on this machine, so measure the thing that bounds
it and is machine-independent:

1. score every query against the corpus in float32 (what the engine returns) and
   in float64 (the reference the float32 paths are both approximating);
2. `err = |fp32 - fp64|` per score -- the real error scale, not an assumed one;
3. for each query take the float64 ordering and the gap between rank k and rank
   k+1 at k in {1, 4, 10};
4. a query is UNDETERMINED at k when that gap is smaller than the perturbation
   any reordering could apply, taken as `2 * max(err)` over the pair.

An undetermined query is one where two BLAS implementations are permitted to
disagree about the result. A determined one is one where they cannot.

Corpus: `structured_rows(4000, dim=128, rank=24)` and 150 queries, the exact
shape `test_search_returns_the_exact_scan_result` builds, plus a 768-dim run at
the width nanomem actually ships with. Seeded, no fixtures, no network.

## Pre-registered decision rule

  * **> 1.0% of queries undetermined at k=4** -- the README's exactness claim is
    wrong as written and must be restated on the first screen, not in a
    footnote. Report it as a defect in the claim.
  * **<= 1.0%** -- the claim stands with a stated tie-band. `test_screen.py` is
    then rewritten to assert the portable property:
      (a) scores equal within the MEASURED error bound, never bitwise;
      (b) ids and order identical whenever the rank-k/k+1 gap exceeds that
          bound -- which is the case the claim is actually about;
      (c) the bitwise property recorded as an observation about one BLAS, in a
          comment, not an assertion.
  * **0 undetermined** -- unlikely, and if it happens the tolerance in (a) is
    still required, because CI has already shown the bits differ.

Whatever the rate, it goes in the README next to the existing fp16 tie-break
note (which already admits "2 of 500 and 5 of 500 questions"), because it is the
same class of caveat and that note set the precedent for stating it.

The bar is fixed here and is not restated afterwards.

## Not in scope

Making the screen faster, or making it engage. The flag ships OFF and
`_ranking_is_inert()` keeps it off in production; this is about what the tests
are allowed to assert and what the README is allowed to claim.
