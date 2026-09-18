# New use cases for nanomem — pre-registration

Written BEFORE `exp_new_usecases.py` exists and before any number is seen.
Predictions are recorded so a miss cannot be reframed as a hit.

## Why these six

Personal memory — the only profile nanomem has been driven through — is: tens of
entities, 2–5 revisions each, distinct human-readable text, one user, low write
rate. Every use case below breaks at least one of those assumptions on purpose.
The differentiator being tested is the only thing nanomem has that FAISS,
sqlite-vec and Chroma do not: `as_of`, `history`, `changes`, `volatility` over
declared revision chains.

| # | Use case | Assumption it breaks |
|---|----------|----------------------|
| A | Infra config drift / incident forensics | many entities; sibling keys with near-identical text |
| B | Price & catalog monitoring | revisions differ by ONE token; high revision rate |
| C | Multi-tenant SaaS state | one user → many; isolation is a security property |
| D | Long-running agent task state | revisions seconds apart, not days |
| E | Policy / compliance versioning | long text that CHUNKS; `as_of` is the product |
| F | Entity-count scale | ~50 entities → 5,000 |

## Bars, fixed now

- **A1** `as_of` returns the value in force at T for a named config key: **≥ 95%** over 200 probes.
- **A2** `search` returns the right key among siblings whose text differs only in
  the service name: **≥ 80%** top-1.
- **B1** `history` reconstructs a price series in correct chronological order with
  no value dropped or duplicated: **100%** of 50 SKUs (it is a sort, not a guess).
- **B2** `search` returns the right SKU among 200 template-identical rows: **≥ 70%** top-1.
- **C1** Cross-tenant leakage with an explicit tenant filter: **exactly 0** over
  all probes. Any leak fails the whole use case.
- **C2** Recorded, not barred: what a caller who OMITS the filter sees.
- **D1** `history` orders revisions 2 s apart correctly: **100%**.
- **D2** `volatility` reports a rate whose unit a reader can act on (not "0"): **100%**.
- **E1** A policy long enough to chunk keeps its entity on every chunk, so `as_of`
  answers from the version in force: **≥ 95%**.
- **F1** 5,000 entities: `as_of` accuracy holds within **5 points** of A1, and
  search latency stays **< 25 ms** at that size.

## Predictions (recorded before measurement)

1. **A1, B1, D1, E1 pass.** These are sort-and-filter over declared entities and
   supplied timestamps; no inference is involved.
2. **B2 FAILS.** 200 rows of `"<SKU> is priced at $X.XX"` are near-identical in
   embedding space. Cosine cannot separate SKU names it has no lexical signal for.
   I expect top-1 well under 70%. This is the prediction most likely to be wrong
   in the interesting direction.
3. **A2 is marginal** — passes only where the service name is lexically present
   in the stored text.
4. **C1 passes; C2 is the finding.** I expect no leak WITH a filter, and I expect
   that omitting the filter silently returns other tenants' rows. If that is so,
   the defect is that nothing makes the safe path the default one.
5. **E1 is the one I am least sure of.** Chunking splits a long record into
   `{id}_chunk_N`. If chunks do not inherit `entity`, `as_of` on a long policy is
   broken, and that breaks the compliance use case entirely.
6. **F1 passes on accuracy, unknown on latency.** The scan is memory-bandwidth
   bound, so 5,000 entities over ~15,000 rows should stay in single-digit ms.

## Rules

- Negatives are deliverables. A use case nanomem is bad at is as useful to know
  as one it is good at, and is reported with the same prominence.
- Every number reported anywhere must appear in `new_usecases_results.json`.
- The vault is driven through its PUBLIC API only, as an outside user would.
- No bar moves after a number is seen. An amendment is recorded as an amendment.

---

## Amendment 1 — recorded after the first run, before the follow-up measurement

The first run failed A1 (52.0%), E1 (66.7%) and F1 (63.0%) against a 95% bar,
and the cause is NOT the one any prediction named. Root cause, isolated in four
probes and confirmed by a single-word edit to the query:

`entities.temporal_question()` gates the whole entity/revision layer on

    intent is not None or is_personal_query(q) or has_temporal_cue(q)

so a third-person question — "what is the checkout timeout" — never engages it,
and `search` falls back to raw cosine over the revision chain. The newest
revision then wins only by luck. Adding "my", "now", "current" or "latest" to
the same query against the same vault flips top-1 from a superseded value to the
correct one, and `as_of` from 2/4 to 4/4.

**Follow-up to measure, bars unchanged:** re-run A1, E1 and F1 with the ONLY
change being a temporal cue in the query. The original bars (95%) still apply.

**Prediction, recorded before running it:** all three rise above 95%. If they do,
the use-case failures have one cause and one workaround, and the defect is that
nothing tells a caller their query missed the gate. If they do not, there is a
second cause still hidden.

F is re-run at 1,000 entities rather than 5,000 to keep the follow-up cheap. That
is a DIFFERENT corpus from F1 above and its number is reported separately as
F2, never as a correction to F1.
