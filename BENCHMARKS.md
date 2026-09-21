# Evidence

**Don't take this page's word for anything.** Most of what is claimed below is a
test that ships inside the package you can install right now, and the command to
run it is printed next to the claim. The rest is a number in a results file,
reported with the machine it was measured on and where nanomem loses.

```bash
pip download --no-deps --no-binary :all: nanomem==0.7.20
tar xzf nanomem-0.7.20.tar.gz && cd nanomem-0.7.20
pip install pytest && python -m pytest tests/ -q
# 681 passed   -- no network, no API key, no fixtures to fetch
```

Every test named on this page runs from that download except one, which is
called out where it appears.

Everything here is nanomem **0.7.20 / engine 3.4.6**. This line read "0.7.9 /
engine 3.4.1" through 0.7.17 -- nine releases stale, on the page the README calls
"Evidence", while the download instructions above fetched a version from before
half the fixes on it. Timings are an Apple M4
Pro, macOS 26.5, 12 logical CPUs, Python 3.12.12, numpy 2.5.3.

---

## The short version

nanomem is an embedded store for **facts that change**. It keeps every revision,
answers with the current one, and can tell you what it believed at a past moment.

* If you want a pure nearest-neighbour index and nothing else, **use FAISS** — at
  71,433 documents it is 1.8× faster than nanomem, uses less peak RAM, and
  reopens 17× quicker.
* nanomem uses **less disk** than every arm measured (149 MB vs FAISS's 209 MB),
  needs no server, and depends only on numpy.
* What it does that none of them do: revisions, `as_of`, `history`, retention
  that knows a superseded value from a current one, and encryption at rest.

---

## 1. Does it answer with the value that is currently true?

The only thing here a plain vector store cannot do, so it is the first thing to
check. Each claim names the test that proves it.

```bash
python -m pytest tests/ -q -k "backfilled or third_person or floor_protects or search_calls_current"
```

| claim | test |
|---|---|
| A record written out of order does not become the current value | `test_a_backfilled_older_revision_does_not_become_the_current_value` |
| …and equal timestamps still resolve to the last write | `test_the_arrival_counter_still_breaks_a_timestamp_tie` |
| A third-person question resolves a declared chain | `test_a_third_person_question_resolves_a_declared_revision_chain` |
| `as_of` answers with the value in force then | `test_as_of_works_for_a_third_person_question` |
| Retention keeps the record `search` calls current | `test_forget_superseded_keeps_the_record_search_calls_current` |
| `prune` protects it too | `test_prune_protects_the_record_search_calls_current` |
| Deleting the current revision promotes the previous one | `test_deleting_the_current_revision_promotes_the_previous_one` |
| A long document keeps every chunk of its current version | `test_forget_superseded_keeps_every_chunk_of_the_current_version` |

**Measured on corpora**, six application shapes with an attribute schema of their
own (`benchmarks/new_usecases_results.json`):

| use case | corpus | result |
|---|---|---|
| Infra config drift | 50 entities, 200 records | `as_of` **100%** (n=200); sibling key top‑1 **100%** |
| Price / catalog tracking | 200 SKUs, 1,600 records | history order **100%**; SKU top‑1 **100%** (n=200) |
| Multi‑tenant SaaS | 3 tenants, 450 records | **0** cross‑tenant rows in 30 filtered probes |
| Agent task state | revisions 2 s apart | order correct; rate renders `2s`, not `0d` |
| Policy versions | 5 policies, 60 chunk records | `as_of` **100%** (n=15) |
| Fleet state | 5,000 entities, 15,000 records | `as_of` **100%** (n=200) |

**Out-of-sample chat**, 3 personas, 36 verification questions, measured
in-session and again after closing and reopening the vault — identical both ways
(`benchmarks/clean_chat_results_floorfix.json`):

```
gold stores  top-1 97.2%  top-3 100.0%  (n=36)
end-to-end   top-1 94.4%  top-3 100.0%
per persona  12/12, 12/12, 11/12
```

---

## 2. Against a real vector store

One corpus, one embedding matrix, one insertion order, one query set, and the
recall function lives in the driver so no arm can score itself. Every arm is
built, closed, and **reopened from disk** before timing — which is where nanomem
is weakest. 71,433 documents, 768-d (`benchmarks/competitors_standard_results.json`):

<!-- GENERATED: headtohead -- rewritten by evidence/refresh_benchmarks.py -->
| arm | recall@4 | p50 query | disk | reopen | peak RSS |
|---|---|---|---|---|---|
| FAISS flat IP | 60.6% | **1.10 ms** | 209 MB | 0.012 s | **230 MB** |
| **nanomem exact** | 60.6% | 1.91 ms | **149 MB** | 0.248 s | 301 MB |
| sqlite-vec brute force | 60.6% | 54.00 ms | 258 MB | **0.001 s** | 266 MB |
| Chroma HNSW (tuned) | 60.6% | 8.54 ms | 514 MB | 0.002 s | 318 MB |
| Chroma HNSW (default) | 55.0% | 1.25 ms | 488 MB | 0.002 s | 290 MB |
<!-- /GENERATED -->

Query decomposition is **on by default** and roughly triples end-to-end latency
on a short identifier-like query, because it embeds sub-queries separately. Pass
`decompose=False` when your queries are already atomic.

<!-- GENERATED: latency -- rewritten by evidence/refresh_benchmarks.py -->
```
store only, query already embedded      2.19 ms   <- the part nanomem owns
embedding round-trip (local Ollama)    11.11 ms   <- your embedder, not the store
Vault.search(decompose=False)          15.23 ms
Vault.search(...)  the DEFAULT         50.33 ms   <- decomposition adds 35.1 ms
```
<!-- /GENERATED -->

These are wall-clock medians over 60 distinct entities and move a few percent
between runs. The two blocks above are **generated** from
`benchmarks/latency_split_results.json` by `refresh_benchmarks.py` rather than
typed, because a page that hand-quotes two decimals from a file that changes
every run is a drift generator, not a cure. `release_preflight.py` additionally
refuses to ship if any figure on this page is absent from a results file at the
precision it is quoted.

Store-side filtered search, isolated from the embedder using the deterministic
offline encoder — and pinned by `test_the_entity_prefilter_changes_no_result`,
which re-runs the same queries down the old path and requires identical ids:

<!-- GENERATED: storeside -- rewritten by evidence/refresh_benchmarks.py -->
```
unfiltered                 0.78 ms       5 record decodes
filtered, before 0.7.9    33.70 ms  15,001 record decodes   (the whole corpus)
filtered, now              2.11 ms       4 record decodes
```
<!-- /GENERATED -->

---

## 4. Will it lose your data?

The question that decides adoption for a single-file store
(`benchmarks/unexplored_surface_results.json`, `benchmarks/fuzz_ops_results.json`).

A writer killed with `SIGKILL` mid-write, then the file truncated to five points:

| file truncated to | opens | records | partial records | warned |
|---|---|---|---|---|
| SIGKILL, no close | yes | all 40 flushed survive | **0** | — |
| 99% / 90% | yes | 100 | **0** | yes |
| 75% / 50% | yes | 50 | **0** | yes |
| 25% | yes | 0 | **0** | yes |

It falls back to block boundaries, never returns a partial record, and warns
every time.

**Four concurrent writer processes**, 150 records each, flushing after every
write: **600 acknowledged, 600 on disk, 0 lost, 0 duplicate ids.**

**Randomised operation fuzzer.** One vault churned through 120 random operations
— add, add_batch, update, delete, prune, forget_superseded, compact, flush,
reopen, merge, export — timestamps deliberately out of order, three users,
invariants checked after every step: **60 of 60 seeds clean, 19 encrypted, 0
plaintext leaks.**

It is required to *fail* against a version known to have the bug it was written
for. Against 0.7.2 it fails **0 of 12 seeds**, each within 3 to 32 operations. A
fuzzer that passes everywhere measures nothing.

**Degenerate input**: 14 of 14 (`benchmarks/edge_cases_results.json`). Unicode round-trips
byte-exact (emoji, CJK, RTL, combining marks); a 200,000-character single token
stores in 0.25 s; `as_of` before the corpus returns empty rather than guessing.

**The same answer in every configuration**: 8 checks across plaintext/encrypted ×
flushed/unflushed — **32 of 32, zero disagreements** (`benchmarks/untested_configs_results.json`).

**Encryption** — `test_export_from_an_encrypted_vault_stays_encrypted` reads the
raw bytes rather than trusting an API. A wrong password gives
`WrongPasswordError`, none gives `PasswordRequiredError`, and the secret is
absent from disk after `prune` and `forget_superseded`, both of which rewrite the
whole file.

**Migration** — every v2 record survives (14 of 14, 845 of 845) and zero are
marked caller-declared, so upgrading cannot silently change a migrated vault's
ranking. `test_migrated_v2_records_are_never_read_as_declared` pins this, but it
**SKIPS** in the sdist: it needs the recorded v2 golden vaults, which are
research fixtures and are not shipped. It is the one claim on this page you
cannot re-run yourself, and it is flagged rather than quietly listed with the
others. (It is what the 17 skips in the run at the top are: tests needing
fixtures that do not ship.)

---

## 5. Where it runs

CI on every push: **Linux 3.9 / 3.11 / 3.13, Windows 3.9 / 3.13, macOS —
7 of 7 green**, plus a release preflight job. numpy is the only runtime
dependency.

---

## 6. What it does badly

- **FAISS is faster, lighter and reopens quicker.** If you need an ANN index and
  nothing else, use it.
- **Query decomposition costs 3.4×** end-to-end on short queries, on by default.
- **A long query is no longer truncated, but it is still diluted.** The
  100-word cap was removed in 0.7.10; what remains is the embedding model's own
  behaviour, and it is not free. Five questions that scored 5/5 on their own
  dropped to 4/5 when each was prefixed with 140 words of unrelated text — no
  truncation involved, just a longer vector average. Put the question first if
  you can. Measured in `benchmarks/query_truncation_results.json`.
- **Decomposition is capped at 8 sub-queries.** Past that the whole text is used
  as one query. 99.95% of 145,051 real queries decompose to 8 or fewer, but a
  genuine question with nine parts will not fan out.
- **Vocabulary-free fact grouping is impossible.** Five candidate signals were
  measured and all sat at chance against sibling attributes. Without declared
  entities, grouping falls back to a lexical tagger: 70/100 recall on canonical
  phrasing, 0/100 on narrative phrasing.
- **An unfiltered search crosses tenants.** With no `user_id` filter, a query
  legitimately matches every owner's rows — 100 of 150 returned rows belonged to
  other tenants in a 3-tenant probe. Pass the filter.
- **The `router_auto` path is not the default and should not be used at scale**:
  at 71,433 documents it shows 1463.0 MB peak RSS and a 7.2206 s reopen, against
  301.2 MB and 0.2482 s for the exact path.

---

## 7. The failure record

The most useful thing on this page is not a number.

**Nine defects were found in a single day, and none of them by the test suite**,
which was green the whole time. They were found by installing the package from
PyPI, driving the MCP server over stdio, and running it on machines that were not
the development laptop. Every one returned a *confident wrong answer* rather than
an error:

| version | what was wrong |
|---|---|
| 0.7.3 | retention deleted most of a chunked document and kept boilerplate |
| 0.7.4 | a backfilled record became the current value; `top_k` capped silently at 50 |
| 0.7.5 | a third-person question never engaged the revision layer at all |
| 0.7.6 | `add_batch` / `ingest_file` made long documents unretrievable |
| 0.7.7 | retention deleted the record `search` called current |
| 0.7.8 | a regression from 0.7.4 — the relevance floor stopped protecting the current value |
| 0.7.9 | filtered search walked the entire corpus |

One more came the following day, and not the same way. 0.7.10 was found by
*measuring an item already on the weakness list above* — the 100-word query cap,
which had been sitting there described as "defensible". It was not: it silently
discarded the question in exactly the shape people send most, context first and
the question last. Measuring it also surfaced a second defect nothing had
noticed, a pasted document fanning out into 400 full scans. A known weakness is
worth measuring; the list above is not decorative.

`CHANGELOG.md` gives each one the wrong answer it produced and the measurement
that caught it. Two of those were found by the fuzzer, and **one of those two was
a regression introduced earlier the same day** by a fix that had passed every
enumerated test, the ranking baseline and the chat corpus.

That is the argument for reading this page sceptically and running the tests
yourself, which is why the command is at the top.

---

## 8. How the numbers are made

Bars are **pre-registered**: the corpus, the metric and the threshold are written
down before the measuring script exists and before any number is seen. Negative
results ship as results. A change that fixes its target case but costs elsewhere
is rejected rather than re-scoped — `group_floor_sim=0.45` was worth +19.0 points
on declared schemas and −13.9 on the chat set, and was not shipped.

Ranking changes are additionally checked against a 520-result recorded baseline
and an out-of-sample chat corpus, and any changed result has to be explained
individually. That is how the 0.7.8 regression was caught: it passed the
automated check, which asked whether rank 1 moved toward the newest revision and
never asked whether rank 1 was still relevant.
