# Changelog

Paths of the form `scratch/refound/…` are relative to the repository root. Every
number below is from one of those files.

---

## 0.7.13 — engine 3.4.1 (untouched)

**Which sub-query came first decided rank 1, so a pasted preamble answered the
question instead of the question.** Found by the second black-box review.

`search` decomposes a composite question, runs one scan per sub-query, and
interleaves the hits. The interleave walked the sub-queries in INPUT ORDER, so
rank 1 was always the best hit of whatever text appeared first — which, for a
pasted email, a chat turn, or any question asked after context, is the preamble.
Five unrelated questions behind one 46-word preamble:

```
wrapper                    words  subs  default  no-decomp  distinct answers
none                           8   1.0      5/5        5/5                 5
short (no clause commas)      13   1.0      5/5        5/5                 5
email with clause commas      46   2.0      0/5        5/5                 1
long thread (x3)             122   4.0      0/5        5/5                 1
```

The control is the finding: the preamble ALONE returns the same record as the
preamble plus any of the five questions. The question contributed nothing to
rank 1. `benchmarks/decompose_order_results.json`.

Only the order WITHIN an interleave level changes, from input position to score.
Each sub-query still contributes its best hit before any contributes a second,
so a genuine composite question still answers every part — pinned by its own
test.

**This corrects 0.7.10.** That release removed the 100-word query cap for
exactly this shape — a question asked after a long preamble — and the defect
survived it, because the cap was not the only thing discarding the question. It
was verified on the one path where this cannot happen: both the regression test
and the release check against the published wheel passed `decompose=False`, to
isolate the truncation. Isolating the mechanism under test is right; letting
that isolated path stand as the verification of a user-facing claim is not. The
0.7.10 test now says so in its own docstring and names the default-path test
beside it.

The measuring script fails against the pre-fix merge (0/5, and it names the two
wrappers whose answers collapse), so it can detect the defect it certifies gone.

Suite 595 -> 598. Engine untouched: the 520-result ranking baseline is bitwise
identical, the clean-chat benchmark is unchanged at 97.2% / 100.0% top-3 / 94.4%
end-to-end, and the adjacent-attribute numbers are unchanged.

**Still open from the same review**, each getting its own release: `history()`
applies the relevance floor and can return a truncated chain reported as
"never changed" — which makes the README's new "history is authoritative" claim
false as written — and query-intent classification can hoist a different
declared attribute past a 0.27 cosine margin.

---

## 0.7.12 — engine 3.4.1 (untouched)

The black-box review's highest-ranked finding: `search` and `history` disagree
about which value is current, on every chain written through MCP or the CLI.

```
nanomem_search('where do I work')  -> [1] I work at Acme Corp.       <- superseded
nanomem_history('where do I work') -> [current] I switched again, ... Globex
```

**The fixable half: neither surface could declare an attribute.** The README's
stated mitigation for the lexical tagger is `metadata={"entity": "employer"}`,
and the tagger is the thing the README itself measures at 0 of 100 on narrative
phrasing. But `nanomem_add`'s MCP schema offered `text` and `source` alone, and
`nanomem add` had `--source`, `--vault`, `--profile`. So the two surfaces
`mcp.py`'s own docstring calls "the only integration path most callers will ever
use" were locked onto the tagger, and nothing said so. `nanomem_add` also
accepted an undeclared `metadata` argument, dropped it, and replied `Stored:
...`, so an agent that reasonably tried got a success message and no entity.

`entity` and `timestamp` are now on the MCP tool, `--entity` on the CLI, and
both confirm what they recorded rather than saying only `Stored`. With the
attribute declared, the two tools agree — measured across three chains in two
timestamp regimes, `benchmarks/entity_declaration_results.json`:

```
                                      search top-1 == history current
declared / spread                              3/3
declared / all-now                             3/3
tagger   / spread                              2/3
tagger   / all-now  (= MCP, CLI before)        2/3
```

**The half that is not a bug.** On a group the TAGGER inferred, the two can
still disagree, and that is a measured trade rather than an oversight.
`history` resolves the tagged chain and reads the revision counter; `search`
also applies a relevance floor, which here drops the current value because
0.593 - 0.532 = 0.0613 against a `GROUP_COS_DELTA` of 0.06. Exempting the newest
member of an inferred group was measured and rejected at 0.6.5: **-19.4 points**
on the 3-persona chat set, with a 0.40-0.80 sweep finding no threshold that
bought one without paying the other. The README now states plainly that
`history` is authoritative about what is current and `search` only usually
agrees, and the MCP `nanomem_history` description says the same to the agent
reading it.

**A measured dead end, recorded rather than quietly dropped.** The reported
mechanism -- `_widen_group` discarding the tagged group it was handed -- is a
real thing that happens, and unioning instead of replacing does fix the reported
case, 4/4 on the probe above. It was rejected: it re-admits records the cosine
window's tag check had dropped, which is the check 3.0.2 added after measuring
40/40 -> 30/40 on adjacent attributes. The adjacent-attribute corpus did not
catch the regression; two unit tests did, a "desk phone" question answered with
the mobile number and a colleague's number answered with my own.

It is also not the root cause. The tagged group is cut to one member FIRST, by
the relevance floor above -- that is what makes `group.size < 2` and triggers
the replacement branch at all.

**The engine is untouched.** This release changes `mcp.py`, `cli.py`, the README
and tests. The 520-result ranking baseline is bitwise identical and no benchmark
moved. Suite 591 -> 595.

---

## 0.7.11 — engine 3.4.1 (unchanged)

Five defects found by an **independent black-box review** — an agent given the
published wheel, the documentation as its only oracle, and no access to this
source tree. It was told to find where the package disagrees with its own docs.
Its full report is reproduced against 0.7.10 before each fix below. A sixth
finding, and the most serious, is not in this release: see the end.

**`add()` handed back an id that nothing accepted.** `add()` documents its
return as "the parent id ... for a chunked document", and the README says you
can `get`, `update` or `delete` by it later. Past 500 words that id is not a
stored row — the rows are `{parent}_chunk_N` — so every by-id call silently
missed:

```
add() returned            doc_61b9e5eadf     (5 chunks stored)
get(doc_61b9e5eadf)    -> None
exists(doc_61b9e5eadf) -> False
update(doc_61b9e5eadf) -> False
delete(id=...)         -> 0        <- 5 rows still on disk
```

Nothing raised, so a caller deleting a document could believe it had been
deleted. The by-id path now resolves a parent id to its chunks: `delete` removes
all of them, `exists` is True, `update` replaces the document, and `get` returns
the whole text — rejoined by finding the real overlap between neighbouring
chunks rather than assuming its width, since `split_large_text` breaks on
paragraph and sentence boundaries and the bridge is rarely exactly 40 words.
Sub-500-word documents are untouched.

**Three documented arguments did not exist.** All three raised `TypeError`, and
all three were implemented further down and simply never plumbed up to `Vault`:

| documented in | argument |
|---|---|
| README, "Embeddings are yours to choose" | `Vault(..., embedder=MyOwnEmbedder())` |
| `Vault` class docstring | `vector_dtype="float32"` |
| README, twice, as "+19.0 points of top-1" | `group_floor_sim=0.45` |

`group_floor_sim` was the costly one: the README presents it as the concrete
tuning step for the declared-entity workload it spends a section recommending,
and there was no way to apply it. `embedder=` now type-checks what it is given
and names the missing method rather than failing later.

**`EmbeddingProvider(dim=N)` reported a width it did not produce.** `dim=` is
documented as the way to skip the startup probe on air-gapped installs. It was
taken purely on trust: `.dim` returned N while `embed()` returned 768, including
in that air-gapped case, where the lexical encoder *is* the model. An object
that advertises one width and produces another mis-sizes everything built from
it. The encoder now honours a declared width (it hashes into as many columns as
it is given, so there was never a reason it could not), and a declared width the
endpoint contradicts raises the new `EmbeddingWidthError` instead of being
quietly overruled. That error is deliberately not caught by the fallback that
degrades to lexical hashing when no daemon answers — swallowing it would hand
the caller lexical vectors at their declared width while they believed they were
using the model, which is worse than the bug it replaced.

One existing test asserted the old behaviour and was rewritten rather than
deleted. It held that the encoder's width is "a property of the encoder, not of
whatever was probed" — true of a probed width, false of a declared one, and the
distinction is the defect. A second test now pins the case that assertion was
really protecting: a width learned from a live model keeps being produced after
the daemon goes away.

**`prune()` returns bytes freed; nothing said so.** `delete()`, documented
immediately above it with the same `-> int`, returns "the number of deleted
records". Pruning 9 records from a small vault returns 15051, so
`print(f"pruned {v.prune(older_than_days=365)} records")` reports fifteen
thousand of them. Documented, not changed: changing the return would break
callers who read it correctly.

**A width-mismatched reopen said less than it should have.** "you get a warning
and the file's own width" reads as *it carries on at the file's width*. It does
not — the file is safe and unchanged, but the handle's encoder is still the
model you named, so every `search`, `history` and `add` raises. The warning and
the README now say that.

Suite 580 -> 591. The clean-chat benchmark is unchanged (97.2% / 100.0% top-3,
94.4% end-to-end). Engine untouched, so every benchmark result stays current.

**Not in this release.** The review's highest-ranked finding is that `search` and
`history` disagree about which value is current on any chain the tagger grouped
— which is every chain written through MCP or the CLI, since neither exposes a
way to declare an entity. That is a ranking-layer fix and it gets its own
release rather than riding along with five unrelated ones.

---

## 0.7.10 — engine 3.4.1 (unchanged)

**A question asked after word 100 of a query was thrown away.** Both `search`
and `search_multihop` did:

```python
clean_query = " ".join(words[:100]) if len(words) > 100 else str(query).strip()
```

Everything past the hundredth word was discarded, and nothing said so. The shape
this ruins is the one people actually send: context pasted first, the real
question LAST. Measured against `nomic-embed-text` over five such queries
(`benchmarks/query_truncation_results.json`, which ships with the package):

```
                              rank 1 correct    rank 1 moved
bare question, no preamble         5/5               --
140-word shared preamble           3/5              2/5
   the same, capped at 100         1/5
140-word distinct preambles        4/5              1/5
   the same, capped at 100         3/5
```

Those accuracy columns understate it. In every one of these queries the cap
removes the question outright, so the text the engine sees contains none of what
was asked — a correct answer there is leftover filler landing near the right
record, not retrieval. With a SHARED preamble the capped queries are byte for
byte identical to each other, so all five must return one document; 1/5 is
chance.

There is no right number to put there. nanomem bundles no weights and embeds
against whatever endpoint answers, so the real limit belongs to a model this
library cannot interrogate. The default `nomic-embed-text` carries 8192 tokens
and enforces that itself: a 50,000-word query returns in 0.18 s at the correct
width. The cap is gone and the model's own limit is the only one left. Embedding
1,000 words rather than 100 costs 56 ms.

**And a pasted document is no longer decomposed into hundreds of scans.** Each
sub-query is its own full linear scan, so decomposition multiplies search cost,
and `decompose_query` splits on `?` and `;` — which a pasted thread is full of.
A 1,800-word FAQ came back as 400 sub-queries. The 100-word cap was incidentally
holding this down, so removing it alone would have traded a correctness defect
for a latency cliff; but the cliff was already reachable, since the same FAQ
capped at 100 words still produced 22 scans. Fan-out is now bounded directly at
`MAX_SUB_QUERIES = 8`, measured rather than chosen: across 145,051 distinct
queries from every non-quarantined corpus on the development tree, 99.95%
decompose to 8 or fewer and the largest real one is 25. Past the bound the whole
text is used as one query rather than keeping the first 8 — dropping sub-queries
silently is the same defect as the one above.

Neither change moves an existing answer: the clean-chat benchmark is identical
before and after apart from latency jitter. Four tests added, three of which
fail against 0.7.9. Suite 576 -> 580.

---

## 0.7.9 — engine 3.4.1 (unchanged)

**Filtered search was 48x slower than unfiltered, and now is not.** The one open
performance item from the use-case sweep, diagnosed rather than re-reported.

`_apply_filter` walks candidates in descending score, decoding one record per
step, and stops early once `top_k` matches are in hand and the score bound rules
out everything below. For a SELECTIVE filter the matches are scattered, so the
bound cannot fire until the walk is nearly over. Measured on 15,000 records
across 5,000 entities:

```
unfiltered            0.70 ms       5 record decodes
filter={'entity':..}  33.70 ms  15,001 record decodes   <- the entire corpus
```

65% of that time was `json` decoding metadata the walk then threw away.

The walk is now narrowed by the interned `entity_id` column before it starts:

```
filter={'entity':..}   2.08 ms       4 record decodes   (16x faster)
```

**Why this is a superset and not a semantic change.** The column holds
`normalize_entity(...)` while `matches_filter` compares the RAW metadata, so the
two are not interchangeable — and they do not have to be. Normalization is a
function, so `raw == filter` implies `normalize(raw) == normalize(filter)`:
restricting to that id can only drop rows that could never have matched. Every
surviving row still goes through `matches_filter` completely unchanged, so a
false positive is filtered out as before and a false negative is impossible.

Proven rather than argued: 123 searches over 41 filters — mixed case, spaces,
unicode, integer values, list values, multi-key filters, absent keys — return
identical ids with the narrowing on and off, with 84 of the 123 non-empty so the
comparison has bite. That comparison is now a test, which disables the column
and re-runs the same queries down the old path. `temporal_baseline` is bitwise
identical across all 520 results and the 3-persona chat set is unchanged at
97.2% and 94.4%.

It applies only when `entity` is a filter key with a scalar value; every other
filter takes the walk it always took.

**The fuzzer was extended and found nothing further.** The revision group key is
`(user_id, project, entity)` and none of that was being varied: every seed wrote
as a single anonymous user and never merged or exported. It now draws an OWNER
for every write from three users, asserts that a search filtered to one user
never returns a row owned by another, and has `merge` (both reconcile modes) and
`export` (write out, reopen, confirm the target holds what the call reported) in
its operation pool. 60 of 60 seeds clean at 120 operations, 19 encrypted, zero
plaintext leaks. Against 0.7.2 it still fails 0 of 8, now also on multi-user
chains and on `merge`, which the first version never reached.

The isolation invariant caught the HARNESS first, and it is recorded because it
looks exactly like a leak and is not one: a write with no `user_id`, queried
with an `{"entity": ...}` filter, is an UNCONSTRAINED query. That filter names
no owner, so matching every owner's rows is correct — the same behaviour already
recorded as use case C2. Asserting isolation against an unowned write was the
harness being wrong, not the vault leaking.

Suite 571 -> 576.

---

## 0.7.8 — engine 3.4.1

**A regression this session introduced, caught by the fuzzer it also produced.**

The relevance floor exempts a declared group's CURRENT VALUE from being dropped
— without that, a floor tuned to remove records which merely carry a tag also
removes the answer. That exemption asked `_revisions_comparable`. 0.7.4 added a
CONCORDANCE condition to that predicate, and so switched the protection off for
every chain whose arrival order and timestamps disagree — which is precisely the
set of chains 0.7.4 was written to fix. The floor then dropped the current value
and a superseded record answered.

Found on a chain whose revisions read 5, 3, 4 in time order. `search` returned
the MIDDLE record of three:

```
ts 1633868800  rev 5   value-38
ts 1736771200  rev 3   value-11   <- returned
ts 1753273600  rev 4   value-12   <- the current value
```

Two different questions had been fused into one predicate: *do these rows belong
to one chain*, which decides whether a current value exists at all, and *does the
counter agree with the clock*, which decides whether the counter may ORDER it.
`_same_group_key` is split back out and gates the floor; `_revisions_comparable`
keeps the concordance test and gates the ranker. The protected record is now
chosen by timestamp with the counter only breaking a tie, rather than by
`rv.max()`, which on an out-of-order chain is the last-WRITTEN record and not the
current one.

**Inert wherever the two orders agree, and measured as such:** the 520-result
`temporal_baseline` is unchanged from 0.7.5 — 40 changed against the 0.7.2
baseline, all five declared chains, zero document queries, every one moving rank
1 correctly and staying inside the floor. The 3-persona chat set is 97.2% gold
top-1 and 94.4% end-to-end, identical to the baseline for the fourth release
running.

Fuzzer: **60 of 60 seeds clean** at 120 operations each, 19 of them encrypted,
zero plaintext leaks — against 59 of 60 before this fix. Design and standing
bars are recorded in `design/fuzz_ops_spec.md`, including the requirement that
the fuzzer must FAIL against a version known to have the bug it was written for,
since a fuzzer that passes everywhere measures nothing.

Suite 570 -> 571.

---

## 0.7.7 — engine 3.4.0 (unchanged)

**Retention deleted the record `search` calls current.** Found by a randomised
operation fuzzer (`scratch/refound/exp_fuzz_ops.py`), which churns a vault
through random sequences of add / add_batch / update / delete / prune /
forget_superseded / compact / flush / reopen and checks invariants after every
single step.

0.7.4 taught the RANKER to prefer the caller's timestamp when it disagrees with
the arrival counter. Retention was never told: `_current_value_ids` and
`forget_superseded` both still ordered by `(revision, timestamp)`, i.e. arrival.
So on a vault written out of order — an import, a replay, an out-of-order sync —
the two halves of the system disagreed about which record was current:

```
arrival A, B, C; timestamps A=+400d, B=+100d, C=+200d

search                     -> value-A   (newest by time)
forget_superseded(keep=1)  -> kept C, DELETED A
search after               -> value-C   (a superseded value)
```

Both now order by `(timestamp, revision)`: time first, the arrival counter only
to break a tie. That keeps the tie-break the counter exists for — three writes
at the identical instant still resolve to the last one — and it has its own
test, because a fix that disabled it would look correct on every other case.

**The fuzzer has teeth, measured:** against 0.7.2 as published it fails 0 of 12
seeds, each within 3 to 32 operations. Against this tree, 12 of 12 clean.

Two of its early failures were the HARNESS, not the vault, and are recorded
because the distinction is the whole value of the technique: the model counted
each chunk of a long record as a separate value, and it asserted that the newest
version always wins retrieval even when that version is a paragraph of
boilerplate and a short fact in the same chain answers the question better. The
second is the relevance floor behaving exactly as 0.7.5 requires. The fuzzer now
keeps fact entities and document entities in disjoint pools and asserts of each
the thing that is actually true of it.

Three tests; two fail against 0.7.2 as published and the third is the tie-break
guard, which must pass against both.

Suite 567 -> 570.

---

## 0.7.6 — engine 3.4.0 (unchanged)

**`add_batch()` made long documents unretrievable.** It is the method its own
docstring recommends for "books, datasets, and corpora", and it stored whatever
it was given verbatim while `add()` split anything over 500 words into
overlapping chunks. One embedding then had to stand for a whole document, so the
vector represented the document's BULK and not its details, and a fact stated
once near the end became unreachable.

Measured with `nomic-embed-text` on a 48,466-character manual containing one
shutdown code, against 40 documents that merely shared vocabulary:

```
add()       needle at rank 1
add_batch   needle NOT IN TOP 5  -- a safety bulletin won instead
```

The record was in the vault the whole time and search answered with something
else. The dilution is gradual, which is why nothing caught it: with the document
alone in the vault the needle still came back, at a score falling from 0.7773
chunked to 0.4340 unchunked as the document grew.

`add_batch` now splits fresh text exactly as `add()` does, with the same chunk
metadata. **It does NOT split a transfer:** when the caller supplies embeddings
the rows are already whole, and `merge`, `split` and every rebuild go through
that path — re-splitting there would invent records the source never had.
Verified: export 7 records to 7, merge 7 and 7 to 14, no chunk explosion.

**It reaches further than `add_batch` itself.** `ingest_file` and
`ingest_directory` both route through this method, so document ingestion from
disk had the same defect. Measured on one 17,632-byte file: 1 record before,
5 after. That is the most natural way a caller loads a corpus, and it was
storing each file as a single vector.

Bulk ingestion also agrees with `add()` on a declared revision chain written out
of order — every record declared, correct current value, `as_of` 4 of 4 — which
had never been checked, and is how an application with a schema actually writes.

Three tests; two fail against 0.7.2 as published. The retrieval one asserts
PARITY between the two paths rather than rank 1, because rank 1 depends on the
embedder and the suite runs the offline lexical encoder; asserting rank there
would assert the encoder rather than the fix.

Suite 564 -> 567.

---

## 0.7.5 — engine 3.4.0

**An application with its own attribute schema now gets the feature it adopted
nanomem for.** Pre-registered in `design/gate_declared_spec.md` with the
baseline captured before the change and four bars fixed in advance.

`entities.temporal_question()` gated the entity and revision layer on the
QUESTION's wording — `is_personal_query(q) or has_temporal_cue(q)`. A person
asks "what is MY phone number"; an application asks "what is the checkout
timeout". The second never engaged the layer, so `search` fell back to raw
cosine across the revision chain and returned whichever revision scored highest.
On a four-deep chain the correct current value had the LOWEST cosine of the four
(0.90618 against 0.91836) and ranked last.

A group whose entity the CALLER DECLARED is now exempt from the wording test.
That is the argument 0.6.5 already accepted for the relevance floor: the gate
compensates for TAGGER IMPRECISION, and a caller with its own schema has none.
A chat write with an inferred entity is untouched.

`as_of` with the entity pinned, third-person queries, no temporal cue:

```
config drift (200 entities)     54.5%  ->  100.0%
policy versioning                53.3%  ->  100.0%
fleet state (1,000 entities)     64.0%  ->  100.0%
```

**Cost, measured on the corpus that has rejected changes like this before:**
3-persona chat set 97.2% gold top-1 and 94.4% end-to-end, IDENTICAL to the
baseline, per-persona 12/12, 12/12, 11/12 — because the exemption cannot fire on
an inferred entity. `temporal_baseline.py`: 40 of 520 recorded results changed,
all five declared chains, all `current`/`present`, every one moving rank 1 to
the newest revision and staying inside the relevance floor. Zero document
queries changed.

**The first version of this fix was wrong and is worth recording.** Exempting
declared groups from the wording gate also handed them the floor exemption, and
a question about an unrelated document promoted a declared group's newest
revision to rank 1 at cosine 0.016 over a leader at 0.110 — the exact failure
the gate exists to prevent. It passed the automated check, which asked whether
rank 1 moved toward the newest revision and never asked whether rank 1 was still
relevant. The two exemptions are now separate: reaching the layer by DECLARATION
is weaker evidence than reaching it by wording, so on that path the newest
revision keeps no protection from the relevance floor. That regression has its
own test.

Four tests; two fail against 0.7.2 as published and two are guards that must
pass against both.

**A sweep of the public API nothing had driven found no further defect**
(`design/unexplored_surface_spec.md`, results in
`unexplored_surface_results.json`). Recorded rather than assumed, because a
clean sweep is a result and three of six predictions were wrong in the direction
of the code being better than expected:

* `export` from an encrypted vault inherits the encryption. Checked in raw
  bytes: the secret is absent from the default export and the file will not open
  without a password. `target_password=None` is the only route to plaintext.
* `merge(reconcile_revisions=)` returns the newest value BY TIME and a history
  in time order, in both modes, on vaults whose revisions interleave.
* `compact()` changes no answer and keeps every chunk of a chunked value.
* SIGKILL mid-write loses no flushed record. Truncating a vault to 99, 90, 75,
  50 and 25 percent opens every time, returns only whole records, falls back to
  block boundaries and warns on each one.
* Four processes writing one vault and flushing every record: 600 writes
  acknowledged, 600 on disk, none lost, no duplicate ids.
* v2 migration preserves every record (14 of 14, 845 of 845) and marks ZERO of
  them declared -- so 0.7.5's exemption cannot fire on a migrated vault and its
  ranking is unchanged. Zero revision/timestamp inversions, so 0.7.4's
  concordance check does not fire either. Both pinned by tests.
* Deleting the current revision promotes the previous one; a width-mismatched
  embedding model is refused by name.

Suite 556 -> 564.

---

## 0.7.4 — engine 3.3.5

From a pre-registered edge-case sweep
(`scratch/refound/design/edge_cases_spec.md`, 14 cases, results in
`edge_cases_results.json`). 12 of 14 passed as shipped; the two that did not are
below. Unicode round-trips exactly, a 200,000-character token stores in 0.25 s,
odd metadata types survive JSON, `as_of` outside the range behaves, and reopening
after `prune` gives the same answer — all recorded rather than assumed.

**A backfilled older revision became the current value.** `add_fact` numbers a
new record `group_max + 1`, which is ARRIVAL order. That is correct for a chat
log, where arrival and time agree, and it is the mechanism that breaks ties
between records written at the same instant. It is wrong the moment a caller
writes an older record after a newer one — importing history, replaying a log,
migrating from another store, syncing out of order. Three addresses written
newest, middle, oldest:

```
current value ->  "3 Elm Lane"          the OLDEST of the three, by 350 days
history       ->  Oak, Pine, Elm        arrival order, not time order
```

`_revisions_comparable` already encoded the rule "a revision number only orders
a group under condition X, and otherwise the timestamp is the only order there
is". It required one group key; it now also requires that the counter AGREE with
the caller's timestamps. On disagreement the revision key is dropped exactly as
it is for a multi-key group. A timestamp states when a fact was true; arrival
order is an artefact of how it reached the store.

**This is a no-op whenever the two orders agree, and that is measured, not
argued:** `temporal_baseline.py` reports all 520 recorded query results bitwise
identical before and after. Equal timestamps are not an inversion, so the tie
break the counter exists for still works and has its own test.

**`top_k` did two silent things.** `max(1, min(50, int(top_k)))` meant `top_k=0`
returned ONE row — a pagination loop reaching zero got a phantom result — and
every request above 50 was capped with nothing said, so a caller could not tell
"only 50 matched" from "we truncated you". On a 120-record corpus `top_k=120`
returned 50. The cap bought almost nothing: on 2,000 records the engine returns
`top_k=2000` in 17.0 ms against 7.7 ms for `top_k=50`. Nothing asked for now
returns nothing, and a request above 50 is honoured, bounded only by the corpus.

Four tests, three failing against 0.7.2 as published; the fourth pins the tie
break so the concordance rule cannot quietly disable it.

**Also found, NOT fixed:** a query longer than 100 words is silently truncated
(`" ".join(words[:100])`), so a distinguishing term at the end never reaches the
embedder and the answer changes with no indication. A limit is defensible —
`all-minilm` holds ~256 tokens — but 100 words is far under `nomic-embed-text`'s
8,192 and the truncation is undocumented. Fixing it means choosing a limit per
model, which needs its own measurement rather than a guess.

> Fixed in 0.7.10, and that framing was the reason it sat here for five
> releases. There is no limit to choose: nanomem embeds against whatever
> endpoint answers, so a per-model word limit is not merely unknown here, it is
> unknowable from inside the library. Removing the cap and letting the model
> enforce its own was always available.

Suite 552 -> 556.

---

## 0.7.3 — engine 3.3.4 (unchanged)

Found by driving nanomem through six use cases it had never been used for —
infra config drift, price tracking, multi-tenant SaaS, agent task state, policy
versioning and 5,000-entity fleet state. Pre-registered in
`scratch/refound/design/new_usecases_spec.md`; numbers in
`scratch/refound/usecases_findings.json`.

**Retention treated the chunks of one long value as separate revisions.**
`add()` splits a record over ~2 KB into `{parent}_chunk_N`, and every chunk
inherits `entity` — so four chunks of one policy looked like four revisions of
it. Measured before the fix, on a refund policy stored in three versions:

```
forget_superseded(keep=1)                    12 records -> 1
prune(older_than_days=365, keep_current=True) 9 records -> 1
```

In both cases the single survivor was a paragraph of boilerplate, because the
chunks tie on revision and timestamp and the first one won. `eligibility window
is 90 days` — the thing the vault was being asked for — was gone. This is the
exact promise `forget_superseded` makes in its own docstring and the exact hole
0.7.1 closed for single-record facts and left open for chunked ones.

Worse than stated above: a policy written ONCE and never revised still lost two
of its three chunks, because `min_revisions=2` counted chunk records rather than
versions.

A version is now the unit that `keep`, `min_revisions` and `keep_current` count,
and `keep_current` protects every record of the current version rather than one
arbitrary id. Same corpus after the fix: `12 -> 4`, `9 -> 3`, never-revised
document `deleted 0`, current version readable end to end in all three.

Four tests, three of which fail against 0.7.2 as published; the fourth asserts
the precondition that the fixture chunks at all, so the other three cannot
quietly stop testing anything.

**Also found, NOT fixed here, because it is a ranking change:** the entity and
revision layer is gated on the QUERY's wording rather than on the data.
`entities.temporal_question()` requires `is_personal_query(q)` or
`has_temporal_cue(q)`, so a third-person question — "what is the checkout
timeout" — never engages it and `search` falls back to raw cosine across the
revision chain. On a four-deep chain the correct current value had the LOWEST
cosine of the four (0.90618 against 0.91836) and ranked last. Adding one word:

```
"what is the checkout timeout"          -> "30 seconds"   (superseded)
"what is the current checkout timeout"  -> "90 seconds"   (correct)
```

At corpus scale, `as_of` accuracy with the entity pinned: config drift
54.5% -> 100.0%, policy versioning 53.3% -> 100.0%, 1,000-entity fleet
64.0% -> 100.0%. One cause, one workaround. Until it changes, an application
with a declared schema should put "current", "now" or "latest" in its queries;
the defect is that nothing tells a caller their query missed the gate.

Suite 548 -> 552.

---

## 0.7.2 — engine 3.3.4 (unchanged)

Three defects found by driving the PUBLISHED 0.7.1 from outside — a clean venv
install from PyPI and the MCP server over stdio — rather than by the suite.
That is now the ninth release in a row where that is where the bug was, and the
suite had a test pinning two of these three in place.

**`forget_superseded(dry_run=True)` reported `deleted: 0`.** `ids` held the
records that would go, but the headline number said none. The obvious caller

```python
plan = v.forget_superseded(keep=2, dry_run=True)
if plan["deleted"]:          # 0 -- always
    v.forget_superseded(keep=2)
```

therefore never cleaned anything, and nothing said so. For a full-rewrite delete
with no undo, a preview disagreeing with the run it previews is the one thing it
must never do. `deleted`, `ids`, `groups` and `kept` now describe the same
operation whether or not it runs, and a new `dry_run` key in the result tells
the two apart.

**`kept` echoed the `keep=` argument instead of counting anything.** Three of
the four keys were outcomes and the fourth was the knob you had just passed in,
so `f"deleted {r['deleted']}, kept {r['kept']}"` printed a true number and a
false one. It is now the revisions left across every tagged fact.

**Two MCP tools answered a missing argument with `float() argument must be a
string or a real number, not 'NoneType'`.** `nanomem_changes` without `since`
and `nanomem_as_of` without `as_of` both declare the field required, but nothing
between a model and the handler enforces that, and an LLM client omits an
argument routinely. The message named neither the tool, the argument, nor the
fix. The bad-VALUE path was already right — `since="last week"` has always
answered "could not read 'last week' as a time; use YYYY-MM-DD…" — so this was
the one hole in an otherwise good surface. Both now say which tool wants which
argument and what a usable value looks like.

**An unknown tool name came back as a SUCCESSFUL call.** `{"content": [{"text":
"Unknown tool: recall"}]}` with no `error` and no `isError`, so a client had to
string-match content to learn its call had not happened. It is now a JSON-RPC
error that names the seven tools that do exist.

Six tests, five of which fail against 0.7.1 as published (the sixth covers the
already-correct bad-value path). Two tests that pinned the old behaviour are
gone: `test_unknown_tool_does_not_raise` asserted the bug in its own name.

Suite 543 → 548.

---

## 0.7.1 — engine 3.3.4 (unchanged)

**`prune()` no longer deletes the answer.** 0.7.0 documented this and shipped an
alternative. That was the wrong call — a documented trap is still a trap, and
the whole standard this project has been held to is that a confident wrong
answer is the worst outcome available.

```
before  prune(older_than_days=365):  "what is my blood type" -> "My blood type is O negative."
0.7.0                                "what is my blood type" -> "My locker code is 5555."
0.7.1                                "what is my blood type" -> "My blood type is O negative."
```

`keep_current=True` is now the default and exempts the newest revision of every
tagged fact, whatever its age.

**It costs the case prune was built for nothing.** Only a record that is the
newest revision of a tagged chain is exempt, and an ingested document carries no
entity — so ageing out a corpus prunes exactly as it did. Verified: six
paragraphs in, six paragraphs out. What changed is only the case where it used
to destroy an answer.

It is also not a no-op on facts. Superseded revisions are still the oldest
records in the vault and still go: a five-deep locker-code chain came back four
deep, with the current value intact.

`keep_current=False` restores age-alone selection for a caller who means it.
`forget_superseded` remains the better tool for retention, because it can keep
more than one revision and says what it would remove before removing it.

Four tests replace the one that pinned the old behaviour on purpose. That test
was written yesterday to stop anyone changing `prune` by accident; the right
answer was to change it deliberately.

Suite 540 → 543.

---

## 0.7.0 — engine 3.3.4 (unchanged)

**`forget_superseded()` — retention that knows a revision from a fact.**

Used as a dump, a vault grows a tail of values that were true once: five
addresses, four phone numbers, three employers. Only the newest of each answers
a question; the rest exist so `history` and `as_of` can say what it was before.
Past some point you stop wanting all of them.

```python
v.forget_superseded(keep=1)                      # only the current value of each fact
v.forget_superseded(keep=3)                      # current plus the two before it
v.forget_superseded(keep=1, older_than_days=365) # and only revisions over a year old
v.forget_superseded(keep=1, dry_run=True)        # what would go, without going
```

It will never delete the current value of a fact at any age, and never touches a
record with no entity, because a record in no chain has nothing superseding it.

**Why it had to exist: `prune(older_than_days=N)` selects on age alone.** On a
memory vault that is a trap, and an easy one to walk into while looking for
exactly this feature. Measured:

```
before  prune(older_than_days=365):  "what is my blood type" -> "My blood type is O negative."
after   prune(older_than_days=365):  "what is my blood type" -> "My locker code is 5555."
```

The blood type had not changed in 700 days and was still true. Deleting it is
bad; the query then returning **a different fact** is worse than returning
nothing, because the caller has no way to tell. `prune` is the right call for a
document corpus you are ageing out, and its docstring now says so in those
words instead of the single line it had.

Seven tests, including one that pinned `prune`'s revision-blindness
deliberately. **0.7.1 removed that blindness instead** — see above; pinning the
defect rather than fixing it was the wrong instinct, and the pin lasted one
release.

Suite 533 → 540.

---

## 0.6.9 — engine 3.3.4 (unchanged)

**Two revisions on the same day were indistinguishable through MCP.** Which is
the case this whole store exists for.

`mcp._fmt_when` formatted `%Y-%m-%d` and nothing else, so a fact corrected four
times over nine hours came back as four identical dates:

```
  2026-09-17  [superseded]  My locker code is 1111.
  2026-09-17  [superseded]  Changed it. My locker code is 2222.
  2026-09-17  [superseded]  Changed again. My locker code is 3333.
  2026-09-17  [current]     Final. My locker code is 4444.
```

An agent reading that cannot tell how recent any of them is, or that any time
passed between them. The CLI has always printed the time; only this surface
dropped it — and this surface is the one an assistant actually uses.

**And `volatility` rendered every duration in whole days.** A fact restated
every three hours read `~every 0d, last confirmed 0d ago` — identical to a fact
with no measurable interval at all. That rate IS the answer `volatility` exists
to give: something changing every few hours is the most volatile thing in the
vault, and it was the one case displayed as nothing. Durations now pick a unit:
`45s`, `7min`, `3h`, `5d`, `5mo`, `2.5y`.

```
  locker_code: 4x, ~every 3h, last confirmed 0s ago
```

**The storage was never wrong.** Timestamps are float64 unix seconds and always
have been: `as_of` resolves correctly to the hour inside a single day, `changes`
windows to the hour, and `volatility` computed the true 10,800-second median all
along. Only the rendering rounded it away. Three regression tests, all failing
against the published 0.6.8.

Suite 530 → 533.

---

## 0.6.8 — engine 3.3.4

**Windows.** The first CI run reported 50 failures there, from two POSIX idioms
the package used unconditionally. Both are invisible on macOS and Linux, which
is why one machine's green suite never found them.

*The arena sidecar unlinked a file it still had open.* `os.unlink` on a live
handle is the idiom that makes a temp file crash-proof and unreadable by other
processes — and Windows raises `PermissionError [WinError 32]` for it. It was in
the constructor, so **every float16-residency vault failed to open at all**; 16
of the 50. `os.pwrite` does not exist there either, so the constructor raised
before anything could reach the second problem. On Windows the file is now
deleted in `close()` and written with `lseek` + `write`. The cost is stated in
the docstring rather than hidden: a hard crash can leave one temp file behind,
and another process could open it in the window before deletion. Neither is true
on POSIX; both beat not running.

*`replace_all` replaced a file it was holding open.* `replace_with` was called
inside `exclusive()`, which holds the vault's own handle. POSIX swaps the
directory entry and the open fd keeps the old inode — that is what makes the
rewrite atomic for a concurrent reader. Windows refuses with
`PermissionError [WinError 5]`, and the retry loop already in the code could
never help, because the handle was held for the whole operation rather than
momentarily. 18 failures, plus `compact`, which goes through the same path. The
handle is released there before the replace now, and the window that opens is
named in the docstring: `_assert_same_file` on the next `exclusive` is what
catches a write that lands in it.

**Both fixes are tested on every platform, not just on Windows.** A new
`tests/test_windows_paths.py` forces the Windows branch — `_CAN_UNLINK_OPEN`
off, `os.pwrite` removed, `_REPLACE_NEEDS_CLOSED_HANDLE` on — so a regression
fails on a developer's machine rather than twenty minutes later in CI. Eight
tests, including one that asserts the exclusive handle really is closed when
`os.replace` runs; with the fix removed it fails, which was checked rather than
assumed.

**macOS: the engine was right and the tests were wrong.** Eight failures there
were tests demanding a screen that had correctly declined to exist.
`_screen_ready` calls `_calibrate_gather()` and stands the screen down when a
gathered sub-scan does not give the full scan's arithmetic on that machine —
identical scores being the flag's entire promise. The runner's BLAS is such a
machine. Those tests now skip with that reason, and a new one asserts the half
that actually matters there: with calibration forced to fail on any platform,
the screen never engages and every search returns exactly what the full scan
returns.

**Python 3.9.** `sys.stdlib_module_names` arrived in 3.10, and the static import
audit used it unguarded — so on 3.9 it raised `AttributeError` instead of
checking anything, while the other 504 tests passed. The package was fine; the
test was not. It skips below 3.10, where four other jobs cover the same rule.

**What Windows still cannot do, stated rather than skipped quietly.** Ten tests
now skip there, each with its own reason rather than a blanket marker, and every
one is a thing the platform refuses rather than something nanomem gets wrong:

* it will not `os.replace` or delete a file the process has **mapped**, so a
  live arena cache cannot be rewritten in place. That is recorded as a
  `write_error` naming the limitation, the old cache stays, and the next open
  re-scans the arrears — correctness is untouched and the cost is the
  optimisation;
* it will not truncate a mapped file, so two tests cannot even set up the
  torn-tail case they exist to provoke;
* `chmod 0600` has no meaning there, so the two owner-only assertions have no
  mode bit to check;
* `msvcrt` has no shared lock — `container.file_lock` has always said so and
  warns — so concurrent opens are not serialised the way `fcntl` serialises
  them, and **several processes appending to one vault is not a guarantee
  nanomem makes on Windows**. That is the one limitation here worth planning
  around: a single process, or one writer with readers that tolerate a stale
  view, is the supported shape there.

Four test bugs of our own surfaced with them, all invisible on POSIX:
`os.getloadavg` decorating a diagnostic print, source read with the platform
codec (cp1252) instead of UTF-8, a `process_time()` ratio that divides by zero
where the clock's tick is 15.6 ms and the operation takes 20 µs, and a CONTRACT
test asserting `process_peak_rss_kb` is a number when its own contract says it
is `None` wherever `resource.getrusage` is missing.

Suite 521 → 530.

---

## 0.6.7 — engine 3.3.3 (unchanged)

**CI's first run found two things this laptop could not.**

**`requires-python` said 3.8; the package cannot be built on 3.8.**
`license = "AGPL-3.0-or-later"` is PEP 639, which setuptools supports from
77.0.0 — and setuptools 77 itself requires Python ≥ 3.9. On 3.8 the backend
rejects pyproject.toml outright, so anyone on 3.8 installing from the sdist got
`ValueError: invalid pyproject.toml config: 'project.license'` instead of
nanomem. Nothing in `nanomem/` uses a construct newer than 3.8 — the source
would have run — but it could not be built, which from outside is the same
thing. The floor is `>=3.9`, `build-system` pins `setuptools>=77`, and CI keeps
a job on it.

**`test_screen.py` asserted something BLAS is under no obligation to provide.**
Eleven tests demanded bitwise-identical float32 scores from two matmuls of
different shapes — a gathered sub-scan of m rows against a full scan of n. BLAS
blocks by shape, float addition is not associative, and the same query came back
**2.98e-08 — two ulps —** apart on OpenBLAS from its value here on Apple
Accelerate. Every CI runner failed; this machine passed. The G2 gate learned the
identical lesson in 0.5.0 and was replaced for the identical reason; the same
assertion was sitting in a second file.

Measured before changing anything, pre-registered in
`scratch/refound/design/screen_exactness_spec.md`
(`screen_exactness_results.json`):

| corpus | worst \|fp32 − fp64\| | smallest rank-4 gap | undecided at k=1,4,10 |
| --- | ---: | ---: | ---: |
| 4,000 × 128 | 2.01e-07 | 4.46e-06 | 0 / 150 |
| 4,000 × 768 | 2.01e-07 | 2.99e-05 | 0 / 150 |

The gaps are twenty to a hundred and fifty times the error, so **no accumulation
order can change which documents come back or in what order** — only the last
bits of the score beside them. The bar was set at 1.0% undecided before the
numbers were seen; the result is 0.0%.

So the tests now compare **the score sequence**, position by position, within
the measured 1e-6 band, and the bitwise property is recorded as an observation
about one BLAS rather than promised.

Requiring identical id ORDER was the first attempt and was still too strong,
which only Linux could show. `test_exact_duplicate_rows` builds 200 exact
duplicates on purpose: inside each path those are exactly tied and each breaks
the tie by row id consistently, but an ulp between the paths can separate a
duplicate from a near neighbour that was never tied. Where two documents are
tied inside the band the corpus does not contain a tie-break, and demanding a
particular one of two identical documents is demanding an answer the data does
not have. Ids need no separate assertion: a document at the wrong rank either
has a score outside the band — a real miss, caught — or inside it, which means
it was interchangeable.

Checked for teeth rather than assumed: the contract accepts a two-ulp difference
and a swap of genuinely tied documents, and rejects a wrong order, a score error
of 1e-4, a higher-scoring document that should not be there, and a short result.

A negative control in the same file also stopped being one. It showed that a
plain `argpartition` selection would differ from the row-id rule — on ONE
arrangement, and on the runners' numpy that arrangement happened to agree, so
the control passed by coincidence. It now tries six arrangements, each still
containing every tied row, and asserts the stable rule is invariant across all
of them.

README says it plainly now, beside the fp16 tie-break note it belongs with:
"returns what an exhaustive scan returns" is a claim about which documents come
back and in what order, not about the bit pattern of the float beside them.

---

## 0.6.6 — engine 3.3.3

**Four of the seven MCP tools were dead, and they answered anyway.**

`nanomem/mcp.py` writes with `source="mcp_client"`. That string was not in
`entities.PERSONAL_SOURCES`, which is the gate deciding whether the entity
tagger runs at all. So on the MCP path no entity was ever assigned, no revision
group ever formed, and the temporal tools that server exists to expose had
nothing to work with:

* `nanomem_history` answered **"This has one value and has never changed"**
  about a fact that had just been revised in the previous tool call;
* `nanomem_volatility` could only ever return nothing, because `volatility()`
  excludes records with no entity — correctly, but there were never any;
* `nanomem_search` ranked without the revision layer.

None of them errored. They returned confident, well-formatted, wrong answers,
which is worse than a stack trace and is why this survived the promotion of the
server into the package in 0.6.0.

Found by driving the server over stdio while writing its quickstart — the same
way a first-time user would meet it, and not something the suite could see,
because every MCP test built its vault through the library with a chat source.
`mcp_client` and `mcp` are now personal sources. Two regression tests, both
failing against the published 0.6.5.

**The README is a landing page now, not a folder note.** It opened with
"nanomem — portable standalone folder", then eleven lines of licensing, then a
correction about a test file that never existed. Someone arriving from `pip
install nanomem` read all of that without learning what nanomem does. It now
opens with the one thing it does that a vector store does not, and a sample that
runs.

`release_preflight.py` EXECUTES that sample and diffs stdout against the
comments claiming what it prints. The first draft raised `IndexError` on its
`as_of` line — the facts were all written at `now`, so "200 days ago" preceded
every one of them. A landing sample is the code most likely to be tried and
least likely to be tested.

**Documented: the tagger is the ceiling when you do not declare entities.** The
README now says so on the first screen, because the honest version of the
example proves it. Written with no `metadata={"entity": ...}`, the lexical
tagger reads "I moved jobs, I now work at Initech." as `location` rather than
`career` — "moved" outweighs "work at" — and one chain silently becomes two.
Measured recall is 70 of 100 on plain chains and 0 of 100 on narrative phrasing.
Everything nanomem does beyond a vector store depends on knowing which
statements are about one fact, so an application with its own attributes should
declare them.

Also: a copy-pasteable `claude_desktop_config.json` block, verified by running
the seven tools over stdio rather than by writing it down and hoping.

Suite 519 -> 521.

---

## 0.6.5 — engine 3.3.2

**`search()` returned a superseded value as current.** Recorded as a major
defect on 2026-09-17 in `scratch/refound/finding_floor_drops_current_value.json`,
reproduced identically on 0.5.0, and open through 0.6.4:

    search("where do I work")  ->  "I moved jobs, I now work at Initech."
    the current employer, Globex, ranked THIRD

`GROUP_COS_DELTA = 0.06`. The group's best member sat at cosine 0.6348 and the
current value at 0.5317, a gap of 0.1031, so the relevance floor removed the
newest revision from the group BEFORE `apply_revision_lead` ran. The lead went
to the most recent SURVIVING member. The current value was never a candidate.

The trigger is ordinary phrasing: "I switched again, I work at Globex now."
carries two narrative clauses before the value, so it sits further from the
question than the superseded statement does.

**The floor is compensating for tagger precision, so it now asks who tagged.**
Either the caller named the entity (`metadata={"entity": ...}`) or the lexical
tagger guessed it. The engine has always known which; it just never wrote it
down. It does now, per revision group, and a DECLARED group's newest revision is
exempt from the query-relative floor while an inferred group's is not.

Measured over six arms, pre-registered in
`scratch/refound/design/floor_current_value_spec.md` before the experiment was
written (`floor_current_value_results.json`):

| arm | 0.6.4 | 0.6.5 |
| --- | --- | --- |
| A drifting phrasing, declared tags | 52.0 | **71.0** |
| B canonical phrasing, declared tags | 89.0 | 90.0 |
| C sibling / adjacent-attribute probes | 68.6 | 69.3 |
| D 3-persona chat, tagger-inferred | 97.2 | 97.2 |
| E `historical_value`, declared tags | 64.0 | 64.0 |
| F `previous_value`, declared tags | 92.9 | 92.9 |

**Two cheaper fixes were measured first and rejected.** Exempting the newest
member unconditionally gives the same +19.0 on arm A and costs **-19.4** on the
chat set — all seven of its regressions on `phone number` and `address`, the two
attributes with siblings, which is a tagger-merged pair being promoted rather
than a revision. Gating that exemption on how much the newest member resembles
the group (sweep 0.40 to 0.80) found no threshold that helped A without costing
D the same: at 0.40 it is the unconditional rule, by 0.70 it does nothing.
Provenance is the signal that separates them, so provenance is what is stored.

Arms E and F were added by amendment after F1's numbers on A-D were seen, and
the amendment is recorded as one. Both candidates work by protecting the newest
revision, and `historical_value` and `previous_value` are the two question types
whose answer is explicitly not the newest — 156 questions that the original four
arms never scored. They came back 0.0 and 0.0.

**Exactness.** Gate G1 does not apply: this deliberately changes ranking. What
was required instead is that nothing outside the revision path moves, and that
holds — of 70 queries in the 520-result baseline, 18 changed and every one is
about `employer`, `home address` or `phone number`, the three attributes with
revision chains. All 40 document queries and both non-revising attributes are
bitwise identical.

**The arena sidecar format goes 3 to 4** to carry the new per-group column. The
sidecar is a CACHE: a version it does not recognise is rejected and rebuilt on
the next open, so vault files are untouched and no migration is needed. Records
written before this release carry no provenance marker and read as INFERRED,
which is the behaviour they were written under — an existing vault ranks exactly
as it did until something is written to it declaring an entity.

Five tests pin what no benchmark can see: that the bit survives a reopen, that
it survives a rebuild with the sidecar deleted (or the answer would depend on a
cache being warm), that a pre-0.6.5 record reads as inferred, and that one
declaration makes the whole group declared. Suite 514 -> 519.

`floor_keeps_current` is a three-way override for callers who want the rule
always on or always off; `None`, the default, decides per group.
`group_floor_sim` is untouched and stays 0.0 — it is a different question, one
asked of every member rather than of the one the revision counter already calls
current.

---

## 0.6.4 — engine 3.3.1

**`volatility()` and `staleness()` only counted SEALED blocks.** Found by
installing the published 0.6.3 wheel from PyPI into a clean virtualenv and
using it, which is not something the suite can do for you.

Records live in a memtable until `block_capacity` (50) of them accumulate and
spill to disk. `search`, `history` and `changes` have always scored the pending
ones — each has had a "sees unflushed records" test since it shipped.
`volatility()` read the resident `arena` columns alone, so it saw only what had
already spilled:

| writes, newest made *now* | `stats()['memtable_pending']` | 0.6.3 `volatility()` |
| --- | --- | --- |
| 3 | 3 | `[]` |
| 49 | 49 | `[]` |
| 120 | 20 | `n_revisions=100`, `age=20 days` |

The empty answers are bad; the third row is worse. It reports a fact restated
*today* as three weeks unconfirmed — a confident wrong answer from the method
whose entire purpose is to say which facts need re-confirming, and the exact
failure this project is positioned to fix. `staleness()` is built on
`volatility()`, so it returned the same thing.

Under 50 writes — the whole life of a small vault — the primitive returned
nothing at all.

Now both fold in the pending memtable, so flushing changes durability and never
visibility. A group with records on both sides of the boundary is counted once,
with all of its timestamps. Untagged pending rows are excluded by the same rule
as untagged sealed ones.

Five regression tests. Four fail against the published 0.6.3 wheel and pass
here, which is the only check that matters for a fix; the fifth guards the
untagged-pending path, where 0.6.3 returned the right answer for the wrong
reason. Suite: 509 -> 514.

**Packaging, from the same release.** There is now an sdist — the AGPL
corresponding source, and what conda-forge and Debian build from. A clean
virtualenv built from it alone runs the suite. `MANIFEST.in` is what puts the
tests, the CHANGELOG and the seven linked documents in it; the setuptools
default ships the package and nothing else.

Two more caught by preflight before upload:

* All seven of README's links were relative. GitHub resolves those; PyPI does
  not — they would have been seven dead links on the project page.
* `*.npz` in `.gitignore` had eaten a second shipped file:
  `tests/data/adjacent_attributes.npz`. Unlike the classifier head, this one
  fails silently — `_fixture()` calls `pytest.skip`, so a clone loses the
  coverage and still reports green.

`release_preflight.py` grew four checks for these, each negative-tested to fire
on the defect and stay quiet on the real tree.

---

## 0.6.3 — engine 3.3.0 (unchanged)

**The macOS bundles carried 139 MB of a model nanomem never opened.**
`nanomem/assets/model.bin` is a genuine GGUF of `nomic-embed-text-v1.5` (its
header reads `general.architecture = nomic-bert`), downloaded in some earlier
session. The only code that ever touched it called `os.path.exists` and
`os.path.getsize` to set `has_fused_weights` and `fused_size_mb`, which nothing
read — while the class docstring advertised "bundled offline neural weights".

It could not have been used. GGUF is llama.cpp's format; reading it means a C++
dependency or writing BERT inference plus GGUF dequantisation in numpy, which
would be slower than the daemon it replaces and would cost the
only-dependency-is-numpy property that is the point of the project. It is also
redundant — `ollama pull nomic-embed-text` is the same model, and that is what
nanomem talks to.

Removed from both bundles, along with `FUSED_MODEL_PATH` and the two dead
attributes. One archived copy kept at `scratch/refound/assets/` with a note;
there were four (two bundles, two snapshots, 556 MB).

**The macOS bundle goes 152 MB → 10 MB.**

And the offline fallback is now described honestly. The README said "usable, but
not the real model", which understates it: it hashes words to sine frequencies
and bumps dimensions on character trigrams, so it scores shared WORDS AND
CHARACTERS rather than agreement.

| pair | cosine |
| --- | --- |
| "the server is up" / "the server is down" — opposites | 0.783 |
| "my dog is black" / "my car is black" — unrelated | 0.740 |
| "I drive a car" / "I own an automobile" — same meaning | 0.286 |

Opposites score near-identical and synonyms score unrelated. It is fuzzy string
matching that keeps the pipeline running offline, and anything measured with it
is measuring string overlap. Both the numbers and that sentence are in the
docstring and the README now, because "falls back to a deterministic hash
encoder" invites a reader to assume it degrades gracefully, and it does not.

No behaviour changes. 509 tests.

---

## 0.6.2 — engine 3.3.0 (unchanged)

**`embed_model=` was a parameter you could pass, not a model you could use.**
`EmbeddingProvider.dim` was the constant `768`, and `Vault` sizes its engine
from that number, so naming any other model built a 768-d vault that then
rejected every write:

```
Vault(embed_model="all-minilm")      # a real 384-d model
-> ValueError: embedding has 384 dims, vault has 768     on the first add()
```

The container was never the limitation — it stores whatever width it is given,
and an existing file's header width already won over the requested one. Only
the constant was.

`dim` is now probed once from the model itself, lazily, with a single short
request, and cached from the first real batch if one happens first. Any width
works: `all-minilm` (384), `mxbai-embed-large` (1024), OpenAI
`text-embedding-3-small` (1536). Pass `EmbeddingProvider(dim=…)` to skip the
probe for an air-gapped install or a lookup-table provider. The offline hash
encoder keeps its own fixed 768 — that width is a property of *that encoder*,
not of the store.

Verified end to end against a real 384-d model: sized, written, searched at
cosine 1.0 on identical text, closed and reopened. 502 → 509 tests.

---

## 0.6.1 — engine 3.3.0 (unchanged)

**LLM provider routing was decided by substring matching, and it misrouted three
real cases.** The test was:

```python
is_openai_compat = (base.endswith("/v1") or "/chat" in base
                    or "1234" in base or "8000" in base)
```

Those port numbers were matched against the *whole URL*, not the parsed port:

* **Anthropic silently 404'd.** `https://api.anthropic.com/v1` ends `/v1`, so it
  was POSTed to `/v1/chat/completions`. Claude's API is `/v1/messages`, with a
  top-level `system`, a required `max_tokens`, and `x-api-key` rather than a
  bearer token. Callers saw `[Model not found]` for a model that exists.
* a host named `web8000.internal` on port 11434 was treated as OpenAI-compatible
* **llama.cpp on :8080** was sent to Ollama's `/api/generate`, because "8080"
  does not contain "8000"

`_llm_endpoint()` now resolves on the parsed host and port, most-explicit-first,
and returns the request *shape* as well as the URL. Anthropic is a first-class
flavour. An explicit `/chat/completions` path is used verbatim, which is how an
Azure deployment URL with its `api-version` query is supported.

Verified on the wire against a capturing server, not just by URL: the Anthropic
request carries `x-api-key`, `anthropic-version`, `max_tokens` and a top-level
`system`, and carries no bearer token; the OpenAI request carries
`Authorization: Bearer` and a system+user message pair. 16 provider URLs route
correctly, 443 -> 502 tests.

| routes to | providers |
| --- | --- |
| Ollama native | `localhost:11434` (default) |
| OpenAI-compatible | OpenAI, Groq, Together, Mistral, DeepSeek, OpenRouter, Fireworks, LM Studio, vLLM, llama.cpp |
| Anthropic native | `api.anthropic.com`, or any path ending `/messages` |

Also: `volatility()` and `staleness()` are now on `Vault`, not only on
`VaultEngine`. 0.6.0 shipped `history` and `changes` as `Vault` wrappers and left
these two reachable only through `Vault.engine`, so the README had to tell people
to reach past the public object for the headline capability.

---

## 0.6.0 — engine 3.3.0, container format 3 (unchanged), arena cache format 3 (unchanged)

**The revision layer stops being something only the ranker can reach.** Every
personal-memory query already assembled a *revision group* — the set of records
that are competing statements of one fact — used it to decide which member to
surface, and then discarded it. Three calls return it instead.

Nothing existing changes. `search()` with no `as_of` is bitwise identical to
0.5.0 across 520 recorded query results, and the entire 0.5.0 suite passes
untouched — 443 tests before, 467 after, 0 failures
(`scratch/refound/temporal_g1_results.json`; the gate is stated in
`scratch/refound/design/temporal_api_spec.md`).

**`history(query)`** — every value a fact has held, oldest first, the current one
last, each with `timestamp`, `revision` and `superseded`. No boost is applied, so
`cosine` is the raw similarity. A fact that never changed has a one-element
history, which is an answer rather than an empty result.

It deliberately does **not** apply the ranker's relevance floor. That floor asks
"is this member a plausible answer to THIS QUESTION", which is the right question
for ranking and the wrong one for an audit surface: measured on four restatements
at cosine 0.80 / 0.77 / 0.74 / 0.71, `GROUP_COS_DELTA = 0.06` drops the fourth,
and the chain then reports the **third** as current with `superseded=False` while
a newer value exists (`scratch/refound/temporal_g1_results.json`). The cost of the opt-out is the opposite error — a record
carrying the tag without being a restatement can appear — which is the safer
direction, because the entry arrives with its own text and timestamp and nothing
is hidden.

**`search(..., as_of=<unix ts>)`** — the answer as the vault stood at that
moment. It forces an exhaustive scan and turns the PCA screen off: masking a
routed or screened shortlist would be wrong, because a newer record can crowd an
older one out of selection before the mask is applied, and that older record is
exactly what an as-of query is asking for. Proven against physically truncated
vaults — the admitted row set is exactly `{rows : ts <= t}` over 705 checks with
no tolerance, and 5,670 comparisons give **0 id differences, 0 order differences
and 0 timestamp leaks** (`scratch/refound/temporal_as_of_results.json`).

**`changes(since, until)`** — what was written in a window, with no query vector
and no embedding call, read off the resident timestamp column. The interval is
half-open on `since` so that `changes(t)` reports exactly what `search(as_of=t)`
could not see.

The CLI gains `nanomem history`, `nanomem changes --since …`, and `--as-of` on
`nanomem search`.

**`volatility()`** — how often each fact actually changes, measured from the log
alone: `n_revisions`, the intervals between them, `median_interval`, and how long
the current value has stood unconfirmed. No model, no query, no embedding call —
differenced timestamps off the resident `ts`/`group_id` columns. This is the
capability no vector store can represent, because none of them keeps a revision
history to difference. Records with no entity share the empty group key and are
excluded; pooling them would report one enormous fake fact.

**`staleness()`** returns the same rows plus a modelled `p_superseded` — **and
suppresses it by default**. That is the pre-registered consequence of a gate the
model did not clear, not caution. Held-out last-interval forecast:

| corpus | ECE (bar ≤ 0.15) | Brier | constant-rate baseline |
|---|---|---|---|
| uniform intervals | 0.0771 | 0.1916 | 0.1909 — **ties/wins** |
| exponential intervals | 0.0593 | 0.1704 | 0.1779 — model wins |

Calibration passes on both. The second clause fails: against a single
corpus-wide rate, the per-fact rate wins only on the corpus whose intervals were
generated to match its own memoryless assumption. Elsewhere it is a tie, and a
tie means the per-fact rate earned nothing. `assume_memoryless=True` turns it on
for a caller whose domain justifies it. `scratch/refound/staleness_calibration.json`

An earlier run of that gate used `obs.mean()` as the "constant baseline" —
estimated from the test outcomes, an in-sample oracle with access to the answers
(its ECE was exactly 0.0000, which is the tell). Both runs are in the results
file; the fair baseline is the one quoted.

**A personal-memory search is 47.4% faster, and every result is identical.**
Profiling at 70,000 rows found `_resolve_revisions` was **50.1% of the search**,
and ~90% of *that* was a single line in `_cosine_window`:

```python
order = cand[np.argsort(-cos[cand], kind="stable")[:int(self.window_max)]]
```

A full stable argsort over every candidate, to keep **four** of them. Replaced
with `_top_k_stable`: partition to locate the k-th largest value (O(N)), keep
everything at or above it — a superset of the top k including every boundary tie
— then stable-sort only that.

| at 70,000 rows | before | after | |
|---|---|---|---|
| `_resolve_revisions` p50 | 1.917 ms | **0.194 ms** | −89.9% |
| `search` p50 | 3.899 ms | **2.050 ms** | −47.4% |

Exact, not approximate: `_top_k_stable` is identical to the expression it
replaces element-for-element across **21,200 fuzz cases** — including
all-identical, two-value and three-value tie profiles and arrays of 5k–80k — plus
the 520-query recorded baseline, unchanged. `argpartition`'s own tie order is
unspecified and is never relied on; it is read for a *value*, never to choose
between equal ones. `scratch/refound/window_topk_results.json`

Two corrections to earlier notes in this project:

* This is **not** the "O(top_k) rewrite of `_resolve_revisions`" those notes
  described. No group formation changes and no candidate pool is narrowed, so
  that rewrite's accuracy risk does not apply.
* Those notes also claimed the fix would "re-enable the PCA screen for
  personal-memory workloads". **That is wrong.** The screen is gated by
  `_ranking_is_inert`, which asks whether the entity/temporal layer can fire —
  a pruned row cannot be brought back by a boost — so on a personal corpus the
  screen stays off however fast revision resolution becomes. Unrelated.

**The relevance floor was retuned, and the retune was refused** — but the sweep
produced a measured recommendation. `group_floor_sim` stays at 0.0. On records
whose entity the CALLER declares, 0.45 is worth **+19.0 points** of top-1 on
revision chains whose newest statement is phrased furthest from the question
(canonical +1.0, sibling probes +0.7). On the 3-persona chat set, where the
tagger infers the entity instead, the same 0.45 costs **−13.9 points**; the two
values that cost nothing there (0.65, 0.70) buy +2.0 and −1.0, i.e. nothing.

No single threshold serves both, because the floor is compensating for *tagger
precision* and a caller that declares its own schema has no imprecision to
compensate for. So it stays a knob, now documented in `_apply_group_floor` with
both numbers. This independently reproduces the engine's own recorded finding
that three earlier floor fixes each cost more than they gained — using an arm
its authors did not have. `scratch/refound/floor_retune_results.json`,
`floor_chatcheck_results.json`

An earlier note in `temporal_drift_results.json` dismissed this sweep as
pointless because a tagged group forms in 0 of 100 drifting chains. That was
true of *that* arm only: the floor defect reproduces with **perfect** tags, so
it is independent of tagger recall, and the schema-aware path is exactly where
it bites. The correction is recorded in both files.

**Two mechanisms were built, measured and refused.** Both were attempts to make
fact-grouping work without a vocabulary, which is what bounds every temporal
feature here:

* *Vocabulary-free grouping signals.* Five candidates scored on drifting
  phrasing. Against **sibling** attributes — the discrimination that matters,
  since home vs office address must never merge — direct cosine reaches AUC
  0.456, second-order profile 0.512, neighbour-set Jaccard 0.426 and value-shape
  matching 0.517. All chance. Only the query-relative signal discriminates
  (0.880), and that is what the relevance floor already computes. Siblings share
  value shapes by construction, so shape can never separate them.
  `scratch/refound/grouping_signal_results.json`
* *Write-time anaphora* (change marker + value shape + recency). Attached 41.7%
  of narrative restatements at 78.7% precision, 8.9% wrong — inside the wrong-rate
  bar, well under the 70% attach bar. 105 of 180 restatements carry no specific
  value shape at all, and the confusions are exactly the siblings
  (`primary_email→backup_email` 7, `home_address→office_address` 6): recency does
  not break sibling ties. `scratch/refound/write_time_anchor_results.json`

The conclusion both share is worth stating plainly: **a revision cannot be told
from a sibling attribute without either a vocabulary or the query.** That is a
property of the problem. It is why the tagger is lexical, and it means every
temporal capability here is bounded by tagger recall — 70/100 chains on canonical
phrasing, 0/100 on drifting.

**Cost**: +0.21% on p50 at 20,000 rows over 5 alternating paired runs, against a
pre-registered bar of +1.0% (`scratch/refound/temporal_cost_results.json`). The
first unrepeated pair read +1.88% and was not acted on in either direction; the
stdev across pairs is 1.11%, which is what that single sample was measuring.

**One gate did not survive contact.** G2 was written as "`search(as_of=t)` equals
`search()` on a vault holding only the rows at or before t, 0 mismatches", and
read as bitwise float equality that is not satisfiable by any correct
implementation: the two vaults hold different row counts, so the score matmul has
a different shape and float32 accumulation is not associative (max observed delta
1.49e-08, inside one ulp). That was discovered *after* measuring. The episode is
recorded in the results file rather than erased, and the replacement is strictly
stronger on the property in question — it proves the admitted row set exactly,
with no float comparison in it at all.

**Two defects found, neither introduced here, neither fixed here.**

* `search("where do I work")` on a real vault holding three `employer`
  statements returns the **middle** one and ranks the current employer **third**.
  The relevance floor drops the newest revision from the group before the
  revision lead runs, because it is phrased further from the question (cosine
  0.5317 against the group best's 0.6348, gap 0.1031, floor 0.06). Reproduced
  identically on 0.5.0. The knob that fixes it, `group_floor_sim`, ships off on
  measured evidence that three such fixes each cost more than they gained, so
  retuning it needs its own benchmark run — and the temporal benchmark reports
  100.0% on `current` questions while getting this one wrong, so the coverage
  gap should be closed first. `scratch/refound/finding_floor_drops_current_value.json`
* The five distribution copies of the package are at **0.3.0 / engine 3.0.3**,
  four releases behind, and live under untracked trees. They still answer "what
  was my original address?" with the current value, peak at ~818 MB on a 71k
  ingest, and gate writes at 0.60.
  `scratch/refound/finding_stale_distribution_copies.json`

---

## 0.5.0 — engine 3.2.0, container format 3 (unchanged), arena cache format 3

**The disk axis 0.4.0 gave up is back, and the biggest loss in the product
pipeline was never in the engine at all.** A cached 71,433-document vault goes
**406.15 MiB → 153.05 MiB**, below sqlite-vec's 258.0 and 4.4 MiB above a vault
with no sidecar, with the O(1) reopen kept and every score bitwise identical.
Separately, the write gate's decision point moves **0.60 → 0.05**, worth
**+13.3 pt of end-to-end top-1 [CI +8.7, +18.3]** on a 300-question held-out
split — 15.2 points of the chat pipeline's loss were being destroyed at write
time, before retrieval ran at all.

Two mechanisms are integrated. **Four are not, including one that beat the exact
cosine ceiling** — they are listed at the bottom with the criterion each missed,
because a measured negative is the more useful half of this release.

`format_version` is still 3 and no vault file changes: a vault written by 0.3.x
or 0.4.0 opens unchanged, and the vault this release writes is byte-identical to
one written with every new flag off.

### Changed — the `.arena` sidecar keeps OFFSETS, not a copy (`arena_cache_vectors`, `arena_cache_records`)

Measured by `scratch/refound/bench_sidecar_size.py` →
`sidecar_size_results.json`; independently re-measured on the final tree in
`scratch/refound/quality_summary.json`.

0.4.0's sidecar *was* the resident fp32 arena, so it duplicated the vault: 209.28
MiB of vectors and 43.81 MiB of record text beside a 148.7 MiB vault. It no
longer copies either. The block table the sidecar already stored is enough to
derive, for every block, where its fp16 vector run and its record section live
**inside the vault**, so those two sections cost zero new bytes:

| at 71,433 rows | sidecar | vault + sidecar | reopen | p50 (screen `pca`) |
|---|---|---|---|---|
| 0.4.0 `cache` + records `cache` | 257.48 MiB | 406.15 MiB | 0.000187 s | baseline |
| **0.5.0 default** (`offsets_ram` / `vault`) | **4.38 MiB** | **153.05 MiB** | 0.000232 s | x0.9976 paired |
| `arena_cache="off"` | 0.00 | 148.66 MiB | 0.174820 s | x0.9913 |

**Nothing an answer depends on changed.** 36 of 36 arm × corpus × screen
comparisons share one SHA-256 over the concatenated fp32 score vectors of 500
queries against every row, and 0 of 500 top-10 lists differ, at 71,433 / 10,000
/ 1,190 documents with the screen on and off.

**Two new flags**, both on `VaultEngine`:

* `arena_cache_vectors` — `"offsets_ram"` (default) maps the vault and upcasts
  fp16 → fp32 **once, on the first vector read**; `"offsets"` gathers from the
  mapping on every query; `"cache"` is 0.4.0's fp32 sidecar exactly.
* `arena_cache_records` — `"vault"` (default) reads each block's record section
  from the vault; `"cache"` copies them into the sidecar as 0.4.0 did.

**It is a three-cornered trade, not a free win.** `"cache"` keeps the p50 and a
43.5 MB `phys_footprint` and pays 406 MiB of disk. `"offsets"` has the best disk
*and* the lowest memory of any arm (227 MB peak, 57 MB phys) and pays **x1.73**
(screen on) / **x2.98** (screen off) on p50, because an exact cosine wants fp32
and the vault stores fp16, so a scan would convert 54.9M values per query.
`"offsets_ram"` keeps the disk and the p50 and pays the `phys_footprint`:
**270.5 MB against 43.5 MB**, because an upcast array is dirty anonymous memory
where a mapped sidecar is clean, evictable, file-backed pages. That cost was
**not** priced by the decision rule that chose the default; it is stated here and
`arena_cache_vectors="cache"` is the escape hatch.

**The cache format goes 2 → 3.** Every existing `.arena` is refused once and
rebuilt — one slow open, once, and the same again if the flags are changed.

### Added — `VaultShrankError`, and a crash mode that the default has and `"cache"` does not

Reading through a mapping means a vault truncated **out of band, under a live
engine** is a `SIGBUS` — an uncatchable process kill, exit 138 — where 0.4.0's
copying layout could not notice at all. Every read through the mapping now
checks one `os.fstat` against the last byte the block table can address and
raises `nanomem.errors.VaultShrankError` instead. Cost: **0.486 µs per call**,
about 0.4% of a query that materialises ten records.

It is a guard, not a guarantee: a truncation landing between the check and the
page touch still faults. nanomem's own torn-tail recovery does **not** trip it
(it only removes bytes past the last valid block, which the check does not
address), and that case is a test rather than an assertion.

### Changed — the write gate now keeps 57% more turns (`DEPLOYMENT_THRESHOLD_FULL` = 0.05)

Measured by `prime_4d_unified_engine_2026_09_13/write_policy.py` →
`scratch/refound/write_policy_results.json`, pre-registered and hash-verified
before any number, tuned on a 120-question dev split, scored **once** on a
disjoint 300-question test split. Reproduced end to end through the shipped
classifier in `scratch/refound/quality_summary.json`.

The trainer chose 0.60 by leave-one-persona-out **accuracy/F1**, which prices a
false positive and a false negative the same. Deployment does not: a refused
fact is never written and no retrieval quality can recover it, while a kept
noise turn costs ~1.9 kB and no measurable query time. At 0.60 the gate ran at
**100.0% precision / 64.6% recall** and made **135 of 420 questions (32.1%)
unanswerable before retrieval ran**.

| on the 300-question test split | shipped 0.60 | **0.05** | store everything |
|---|---|---|---|
| end-to-end top-1 | 35.7% | **49.0%** | 49.7% |
| delta vs 0.60 (paired bootstrap) | — | **+13.3 [+8.7, +18.3]** | +14.0 [+8.3, +19.7] |
| persona-clustered CI | — | **[+9.7, +16.7]** | [+9.7, +18.3] |
| gate recall / precision | 64.5 / 100.0 | **94.0 / 92.9** | 100.0 / 35.6 |
| documents per 10 personas | 271 | **425** | 1,180 |
| vault bytes | 0.492 MiB | **0.764 MiB** | 2.014 MiB |
| questions left unanswerable | 100 | **16** | 0 |

Storing everything buys 0.7 pt more for 2.8x the rows and 5.6x the added bytes,
so this is the efficient point rather than the extreme one: **0.0205 MiB per
point of accuracy against 0.1087**. The gate's own F1 is also better here
(0.935) than at 0.60 (0.784). Both curves are flat from 0.10 to 0.02 — **0.05 is
a region, not a tuned constant.**

Not changed, deliberately: the **surface** head still decides at 0.60 (the study
handed a real embedding to every turn, so it measured the full head and only the
full head), and the **rule layers** are untouched — with the learned head fully
off they still drop 52 of 1,180 turns, 14 of them answers, which is a second
loss worth ~3.4 pt that needs its own study.

`WriteClassifier(threshold=0.60)` restores 0.4.0's gate exactly, and the trained
value is still readable as `inspect()["model_info"]["threshold_trained_full"]`.

### Measured and NOT shipped

Every one of these was pre-registered before it was measured, and each is
reported against the criterion it was registered against rather than a criterion
chosen afterwards.

1. **BM25 + dense fusion — the first mechanism in this project to beat the exact
   cosine ceiling, and it is not in the build.** A numpy+stdlib inverted index
   (`hybrid_results.json`) lifts evidence recall@10 from **71.2% → 77.4%,
   +6.2 pt [+4.0, +8.6]**, still positive after Bonferroni over 7 fusion
   families, because BM25's top-200 holds 3.0 pt of gold the cosine never
   returns at any depth. It failed the bar it registered: recall**@4**, where
   the pre-committed arm scores +2.60 [-0.20, +5.20] and no arm survives
   multiplicity correction. It also failed the registered latency cost gate
   (**+1.0611 ms** against +1.0 ms) and the index is **50.39 MiB = 32.9%** of
   the vault this release now writes, against a registered 25% limit. And the
   pipeline this engine actually serves reads a **top-1 to top-3 window**
   (`chat.py`, `server.py`), not a top-10, so the gain sits outside the window.
   Not shipped, not behind a flag, and it should not be the default. What would
   change that: a caller that consumes a top-10, a re-measured single-query
   fusion path against the same +1.0 ms bar, and a cost bar re-registered
   against 153.05 MiB.
2. **Learned attribute selection** (`selection_results.json`). The ensemble
   clears its bar — +11.0 pt [+6.7, +15.7] overall, +10.7 [+3.8, +18.5] on the
   questions that never name their attribute — but leave-one-ATTRIBUTE-out
   collapses the two components that carry that gain to 3.7% and 1.9% selection
   accuracy, i.e. the gain is per-attribute supervision transferring through a
   shared question-template table, not language understanding. The
   attribute-agnostic arm that survives LOAO clears the overall bar (+5.3
   [+1.7, +9.3]) and does **nothing** on the head-dropped questions the study
   exists for (-1.6 [-6.2, +3.1]). It replaces `entities.query_intents`
   wholesale and was never measured against the temporal set that hook was tuned
   for. Not shipped; the measurement that would decide it is named in
   COMPETITIVE_POSITION.md.
3. **Write-confidence down-weighting at retrieval time** — dead. Its dev argmax
   is weight 0, i.e. the mechanism switched off; every strictly positive weight
   scores at or below store-everything on dev and on test.
4. **A two-tier confidence shard** — passes its bar (+8.0 [+3.7, +12.3]) and is
   Pareto-dominated on every axis: lower accuracy than either arm above, the
   largest footprint and the highest p50. The mechanism is measured, and it is a
   general warning: splitting one vault into two costs **-7.5 pt on dev even
   when both shards are always searched and merged by raw score**, because
   nanomem's entity boosts are per-vault statistics. A confidence shard is not a
   free index split.

### Tests

**443 passed** (408 at 0.4.0): 24 for the offset layout and its attacks, 6 for
the truncation guard, 5 pinning the write gate's decision point — nothing in the
suite touched the classifier before this release, so that constant was a single
unpinned float.

### Known gaps in this release

* The p50 gate that chose the sidecar default was first reported against a
  substituted criterion after failing as written. It was then re-measured once
  on a quiet machine under a rule fixed in advance — including the branch that
  would have reverted the default — and passes as written (baseline 1.2055 ms
  against a 1.2643 ms bar; only `"offsets"` fails, at 2.0857 ms). Both readings
  and the superseded verdict are in `sidecar_size_results.json`. The margin is
  thin in both directions, ~4-5%, so that gate separates `"offsets"` from
  everything else and resolves nothing finer.
* `phys_footprint` 43.5 → 270.5 MB was not in the decision rule that chose the
  default and is not being added to it after the fact.
* The write-gate study is one 14-persona synthetic fixture. The classifier's own
  held-out generalisation set is quarantined and was not opened, so what 0.05
  does to its published 90.5% held-out accuracy is **unmeasured**; whoever
  publishes should re-run `train_write_classifier.py --heldout` at the new
  threshold.
* No cold-page-cache number exists anywhere: `purge` needs root. Every reopen,
  p50 and footprint figure here is warm.
* Landmark tables (`m > 0`) are exercised by a unit test only; no benchmark
  corpus produces one. `residency="int8"` and `"float16_mmap"` still get no
  cache, and encrypted vaults still get no sidecar and no mapping.

---

## 0.4.0 — engine 3.1.0, container format 3 (unchanged)

**The one axis nanomem lost worst is now a win, and it was paid for on disk.**
Reopening a 71,433-document vault went **0.1634 s → 0.000193 s (846x)**, against
sqlite-vec's 0.0014 s — from **115x behind the winner to 7.3x ahead of it**. The
same vault now occupies **406.2 MiB instead of 148.7 MiB**, which loses an axis
nanomem used to lead. Both numbers are below; neither is a rounding error and
neither is optional reading.

Two mechanisms are integrated. Both are exactness-preserving, and that is
measured, not asserted. One is ON by default (`arena_cache`), one is OFF
(`screen`). `format_version` is still 3: a vault written by 0.3.x opens
unchanged, and a vault written by this release is **byte-identical** to one
written with both features off.

### Added — `arena_cache="map" | "copy" | "verify" | "off"` (default `"map"`)

The resident fp32 arena is kept in a `<vault>.arena` sidecar laid out so every
array is a page-aligned section `mmap` can hand to numpy with no copy and no
parse. An open becomes a stat, a 4 KiB header read, a bind and an `mmap`
instead of a replay of every block.

Measured by `scratch/refound/bench_reopen.py` → `reopen_results.json`,
regenerated against this build at machine load 2.1. Reopen is the median of 25
opens in one warm interpreter — the protocol `competitors_standard_results.json`
used, and the agreement is checked rather than assumed: a cache-less open here
reads 0.1634 / 0.0226 / 0.0027 s against that file's 0.1609 / 0.0218 / 0.0027.

| rows | scan open | mapped open | speedup |
|---|---|---|---|
| 1,190 | 0.002657 s | 0.000186 s | 14.3x |
| 10,000 | 0.022559 s | 0.000206 s | 109.5x |
| 71,433 | 0.163351 s | **0.000193 s** | **846.4x** |

It also changes what a serving process's resident bytes *are*. At 71,433 rows
`phys_footprint` is **9 MB against 288 MB** and peak `ru_maxrss` **258.2 MB
against 287.5 MB**, because the vectors become clean file-backed pages the
kernel may evict rather than dirty anonymous ones. Steady-state latency does not
move: the paired duel, arms alternated cycle by cycle on the same vault, reads
mapped/scanned at **0.9983**.

**Nothing an answer depends on changed.** Same build, cache on against cache
off, 500 queries scored against every row, at all three corpus sizes: the
SHA-256 of the concatenated fp32 score vectors is **identical** and **0 of 500**
top-10 lists change, for `map`, `copy` and `verify` alike. All four modes
produce one digest (`post_fix_check`).

**What it costs, all four things.**

1. **Disk, and this is a trade rather than an oversight.** 257.5 MiB beside a
   148.7 MiB vault at 71,433 rows = **406.2 MiB**, against sqlite-vec's 258.0
   (1.57x) and nanomem's own previous 148.7, which was the smallest of every arm
   measured. The sidecar is 1.73x the vault because the vault stores **fp16**
   and the sidecar **is** the fp32 arena — storing fp16 there would halve it and
   put the O(rows) upcast back into the open, which is the thing being deleted.
   `residency="float16"` gets an fp16 sidecar for the same reason.
   `arena_cache="off"` writes nothing.
2. **The first query**, which now pays the page faults the open skipped:
   0.0053 s → 0.0107 s at 71,433. Time to first answer is therefore
   **0.1687 s → 0.0109 s, 15.4x** — the honest secondary number, and it is
   reported next to the 846x rather than instead of it.
3. **A fresh process on a SMALL vault is slower**: 0.00764 s against 0.00321 s
   at 1,190 rows. It wins from ~10,000 rows up (0.00475 s against 0.02299 s).
   `ARENA_CACHE_MIN_ROWS` stays at 256 and was deliberately **not** re-tuned:
   the crossover is bracketed only by those two measured sizes, and picking a
   floor inside that bracket after seeing the result would be choosing a
   criterion to pass it. Small vault, short-lived processes: pass
   `arena_cache="off"`.
4. **The open that writes it** costs 0.3051 s against a 0.1659 s plain open,
   of which 0.1315 s is the write — 1.84x, paid once. The first *write* after a
   cached open pays for the lazy tables the open skipped: 0.0474 s against
   0.0124 s.

**On integrity, which is where the first version of this work was wrong.**
Two claims made earlier in development are **withdrawn**, in the source, the
tests and the results file:

* *"`verify` is safe."* It is not. The cache's `content_sha256` is an **unkeyed**
  digest stored **inside the header of the file it authenticates**, behind a
  CRC32. A ~10-line forgery updates both, and `verify` then serves the planted
  row at **cosine 1.0** (`cache_integrity.cache_tamper.forged_digest`, all three
  corpora; `tests/test_arena_cache.py::test_verify_mode_is_hijacked_by_a_forged_content_digest`).
* *"The sidecar is a strictly weaker, unauthenticated path to the same answers."*
  False in the only configuration a sidecar can exist in. A cache is refused
  outright for an encrypted vault, so every vault that has one is **plaintext**,
  and a plaintext block trailer is an unkeyed SHA-256 that `crypto.py`'s own
  `THREAT_MODEL` says "can be recomputed by anyone". With **no sidecar on disk
  and `arena_cache="off"`** — the full scan that re-reads every block — a forged
  row is served at **cosine 0.999997**
  (`tests/test_arena_cache.py::test_a_plaintext_vault_is_forgeable_with_no_cache_in_sight`).

So: the sidecar does not lower the vault's threat model, it inherits it, and in
plaintext that model is **corruption, not adversaries**. `verify` checks the
cache's digest and re-reads every vault block; it costs **0.1478 s against
0.1656 s** for simply rescanning — *cheaper* than a scan, because it skips the
decode, the record parse and the per-row interning — so what it buys is the
mapped memory profile at a scan's price, with a scan's corruption checking. The
word "authentication" no longer appears anywhere in this feature except where a
passphrase exists, and a passphrase disqualifies a vault from having a cache at
all, which is exactly why no keyed fix is available here.

**One integrity case was closed rather than described.** A vault edited in place
at the same length with its trailer left stale was refused by `off` and `verify`
(`IntegrityError`) and served by `map` with no error, no warning and no
`integrity_errors` entry — a silent downgrade of the **default**. The cache
header now records the vault's `(size, mtime)` as of the scan it was built from,
captured the instant the scan returns rather than at write time, so a concurrent
in-place rewrite cannot bless a cache built from the old bytes. All three modes
now raise. Cost: **+4.08 us** at 71,433 rows and +6.42 us at 10,000, on a
~0.000175 s open, measured paired against the same package with the check
deleted. It is one `os.stat` and it is a corruption check: **`os.utime` defeats
it**, which is itself a passing test, and so does an edit to the prefix of a
vault that is afterwards appended to.

`ARENA_CACHE_VERSION` 1 → 2 for the two new header fields. The header is a fixed
4 KiB page, so the file is not one byte larger; v1 caches are dropped and
rebuilt silently.

**Added with it:** `VaultEngine.arena_cache_path()`,
`VaultEngine.arena_cache_info()` — including `vault_changed_since_cache` and
`vault_blocks_checked` (`"all"` / `"appended tail only"` / `"none"`, so a
default that stops re-reading blocks is *answerable* instead of silent) — and
`arena_cache_refresh_rows`.

### Added — `screen="pca"` (default OFF), an exactness-preserving latency option

A **latency** flag with no recall knob and no accuracy trade-off: its recall
delta is identically 0 by construction. It bounds every document's cosine from
above in a 256-dimensional subspace, skips the rows whose bound provably puts
them below the `top_k`-th score already in hand, and scores the survivors with
the same fp32 kernel the full scan uses. `nanomem/screen.py` derives the bound;
it holds for *any* basis, which is why an append cannot invalidate it.

71,433 HotpotQA paragraphs, 200 questions × 5 paired interleaved cycles, one
engine with the flag toggled per query (`scratch/refound/bench_screen.py` →
`pca_screen_results.json`, `phaseC_latency_full`, re-run against this build at
load 1.28):

| arm | p50 | p95 | mean |
|---|---|---|---|
| `screen="off"` | 1.9495 ms | 2.0898 ms | 1.9328 ms |
| `screen="pca"` | 1.1350 ms | 1.7318 ms | 1.1829 ms |
| **speedup** | **1.718x** | 1.207x | 1.634x |

Per-cycle p50 ratios 1.709–1.731; bootstrap CI [1.695, 1.736]. An earlier run on
a machine at load 4.5–5.4 read 1.705x [1.689, 1.725] — the same answer, which is
the point of interleaving. **Read that file's `phase*` blocks, not its
`verdict`**: `bench_screen.py` merges a new run over the old file with
`prior.update(res)`, so `verdict.C2_speed` still holds the previous run's
numbers.

**Exactness**, proven and then measured anyway: 0 bound violations in
**71,433,000** document checks, and over 3,000 searches at k = 1, 4 and 10, zero
with a different id list, zero with a non-bitwise-identical float32 score list,
zero real misses. A basis fitted from 6,000 rows with 24,000 rows appended
afterwards and never refitted: 0 and 0 again over 2,400,000 checks.

**Cost**: 1032 bytes per document resident (71.6 MiB at 71,433, +34.2% over the
209.3 MiB arena) and a one-off 0.191 s build. The basis is **recomputed**, not
persisted — it is admissible however stale it is, so persisting it would buy
latency only.

**Stands down automatically**, changing no result and raising nothing, below
`screen_min_rows=20_000`, under `residency="int8"`, with a `metadata_filter`, on
an entity-tagged vault, for an explicit `temporal_direction="historical"`, when
survivors would exceed `screen_max_frac`, and if the basis cannot be built. The
floor comes from a measured sweep: forced on, the screen is 0.478x at 1,000
documents, 0.919x at 5,000, 0.977x at 10,000, 1.370x at 20,000, 1.468x at
40,000. At the shipped default the p50 ratio at 1,000/5,000/10,000 is
1.0005/1.0002/0.9949.

**What it does not claim**: any recall improvement. The pre-registered rule in
`exotic_routing_results.json` was a confidence interval on a recall delta, which
is unreachable for this mechanism **by construction** rather than merely unmet.
A new criterion — exactness, ≥1.25x p50 at 71,433, no regression below 10,000 —
was written before the mechanism went into the engine, and that substitution is
stated here rather than buried.

### Changed — top-10 output can REORDER against 0.3.2, always between tied rows

`_select_top_k` now breaks ties on the **row id** instead of inheriting
`np.argpartition`'s unspecified order among equal elements. This was required to
make the screen's "identical top-k" clause true as written rather than restated
as "identical up to ties": `argpartition` resolves a tie differently depending
on how long the array it is handed is, and the screen hands it a shorter one.

Consequences, both real:

* **8 of 500** top-10 lists at 71,433 documents and **1 of 500** at 10,000 come
  back in a different order than 0.3.2 gave. Every changed position was scored
  and the maximum |score gap| is **exactly 0.0** — bit-identical duplicate
  paragraphs, where both orders were always correct
  (`reopen_results.json`, `tie_forensics`, which fails loudly if one is not).
  The count is insertion-order dependent: an independent re-derivation with a
  different insertion order measured 7.
* It costs **16 us per search on the default exact path** (0.8% of its p50),
  paid whether or not `screen="pca"` is ever turned on. What it buys is a result
  that is a function of the *set* of rows scored and nothing else, which is also
  reproducibility across numpy versions.

`search_batch` is untouched and its tie order remains numpy-dependent.

### Changed — `stats()["arena_bytes"]` reads 0 on a cached open

`arena_bytes` counts **anonymous** memory only, deliberately — counting clean
evictable pages as RAM is the over-reporting 0.3.2 removed. On a cached open the
vectors are a mapping, so it reads 0 and the 219,442,176 bytes appear under
`arena_cache_mapped_bytes` / `arena_mapped_bytes`, with `arena_used_bytes`
unchanged. `arena_from_cache` and `arena_vectors_mapped` say which shape you
have. Anything trending `arena_bytes` will see a step change.

### Also in this entry

`tests/test_round5_temporal.py` no longer reads the chat fixtures. It globbed
`scratch/refound` for `*chat_benchmark*.json` and `*personas*.json` and read
every match, so **every `pytest` run opened the quarantined
`clean_chat_benchmark_persona4.json` and `clean_chat_benchmark_heldout.json`**.
`test_entities.py` had already been moved onto the committed digest table
`tests/data/fixture_vocab.txt`; this sibling was missed. It now uses the same
table and names no path under `scratch/`. A new suite-wide guard,
`tests/test_screen.py::test_no_test_module_names_a_quarantined_fixture`, parses
every test module so the next one cannot be missed the same way — the old audit
only inspected itself. Found by wrapping `builtins.open` for a whole suite run
and reading the paths back; an access timestamp would not have settled it, since
reads do not reliably bump `atime` on this filesystem. Verified again on this
build: a full instrumented suite run opens exactly 6 paths under
`scratch/refound`, none of them quarantined.

`tests/test_arena_residency.py::test_the_allocator_not_a_narrower_dtype_is_what_took_the_ram_off_the_default`
was amended, because integrating the arena cache made its assertion false
without making its claim false. It read `arena_bytes` on a reopened vault, which
is now 0 (above); it now reads whichever key carries the vectors and pins
**both** open paths. Its `legacy_303_fp32_doubling` proxy is also asserted only
when both arms took the same open path: `bench_memory.py` does not pin
`arena_cache=`, so whichever arm runs first for a given dtype scans and writes
the sidecar and the next arm with that dtype maps it, which makes its
`reopen_only` RSS column no longer a clean residency comparison. Repairing that
harness is not work this release did.

**Not shipped, and recorded as a negative:** four mechanisms — conversation
context, per-user attribute priors, multi-intent retrieval and session recency —
pre-registered against a +5.0 pt bar on the personal-memory chat task and scored
on a held-out persona split. None cleared it; the best reached +1.0 pt
[-0.3, +2.7] (not significant) and conversation context was significantly
**negative** at -3.3 pt [-6.0, -0.7]. See COMPETITIVE_POSITION.md, Axis 6, and
`scratch/refound/context_lever_results.json`.

---

## 0.3.2 — engine 3.0.5, container format 3 (unchanged)

One change, in one place: **the resident arena stopped paying for its own
growth.** No stored byte, no answer and no public default moves. `format_version`
is still 3 and files written by 0.3.0/0.3.1 open unchanged.

### The measurement

`ru_maxrss` high-water mark over the whole loader shape — build 71,433
documents, close, reopen, answer 500 queries — measured by
`scratch/refound/bench_memory.py` (arm `fp32_reserved`, the shipped default),
today's harness on today's machine against a checkout of the committed 0.3.1
build whose `arena.py` is byte-identical to `git show 6aa6923:`:

| corpus | 0.3.1 (engine 3.0.4) | 0.3.2 (engine 3.0.5) | ratio |
|---|---|---|---|
| 71,433 documents | 819.1, 818.6 MB | 286.4, 286.9 MB | **0.35x** |
| 10,000 documents | 102.0, 103.0 MB | 41.2, 41.1 MB | **0.40x** |

Two runs per cell, both printed rather than averaged. The same file's single
driver pass reads 305.1 MB at 71,433, so the honest range for the new arm at
that size across today's three readings is **286.4–305.1 MB**; a separate
harness (`scratch/refound/ingest_ram_results.json`, four runs plus two
post-report checks) puts it at **287.9–310.0 MB** with a median of 298.4. Quote a
range, not a point.

**The server shape did not move, and was not supposed to.** Opening a vault
another process wrote and serving from it: 286.0 MB → 285.2 MB at 71,433
(`memory_results.json`, `reopen_only`). 0.3.1 already fixed that shape; this
release fixes the *loader* shape, which was the one still losing.

### Nothing an answer depends on changed

* The SHA-256 of the concatenated fp32 score vectors of all 500 queries against
  all rows is **identical** between the two builds at both corpus sizes
  (`ingest_ram_results.json`, `exactness`). That check does not depend on the
  machine.
* **0 of 500** top-10 lists changed, in all 20 (arm, corpus, phase) cells.
* Evidence recall@4 is **70.4%** at 10,000 and **60.6%** at 71,433 — the same
  numbers, still equal to exhaustive numpy and FAISS `IndexFlatIP`.
* p50 **1.7495 ms** at 71,433 against a 0.9605 ms bare-numpy floor measured in
  the same process (`memory_results.json`, `arms.n71433.fp32_reserved.timing`);
  published 0.3.1 figure was 1.774 ms. A 7-cycle paired duel reads the change as
  0.9966x (`ingest_ram_results.json`, `latency_duel`) — no measurable cost.
* The temporal-supersession benchmark re-run against this build reproduces
  **all 19 arms to the digit**: 90.5% top-1 at shipped defaults, 47.9% ranking-off
  floor, 86.2% for the strongest competitor arm
  (`scratch/refound/temporal_bench_results.json`).

### How it works

`nanomem/arena.py` holds the fp32 vectors in a lazily committed anonymous
mapping (`mmap.mmap(-1, ...)`, `MAP_PRIVATE|MAP_ANON`) reserved larger than the
rows in it; growth is a **new view over the same pages** — no allocation, no
copy, nothing discarded. The array itself is still cut to the exact row count.
When a reservation is outgrown a larger one is taken, the live rows are copied
once and the old mapping is `munmap`ped — which matters because a freed numpy
buffer measurably is *not* returned to the OS on this platform, and that is why
doubling's peak tracked live-plus-everything-ever-discarded. Reaching 71,433
rows costs **2 copies**, of 2.3 MB and 37 MB.

Allocator in isolation, 71,433 x 768 fp32 grown 50 rows at a time, one
subprocess per policy (`memory_results.json`, `growth_policy`, and
`ingest_ram_results.json`):

| policy | final capacity | growths | bytes discarded | peak RSS delta |
|---|---|---|---|---|
| capacity doubling (0.3.1) | 131,072 rows | 12 | 383.8 MB | 592.2 MB |
| exact fit, reallocating | 71,433 rows | 1,429 | 149,458.9 MB | 11,729.9 MB |
| **reserved view (0.3.2)** | 71,433 rows | **2 copies** | **0 MB** | **208.9 MB** |
| explicit `reserve` up front | 71,433 rows | 1 | 0 MB | 209.3 MB |

Doubling is **kept** for the small per-row column arrays (2.7 MiB live at 71k).

**Fallbacks, all tested.** No anonymous mapping available → a plain zero-filled
array with the same view discipline. No headroom available (strict overcommit,
an rlimit) → retry at exactly the rows needed. An impossible hint (10^13 rows)
is dropped, not raised.

### Added

* **`VaultEngine.reserve_additional_rows(n)`** — batch-relative pre-sizing, for
  callers that know a batch length but not the eventual total. `reserve_rows(n)`
  remains exact and total-relative.
* **`nanomem ingest --expect-docs N`**.
* **Automatic hinting on every bulk path.** `Vault.add_batch` hints its own
  `len(records)`; `ingest_file` flows through it; `ingest_directory` hints once
  up front from total bytes / `BYTES_PER_CHUNK_ESTIMATE` (2048, chosen from a
  measured 1,640 / 1,755 / 2,339 / 9,703 bytes per chunk on four real trees and
  leaning low on purpose); `merge`, `export` and `split` reach it through
  `add_batch`; `_rebuild` → `replace_all` gets the container's exact count.
  `Vault.add()`'s auto-split path (a handful of chunks per call) is **not**
  hinted.
* **Three `stats()` keys**: `arena_reservation_bytes`,
  `arena_reservation_is_mapped`, `arena_growth_copies`.

The hint is now worth **3%** (298.4 → 289.5 MB at 71,433), not the 2.85x it was
worth under doubling. It is wired anyway because it also guarantees zero growth
copies and a reservation no larger than the rows — on a reopen the arena comes
back with `arena_bytes == arena_used_bytes == arena_reservation_bytes` =
219,442,176 and `arena_growth_copies` = 0.

### Changed

* **`stats()["arena_bytes"]` no longer overstates.** Under doubling it reported
  the *capacity*, up to **1.83x** the rows that existed. It now reports the rows.
  Anything trending that key will see a step change that is a reporting fix.
* **`Arena.vec` is a read-only property.** Assigning to it was never supported
  and now raises.
* `arena_reservation_bytes` is **address space, not RAM** — an untouched page of
  an anonymous mapping is not resident. Unhinted, a 209 MB arena sits behind
  552 MB of reserved VM (721 MB via `add_batch`); hinted, exactly the rows.
  Never add it to `arena_bytes`.

### Considered and declined

**Fixed-size chunk list.** Same RAM, no reservation needed — and it turns one
BLAS call into `ceil(n/chunk)`. Paired in cycles at 71,433 rows: 0.9634 ms
contiguous against 1.0207 / 1.0847 / 1.1915 / 1.5541 ms at 32,768 / 16,384 /
8,192 / 4,096-row chunks, i.e. **1.06x to 1.61x** on latency, the axis nanomem
is already second on. Scores bitwise identical at every chunk size, so it is a
pure latency-for-simplicity trade. Declined: the reserved view buys the same RAM
at 0.997x the p50. (`ingest_ram_results.json`, `chunked_alternative`.)

### Tests

313 → **322** (`python3 -m pytest -q`, 0 failed).
`test_growth_is_still_capacity_doubling` is deleted — it encoded the policy that
was replaced — and five tests encode the new one (no copies, exact fit, repeated
hints, refused reservation, impossible hint) and five cover the bulk wiring. The
directory-estimate test asserts a *bound* (0.2 ≤ hint/chunks ≤ 5.0), not a
fitted constant.

One further test was **rewritten, not added**:
`test_reserve_is_what_actually_took_the_ram_off_the_default` compared the
pre-sized arm against the harness's `legacy_303_fp32_doubling` arm and required
it to be under 55% of it. Re-running `bench_memory.py` against this build made
that assertion fail — correctly, because the legacy arm stopped being a baseline
(see "Known limits" below). It is now
`test_the_allocator_not_a_narrower_dtype_is_what_took_the_ram_off_the_default`
and asserts the claim directly: on a reopened 71,433-document vault
`arena_bytes == arena_used_bytes == arena_reservation_bytes` = rows x 768 x 4
with `arena_growth_copies` = 0, resident cost below 1.45x the vector bytes, the
arena still `float32`, and zero changed answers. Its last assertion pins the
dead proxy — `legacy ≈ presized` within 5% — so the arm cannot quietly be
re-read as a 3.0.4 baseline, and so that the test fails loudly if anyone repairs
it into a real one.

### Known limits in 0.3.2

* **Peak RSS is better, not won.** 298.4 MB at 71,433 against FAISS's 230.2 MB
  is still **1.296x** — see `COMPETITIVE_POSITION.md`, "Where nanomem loses".
  The remaining ~290 MB is no longer the allocator: the loader now peaks at the
  *server* floor, i.e. what a process that only reads the file pays.
* **The reserved arm is bimodal at 71,433 rows.** Readings cluster near 288 and
  near 305–310 MB, a 7% spread. The committed-plus-hint arm shows the same jump,
  so it is an ingest transient rather than the reservation. Not chased.
* **`scratch/refound/bench_memory.py`'s `legacy_303_fp32_doubling` arm no longer
  reconstructs what it claims to.** It disables `Arena.reserve`, which was the
  whole difference under 3.0.4, but the reserved view sizes exactly regardless;
  its own `_next_capacity` assertion still passes because the *column* arrays
  still double, so the harness does not self-detect this. Measured: that arm's
  server-shape figure went 643.6 MB → 285.7 MB between the two builds, which is
  the proof. **Do not read that arm as a 0.3.1 baseline.** This changelog's
  before-column comes from a real checkout instead.

---

## 0.3.1 — engine 3.0.4, container format 3 (unchanged)

Recorded late; this entry was missing when 0.3.2 was written. Three measured
default-behaviour changes, each named in `nanomem/engine.py`'s `ENGINE_VERSION`
comment and each citing its own results file.

* **The temporal direction is read off the question's own wording.** A default
  `search()` returns a different record for a historically-worded question than
  0.3.0 did, with no argument change. Temporal-supersession top-1 at shipped
  defaults **44.2% → 90.5%**; historical questions 9.0% → 75.0%; previous-value
  questions 1.8% → 98.2%; sibling attributes 57.5% → 90.0%
  (`scratch/refound/temporal_bench_results.json`, before-column via
  `final_scorecard.json`). Passing the two-way `temporal_direction` argument
  explicitly now *costs* 13.2 points (90.5% → 77.3%) because it overrides that
  reading — see `COMPETITIVE_POSITION.md`.
* **The arena is pre-sized on reopen** from the container's header row count.
  Server shape at 71,433 documents: **643.6 MB → 286.0 MB**, 0 of 500 top-10
  lists changed, no latency cost (`scratch/refound/memory_results.json`,
  `reopen_only`). This did **not** fix the loader shape; 0.3.2 does.
* **The router gate was re-run and the default confirmed OFF**
  (`scratch/refound/router_gate_results.json`).

---

## 0.3.0 — engine 3.0.3, container format 3

A rebuild of the storage, retrieval and ranking layers, and a rewrite of the
documentation to match what is measured. **This release is not source-compatible
with 0.1.x for anything that reads `score`.**

### Breaking

* **`score` is a cosine.** In 0.1.x it was a squashed non-linear value. A hit now
  carries `score` (cosine plus explicit, documented boosts, bounded by
  `stats()['max_boost']` = 0.70) and `cosine` (the plain cosine). Every shipped
  threshold was retuned: **0.25 → 0.42, 0.32 → 0.53, 0.35 → 0.58**
  (`cli.py`, `chat.py`, `vault.py` forget default, `proxy.py`, `server.py`).
  `nanomem.engine.legacy_score_to_cosine(old_threshold)` converts any other one.
  `min_score` is applied to the pre-boost cosine, so a threshold means the same
  thing whether or not the entity/temporal layer fires.
* **`Vault.add()` and `VaultEngine.add_fact()` return the document id (`str`).**
  They returned `None` before. `POST /v1/memory/add` now returns the id the
  record is really stored under; in 0.1.x the response invented an id nothing
  could look up. The CLI's `add` prints the id and its own elapsed time.
* **Plaintext is the default.** A vault is a plain file unless you supply a
  passphrase. `stats()['encrypted_at_rest']` reports which one you have. In
  0.1.x `stats()` hard-coded `encrypted_at_rest: true` regardless.
* **`stats()` values are measured.** `active_heap_ram_kb` was the constant
  `160.0`; it is now a real sum of allocated buffers, with an
  `active_heap_ram_method` string stating what it excludes and a
  `process_rss_kb` beside it. `cipher` reports the real construction. New keys:
  `engine_version`, `format_version`, `vector_dtype`, `router`, `n_exhaustive`,
  `resident_arena_mb`, `max_boost`, `integrity_errors`, `truncated_tail_bytes`.
* **`search(multihop=True)` uses a text bridge.** `alpha`, `num_hops` and
  `beam_width` are accepted for compatibility and no longer change the result;
  alpha-steering was removed after measuring significantly worse than the query
  alone.
* **`prune()` returns bytes freed**, not a record count. (`delete()` returns a
  count.)
* **`password=""` is an error**, not a silently plaintext vault. A `bytes`
  password is an error rather than being coerced with `str()`.

### Migration from 0.1.x (`format_version` 2)

Open the file. Migration runs once and the original survives beside it as
`<path>.v2.bak`, byte-identical to the source, readable by the archived v2
reader. `stats()['format_version']` is `3` afterwards, and the second open is an
ordinary v3 open.

Verified on two golden fixtures in `scratch/refound/golden/`: a chat vault
answers **12 of 12** expected top-1 queries after migration, against 10 of 12 for
the v2 engine on the same fixture (`scratch/refound/ranking_dev_r4_shipped.json`,
`golden_chat_v2`); and a book vault migrates **845 of 845** documents with **0 of
20** top-4 differences from exhaustive fp32 cosine computed over
`iter_records()` — that second arm was run by hand and is not in a results JSON.

Junk v2 entity tags are re-derived with the generic tagger; the original value
is preserved in `metadata["entity_v2"]`. `nanomem user delete` ignores and
removes `.v2.bak` and `.tmp-*` siblings. `Vault(..., migrate=False)` opens a v2
file read-only through the legacy reader; supplying a passphrase to that path is
now an error instead of being silently dropped.

### Retrieval

Real HotpotQA paragraphs in random insertion order, 500 held-out questions,
`top_k=4`, ingest → close → re-open → search. Before:
`scratch/refound/scale_results_current_engine.json`. After:
`scratch/refound/scale_results_v3r4.json`.

| Corpus | recall@4 | p50 | index |
| :--- | :--- | :--- | :--- |
| 10,000 | 24.0 % → **70.4 %** | 44.01 ms → **0.345 ms** | 71.8 MB → **22.0 MB** |
| 71,433 | 5.8 % → **60.6 %** | 461.22 ms → **1.762 ms** | 511.5 MB → **155.9 MB** |

70.4 % and 60.6 % are exactly what an exhaustive fp32 numpy scan and FAISS
`IndexFlatIP` score on the same data. At 1,190 documents the engine matches
exhaustive fp32 cosine in *ordering* as well: 0/120 top-4 order differences and
0/120 rank-1 differences with the real question strings
(`scratch/refound/exactness_v3r2.json`, `headtohead_v3.json`). At 10,000 and
71,433 documents recall is still identical, but 2 of 500 and 5 of 500 questions
differ in top-4 *order* — every case a tie or near-tie, largest cosine gap
2.2e-05, caused by the fp16 vectors on disk
(`scratch/refound/verify_round3_v3r3.json`).

The 0.1.x collapse was the block-page router on randomly ordered blocks: in
random insertion order it recalled 3.3–21.7 % against 68.3 % exhaustive
(`scratch/refound/sweep_routing_1190.txt`). 3.0 scans exhaustively below
`n_exhaustive` (50,000) instead.

### Ranking

* Revision resolution: the current revision is ranked first in **14 of 16**
  generic probes against **3 of 16** for plain cosine, while **40 of 40**
  adjacent-but-different attributes are left exactly where plain cosine puts
  them. Historical lookups 16/16. `scratch/refound/ranking_dev_r4_shipped.json`.
* Third-party statements are namespaced separately, so "his number is …" can no
  longer be stored as revision 2 of your own number and returned as the answer to
  your own question (`scratch/refound/third_party_v3r4.json`).
* Third-person questions now resolve into that namespace; in 0.1.x and 3.0.2 they
  received a boost of exactly +0.000 and fell back to raw cosine.
* Score contract: 0 violations over a 5,856-hit fuzz across 120 randomly shaped
  vaults; `cosine` matched the true stored-vector cosine on 5,856 of 5,856.

Chat benchmarks, gold-store top-1 (the expected memory ranked first, with the
correct records already stored):

| Set | before | after | top-3 after |
| :--- | ---: | ---: | ---: |
| 3-persona selection (n=36) | 50.0 % | **91.7 %** | 100.0 % |
| 2-persona held out (n=24) | 33.3 % | **75.0 %** | 95.8 % |
| 3-persona dev set (n=24) | — | 95.8 % | 100.0 % |

`scratch/refound/clean_chat_results_current_engine.json`,
`clean_chat_results_heldout_baseline.json`,
`clean_chat_results_v3r4_engine*.json`.

**The release target was ≥ 80 % on both persona sets; the held-out set missed it
at 75.0 % (18 of 24).** Five of the six failures are ordering errors inside a
group that was retrieved (top-3 is 23 of 24); the sixth misses the top 3. The held-out set was also inspected during development this
round, so 75.0 % is an upper bound rather than a clean out-of-sample estimate.

### Write gate (classifier)

Replaced. The 0.1.x gate had four rule layers whose phrases were copied verbatim
from a benchmark fixture, plus a prototype asset built from paraphrases of the
same benchmark; both are gone, and the asset was deleted. The new gate is a numpy
logistic head over the embedding plus generic surface features.

Out of sample, on two unseen personas, 231 turns
(`scratch/refound/write_classifier_v2_results.json`):

| | accuracy | F1 |
| :--- | ---: | ---: |
| 0.1.x gate | 76.2 % | 69.6 |
| **3.0 gate** | **90.5 %** | **87.2** |
| 3.0 surface-only fallback (no embedder) | 87.0 % | 82.8 |

The decision itself costs 0.06 ms given an embedding; 13.2 ms p50 end to end
through `Vault`, dominated by the embedding call.

### Storage, durability, concurrency

The storage and concurrency figures in this subsection, and the tamper battery
under *Password mode*, came from one-off scripts run during implementation and
reproduced during verification. They are not written into a results JSON in
`scratch/refound/`; the performance and recall tables elsewhere in this file all
are.

* fp16 vectors on disk, fp32 arena in RAM. Worst per-row cosine error
  0.99999988; no ordering changes.
* 2,209 bytes per document at 1,190 docs — 0.609× the raw text plus fp32 vectors
  it replaces. Holds at scale: 2,307 B/doc at 10k, 2,289 at 71,433.
* Rebuilds (`update`, `delete`, `prune`, `unmerge`, `export --purge`) go through
  one atomic `engine.replace_all()`: temp file plus `os.replace`.
* `SIGKILL` mid-ingest at 809,000 records: reopens in 887 ms, all records
  contiguous and in insertion order, `integrity_errors` empty,
  `truncated_tail_bytes` 0, appendable afterwards.
* A single-byte flip at four different structural offsets raises four different
  specific `NanomemError` subclasses; no silent wrong data.
* 4 processes × 150 appends, 8 × 80, and 4 × 60 racing to *create* a vault that
  does not exist: every record present, unique ids, no torn tail.
* `VaultEngine.__exit__` flushes; a `with` block no longer discards pending
  records.
* Non-finite embeddings and timestamps are refused on the write path instead of
  being stored unreachable.

### Password mode

Optional, off by default. scrypt (n=2¹⁶, r=8, p=1) → three HMAC-SHA256 sub-keys →
SHAKE256 keystream XOR → HMAC-SHA256 encrypt-then-MAC bound to the vault uuid,
verified with `compare_digest` before decryption. Header key-check rejects a wrong
passphrase before any block is read.

Measured (`scratch/refound/crypto_overhead_v3r3.json`): 95.67 ms per scrypt
derivation (≈ 10.5 offline guesses/s/core); open +99.67 ms at 1,190 docs,
+104.88 at 5,000, +165.36 at 40,000; per-search −0.0003 / +0.002 / −0.0043 ms.
Encrypted and plaintext files are byte-for-byte the same size.

18 tamper mutations — ciphertext flips, nonce flips with repaired CRC, reserved
header bytes, block swaps, replays, appended duplicates, deleted blocks, blocks
spliced from another vault and from an earlier generation of the same vault, KDF
downgrade, flag stripping — were all rejected, each with a specific
`NanomemError`.

`nanomem.THREAT_MODEL` now states the three things it does not do: truncation and
rollback are undetectable and `on_torn_tail="raise"` covers neither; the size
leak is exact rather than approximate; and a block tag binds the vault uuid, not
the file path.

### HTTP services

* `POST /v1/vault/init` was unauthenticated, took an arbitrary filesystem path
  and opened it with a bare `Vault(name)` — a remote caller could downgrade a
  password-protected proxy to a plaintext vault anywhere on disk. Now: names are
  confined to the active vault's directory (absolute paths and `../` → `400`),
  the operator's passphrase is applied to every `Vault()` the handler opens, and
  changing the *active* vault requires `--allow-vault-switch` (otherwise `403`).
  `server.py`'s `/load` is confined the same way.
* The proxy binds `127.0.0.1` by default and warns otherwise. Neither service
  authenticates; that is stated in the docs rather than implied away.
* `/v1/memory/stats` answered `500` on a freshly created empty vault, because
  `Vault.__len__` made `if self.vault:` false. Fixed; a new vault can now learn.
* `server.py`'s `/health` service string is "NanoMem Continuous Memory Engine".
  It was "NanoMem 4D Latent Continuous Memory Engine"; there is no 4D component
  in this package.

### Multi-hop

* The shipped bridge is a text hop: re-embed the question with the hop-1 winner's
  text, search again, merge under the same `top_k`. 68.3 % → 79.2 % evidence
  recall@4 at `top_k=4` on 1,190 documents
  (`scratch/refound/multihop_texthop_v3r2_1190.json`). Single-pass `top_k=8` is
  90.0 %, so widening `top_k` remains the larger lever.
* Alpha-steering removed: 95 % CI [−0.050, −0.033] on MRR against the query
  alone, i.e. significantly worse.
* No trained latent bridge shipped. Linear residual, residual MLP and an RK4
  neural ODE (3 seeds each) all had CIs at or below zero on the primary held-out
  set of 1,600 questions over 71,433 documents
  (`scratch/refound/experiment_4d_bridge_results.json`).

### Routing

`router="auto"` (opt-in) uses global spherical k-means cells and triggers one
`compact(recluster=True)`, recorded in the file header so later opens do not
rewrite again (12 of 12 concurrent first opens succeed;
`scratch/refound/router_persist_v3r4.json`). On the reclustered 71,433-document
corpus it reaches 78.60 % recall@4 against exhaustive 78.65 % (−0.05 pt, CI
[−0.20, +0.10]) scanning 29 % of the corpus — the recall gate passes. **It ships
off**, because in the engine it is slower at equal recall: p50 3.604 ms versus
1.919 ms, plus 7.51 s on every open (`scratch/refound/router_gate_v3r3.json`).

### Tests

A pytest suite ships at `nanomem_standalone/tests/`: 237 tests, `python3 -m
pytest -q`, no network. 0.1.x had no tests at all.

### Documentation claims withdrawn

The following appeared in 0.1.x documentation and are not supported by any
results file in this repository. They have been removed, and where a real
measurement exists it replaces them:

* "160 KB active heap", "< 500 KB RAM", "strictly 160 KB", "~160 KB per open
  vault" → measured 8–9 KB resident **per document** (`rss_v3r4.json`).
* "256-bit encrypted", "AES-256", "Projected Stream Cipher (256-Bit)", "Zero
  plaintext leakage", "military/bank-grade" → plaintext by default; the optional
  mode is scrypt + SHAKE256 + HMAC-SHA256 and is explicitly not AES.
* `"cipher": "256-bit Projected Stream Cipher"` as a documented `stats()` value →
  the code never produced it.
* "Sub-millisecond retrieval", "0.44 ms", "34–100× faster", "retrieval scales
  sub-linearly", "searches 20+ projects in under 20 ms", "< 10 ms at 20,000
  chunks" → the measured p50 table, which is linear in corpus size.
* "100 % recall", "100 % Hop-2 recall@1", "100 % rank-1 needle precision",
  "98 %+ accuracy", "95.8 % LLM factuality" → the measured recall tables.
* The 321 ms / 648 ms / 978 ms multi-hop latency table and its 70.8 % / 70.8 % /
  67.5 % recall row → `multihop_texthop_v3r2_1190.json` and
  `experiment_4d_bridge_results.json`.
* The "single vault vs divided vs re-merged" table (586.5 / 1,221.5 / 580.9 ms,
  160 KB, 95.8 % factuality, 100 % composite recall) → removed; its benchmark's
  fixtures had leaked into the engine's own rule layers.
* "α = 0.35 balanced optimal default" for multi-hop steering → measured
  significantly worse than no steering; the mechanism is gone.
* "prime kernel", "manifold resonance", "non-linear harmonic scores",
  "4D Latent Continuous Memory Engine" as a service name → removed; none names
  anything in this package.
* "Run `python3 test_security.py`" → that file never existed. Run
  `python3 -m pytest -q`.
* "MCP: reserved for Pro (v2)" → there is no `mcp.py` in this package; the
  capability is absent, not withheld.
* "+16.7 % EM multi-hop improvement" → a measurement of a different, MLX-based
  research prototype on synthetic data, not of `nanomem.Vault`.

### Packaging and distribution

The 0.1.0 wheel was built on 2026-09-15 from a *v2* source tree and then
hand-copied to five places, while three older generations of it sat in the
platform folders. Eleven `nanomem-0.1.0-py3-none-any.whl` files were on disk in
three mutually different generations (md5 `a7a29809` ×5, `84bf1848` ×3,
`513a1f1c` ×3) and none of them contained this engine. All eleven are deleted.

* **One wheel: `nanomem-0.3.0-py3-none-any.whl`**, 139,166 B, md5
  `900c16a08b86c1da588ef28fcf9ace4b`, 22 entries, built from `Launch 1/shared`.
  It is byte-identical in all twelve locations that previously held a 0.1.0
  wheel. Every `nanomem/*.py` inside it md5-matches the canonical
  `nanomem_standalone/nanomem/` source.
* **There is no 0.2.0.** The version is not written in `pyproject.toml` or
  `setup.py` any more; both read `nanomem.__version__`, so the distribution
  version and the imported version cannot disagree again. That disagreement is
  what made a 0.1.0 install indistinguishable from this one.
* **`assets/write_classifier.npz` is now declared package data.** A wheel built
  without `[tool.setuptools.package-data]` contains no `nanomem/assets/` entry
  at all — verified by building one — and the classifier then falls back to its
  surface-only constants without saying so. On the held-out 2-persona set that
  fallback costs 3.5 pts of accuracy and 4.5 pts of F1 (90.48 % / 87.21 % with
  the head, 87.01 % / 82.76 % without; `scratch/refound/write_classifier_v2_results.json`,
  keys `heldout.full` and `heldout.surface`).
* **`assets/manifold_prototypes.npz` is gone** (271,607 B in every 0.1.x copy).
  It cached the 95 archetype embeddings of the retired manifold classifier,
  30 of which were paraphrases of a single benchmark's turns. Nothing reads it.
* **Both build paths agree.** `python3 -m pip wheel . --no-deps -w dist` and the
  `python3 setup.py bdist_wheel` that `Mac/build_mac.sh`, `Linux/build_linux.sh`
  and `Windows/build_windows.py` invoke produce the same 22-entry archive. The
  package is pure Python; all three platform folders hold the same
  `py3-none-any` wheel.
* **Install check**: `pip install --force-reinstall` of the wheel into a clean
  CPython **3.9.6** venv (the floor of `requires-python = ">=3.8"` available
  here), then `import nanomem; from nanomem.vault import Vault` →
  `0.3.0 3.0.3`, `classifier.model_info['source'] == 'write_classifier.npz'`,
  and the `nanomem` console script resolves.
* **MCP**: `nanomem/mcp.py` is still absent from the package and from the wheel.
  A 0.1.x-era copy of it survives in the two `nanomem_mac_bundle/` folders only,
  where it was preserved rather than shipped as a supported feature.

### Shipped fixture vaults migrated

Opened once with this engine, in place, each leaving its original beside it as
`<path>.v2.bak` (verified byte-identical to the pre-migration file):

| fixture | docs | v2 bytes | v3 bytes | migrate ms |
|---|---:|---:|---:|---:|
| `hands_on_llm_vault.dat` (×3 copies) | 845 | 6,352,820 | 2,159,838 | 205 / 84 / 83 |
| `Launch 1/Mac/nanomem_mac_bundle/my_demo_vault.dat` | 4 | 35,956 | 7,276 | 1 |
| `personal_memory.dat` | 0 | 64 | 256 | 1 |

The book vault is **66.0 % smaller** (fp16 vectors on disk plus block
compression) with all 845 documents intact: reading every record back out of the
three migrated copies gives one identical content digest. The three files are
*not* byte-identical to each other any more — each migration stamps its own
creation time and vault uuid into the header — which is a change from 0.1.x,
where the three copies had one md5.

### Known limits in 0.3.0

1. Held-out chat ranking is 75.0 % gold-store top-1 against an 80 % target.
2. Search is linear; no sub-linear index ships. The opt-in router is slower
   in-engine at equal recall.
3. `delete` / `update` / `prune` are full rewrites (7.5 ms at 1k, 70.5 ms at 10k,
   ≈ 1 s at 71k) and block appenders. Tombstones are planned for 3.1.
4. Resident memory is 8–9 KB per document; there is no fixed ceiling.
5. Password mode does not detect truncation or rollback and leaks exact sizes.
6. The write gate is English-only.
7. `durable="full"` (F_FULLFSYNC) is implemented but unmeasured.
8. No MCP server.
9. A filter that matches nothing still costs a full scan.
10. `_ContainerView` (`.toc`, `.read_payload`, `.close`) is a deprecated compat
    shim for 3.0 and will be removed in 3.1.

---

## 0.1.0

Initial release. Superseded; see the withdrawn-claims list above before relying
on anything written about it.
