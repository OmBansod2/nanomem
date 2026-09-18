# nanomem — Developer Manual

An embedded memory engine: one `.dat` file, no daemon, `numpy` as the only
third-party runtime dependency. Search is an **exact linear scan** over the
corpus — it returns what an exhaustive fp32 cosine scan returns, and its latency
grows with the corpus.

Package 0.7.14 · engine 3.4.2 · container format 3 · arena cache format 3.

Every performance number in this manual comes from a results JSON in
`evidence/` produced by a script in this repository, and the file is named
next to the number.

---

## 1. Install

```bash
pip install -e nanomem_standalone
```

Or use it in place:

```python
import sys
sys.path.insert(0, "nanomem_standalone")
from nanomem import Vault
```

Check what you actually imported:

```python
import nanomem
print(nanomem.__version__, nanomem.ENGINE_VERSION)   # 0.7.14 3.4.2
```

If that prints `0.1.0`, an older editable install is shadowing this package.

Embeddings come from a local Ollama-compatible endpoint (`nomic-embed-text` by
default). `EmbeddingProvider` falls back to a deterministic hash encoder when no
daemon answers; that fallback produces vectors that are internally consistent but
are not the real model, so benchmark with the daemon up.

---

## 2. Breaking changes from 0.1.x

| Change | What to do |
| :--- | :--- |
| `score` is a cosine (was a squashed non-linear value) | re-tune thresholds: 0.25 → 0.42, 0.32 → 0.53, 0.35 → 0.58, or call `nanomem.engine.legacy_score_to_cosine(old)` |
| hits carry both `score` and `cosine` | compare against `cosine` for a pure similarity; `score` = cosine + documented boosts, bounded by `stats()['max_boost']` (0.70) |
| `Vault.add()` and `engine.add_fact()` return the document id (`str`) | previously `None`; the REST add response now returns a usable id |
| plaintext is the default | pass `password=` (or `NANOMEM_PASSWORD`) to encrypt; check `stats()['encrypted_at_rest']` |
| v2 files migrate on first open | the original is preserved as `<path>.v2.bak` |
| `stats()` values are measured | `active_heap_ram_kb` is no longer the constant 160.0; `cipher` reports the real construction |
| `search(multihop=True)` uses a text bridge | `alpha`, `num_hops`, `beam_width` are accepted for compatibility and no longer change the result |

See `nanomem_standalone/CHANGELOG.md`.

---

## 3. Core SDK

### Open a vault

```python
from nanomem import Vault

with Vault("knowledge.dat") as vault:              # plaintext
    ...

with Vault("knowledge.dat", password="…") as vault:  # optional password mode
    ...
```

`Vault.__init__(path, embed_model="nomic-embed-text", base_url=None,
llm_base_url=None, api_key=None, password=…, on_torn_tail="warn")`.

`close()` flushes. `VaultEngine.__exit__` also flushes, so nothing pending is
lost when a `with` block ends.

### Add

```python
doc_id = vault.add(
    text="Acme Corp acquired Robotics AI for $45M in July 2024.",
    metadata={"category": "mergers", "year": 2024},
    source="press_release",
)
# doc_id -> 'doc_1f4c9a30b2'
```

Batch:

```python
n = vault.add_batch([
    {"text": "...", "metadata": {"page": 82}, "source": "page_82"},
    {"text": "...", "metadata": {"page": 130}, "source": "page_130"},
], batch_size=64)
```

**Cost of a write.** The storage side is 19.81 µs mean over the first 500 of
4,000 adds and 19.32 µs over the last 500 — a ratio of 0.975, i.e. flat as the
vault grows (median 10.63 µs, p95 12.08 µs, max 598.29 µs on a block spill;
`evidence/headtohead_v3.json`). On a re-opened 71,433-document vault it is
17.9 µs mean / 7.0 µs p50 (`evidence/scale_results_v3r4.json`). Those
exclude embedding, which dominates: one local `nomic-embed-text` call is roughly
9–16 ms. With precomputed vectors, ingesting 71,433 documents takes 1.3 s of
storage time; through the embedding daemon it is bounded by the daemon.

### Files

```python
chunks = vault.ingest_file(
    "docs/security_compliance.txt",
    lines_per_chunk=50, overlap_lines=10,
    metadata={"category": "compliance", "version": "2.1"},
    source="security_compliance_v2",
)
```

`metadata` may be a callable `(chunk_text, start_line, end_line) -> dict` for
per-chunk tagging. Chunk ids are `filename:start_line-end_line`. Binary
extensions are refused with a clear error; run OCR or `pdftotext` first.

### Search

```python
hits = vault.search(
    query="How does KV caching optimize attention?",
    top_k=3,
    filter={"category": "mergers", "year": 2024},
    min_score=0.42,                 # applied to the pre-boost cosine
    temporal_direction="current",   # or "historical"
    decompose=True,                 # split composite multi-part questions
)

for h in hits:
    print(h["cosine"], h["score"], h["revision"], h["text"][:80])
```

A hit is a plain dict: `id`, `doc_id`, `text`, `source`, `metadata`, `score`,
`cosine`, `timestamp`, `revision`. All values are JSON-serialisable.

`min_score` is compared against the pre-boost cosine, so a threshold means the
same thing whether or not the entity/temporal layer fires.

**Latency and recall** — real HotpotQA paragraphs in random insertion order, 500
questions, `top_k=4`, measured on a re-opened vault
(`evidence/scale_results_v3r4.json`, `headtohead_v3.json`):

| Corpus | p50 | p95 | recall@4 | exhaustive numpy | FAISS `IndexFlatIP` | index |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,190 | 0.052 ms | 0.063 ms | 68.3 % | 68.3 % (0.017 ms) | 68.3 % (0.050 ms) | 2.5 MB |
| 4,760 | 0.143 ms | 0.171 ms | 45.8 % | 45.8 % | 45.8 % | 10.1 MB |
| 10,000 | 0.345 ms | 0.378 ms | 70.4 % | 70.4 % (0.271 ms) | 70.4 % (0.392 ms) | 22.0 MB |
| 71,433 | 1.762 ms | 2.010 ms | 60.6 % | 60.6 % (1.450 ms) | 60.6 % (1.075 ms) | 155.9 MB |

The engine matches exhaustive fp32 cosine in recall at every size above, and in
*ordering* too at 1,190 and 4,760 documents: 0 of 120 top-4 order differences and
0 of 120 rank-1 differences with the real question strings, on a re-opened vault
(`evidence/exactness_v3r2.json`, `headtohead_v3.json`).

At 10,000 and 71,433 documents the ordering is not quite identical: 2 of 500 and
5 of 500 questions have a different top-4 order, every case a tie or near-tie with
a cosine gap of at most 2.2e-05 and two of them exactly 0.0
(`evidence/verify_round3_v3r3.json`, `F_top4_differences_are_ties`). The
cause is the fp16 vectors on disk, whose worst per-row cosine against the fp32
original is 0.99999988. Recall is unaffected.

There is no sub-linear index in this release. Above `n_exhaustive` (default
50,000) an IVF-style router can be switched on, and it is measured in §8.

### Metadata filtering

```python
hits = vault.search("merger valuation", top_k=5,
                    filter={"category": "mergers", "year": 2024})
```

Filtered search matched a brute-force filtered scan on 120 of 120 queries
(`exactness_v3r2.json`). A filter that matches **nothing** costs a full scan
(p50 3.09 ms at 1,190 docs) because proving absence requires scanning everything.

### Composite questions

`decompose=True` (the default) splits multi-part questions into sub-queries and
round-robins the results so one dominant topic cannot take every slot:

```python
hits = vault.search("What is BPE, and how does KV caching accelerate attention?")
for h in hits:
    print(h.get("sub_query"), "->", h["text"][:60])
```

This is a mechanism, not a measured accuracy claim. The earlier "100 % composite
recall vs 50 % federated" table has been removed; no results JSON supports it.

### Multi-hop

```python
hits = vault.search_multihop("Which university did the creator of Python attend?",
                             top_k=4)
# or: vault.search(..., multihop=True)
```

Pass 2 re-embeds the question together with the hop-1 winner's text and merges
under the same `top_k` budget. Measured, equal budget, 1,190 documents, evidence
recall@4 (`evidence/multihop_texthop_v3r2_1190.json`): 68.3 % single pass
→ **79.2 %** with the bridge at `top_k=4`; 90.0 % → 92.5 % at `top_k=8`.
Raising `top_k` is the larger lever, and the bridge *hurts* when the hop-1 anchor
is wrong. Full analysis, including the trained latent bridges that did not work:
[MULTIHOP_REASONING_AND_TOPOLOGY_GUIDE.md](MULTIHOP_REASONING_AND_TOPOLOGY_GUIDE.md).

### Temporal revisions

```python
vault.add("Server active in us-east-1",
          metadata={"entity": "server_location", "source": "chat_session"})
vault.add("Server migrated to eu-central-1",
          metadata={"entity": "server_location", "source": "chat_session"})

vault.search("Where is the server?", temporal_direction="current")     # revision 2
vault.search("Where was the server?", temporal_direction="historical") # revision 1
```

> **Do not set `metadata["entity"]` by hand.** The example above is the shape of
> the data, not a recommendation. Measured on the fixture-free temporal set,
> writing facts with an explicit entity tag scores **59.2 %** against **90.5 %**
> for writing them with no tag at all and letting the engine detect one
> (`evidence/temporal_bench_results.json`). Earlier revisions of this
> manual recommended the tag; following that advice makes the engine worse. The
> tag remains supported because a caller that has a *reliable* extractor can
> still beat detection — but it overrides the engine's own grouping, so a tag
> that is merely plausible is worse than none.

Revision groups are scoped by `(user_id, project, entity)`. On the release's own
generic probes the current revision is ranked first in 14 of 16 cases against 3
of 16 for plain cosine, and historical lookups are 16 of 16
(`ranking_dev_r4_shipped.json` (dev-persona probes, not published: an evaluation corpus whose value depends on not being public, and it carries realistic contact-shaped strings)).

**On adjacent attributes the layer is worse than doing nothing, and this manual
used to claim parity.** On the fixture-free set — "backup email" against
"primary email" — nanomem scores **90.0 %** where a plain cosine scan scores
**100.0 %**, with a 10.0 % sibling-confusion rate against plain cosine's 0.0 %
(`evidence/temporal_bench_results.json`). Third-party facts are the same
shape: 96.7 % against 100.0 %. If your corpus is mostly near-identical sibling
attributes and never supersedes anything, this layer costs you.

Statements about third parties are namespaced separately, so "his number is …"
cannot become revision 2 of your own (`evidence/third_party_v3r4.json`).

### CRUD

```python
chunk = vault.get("docs/security_compliance.txt:1-50")
vault.update(id=..., text=..., metadata=...)     # bumps revision
vault.delete(id=...)                              # or ids=, text_exact=,
                                                  # text_contains=, where=, source=
vault.forget("credit card number", min_score=0.42)
vault.prune(older_than_days=365)                  # returns BYTES FREED
info = vault.inspect()                            # sources + metadata summary
```

**`delete`, `update` and `prune` are full atomic rewrites.** There are no
tombstones in 3.0 (planned for 3.1). Deleting one record costs **7.5 ms at 1,000
records and 70.5 ms at 10,000** (`evidence/rewrite_cost_v3r4.json`),
roughly a second at 71k, and the rewrite holds the exclusive lock throughout, so
appenders block. Batch your deletions.

Note the return types: `delete` returns the number of records removed; `prune`
returns **bytes freed**, not a count.

### Merge, split, export

```python
with Vault("master.dat") as master:
    master.merge("project_a.dat", incoming_project="project_a")
    # -> {'incoming': 150, 'added': 150, 'duplicates_skipped': 0}

    master.unmerge("project_b.dat", target_vault_path="project_b_restored.dat")
    master.split_by_doc("handbook.pdf", target_vault_path="handbook.dat", purge=True)
    master.export("legal.dat", where={"category": "legal"}, purge=True)
```

Merging preserves every text, metadata dict and vector exactly, stamps
provenance (`project`, `source_vault`, `file_path`), and reconciles revisions
chronologically. `deduplicate=False` is the default, so nothing is silently
dropped.

An earlier version of this manual recommended merging "5 to 15 vaults or ~20,000
chunks" for "< 10 ms retrieval and 98 %+ accuracy". Both numbers are withdrawn.
What is measured is the size/latency curve in the table above: pick a corpus size
from it. At 20,000 documents, expect a p50 between the 10,000 and 71,433 rows.

### Read-only federation

```python
hits = Vault.search_multi(["company_docs.dat", "local_code.dat"],
                          query="OAuth token refresh flow", top_k=3)
```

`search_multi` is read-only and sorts the merged hits naively. Two mechanical
caveats: cosine scores are relative to one corpus, so a 0.78 in a 50-document
file is not comparable to a 0.78 in a 10,000-document file; and each file keeps
its own revision clock, so two versions of one fact in two files both look
current. Merge when you want one calibrated ranking.

---

## 4. Concurrency, durability, crash safety

All measured on this build. Unlike the performance tables above, these came from
one-off scripts run by hand during implementation and independently reproduced
during verification; they are not written into a results JSON in
`evidence/`, so treat them as reproducible procedures rather than citable
numbers. The commands are in the release notes.

* **Threads.** 4 writer + 4 reader threads on one engine: 1,200/1,200 records,
  1,200 unique ids, 0 errors. `stats()`, `count()`, `search()` and
  `iter_records()` all take the engine lock.
* **Processes.** 4 processes × 150 appends and 8 × 80 appends against one vault:
  every record present, unique ids, `integrity_errors` empty, torn tail 0.
  4 processes racing to *create* a vault that does not exist: 240/240.
* **Crash.** `SIGKILL` mid-ingest at 809,000 records: reopens in 887 ms with all
  records contiguous and in insertion order, `integrity_errors` empty,
  `truncated_tail_bytes` 0, and the vault is appendable afterwards.
* **Corruption.** A single-byte flip at four different structural offsets raises
  four different specific `NanomemError` subclasses. No silent wrong data.
* **Truncation.** Every byte offset of a small vault truncated in turn (4,971
  cases): each reopened to an exact record prefix and accepted an append.

`durable="fsync"` is the default. `durable="full"` (F_FULLFSYNC) is implemented
but **unmeasured**.

Rebuilds go through `engine.replace_all()`: a temp file plus `os.replace`, so a
crash mid-rewrite leaves the old file intact.

---

## 5. Memory

`stats()['active_heap_ram_kb']` is a measured sum of the buffers the vault has
allocated. It is not the process cost.

**The table that used to be here (612.2 MB at 71,433 documents, 8–9 KB per
document) was measured before 0.3.2 and is gone.** The arena no longer
over-allocates as it grows, so the per-document figure fell with it. Measured on
this build, a process that opens a vault another process wrote and serves 500
queries (`evidence/reopen_results.json`, `headline`;
`memory_results.json`, `reopen_only`):

| Documents | peak RSS, scanned arena | peak RSS, mapped arena (0.4.0 layout) | `phys_footprint`, mapped |
| ---: | ---: | ---: | ---: |
| 10,000 | 41.5 MB | 37.6 MB | 1 MB |
| 71,433 | 287.5 MB | **258.2 MB** | **9 MB** |

> **0.5.0 changes this table's last column and it is the one regression in the
> release.** The sidecar no longer holds fp32 vectors to map, so the upcast is
> anonymous memory again: at 71,433 rows `phys_footprint` reads **270.5 MB**
> where 0.4.0's layout read 43.5 MB, and peak `ru_maxrss` 331.0 MB against
> 318.4 MB in the same harness (`evidence/sidecar_size_results.json`).
> That bought 253 MiB of disk. `arena_cache_vectors="cache"` restores 0.4.0's
> profile; `"offsets"` is lower than either (57.1 MB) at ×1.73 on p50. The rows
> above were measured on 0.4.0 and were not re-run.

Budget roughly **3.5 KB of resident memory per document** at 768 dimensions, of
which the fp32 vector is 3.0 KB. A one-process **ingest** peaks higher than a
server does — 298.4 MB at 71,433 documents, range 286–310 across harnesses — and
`reserve_rows(n)` / `--expect-docs` remove the growth copies from that shape.
`stats()['process_rss_kb']` gives the process figure;
`stats()['arena_bytes']` counts **anonymous** memory only, so on a cached open
it reads 0 and the vectors appear under `arena_cache_mapped_bytes`.

The "160 KB active heap" and "< 500 KB RAM" claims in earlier documentation were
constants printed by `stats()`, not measurements. They are gone.

---

## 6. Password mode

```python
with Vault("app.dat", password="…") as v:   # or NANOMEM_PASSWORD
    ...
```

```bash
python3 -m nanomem.cli rekey --vault app.dat --new-password-stdin
python3 -m nanomem.cli rekey --vault app.dat --remove
```

Construction: scrypt (n=2¹⁶, r=8, p=1, parameters in the header) over an
NFKC-normalised passphrase and a 16-byte per-vault salt; three HMAC-SHA256
sub-keys for encryption, authentication and a header key-check; SHAKE256 keystream
XOR; HMAC-SHA256 over the full cleartext block header plus ciphertext, bound to
the vault uuid, compared with `compare_digest` before any decryption.

**Not AES. Not a NIST AEAD. Not audited.** `stats()['cipher']` says so verbatim.

Measured overhead (`evidence/crypto_overhead_v3r3.json`): one scrypt
derivation 95.67 ms ≈ 10.5 offline guesses/s/core; open +99.67 ms at 1,190 docs,
+104.88 at 5,000, +165.36 at 40,000 (it grows — every block's MAC is verified);
per-search −0.0003 / +0.002 / −0.0043 ms, i.e. noise. Encrypted and plaintext
files are **byte-for-byte the same size**, so the size leak is exact.

An empty-string password is an error, not a silently plaintext vault. A `bytes`
password is an error. A passphrase handed to a v2 read-only open is an error
rather than being dropped.

Read `nanomem.THREAT_MODEL` before relying on this. The short version of what it
does not do: it does not stop truncation or rollback by someone who can write to
the file; `on_torn_tail="raise"` catches neither; `on_integrity_error="skip"` is
an integrity downgrade; it hides no sizes or counts; it does not protect a
running process; and losing the passphrase loses the vault.

---

## 7. Migrating a 0.1.x vault

Open it. Migration happens once, on first open, and the original is preserved
byte-identically as `<path>.v2.bak`.

```python
with Vault("old_memory.dat") as v:
    print(v.stats()["format_version"])   # 3
```

Verified on two golden fixtures in `evidence/golden/`: a chat vault
answers 12 of 12 expected top-1 queries after migration — that arm is recorded in
`ranking_dev_r4_shipped.json` (dev-persona probes, not published: an evaluation corpus whose value depends on not being public, and it carries realistic contact-shaped strings) (`golden_chat_v2`), and the v2
engine answered 10 of 12 on the same fixture. A book vault migrates 845 of 845
documents with 0 of 20 top-4 differences from exhaustive fp32 cosine computed
over `iter_records()`; that second arm was run by hand and is not in a results
JSON.

Junk v2 entity tags are re-derived with the generic tagger; the original is kept
in `metadata["entity_v2"]`.

`Vault(..., migrate=False)` opens a v2 file read-only through the legacy reader.
`nanomem user delete` ignores and removes `.v2.bak` and `.tmp-*` siblings.

---

## 7a. Reopen: the arena cache (`arena_cache=`, ON by default)

Opening a vault used to rebuild the resident fp32 arena block by block — O(rows),
**0.1634 s** at 71,433 documents, which lost to sqlite-vec's 0.0014 s by 115x.
Since 0.4.0 the arena is kept in a `<vault>.arena` sidecar and an open is a stat,
a 4 KiB header read, a bind and an `mmap`.

```python
VaultEngine(path)                        # arena_cache="map" — the default
VaultEngine(path, arena_cache="off")     # 0.3.2 behaviour: no sidecar at all
```

Measured, in-process median opens
(`evidence/reopen_results.json` for 0.4.0's layout,
`evidence/quality_summary.json` for 0.5.0's):

| rows | `arena_cache="off"` | `"map"` | speedup |
| ---: | ---: | ---: | ---: |
| 1,190 | 0.002657 s | 0.000186 s | 14.3× |
| 10,000 | 0.021239 s | 0.000206 s | 103× |
| 71,433 | 0.160 s | **0.000228 s** | **702×** |

It changes no answer: same build, cache on against cache off, 500 queries scored
against every row at all three sizes — the SHA-256 of the full fp32 score vectors
is identical and 0 of 500 top-10 lists change. Steady-state p50 is unmoved
(paired duel 0.9983). A serving process's `phys_footprint` at 71,433 rows drops
to **9 MB from 288 MB**, because the vectors become clean file-backed pages.

### What the sidecar contains, and the two flags that decide (0.5.0)

In 0.4.0 the sidecar **was** the resident fp32 arena, so it duplicated the vault:
257.5 MiB beside a 148.7 MiB file that already held the same vectors in fp16 and
the same text. Since 0.5.0 it stores **offsets** into the vault's own regions
instead, derived from the block table it already carried — so they cost zero new
bytes and the sidecar is **4.38 MiB**.

```python
VaultEngine(path)                                  # offsets_ram / vault — default
VaultEngine(path, arena_cache_vectors="offsets")   # no upcast: least RAM, slowest
VaultEngine(path, arena_cache_vectors="cache",
                  arena_cache_records="cache")     # 0.4.0 exactly
```

| `arena_cache_vectors` | sidecar @71,433 | vault + sidecar | p50 (paired) | `phys_footprint` |
| :--- | ---: | ---: | ---: | ---: |
| `"offsets_ram"` **(default)** | 4.38 MiB | **153.05 MiB** | ×0.998 | 270.5 MB |
| `"offsets"` | 4.38 MiB | 153.05 MiB | **×1.73** (screen on) / ×2.98 (off) | 57.1 MB |
| `"cache"` (0.4.0) | 257.48 MiB | 406.15 MiB | baseline | 43.5 MB |

`arena_cache_records` is `"vault"` (read record text from the vault) or
`"cache"` (copy it into the sidecar, 0.4.0's behaviour; +43.8 MiB).

**Pick by what is scarce.** Disk is scarce → the default. RAM is scarce and
queries are rare → `"offsets"`. You are measuring against 0.4.0 → `"cache"`.

**The cache format moved 2 → 3**, so the first open after upgrading rebuilds the
sidecar once. Changing either flag costs the same one slow open.

**Five costs, so you can decide rather than discover.**

1. **Resident memory, which is where 0.5.0 moved the cost.** The fp32 upcast did
   not go away; it happens once, on the first vector read, into anonymous memory
   this process dirties. `phys_footprint` at 71,433 rows is **270.5 MB against
   0.4.0's 43.5 MB**. Disk fell 253 MiB and RAM rose 227 MB — that is the trade,
   and `arena_cache_vectors="cache"` reverses it.
2. **The first query** pays the page faults *and* the upcast the open skipped:
   0.0117 s → 0.0227 s at 71,433. Time to *first answer* still improves ~15×,
   not 702×.
3. **A fresh process on a small vault is slower**: 0.00764 s against 0.00321 s at
   1,190 rows. It wins from ~10,000 rows up. **Small vault + short-lived
   processes (a CLI loop): pass `arena_cache="off"`.**
4. **The open that writes it** costs 0.3051 s against 0.1659 s, once.
5. **A vault damaged out of band, under a live engine, is now an error instead
   of a crash — and it used to be neither.** The arena reads through a mapping of
   the vault, so truncating the vault file while an engine holds it open is a
   `SIGBUS`, which no `except` can catch. Every read now checks one `os.fstat`
   against the last byte the block table can address and raises
   `nanomem.errors.VaultShrankError`, at 0.486 µs per call. **It is a guard with
   a window, not a guarantee**: a truncation landing between the check and the
   page touch still faults, and `arena_cache_vectors="cache"` has no window at
   all. nanomem's own torn-tail recovery does not trip it.

```python
from nanomem import VaultShrankError
try:
    hits = v.search("…", top_k=5)
except VaultShrankError:
    # something outside this process truncated the vault file
    ...
```

The four modes:

| mode | what it does | when |
| :--- | :--- | :--- |
| `"map"` | map the sidecar, bind it to the vault, re-read neither. O(1). | default |
| `"copy"` | as `map`, then copy the vectors into anonymous RAM once, so no query takes a page fault. 219 MB dirty at 71,433. | first query must be as fast as the second |
| `"verify"` | as `map`, plus check the cache against its own digest **and** re-read every vault block. 0.1478 s at 71,433, against 0.1656 s for a plain rescan. | you want a scan's corruption checking with a mapped memory profile |
| `"off"` | no sidecar is read or written. | small vaults, short-lived processes, or you want 0.3.2's bytes-on-disk |

A cache that is missing, stale, truncated, from another vault, of the wrong
dtype or unparseable is dropped and rebuilt in silence — the container is always
the source of truth. Encrypted vaults **never** get one, in any mode, because the
sidecar would hold vectors and document text in the clear. `arena_cache_info()`
reports what the last open actually did, including
`vault_blocks_checked` (`"all"` / `"appended tail only"` / `"none"`).

> **`verify` is not a security control, and neither is anything else here.**
> Its digest is unkeyed and sits in the header of the file it describes, so a
> forgery that updates both makes `verify` serve a planted row at cosine 1.0.
> Nor does the sidecar weaken the vault: a **plaintext** vault's block trailer is
> an unkeyed SHA-256 that anyone holding this library can recompute, and a forged
> row is served by a full `arena_cache="off"` scan at cosine 0.999997. Both are
> passing tests in `tests/test_arena_cache.py`. These modes detect **corruption**.
> If you need tamper resistance, set a password — and note that a password
> disqualifies the vault from having a cache at all.

## 7b. Search latency: the PCA screen (`screen=`, OFF by default)

```python
VaultEngine(path, screen="pca")          # opt-in
```

A **latency** flag with no recall knob and no accuracy trade-off. It bounds each
document's cosine from above in a 256-dimensional subspace, skips rows that
provably cannot reach the top-k, and scores the survivors with the same fp32
kernel the full scan uses. Its recall delta is **identically 0 by construction**;
no recall improvement is claimed.

71,433 documents, 200 questions × 5 paired interleaved cycles
(`evidence/pca_screen_results.json`, `phaseC_latency_full`):

| arm | p50 | p95 |
| :--- | ---: | ---: |
| `screen="off"` | 1.9495 ms | 2.0898 ms |
| `screen="pca"` | **1.1350 ms** | 1.7318 ms |
| speedup | **1.718×** [1.695, 1.736] | 1.207× |

Exactness was proven and then measured: 0 bound violations in 71,433,000
document checks; 0 of 3,000 searches with a different id list or a
non-bitwise-identical float32 score list. Appends cannot invalidate it — the
bound uses no property of the basis — and that was measured too, with a basis
fitted from 6,000 rows and 24,000 rows appended after it.

**Costs** 1032 bytes per document resident (71.6 MiB at 71,433, +34.2% over the
arena) and a one-off 0.191 s basis build per process. The basis is recomputed,
never written to disk.

**Stands down automatically**, changing no result and raising nothing: below
`screen_min_rows` (20,000), under `residency="int8"`, with a `metadata_filter`,
on an entity-tagged vault, for an explicit `temporal_direction="historical"`,
when survivors would exceed `screen_max_frac` (0.35), and if the basis cannot be
built. Forced on below the floor it genuinely loses (0.478× at 1,000 documents),
which is why the floor is there. `search_batch` does not use it.

Knobs: `screen_dims=256`, `screen_min_rows=20_000`, `screen_max_frac=0.35`,
`screen_seed_pool=128`.

> Turning this flag on made the **default** path 16 µs slower per search (0.8% of
> its p50), because reaching "identical top-k" required breaking ties on the row
> id rather than inheriting `np.argpartition`'s unspecified order. That cost is
> paid whether or not you use the flag, and it is why top-10 lists containing two
> documents with the identical float32 cosine can come back in a different order
> than 0.3.2 gave — 8 of 500 at 71,433 documents, every one a bitwise tie.

**Neither flag is exposed on `Vault`**, whose `__init__` takes no engine tuning
arguments (nor does it expose `router` or `residency`). Use `VaultEngine`
directly, or `vault.engine`.

---

## 7c. The write gate (`Vault.should_store`, `Vault.chat`) — its decision point moved in 0.5.0

`Vault.should_store(text)` and `Vault.chat(...)` run a trained logistic head over
[embedding ; 23 surface features] plus four documented rule layers, and decide
whether a conversational turn is worth persisting. **In 0.5.0 the full head's
decision point moved from 0.60 to 0.05** (`nanomem.classifier.DEPLOYMENT_THRESHOLD_FULL`).
The same call now keeps about **57 % more turns**.

**Why.** The trainer picked 0.60 on leave-one-persona-out accuracy/F1, where a
false positive and a false negative cost the same. In deployment they do not: a
refused fact is never written and no retrieval quality can recover it, while a
kept noise turn costs ~1.9 kB and no measurable query time. At 0.60 the gate ran
at 100.0 % precision and **64.6 % recall**, which made **32.1 % of a 420-question
benchmark unanswerable before retrieval ran**.

Measured end to end on a 300-question held-out split
(`evidence/write_policy_results.json`, pre-registered, scored once;
reproduced through the shipped classifier in `evidence/quality_summary.json`):

| | top-1 | gate recall / precision | docs / 10 personas | MiB |
| :--- | ---: | ---: | ---: | ---: |
| 0.4.0 (threshold 0.60) | 35.7 % | 64.5 % / 100.0 % | 271 | 0.492 |
| **0.5.0 (threshold 0.05)** | **49.0 %** | 94.0 % / 92.9 % | 425 | 0.764 |
| storing everything | 49.7 % | 100.0 % / 35.6 % | 1,180 | 2.014 |

+13.3 pt, paired bootstrap 95 % CI [+8.7, +18.3]. Storing everything buys 0.7 pt
more for 2.8× the rows, so this is the efficient point, not the extreme one.

```python
from nanomem.classifier import WriteClassifier, get_classifier

get_classifier().threshold_full            # 0.05  (deployment)
get_classifier().threshold_trained_full    # 0.60  (what the trainer picked)

old = WriteClassifier(threshold=0.60)      # 0.4.0's gate, exactly
paranoid = WriteClassifier(threshold=0.0)  # store everything the rules allow
```

**Four things to know before you rely on it.**

1. **Pass the embedding you already have.** `classify(text, embedding=vec)` skips
   an embedding call. The classifier dials the daemon itself on 95.8 % of turns,
   which is **11.5 ms — 49 % of a write turn** (`turn_latency_results.json`).
   `Vault.chat` does not pass it yet; if you have the vector, call the
   classifier directly.
2. **The rule layers are not affected by the threshold** and they are a second
   loss: with the head fully off they still drop 52 of 1,180 turns, 14 of them
   answers to questions, capping gate recall at 95.0 % (~3.4 pt).
3. **With no embedding daemon reachable** the gate falls back to a surface-only
   head that still decides at 0.60. That path was not in the study that moved the
   full head, so it was not moved either.
4. **This is measured on one 14-persona synthetic fixture.** The classifier's own
   held-out generalisation set is quarantined and was not opened, so the effect
   of 0.05 on its published 90.5 % held-out accuracy is unmeasured.
5. **6.0 % of real facts are still refused**, and lowering the threshold further
   will not get them. A spot check while writing this section found durable
   third-person statements ("the spare key is in the blue tin on the third
   shelf") scoring **p = 0.006** — below any usable threshold, and dropped
   identically at 0.60 and 0.05. The residual loss is head capacity, not the
   decision point. If you cannot tolerate losing any fact, pass
   `WriteClassifier(threshold=0.0)` and accept 2.8× the rows, or gate on your
   own signal.

---

## 8. Routing above 50,000 documents

`n_exhaustive` defaults to 50,000. Below it, search is an exhaustive fp32 scan.
Above it, an IVF-style global spherical k-means cell router is available:

```python
VaultEngine(path, router="auto")    # opt-in
```

Switching it on triggers a one-time `compact(recluster=True)` — random-order
blocks are unroutable. That is recorded in the file header, so successive opens
do not rewrite the file again (measured: open 0 rewrites once at 45.2 ms, opens 1
and 2 do not, same inode and uuid, 16.6 / 16.9 ms; 12 of 12 concurrent first
opens succeed — `evidence/router_persist_v3r4.json`).

Measured on the reclustered 71,433-document corpus, 1,000 held-out questions,
paired bootstrap (`evidence/router_gate_v3r3.json`):

| Arm | recall@4 | Δ vs exhaustive | 95 % CI | scan fraction |
| :--- | ---: | ---: | :--- | ---: |
| exhaustive | 78.65 % | — | — | 100 % |
| cell router, `beam_frac=0.10` | 78.05 % | −0.60 pt | [−1.00, −0.25] | 12.1 % |
| **cell router, `beam_frac=0.25` (default)** | **78.60 %** | **−0.05 pt** | **[−0.20, +0.10]** | 28.97 % |
| cell router, `beam_frac=0.50` | 78.65 % | 0.00 pt | [0.00, 0.00] | 55.6 % |
| page-landmark router (ablation only) | 78.50 % | −0.15 pt | [−0.35, +0.05] | 25.1 % |

The recall gate passes. **It still ships off**, because in the engine it is
slower for the same recall: p50 3.604 ms with `router="auto"` against 1.919 ms
exhaustive, and 7.51 s added to every open against 0.15 s. Turn it on only if
you have measured your own corpus and found otherwise. The page-landmark
power-mean router in `nanomem/routing.py` is kept as a measured ablation; it is
not a shipped path.

---

## 9. CLI

```bash
python3 -m nanomem.cli init company.dat
python3 -m nanomem.cli add "DB port is 5433" --vault company.dat
python3 -m nanomem.cli search "DB port" -k 3 --vault company.dat
python3 -m nanomem.cli ingest docs.md --vault company.dat
python3 -m nanomem.cli merge new.dat --into master.dat --project new
python3 -m nanomem.cli split master.dat --key category --dir ./splits
python3 -m nanomem.cli forget "old address" --vault company.dat
python3 -m nanomem.cli stats --vault company.dat
python3 -m nanomem.cli inspect --vault company.dat
python3 -m nanomem.cli rekey --vault company.dat --new-password-stdin
python3 -m nanomem.cli proxy --port 5000 --upstream http://localhost:11434
python3 -m nanomem.cli user list | create <name> | delete <name>
```

`add` prints the document id and its own elapsed time. Every subcommand accepts
`--password-stdin`. Errors go to stderr; exit 1 for a `NanomemError`, 2 for a
usage error.

---

## 10. HTTP

Two servers ship, both loopback-bound by default and **neither authenticates**.
Do not expose either to a network you do not control.

```bash
python3 -m nanomem.cli proxy --port 5000 --upstream http://localhost:11434
python3 server.py --host 127.0.0.1 --port 8080
```

Endpoints, limits and payloads: [SERVICES_AND_API_SPECIFICATION.md](SERVICES_AND_API_SPECIFICATION.md).

---

## 11. Integrating

```python
from nanomem import Vault

class NanoMemRetriever:
    def __init__(self, path):
        self.vault = Vault(path)

    def get_relevant_documents(self, query):
        return [h["text"] for h in self.vault.search(query, top_k=4)]
```

One-line grounded answer through a local model:

```python
with Vault("knowledge.dat", llm_base_url="http://localhost:11434/v1") as v:
    res = v.ask("What is the return policy?", llm="llama3.2:3b")
    print(res["answer"], res["citations"])
```

`ask` runs retrieval and one chat completion; its latency is the model's, not
nanomem's.

---

## 12. Testing

```bash
cd nanomem_standalone && python3 -m pytest -q
```

**602 tests**, no network required, nothing skipped when a local embedder is
running. There is no `test_security.py`; earlier documentation told you to run
one and it never existed.

---

## 13. Limits, stated plainly

1. Search is linear. 0.37 ms at 10k, 1.78 ms at 71k. No sub-linear index ships.
   `screen="pca"` makes the linear scan 1.718× cheaper at 71k without making it
   approximate; it is still linear. **Before optimising around this, read
   COMPETITIVE_POSITION.md's turn-latency section**: with a real embedder in the
   loop the whole store is **17 % of a read turn at 71k and 0.8 % at 1,190**, and
   one embedding call is 80 %. Below ~10,000 documents this line does not
   describe anything a user can perceive.
2. Resident memory is 8–9 KB per document. There is no fixed RAM ceiling.
3. `delete` / `update` / `prune` are full rewrites and block appenders.
4. A filter that matches nothing still costs a full scan.
5. Password mode does not detect truncation or rollback, and leaks exact sizes.
6. On the personal-memory chat task, gold-store top-1 is **63.9 %** on the clean
   three-persona set (70.8 % on the older held-out pair). The 80 % release target
   this manual used to quote is **withdrawn as unreachable**: 43.3 % of the
   questions never name the attribute they ask about, and the measured achievable
   ceiling from query text is **66.4 % [61.8, 70.8]**. When the question *does*
   name the attribute the same build scores 84.3 % [80.5, 87.5], so the lever is
   the question, not the engine. Four mechanisms aimed at that ceiling were
   pre-registered and all failed (`evidence/context_lever_results.json`).
   End to end — with the write gate deciding rather than gold labels — the same
   build scored 35.7 % in 0.4.0 and **49.0 % in 0.5.0** on a 300-question
   held-out split (§7c); a perfect write gate reaches 52.3 %. **The deployment
   number was limited by what got written, not by what got ranked.**
7. The write-gate classifier is English-only.
8. `durable="full"` is implemented but unmeasured.
9. No MCP server ships in this package.
10. Third-person questions are resolved from `name`-tagged records in a personal
    vault; a vault where the owner never stated their name falls back to plain
    cosine for those.
11. **A vault costs 4.4 MiB more disk than a vault with no sidecar** — 153.05
    MiB at 71,433 documents against 148.66 — because 0.5.0's sidecar stores
    offsets rather than a second copy of the vectors. (0.4.0's cost 406.2 MiB;
    that regression is gone.) **The cost moved to RAM**: a serving process's
    `phys_footprint` is 270.5 MB against 0.4.0's 43.5 MB, because the fp32
    upcast is now anonymous memory rather than a mapped file.
    `arena_cache_vectors="offsets"` gives the lowest memory of any mode at
    ×1.73 on p50; `arena_cache="off"` writes no sidecar at all.
12. **A plaintext vault has no tamper resistance**, with or without the sidecar,
    and no `arena_cache` mode provides any. Its block trailer is an unkeyed
    SHA-256 that anyone holding this library can recompute. Set a password if
    that matters — and note that encrypted vaults never get a cache, so you are
    choosing between §7a's reopen and tamper resistance, not getting both.
