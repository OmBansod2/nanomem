# nanomem — Documentation Hub

nanomem is an embedded, single-file memory engine for AI applications. A vault is
one `.dat` file. There is no daemon, no server and no container to run. The only
third-party runtime dependency is `numpy`; everything else is the Python standard
library.

Retrieval is an **exact linear scan** over the whole corpus below
`n_exhaustive` (default 50,000 documents). For the text it scans it returns the
same documents an exhaustive fp32 cosine scan returns — that is measured, not
asserted — and its latency therefore grows with the corpus. The default
`decompose=True` splits a multi-clause question and scans each clause, so for
such a question the result is the exhaustive answer to each clause rather than to
the whole sentence; `decompose=False` scans the string as one query. There is no sub-linear index in this
release and no fixed millisecond guarantee. Every number below comes from a
results JSON produced by a script in this repository, and the file is named.

* Package version 0.7.21, engine `ENGINE_VERSION` 3.4.6, container format 3, arena cache format 4.
* Embeddings: `nomic-embed-text` (768-d) through a local Ollama-compatible daemon.
* Measurements on Apple M4 Pro, macOS 26.5, Python 3.12, numpy 2.5.3.
* Every `evidence/…` path in these documents is relative to the
  repository root; the results JSONs and the scripts that wrote them live there.

---

## Choose your track

| Track | For | Interface |
| :--- | :--- | :--- |
| [Personal AI](USER_MANUAL_PERSONAL.md) | everyday users | `python chat.py`, or the local proxy in front of Ollama / LM Studio / vLLM |
| [Developer & RAG](USER_MANUAL_DEVELOPER.md) | engineers building agents and RAG | `from nanomem import Vault`, plus the `nanomem` CLI |
| [Multi-hop guide](MULTIHOP_REASONING_AND_TOPOLOGY_GUIDE.md) | anyone chaining evidence | what the two-pass bridge does, and what it costs |
| [Services & REST API](SERVICES_AND_API_SPECIFICATION.md) | non-Python callers | proxy (port 5000) and `server.py` (port 8080) |
| Changelog (`nanomem_standalone/CHANGELOG.md`) | upgraders from 0.1.x | the breaking changes in 3.0 |

---

## Measured performance

### Retrieval, deployed shape (ingest → close → re-open → search)

Real HotpotQA paragraphs in **random insertion order**, 500 held-out questions,
`top_k=4`, evidence recall@4 (all gold paragraphs must appear in the top 4).
"Before" is the 0.1.x engine on the identical corpus and questions.

| Corpus | Recall@4 before | Recall@4 after | p50 before | p50 after | p95 after | Index before | Index after |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,190 docs | not measured¹ | **68.3 %** | — | **0.052 ms** | 0.063 ms | — | 2.5 MB |
| 10,000 docs | 24.0 % | **70.4 %** | 44.01 ms | **0.345 ms** | 0.378 ms | 71.8 MB | **22.0 MB** |
| 71,433 docs | 5.8 % | **60.6 %** | 461.22 ms | **1.762 ms** | 2.010 ms | 511.5 MB | **155.9 MB** |

Sources: before — `evidence/scale_results_current_engine.json`; after —
`evidence/scale_results_v3r4.json` (10k, 71k) and
`evidence/headtohead_v3.json` (1,190).

¹ The 1,190-document corpus was never run against the 0.1.x engine as a shipped
arm. The mechanism behind the 24.0 % and 5.8 % figures is measured there instead:
in random insertion order, the block-page router of 0.1.x recalled 3.3–21.7 %
against 68.3 % exhaustive (`evidence/sweep_routing_1190.txt`, random-order
table).

### The recall is exactly exhaustive recall

| Corpus | nanomem | exhaustive fp32 numpy | FAISS `IndexFlatIP` |
| :--- | ---: | ---: | ---: |
| 1,190 | 68.3 % | 68.3 % | 68.3 % |
| 4,760 (4× perturbed distractors) | 45.8 % | 45.8 % | 45.8 % |
| 10,000 | 70.4 % | 70.4 % | 70.4 % |
| 71,433 | 60.6 % | 60.6 % | 60.6 % |

`evidence/headtohead_v3.json`, `evidence/scale_results_v3r4.json`.
At 1,190 and 4,760 documents there are 0 of 120 top-4 order differences against
exhaustive fp32 cosine, and 0 of 120 differences between searching with a query
vector and searching with the real question string.
`evidence/exactness_v3r2.json` repeats this on a re-opened vault with the
real question strings: 0/120 top-4 order differences and 0/120 rank-1
differences. A `metadata_filter` search matches a brute-force filtered scan on
0/120 queries; a filter matching nothing costs a full scan (p50 3.09 ms at 1,190
docs) because proving absence requires one.

At 10,000 and 71,433 documents the *sets* still match exhaustive exactly — the
recall figures above are identical — but a handful of top-4 **orderings** differ:
2 of 500 questions at 10,000 and 5 of 500 at 71,433, every one of them a tie or a
near-tie (largest cosine gap 2.2e-05, two of them exactly 0.0), caused by the
fp16 vectors on disk. `evidence/verify_round3_v3r3.json`,
`F_top4_differences_are_ties`. "Exactly exhaustive" is measured at 1,190 and
4,760 documents; at scale it holds for recall and up to tie-breaking for order.

These are recall numbers for a retriever, not a score for nanomem against a
different product. 68.3 % and 60.6 % are what this embedding model gets on this
benchmark; nanomem's claim is that it loses nothing relative to scanning
everything in fp32.

### Memory

nanomem holds an fp32 arena in RAM and fp16 vectors on disk. Measured in a
process that does nothing but open the vault and read `stats()`
(`evidence/rss_v3r4.json`):

| Documents | RSS growth on open | `stats()['active_heap_ram_kb']` | ratio | RSS per document |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 81.6 MB | 50.3 MB | 1.62× | **8.4 KB** |
| 71,433 | 612.2 MB | 401.6 MB | 1.52× | **8.8 KB** |

Read that as **roughly 8–9 KB of resident memory per document** at 768
dimensions. The fp32 vector alone is 768 × 4 = 3.0 KB per document, and the
engine over-allocates arena capacity as it grows (`resident_arena_mb` works out
at 4.9–5.5 KB per document in `evidence/scale_results_v3r4.json`); the
rest is record text, metadata and Python objects. `active_heap_ram_kb` is a
measured value — a sum of the buffers the vault has actually allocated — and it
under-reports the process by the ratio in the table, which the
`active_heap_ram_method` string in `stats()` states verbatim. Use
`stats()['process_rss_kb']` for the process figure.

There is no configuration of nanomem that uses a fixed small amount of RAM
regardless of corpus size.

### Write path

Appending is a memtable write plus a periodic block flush; it does not slow down
as the vault grows. Over 4,000 chat-shaped adds the first 500 averaged 19.81 µs
and the last 500 averaged 19.32 µs, a ratio of 0.975 (median 10.63 µs, p95 12.08
µs, max 598.29 µs on a block spill) — `evidence/headtohead_v3.json`. On a
re-opened 71,433-document vault the write path is 17.9 µs mean / 7.0 µs p50
(`evidence/scale_results_v3r4.json`). These figures exclude embedding,
which is the dominant cost in practice (one local `nomic-embed-text` call is
roughly 9–16 ms).

### Index size on disk

2,208.8 bytes per document at 1,190 docs — 0.609× the raw UTF-8 text plus fp32
vectors it stands in for (`evidence/headtohead_v3.json`). It holds at
scale: 2,307 B/doc at 10,000 and 2,289 B/doc at 71,433
(`evidence/scale_results_v3r4.json`).

### Deletes and rewrites

There are no tombstones in 3.0. `delete`, `update` and `prune` are full atomic
rewrites of the container. Deleting one record costs **7.5 ms at 1,000 records
and 70.5 ms at 10,000** (`evidence/rewrite_cost_v3r4.json`), and roughly a
second at 71k. The rewrite holds the exclusive lock for its whole duration, so
concurrent appenders block.

### Password mode

Optional and off by default. Measured overhead
(`evidence/crypto_overhead_v3r3.json`):

| Documents | extra open time | extra per-search time | file size difference |
| ---: | ---: | ---: | ---: |
| 1,190 | +99.67 ms | −0.0003 ms | 0 bytes |
| 5,000 | +104.88 ms | +0.002 ms | 0 bytes |
| 40,000 | +165.36 ms | −0.0043 ms | 0 bytes |

One scrypt derivation costs 95.67 ms (about 10.5 offline guesses per second per
core). It is paid once per open, not per query. The open cost grows with the file
because every block's MAC is verified.

### Chat: does the vault return the right memory?

Two persona benchmarks of scripted conversations, scored on whether the expected
answer is the top-1 hit. "Gold stores" scores retrieval with the correct records
already stored, isolating ranking from the write gate.

| Set | Gold-store top-1 before | after | top-3 after | end-to-end after |
| :--- | ---: | ---: | ---: | ---: |
| 3-persona selection set (n=36) | 50.0 % | **91.7 %** | 100.0 % | 91.7 % |
| 2-persona held-out set (n=24) | 33.3 % | **75.0 %** | 95.8 % | 79.2 % |
| 3-persona dev set (n=24, new) | — | 95.8 % | 100.0 % | 87.5 % |

Before: `evidence/clean_chat_results_current_engine.json` and
`clean_chat_results_heldout_baseline.json` — derived from a fixture held out of development so a score against it means something, and therefore not published. After:
`clean_chat_results_v3r4_engine.json`, `clean_chat_results_v3r4_engine_heldout.json`,
`clean_chat_results_v3r4_engine_devpersona.json`, in `evidence/`.

**The release target was ≥ 80 % gold-store top-1 on both persona sets. The
held-out set missed it at 75.0 % (18 of 24).** Five of the six failures are
ordering errors inside a group that *was* retrieved — the expected answer is in
the top 3 for 23 of 24 questions — and the sixth misses the top 3 entirely. The
implementer also reports that the held-out set was inspected during development
this round, so 75.0 % should be read as an upper bound rather than a clean
out-of-sample estimate.

### Write gate (the classifier that decides what to remember)

A numpy logistic head over the embedding plus generic surface features, trained
on the 3-persona set and scored once on two unseen personas
(`evidence/write_classifier_v2_results.json`, `heldout`):

| Arm | Accuracy | F1 |
| :--- | ---: | ---: |
| 0.1.x classifier (out of sample) | 76.2 % | 69.6 |
| **3.0 classifier (out of sample)** | **90.5 %** | **87.2** |
| 3.0 surface-only fallback, no embedder | 87.0 % | 82.8 |

231 turns, 88 of them worth storing. Leave-one-persona-out on the training set is
94.9 % / 92.8 F1. The decision costs 0.06 ms once the embedding exists; end to
end it is 13.2 ms p50 through `Vault`, dominated by the embedding call
(`write_classifier_v2_results.json`, `via_vault.heldout_2_personas`).

The rule layers that copied phrases from a benchmark fixture are gone, and so is
the prototype asset built from paraphrases of it.

### Multi-hop

The shipped bridge is a **text hop**: re-embed the question together with the
hop-1 winner's text and search again, then merge under the same `top_k` budget.
Equal budget, 1,190 documents, evidence recall@4
(`evidence/multihop_texthop_v3r2_1190.json`):

| | single pass | text hop |
| :--- | ---: | ---: |
| `top_k=4` | 68.3 % | **79.2 %** |
| `top_k=8` | 90.0 % | 92.5 % |

Widening `top_k` is the bigger lever, and the bridge hurts when the hop-1 anchor
is wrong. Trained latent bridges — linear residual, residual MLP and an RK4
neural ODE, three seeds each — did not beat the query alone on the primary
held-out set; every confidence interval was at or below zero. Alpha-steering
(`q + α·v_anchor`) measured significantly worse and has been removed. Details and
the full arm table: [MULTIHOP_REASONING_AND_TOPOLOGY_GUIDE.md](MULTIHOP_REASONING_AND_TOPOLOGY_GUIDE.md).

### Routing above 50,000 documents

An opt-in IVF-style spherical k-means cell router exists (`router="auto"`). On
the reclustered 71,433-document corpus with 1,000 questions it reaches 78.60 %
recall@4 against exhaustive 78.65 % (−0.05 pt, 95 % CI [−0.20, +0.10]) while
scanning 29 % of the corpus — it passes the recall gate. **It still ships off**,
because inside the engine it is *slower* for the same recall: p50 3.604 ms versus
1.919 ms exhaustive, plus 7.51 s added to every open.
`evidence/router_gate_v3r3.json`.

---

## What changed in 3.0 (and will break 0.1.x code)

* **`score` is now a cosine.** It used to be a squashed non-linear value. Hits
  carry both `score` (cosine plus explicit, documented boosts, bounded by
  `stats()['max_boost']` = 1.10) and `cosine` (the plain cosine). Thresholds
  calibrated against the old scale must be re-tuned: 0.25 → 0.42, 0.32 → 0.53,
  0.35 → 0.58. `nanomem.engine.legacy_score_to_cosine()` converts any other
  threshold. The shipped call sites are already retuned.
* **`Vault.add()` and `engine.add_fact()` return the document id (`str`).** They
  used to return `None`. The REST `/v1/memory/add` response now carries the id the
  record is really stored under.
* **Plaintext is the default.** A vault is a plain file unless you pass a
  passphrase. `stats()['encrypted_at_rest']` tells you which one you have.
* **0.1.x (`format_version` 2) files migrate on first open.** The original is kept
  beside the new one as `<path>.v2.bak`, byte-identical to the source. Migration
  is verified on two golden vaults: 12/12 expected top-1 answers on a chat vault
  (`ranking_dev_r4_shipped.json` (not published, see evidence/INDEX.md), `golden_chat_v2`), and 845/845
  documents with 0/20 top-4 differences from exhaustive fp32 cosine on a book
  vault (run by hand; not in a results JSON).
* **`stats()` values are measured.** `active_heap_ram_kb` is no longer the
  constant 160.0; `cipher` reports what the file actually uses.

Full list: `nanomem_standalone/CHANGELOG.md`.

---

## Encryption: what the password mode is and is not

Password mode is **off by default**. When you supply a passphrase, nanomem builds
the scheme from the Python standard library: scrypt (n=2¹⁶, r=8, p=1) over an
NFKC-normalised passphrase and a 16-byte random per-vault salt derives a 32-byte
master key; three independent HMAC-SHA256 sub-keys are taken from it for
encryption, authentication and a header key-check. Each block's payload is XORed
with a SHAKE256 keystream over (K_enc, vault uuid, a fresh 16-byte nonce, block
sequence number) and then authenticated with HMAC-SHA256 over the whole 96-byte
cleartext block header and the ciphertext, bound to the vault uuid. Tags are
compared with `hmac.compare_digest` before anything is decrypted.

**This is not AES, not 256-bit AES and not a NIST-approved AEAD.** It is a
standard-library keyed-sponge stream with a separate HMAC tag. It has not been
independently audited or FIPS validated.

It protects the confidentiality of your text, ids, metadata and vectors against
someone who obtains the `.dat` file without the passphrase, and the integrity of
any complete block against modification, against substitution from another vault,
against re-use of a block from an earlier generation of the same vault, and
against replay or reordering inside one generation.

It does **not** protect against an attacker who can write to the file: a
truncation on a block boundary and a byte-for-byte rollback both open silently,
with no warning and `truncated_tail_bytes` 0. `on_torn_tail="raise"` covers
neither — it only catches an unreadable trailing fragment.
`on_integrity_error="skip"` is an integrity downgrade, not a repair.

It does **not** hide sizes, and the leak is exact rather than approximate: the
keystream is length-preserving, so an encrypted vault is byte-for-byte the same
size as a plaintext one (measured at 1,190 / 5,000 / 40,000 records), and every
block header states its record count, byte lengths and write time in the clear.

It does **not** protect a running process, and it does not defend a weak
passphrase beyond one scrypt guess (95.67 ms, about 10.5 guesses/s/core).

Losing the passphrase means losing the vault. The full statement is in
`nanomem.THREAT_MODEL`; the text above is a summary of it.

---

## Quick start

Personal:

```bash
python chat.py                       # connects to a local Ollama
python chat.py --lmstudio            # LM Studio on port 1234
python chat.py --vault work.dat      # a separate profile
```

Developer:

```python
from nanomem import Vault

with Vault("knowledge.dat") as vault:
    doc_id = vault.add("Server backup runs daily at 02:00 UTC",
                       metadata={"env": "prod"})
    hits = vault.search("When does the backup run?")
    print(hits[0]["text"], hits[0]["cosine"])
```

CLI:

```bash
python3 -m nanomem.cli init company.dat
python3 -m nanomem.cli add "DB port is 5433" --vault company.dat
python3 -m nanomem.cli search "DB port" -k 3 --vault company.dat
python3 -m nanomem.cli stats --vault company.dat
```

---

## Known limits

1. **Held-out chat ranking is 75.0 %, below the 80 % target.** Stated above with
   the measured value.
2. **Latency grows linearly with the corpus.** 0.345 ms at 10k, 1.762 ms at 71k.
   Above `n_exhaustive` (50,000) the opt-in router is available but is slower in
   the engine than the exhaustive scan it replaces.
3. **Deletes are full rewrites** (7.5 ms at 1k, 70.5 ms at 10k, ~1 s at 71k) and
   they block appenders.
4. **Resident memory is 8–9 KB per document,** not a fixed ceiling.
5. **Truncation and rollback of an encrypted vault are undetectable.**
6. **The write gate is English-only.** Its features and its training data are
   English chat; a non-English statement of fact can be rejected.
7. **`durable="full"` (F_FULLFSYNC) is implemented but unmeasured.**
8. **An MCP server DOES ship**, and this line used to say the opposite.
   `nanomem/mcp.py` is in both the wheel and the sdist and exposes 7 tools over
   stdio (`nanomem_add`, `nanomem_search`, `nanomem_history`, `nanomem_as_of`, `nanomem_changes`, `nanomem_volatility`, `nanomem_stats`). Run it with `python -m nanomem.mcp --vault memory.dat`.
   The claim was false from the release that added the module until 0.7.18.

---

## Multiple vaults: what is actually known

Chat sessions bind to exactly one vault. That is a design decision about write
ambiguity — if two vaults are open, there is no principled way to decide which
one a new fact belongs to — not a measured performance claim. The earlier
comparison table of "single vault vs divided vs re-merged" (586.5 ms / 1,221.5 ms
/ 580.9 ms, 95.8 % factuality, 160 KB heap) has been removed: those numbers came
from a benchmark whose fixtures leaked into the engine's own rule layers, and no
results JSON in this repository supports them. Use `Vault.search_multi()` for
read-only federation across a handful of files, and `merge()` when you want one
ranking over one corpus.
