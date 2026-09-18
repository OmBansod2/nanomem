# nanomem — Services and REST API Specification

Package 0.3.0 · engine 3.0.3 · container format 3.

nanomem is an embedded memory engine: one `.dat` file per vault, no daemon,
`numpy` as the only third-party runtime dependency. Two optional HTTP services
ship with it. Neither authenticates; both bind to loopback by default.

Every performance figure in this document names the results JSON in
`evidence/` it comes from.

---

## 1. Who the services are for

### Everyday users of local LLMs

Local models forget everything when the window closes. nanomem gives them a
persistent store.

* `nanomem proxy` sits in front of Ollama / LM Studio / vLLM and speaks the
  OpenAI chat API, so Open-WebUI, Chatbox, Jan or LibreChat get memory without
  any code change. Point the app at `http://localhost:5000/v1`.
* `chat.py` is a terminal client with the same behaviour.
* Profiles: `nanomem user list | create <name> | delete <name>`, and
  `--user <name>` on most commands.
* One file per profile. Backing up is `cp`; resetting is `rm`.

### Developers

* Language-agnostic REST endpoints with CORS, so JavaScript, Go, Rust or curl can
  use the same vault as Python.
* An in-process Python SDK:

  ```python
  from nanomem import Vault
  with Vault("app.dat") as v:
      doc_id = v.add("Client requested enterprise tier")
      hits = v.search("pricing questions")
  ```
* `merge` / `split` / `unmerge` / `export` for vault lifecycle.
* Oversized text is auto-split at 500 words (1,500 characters for non-spaced
  scripts) with a 40-word overlap.

### Privacy-sensitive and offline deployments

* No telemetry and no outbound network calls. The only traffic is to the local
  model and embedding daemon you configure.
* **A vault is a plaintext file by default.** Pass a passphrase to turn on the
  optional password mode, and read §6 for exactly what it protects.
* No Docker, no database server, no cloud account.
* Resident memory is **8–9 KB per document** (`evidence/rss_v3r4.json`) —
  fine for a personal or team corpus on ordinary hardware, not a fixed ceiling.
  The "160 KB RAM" figure in earlier revisions of this document was a constant
  printed by `stats()`, not a measurement.

---

## 2. The two services

```
 Memory Proxy (port 5000)                Direct REST server (port 8080)
 nanomem.proxy                           server.py
 ------------------------------          ------------------------------
 GET  /health                            GET  /health
 GET  /v1/memory/stats                   POST /ingest
 POST /v1/vault/init                     POST /search
 POST /v1/memory/ingest                  POST /ask
 POST /v1/memory/add                     POST /save
 POST /v1/memory/search                  POST /load
 POST /v1/chat/completions  (OpenAI)
 GET  /v1/models            (passthrough)
```

Both refuse request bodies over 10 MB with `413`.

### Security posture (read before binding to anything but localhost)

* **Neither service authenticates any endpoint.** Anyone who can reach the port
  can read and write your memories.
* Both default to `127.0.0.1`. The proxy warns if you bind elsewhere.
* Client-supplied vault names are confined to the active vault's directory.
  An absolute path or a `../` traversal is rejected with `400`; no file or
  directory is created outside the root.
* `POST /v1/vault/init` cannot change the *active* vault unless the proxy was
  started with `--allow-vault-switch`; otherwise it answers `403`.
* If the proxy was started with a passphrase, every vault it opens — including
  one created through `/vault/init` — uses that passphrase. It cannot be
  downgraded to plaintext by a remote caller.

---

## 3. Memory Proxy (port 5000)

```bash
python3 -m nanomem.cli proxy --port 5000 --vault memory.dat \
    --upstream http://localhost:11434 [--password-stdin] [--host 127.0.0.1] \
    [--allow-vault-switch] [--no-cite]
```

### `POST /v1/memory/add`

```json
{ "text": "User prefers concise bulleted answers.",
  "source": "web_settings_modal",
  "metadata": { "user_id": "usr_9410", "category": "preferences" } }
```

```json
{ "status": "success",
  "doc_id": "doc_644c9384f905",
  "text": "User prefers concise bulleted answers." }
```

`doc_id` is the id the record is really stored under — `Vault.add()` returns it,
so `GET`/`update`/`delete` by that id work. Through 0.1.x this field was
invented client-side and nothing could look it up.

Text over 500 words is auto-split into linked chunks.

### `POST /v1/memory/search`

```json
{ "query": "coding style and python version", "top_k": 3 }
```

```json
{ "query": "coding style and python version",
  "top_k": 3,
  "hits": [
    { "text": "User prefers concise bulleted answers.",
      "source": "web_settings_modal",
      "metadata": { "user_id": "usr_9410", "category": "preferences" },
      "score": 0.8412,
      "cosine": 0.8412,
      "timestamp": 1726301400.0,
      "revision": 1 } ] }
```

`top_k` is clamped to 1–50 (default 3).

**`score` changed scale in 3.0.** It is now a cosine plus explicit, documented
boosts, bounded by `stats()['max_boost']` (0.70); `cosine` is the plain cosine.
Thresholds calibrated against the 0.1.x scale must be re-tuned — 0.25 → 0.42,
0.32 → 0.53, 0.35 → 0.58 — or converted with
`nanomem.engine.legacy_score_to_cosine()`.

**Latency.** In-process search is an exact linear scan, so it depends on corpus
size (`evidence/scale_results_v3r4.json`, re-opened vault, 500 questions):
p50 0.052 ms at 1,190 documents, 0.345 ms at 10,000, 1.762 ms at 71,433. Over
HTTP, add the embedding call (roughly 9–16 ms to a local `nomic-embed-text`) plus
HTTP overhead. There is no sub-millisecond end-to-end guarantee.

### `GET /v1/memory/stats`

Returns `Vault.stats()` verbatim. The values are measured, not constants:

```json
{ "file_path": "/var/data/memory.dat",
  "file_size_mb": 0.42,
  "total_documents": 318,
  "compacted_blocks": 3,
  "memtable_pending": 0,
  "active_heap_ram_kb": 1846.2,
  "process_rss_kb": 91344,
  "encrypted_at_rest": false,
  "cipher": "none (plaintext)",
  "engine_version": "3.0.3",
  "format_version": 3,
  "vector_dtype_on_disk": "float16",
  "score_scale": "cosine",
  "router": "off",
  "routing_mode": "exhaustive",
  "n_exhaustive": 50000,
  "max_boost": 0.7,
  "integrity_errors": [],
  "truncated_tail_bytes": 0 }
```

On a password-protected vault, `encrypted_at_rest` is `true` and `cipher` reads:

```
SHAKE256-XOF stream + HMAC-SHA256 tag (encrypt-then-MAC), scrypt-derived keys;
stdlib construction, not AES, not a NIST AEAD
```

Earlier revisions of this document showed `"cipher": "256-bit Projected Stream
Cipher"`. The code has never produced that string in 3.0, and the label was
wrong in 0.1.x as well.

`active_heap_ram_kb` is the sum of the buffers the vault has allocated. It
under-reports the process by a measured 1.52–1.62× — use `process_rss_kb` for
the process figure, and see `active_heap_ram_method` in the response for the
statement of what it excludes.

### `GET /health`

```json
{ "status": "ok", "engine": "nanomem", "version": "0.3.0" }
```

### `POST /v1/vault/init`

```json
{ "vault": "support_docs.dat", "set_active": false }
```

The name is resolved inside the active vault's directory; absolute paths and
traversal are rejected with `400`. `set_active: true` requires
`--allow-vault-switch`, otherwise `403`. The response reports whether the vault
is encrypted.

### `POST /v1/memory/ingest`

```json
{ "file": "./policies.md" }
```
or
```json
{ "directory": "./docs" }
```

### `POST /v1/chat/completions`

An OpenAI-compatible passthrough that, on each turn: retrieves relevant memories,
injects them into the system prompt, forwards to the upstream model, returns the
response, and writes any new durable fact from the user turn.

The injected memory block is capped at **500 words** so it cannot swamp a small
model's context.

Whether a turn is stored is decided by a trained write gate. Out-of-sample, on
two personas it had never seen (231 turns, 88 storable), it scores **90.5 %
accuracy / 87.2 F1**, against 76.2 % / 69.6 for the 0.1.x gate
(`evidence/write_classifier_v2_results.json`). With no embedding daemon
reachable it falls back to a surface-feature head at 87.0 % / 82.8.

### `GET /v1/models`

Passthrough to the upstream server.

---

## 4. Direct REST server (port 8080)

```bash
python3 server.py --host 127.0.0.1 --port 8080
```

| Method | Endpoint | Request | Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | — | status, service name, indexed chunk count, measured `active_heap_ram_kb`, file size, `cipher` |
| `POST` | `/ingest` | `{"documents": ["...", ...]}` or `{"chunks": [{"text","source","metadata"}]}` | `{"status","chunks_ingested","total_indexed_chunks","time_seconds"}` |
| `POST` | `/search` | `{"query","top_k","temporal_direction"}` | ranked hits with `rank`, `id`, `score`, `text`, `revision`, `metadata`, plus the measured `retrieval_latency_ms` for that call |
| `POST` | `/ask` | `{"query","top_k"}` | grounded answer from the configured local model |
| `POST` | `/save` | — | flushes and confirms the path |
| `POST` | `/load` | `{"vault": "name.dat"}` | reopens, and returns the real `cold_start_load_ms` for that open |

`top_k` is clamped to 1–50 (default 5). `/load` confines the requested name to
the server's vault root exactly as the proxy does.

`/load` reports the load time it actually measured rather than asserting a
budget. For reference, re-opening a vault takes 24.8 ms at 10,000 documents and
172.1 ms at 71,433 (`evidence/scale_results_v3r4.json`).

The service name in `/health` is "NanoMem Continuous Memory Engine". Earlier
revisions called it the "4D Latent Continuous Memory Engine"; there is no 4D
component in this package.

---

## 5. Document ingestion

`nanomem ingest <file>` and `Vault.ingest_file()` handle text:

| Category | Extensions | Chunking | Metadata kept |
| :--- | :--- | :--- | :--- |
| Plain text | `.txt`, `.log`, `.transcript` | 50 lines / 10-line overlap | file name, line range |
| Markdown, notes | `.md`, `.rst`, `.org` | paragraph-aware | file name, headings |
| Source code | `.py`, `.js`, `.ts`, `.rs`, `.go`, `.c`, `.cpp`, `.html`, `.css` | indentation- and line-aware | `filename`, `start_line`, `end_line`, `is_code` |
| Config | `.json`, `.jsonl`, `.yaml`, `.toml`, `.xml` | line / key-value | `filename`, tags |
| Tabular | `.csv`, `.tsv` | header-anchored row windows | line numbers, column headers |

Chunk ids are `filename:start_line-end_line`, so a chunk can be fetched, updated
or deleted individually.

**Tables.** The header row is kept as schema context for each row window. This
is semantic lookup over knowledge tables, catalogues and parameter sheets — for
analytical queries over large tables, use SQL or DuckDB.

**Images, PDFs and audio are not embedded.** A multimodal encoder would pull in a
deep-learning runtime and gigabytes of weights, which is the opposite of what
this package is. Binary extensions are refused with an explicit error. Extract
text first:

```
image/screenshot  -> Tesseract or Apple Vision OCR -\
PDF               -> pdftotext or pypdf             -> nanomem ingest output.txt
audio             -> Whisper transcription         -/
```

```bash
pdftotext paper.pdf paper.txt && nanomem ingest paper.txt --vault my_vault.dat
```

---

## 6. Password mode

Off by default. `stats()['encrypted_at_rest']` tells you which mode a file is in.

Construction: scrypt (n=2¹⁶, r=8, p=1, parameters stored in the header) over an
NFKC-normalised passphrase and a 16-byte random per-vault salt; three independent
HMAC-SHA256 sub-keys for encryption, authentication and a header key-check; a
SHAKE256 keystream XOR over (K_enc, vault uuid, a fresh 16-byte nonce, block
sequence number); an HMAC-SHA256 tag over the full 96-byte cleartext block header
plus the ciphertext, bound to the vault uuid, compared with `compare_digest`
before anything is decrypted. The header reserves a cipher id for an HMAC-CTR
suite; it is not implemented.

**This is not AES, not "256-bit encryption", and not a NIST-approved AEAD.** It
is a standard-library keyed-sponge stream with a separate HMAC tag, and it has
not been independently audited or FIPS validated.

Protects: confidentiality of text, ids, metadata and vectors against someone
holding the file without the passphrase; integrity of any complete block against
modification, substitution from another vault, re-use of a block from an earlier
generation of the same vault, and replay or reordering within one generation.

Does not protect against:

* **Truncation and rollback** by anyone who can write to the file. Both open
  silently, with `truncated_tail_bytes` 0 and no warning.
  `on_torn_tail="raise"` catches neither — it only catches an unreadable
  trailing fragment.
* **`on_integrity_error="skip"`**, which is an integrity downgrade, not a repair:
  a whole block can be excised cleanly. The default `"raise"` refuses the file.
* **Size and count disclosure.** The keystream is length-preserving: an encrypted
  vault is byte-for-byte the same size as a plaintext one (measured at 1,190 /
  5,000 / 40,000 records, `evidence/crypto_overhead_v3r3.json`), and every
  block header states its record count, section lengths and write time in the
  clear. The leak is exact, not approximate.
* **File swapping.** A block tag binds the vault uuid, not the path, so two of
  your own vaults can be exchanged on disk undetectably.
* **A running process.** Keys live in memory for the vault object's lifetime.
  `close()` wipes them.
* **A weak passphrase**, beyond one scrypt guess: 95.67 ms, about 10.5 offline
  guesses per second per core.

Measured cost (`evidence/crypto_overhead_v3r3.json`): +99.67 ms to open at
1,190 documents, +104.88 ms at 5,000, +165.36 ms at 40,000 — it grows, because
every block's MAC is verified — against −0.0003 / +0.002 / −0.0043 ms per search,
which is noise. The scrypt cost is paid once per open, never per query.

In plaintext mode there is no protection at all: the block trailer is an unkeyed
SHA-256 that detects accidental corruption and can be recomputed by anyone.

Losing the passphrase loses the vault. `nanomem.THREAT_MODEL` is the full
statement; the above is a summary of it.

---

## 7. Limits and guardrails

| Limit | Value | Enforced by | Why |
| :--- | :--- | :--- | :--- |
| Max fact length | 500 words (1,500 chars for non-spaced scripts) | auto-splitter in `Vault.add` | keeps a record retrievable as one unit |
| Max HTTP body | 10 MB | both servers, `413` | bounds heap use |
| Memory injected into a prompt | 500 words | proxy word counter | protects a small model's context |
| `top_k` | 1–50 (default 3 proxy / 5 server) | clamp in `Vault.search` | bounds result construction |
| Exhaustive search ceiling | `n_exhaustive` = 50,000 | engine | above it, the opt-in router is available |
| Batch ingestion | 1,000 records per batch | batch chunker | bounds flush size |
| Upstream LLM timeout | 120 s | `urllib` | prevents worker starvation |
| Vault file permissions | `0600` | container create | verified after init / add / rekey |

Measured storage cost per record: 2,209 bytes per document at 1,190 docs, 2,307
at 10,000, 2,289 at 71,433 — 0.609× the raw text plus fp32 vectors it replaces
(`evidence/headtohead_v3.json`, `scale_results_v3r4.json`).

Measured rewrite cost: `delete`, `update` and `prune` are full atomic rewrites —
7.5 ms at 1,000 records, 70.5 ms at 10,000
(`evidence/rewrite_cost_v3r4.json`), roughly a second at 71k — and they
hold the exclusive lock throughout, so appenders block. Tombstones are planned
for 3.1.

---

## 8. What is in this package, and what is not

| Capability | In `nanomem_standalone` |
| :--- | :--- |
| Terminal assistant (`chat.py`) | yes |
| Memory proxy for Ollama / LM Studio / vLLM / Open-WebUI | yes |
| Python SDK: `add`, `search`, `merge`, `split`, `unmerge`, `export`, CRUD | yes |
| Direct REST endpoints | yes |
| Optional password mode | yes |
| Text-hop multi-hop retrieval | yes, opt-in |
| Opt-in IVF cell router above 50,000 docs | yes, ships off — measured slower in-engine at equal recall |
| MCP server for IDE assistants | **not included**; there is no `mcp.py` in this package |
| Native image / audio embedding | **not included**; extract text first |
| Multi-tenant auth, rate limiting, cluster sync | **not included**; neither HTTP service authenticates |

Earlier revisions of this document listed MCP and multimodal embedding as
"Reserved for Pro (v2)". They are simply absent; nothing in this repository
implements them.

---

## 9. Where the numbers come from

| Claim | File |
| :--- | :--- |
| recall, p50/p95, index size at 10k and 71,433 docs | `evidence/scale_results_v3r4.json` |
| 0.1.x engine baseline on the same corpora | `evidence/scale_results_current_engine.json` |
| 1,190 and 4,760 docs vs exhaustive and FAISS; write path; index bytes/doc | `evidence/headtohead_v3.json` |
| ordering identical to exhaustive fp32; filter exactness | `evidence/exactness_v3r2.json` |
| resident memory per document | `evidence/rss_v3r4.json` |
| delete / rewrite cost | `evidence/rewrite_cost_v3r4.json` |
| password-mode overhead and the exact size leak | `evidence/crypto_overhead_v3r3.json` |
| router recall gate and in-engine latency | `evidence/router_gate_v3r3.json` |
| write-gate accuracy, in and out of sample | `evidence/write_classifier_v2_results.json` |
| chat ranking, before and after | `evidence/clean_chat_results_current_engine.json`, `clean_chat_results_heldout_baseline.json`, `clean_chat_results_v3r4_engine*.json` |
| multi-hop | `evidence/multihop_texthop_v3r2_1190.json`, `experiment_4d_bridge_results.json` |

---

*Specification 3.0.3, verified against `nanomem_standalone/` at package version
0.3.0.*
