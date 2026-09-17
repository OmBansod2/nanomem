# nanomem

[![CI](https://github.com/OmBansod2/nanomem/actions/workflows/ci.yml/badge.svg)](https://github.com/OmBansod2/nanomem/actions/workflows/ci.yml)

**An embedded store for facts that change.**

A fact your application remembers is not a document. It gets corrected — people
move, change jobs, switch phone numbers. A vector store keeps both statements
and returns whichever one is worded closer to the question, which is how an
assistant ends up confidently repeating an address you left two years ago.
nanomem keeps the chain and knows which end of it is current.

```bash
pip install nanomem
```

```python
import time
from nanomem import Vault

DAY, now = 86400, time.time()
job = {"entity": "employer"}

v = Vault("memory.dat")
v.add("I work at Acme Corp.",                    metadata=job, timestamp=now - 300*DAY)
v.add("I moved jobs, I now work at Initech.",    metadata=job, timestamp=now - 155*DAY)
v.add("I switched again, I work at Globex now.", metadata=job, timestamp=now - 10*DAY)

print(v.search("where do I work")[0]["text"])
# I switched again, I work at Globex now.

for r in v.history("where do I work"):
    print(r["revision"], r["superseded"], r["text"])
# 1 True I work at Acme Corp.
# 2 True I moved jobs, I now work at Initech.
# 3 False I switched again, I work at Globex now.

print(v.search("where do I work", as_of=now - 200*DAY)[0]["text"])
# I work at Acme Corp.

f = v.volatility()[0]
print(f["entity"], f["n_revisions"], round(f["median_interval"]/DAY))
# employer 3 145
```

One file on disk. One runtime dependency (`numpy`). No server, no daemon, no
index to rebuild. Search is **exact** — a full cosine scan, not an approximate
index — so recall is 100% by construction and every interesting question is
about time rather than ranking.

## Tell it what an attribute is

`metadata={"entity": "employer"}` is doing real work above, and it is worth a
paragraph because little else here matters as much.

Name the attribute and nanomem knows those three statements are one fact, so it
keeps them as a chain. Leave it out and a lexical tagger guesses from the text —
measured recall 70 of 100 on plainly-worded chains and **0 of 100 on narrative
phrasing**. In the example above it tags "I moved jobs, I now work at Initech."
as `location` rather than `career`, because "moved" outweighs "work at", and the
chain silently splits in two.

So if your application has attributes of its own, declare them. Everything
nanomem does that a vector store does not rests on knowing which statements are
about the same thing — and you know that, while the tagger is guessing.

## Use it from Claude, Cursor or Zed (MCP)

```bash
pip install nanomem
```

Then add this to your MCP client's config — `claude_desktop_config.json` on
macOS lives at `~/Library/Application Support/Claude/`:

```json
{
  "mcpServers": {
    "nanomem": {
      "command": "python3",
      "args": ["-m", "nanomem.mcp", "--vault", "/absolute/path/to/memory.dat"]
    }
  }
}
```

Restart the client. It gains seven tools: `nanomem_add`, `nanomem_search`,
`nanomem_history`, `nanomem_as_of`, `nanomem_changes`, `nanomem_volatility`,
`nanomem_stats` — so the assistant can ask what a fact USED to be, what the
memory believed at a past moment, what changed last week, and which of its own
beliefs have gone stale.

The vault is an ordinary file. Point the CLI or a Python script at the same path
to read what the assistant wrote.

---

Package 0.6.6 · engine 3.3.3 · container format 3 · arena cache format 3.

**Licence: AGPL-3.0-or-later, or a commercial licence.** Free for personal,
academic and open-source use, and for running internally on your own machines.
If you offer nanomem to users over a network, AGPL section 13 requires you to
offer them your source — see [COMMERCIAL-LICENSE.md](https://github.com/OmBansod2/nanomem/blob/main/COMMERCIAL-LICENSE.md) for
the alternative. Up to 0.6.0 the wheel metadata said Apache-2.0 while the
LICENSE file said All Rights Reserved; that contradiction is resolved here and
the superseded terms are kept in `LICENSE.preview-v1.0.md`.


---

## Run the demo

```bash
cd nanomem_standalone
python3 demo.py
```

It stores a few facts, updates one of them to show revision handling, searches
with citations, and prints the vault's real `stats()` — document count, file
size, the measured `active_heap_ram_kb`, and whether the file is encrypted (by
default it is not).

## Run the tests

```bash
python3 -m pytest -q
```

521 tests, no network needed.

There is no `test_security.py`. Earlier versions of this README told you to run
one to "prove that zero plaintext exists on disk"; that file never existed, and
the claim was wrong anyway — **a vault is a plaintext file unless you give it a
passphrase**. To check for yourself:

```bash
python3 -m nanomem.cli init demo.dat
python3 -m nanomem.cli add "the office wifi password is hunter2" --vault demo.dat
strings demo.dat | grep hunter2          # plaintext vault: it is there
python3 -m nanomem.cli rekey --vault demo.dat --new-password-stdin
strings demo.dat | grep hunter2          # password mode: it is not
```

---


## Use it from your own script

```python
from nanomem import Vault

with Vault("my_knowledge.dat") as vault:            # plaintext by default
    doc_id = vault.add("Server backup runs daily at 02:00 UTC")
    vault.add("The staging database is on port 5433")

    hits = vault.search("When does the backup run?")
    print(hits[0]["text"], hits[0]["cosine"])
```

`add()` returns the document id, so you can `get`, `update` or `delete` by it
later. For an encrypted vault, pass `password="…"` (or set `NANOMEM_PASSWORD`).

**Embeddings are yours to choose.** The default is `nomic-embed-text` on a
local Ollama-compatible daemon, but any model of any width works — the width is
probed from the model itself, and the vault is sized from what it returns:

```python
Vault("m.dat", embed_model="all-minilm")                    # 384-d
Vault("m.dat", embed_model="mxbai-embed-large")             # 1024-d
Vault("m.dat", embed_model="text-embedding-3-small",
      base_url="https://api.openai.com/v1/embeddings")      # 1536-d
Vault("m.dat", embedder=MyOwnEmbedder())                    # anything with
                                                            # .embed/.embed_batch/.dim
```

Point it anywhere with `base_url=` or `NANOMEM_EMBED_URL`. Pass `dim=` to
`EmbeddingProvider` to skip the probe entirely (air-gapped installs). An
existing vault's width always wins over the one you request, so you cannot
silently corrupt a vault by naming a different model later — you get a warning
and the file's own width.

**Without a daemon** there is a fallback, and it is worth knowing what it is:
a deterministic *lexical* encoder at 768-d — words and character trigrams
hashed, no semantics. It scores `"the server is up"` against `"the server is
down"` at **0.783**, and `"I drive a car"` against `"I own an automobile"` at
**0.286**. Opposites look identical, synonyms look unrelated. It keeps the
pipeline running offline and is fine for a smoke test; it is not a substitute
for an embedding model, and there are no bundled neural weights.

---

## What makes it different from a vector store

nanomem is an **append-only log**, so it keeps every value a fact has ever had,
not just the current one. That makes four questions answerable that a vector
index cannot represent, because none of them keeps the history to answer from.

```python
with Vault("memory.dat") as v:
    v.history("where do I work")          # every value, oldest first, current last
    v.search("where do I work",
             as_of=1735689600.0)          # the answer as the memory stood back then
    v.changes(since=1735689600.0)         # what was written in a window, no query
    v.volatility()                        # how often each fact actually changes
```

`as_of` and `since` are unix timestamps in Python; the CLI below takes
`YYYY-MM-DD` as well. `history()` returns each value with its timestamp and a `superseded` flag; the
last entry is the current one. A fact that never changed returns one entry,
which is an answer, not an empty result.

`volatility()` is the one to look at. It reads the revision log and reports, per
fact, how many times it has been restated, the typical interval between changes,
and how long the current value has stood unconfirmed — so an agent can work out
which of its own beliefs have gone stale and ask again:

```
fact             restated   changes every   last confirmed
mobile_phone            4          195 d           2237 d   ← ask again
employer                3          807 d             47 d
```

No model and no query: differenced timestamps off two resident columns, off the
search path. `staleness()` will turn that into a probability, but it returns
`None` unless you pass `assume_memoryless=True` — the per-fact rate only beat a
single corpus-wide rate on data generated to match its own assumption, so it is
not on by default (`scratch/refound/staleness_calibration.json`).

**What bounds this.** Two records are only treated as one fact when nanomem can
tell they are the same attribute, and that needs either a vocabulary or the
query — five vocabulary-free signals were measured and all are at chance against
sibling attributes like home-vs-office address
(`scratch/refound/grouping_signal_results.json`). With canonical phrasing the
tagger groups 70 of 100 chains; with narrative phrasing ("ported the line over
the weekend, reach me on …") it groups 0 of 100. **If your application knows its
own attributes, declare them** — pass `metadata={"entity": "employer"}` on write
— and set `group_floor_sim=0.45`, worth +19.0 points of top-1 on exactly the
phrasing the tagger struggles with (`floor_retune_results.json`). Leave both
alone if you are relying on the tagger; the same setting costs 13.9 points there.

---

## Building RAG on it

Retrieval is the core; generation is optional and uses no extra dependency
(stdlib `urllib` to whatever endpoint you point it at).

```python
v.search("what port does staging use?", top_k=3)     # retrieval only
v.ask("What port does staging use?", llm="llama3.2") # -> {question, answer, citations}
v.ingest_file("runbook.md")                          # chunk + store a document
```

`nanomem proxy` additionally serves an OpenAI-compatible `/v1/chat/completions`
that injects memory, so an existing app can point at nanomem instead of its
provider and gain memory without code changes.

**Providers.** Pass `base_url=` and `api_key=` to `ask()`/`chat()`, or set them
on the `Vault`. The base URL decides the request shape:

| | |
| --- | --- |
| Ollama (default) | `http://localhost:11434` → native `/api/generate` |
| OpenAI, Groq, Together, Mistral, DeepSeek, OpenRouter, Fireworks | any base ending `/v1` |
| LM Studio, vLLM, llama.cpp | `:1234`, `:8000`, `:8080` |
| Anthropic | `https://api.anthropic.com/v1` → `/v1/messages`, `x-api-key` |
| Azure OpenAI | pass the full deployment URL ending `/chat/completions` |

Embeddings come from `NANOMEM_EMBED_URL` (default `nomic-embed-text` on
Ollama). Routing is decided by the PARSED PORT, not by a substring of the URL —
before 0.6.1 a host named `web8000.internal` was misrouted and Anthropic was
POSTed to `/v1/chat/completions`, which 404s.

---


---

## CLI

```bash
python3 -m nanomem.cli init company.dat
python3 -m nanomem.cli add "DB port is 5433" --vault company.dat
python3 -m nanomem.cli search "DB port" -k 3 --vault company.dat
python3 -m nanomem.cli history "DB port" --vault company.dat
python3 -m nanomem.cli search "DB port" --as-of 2026-03-01 --vault company.dat
python3 -m nanomem.cli changes --since 2026-01-01 --vault company.dat
python3 -m nanomem.cli stats --vault company.dat
python3 -m nanomem.cli rekey --vault company.dat --new-password-stdin
```

`add` prints the document id and its own elapsed time.

---

## What it does, measured

Search is an **exact linear scan**. It returns what an exhaustive fp32 cosine
scan returns — 0 of 120 top-4 order differences on a 1,190-document corpus
(`scratch/refound/exactness_v3r2.json`) — and its latency therefore grows with
the corpus.

| Corpus | p50 | recall@4 | same as exhaustive numpy? | same as FAISS flat? | index | RSS per doc |
| ---: | ---: | ---: | :---: | :---: | ---: | ---: |
| 1,190 | 0.052 ms | 68.3 % | yes | yes | 2.5 MB | — |
| 10,000 | 0.345 ms | 70.4 % | yes | yes | 22.0 MB | 8.4 KB |
| 71,433 | 1.762 ms | 60.6 % | yes | yes | 155.9 MB | 8.8 KB |

`scratch/refound/scale_results_v3r4.json`, `headtohead_v3.json`,
`rss_v3r4.json`. Measured on Apple M4 Pro / Python 3.12, `nomic-embed-text`
768-d, random insertion order, 500 held-out questions, ingest → close → re-open
→ search. (Every `scratch/refound/…` path in these documents is relative to the
repository root, not to this folder; the JSONs and the scripts that wrote them
live there.)

The "same as exhaustive" columns are recall. Ordering is identical too at 1,190
documents (0 of 120 top-4 order differences); at 10,000 and 71,433 a few
tie-breaks differ — 2 of 500 and 5 of 500 questions, largest cosine gap 2.2e-05 —
because the on-disk vectors are fp16 (`scratch/refound/verify_round3_v3r3.json`).

Resident memory is roughly **8–9 KB per document**. Earlier documentation claimed
a "< 500 KB RAM" or "160 KB active heap" footprint; those were constants printed
by `stats()`, not measurements, and they are gone.

Writes are flat as the vault grows: 19.81 µs mean over the first 500 of 4,000
adds, 19.32 µs over the last 500, excluding embedding
(`scratch/refound/headtohead_v3.json`).

Deleting one record is a **full atomic rewrite** — 7.5 ms at 1,000 records, 70.5
ms at 10,000 (`scratch/refound/rewrite_cost_v3r4.json`). There are no tombstones
in 3.0.

---

## Documentation

* [USER_MANUAL.md](https://github.com/OmBansod2/nanomem/blob/main/USER_MANUAL.md) — hub, and the full measured-performance tables
* [USER_MANUAL_PERSONAL.md](https://github.com/OmBansod2/nanomem/blob/main/USER_MANUAL_PERSONAL.md) — chat, profiles, privacy
* [USER_MANUAL_DEVELOPER.md](https://github.com/OmBansod2/nanomem/blob/main/USER_MANUAL_DEVELOPER.md) — SDK, CLI, concurrency, migration
* [SERVICES_AND_API_SPECIFICATION.md](https://github.com/OmBansod2/nanomem/blob/main/SERVICES_AND_API_SPECIFICATION.md) — REST endpoints and limits
* [MULTIHOP_REASONING_AND_TOPOLOGY_GUIDE.md](https://github.com/OmBansod2/nanomem/blob/main/MULTIHOP_REASONING_AND_TOPOLOGY_GUIDE.md) — the bridge, and what did not work
* [CHANGELOG.md](https://github.com/OmBansod2/nanomem/blob/main/CHANGELOG.md) — breaking changes from 0.1.x

Read `nanomem.THREAT_MODEL` before relying on the optional password mode. It is
scrypt + a SHAKE256 keystream + an HMAC-SHA256 tag, built from the Python
standard library. **It is not AES, not "256-bit encryption", and it has not been
audited.**
