# nanomem — Personal AI Manual

nanomem gives a local AI a memory that survives closing the window. It runs on
your machine, talks to your local model, and keeps everything in one file. No
account, no cloud, no code.

Read this first: **your vault is a plain file by default.** It is not encrypted
unless you ask for a passphrase. See [Privacy and security](#privacy-and-security)
below for exactly what a passphrase does and does not protect.

---

## Quick start

Open a terminal in this folder.

**Ollama**

```bash
python chat.py
python chat.py --model llama3.2:3b     # pick a specific model
```

**LM Studio** (turn its local server on first, port 1234)

```bash
python chat.py --lmstudio
```

**vLLM** (port 8000)

```bash
python chat.py --vllm --model my-model-name
```

**Not sure what you have installed**

```bash
python chat.py --select
```

You will see a header naming the backend, the model, the vault file and whether
that vault is encrypted. If it says plaintext, it is plaintext.

---

## How the memory works

You never press save. Every turn goes through a write gate that decides whether
the sentence is a durable fact about you or conversational flux.

**Kept:** who you are and what you do, people and relationships, places, numbers
you state about yourself, plans and milestones, preferences, reflections.

**Dropped:** greetings, acknowledgements, jokes, requests to the model ("write me
a poem"), and questions.

When something is stored you see a one-line confirmation.

### How good is the gate

Measured on two personas the gate had never seen — 231 turns, 88 of which should
be stored (`scratch/refound/write_classifier_v2_results.json`):

| | accuracy | F1 |
| :--- | ---: | ---: |
| the 0.1.x gate | 76.2 % | 69.6 |
| **this release** | **90.5 %** | **87.2** |
| this release with no embedding model reachable | 87.0 % | 82.8 |

So roughly one turn in ten is still filed wrongly: 9 things stored that did not
need to be, 13 missed, out of 231. It is a filter, not a transcript. If something
matters, say it plainly as a statement of fact — short fragments and
questions are the cases it misses most (measured: it keeps 12 of 15 very short
canonical facts, missing sentences like "I have a cat.").

The gate needs your embedding model to be running for its best accuracy. If the
daemon is down it falls back to a text-only model, which is 3.5 points worse; it
does not fail.

### How it recalls

When you ask a question, nanomem scans every memory you have and returns the
closest ones. On a benchmark of scripted personal conversations, with the right
facts stored, the expected answer is the top hit:

| Persona set | before | this release |
| :--- | ---: | ---: |
| 3 personas the engine was developed against (n=36) | 50.0 % | **91.7 %** |
| 2 personas held out (n=24) | 33.3 % | **75.0 %** |

`scratch/refound/clean_chat_results_current_engine.json`,
`clean_chat_results_heldout_baseline.json`,
`clean_chat_results_v3r4_engine.json`,
`clean_chat_results_v3r4_engine_heldout.json`.

**The target for this release was 80 % on both sets and the held-out set missed
it at 75 %.** In five of the six failures the right memory *was* found — the
expected answer was in the top 3 for 23 of the 24 questions — but something
adjacent was ranked first. In one it was not in the top 3 at all.
In practice that means a follow-up question usually gets you there, and the
assistant may occasionally answer with a near-miss fact.

Speed: the search itself is well under a millisecond for a personal vault of a
few thousand memories (0.052 ms at 1,190 documents,
`scratch/refound/headtohead_v3.json`). What you actually wait for is your local
model generating the reply.

### Facts that change

If you say your phone number today and a different one next year, nanomem keeps
both and marks the newer one as the current revision. Asking "what is my phone
number" gets the current one; asking about the old one can still reach the
historical record. On the release's own revision probes the current revision is
ranked first in 14 of 16 cases, against 3 of 16 for plain similarity
(`scratch/refound/ranking_dev_r4_shipped.json`).

Statements about other people are kept separate from statements about you. "His
number is …" does not overwrite yours.

---

## In-chat commands

| Command | What it does |
| :--- | :--- |
| `/users`, `/user create <name>`, `/user switch <name>` | manage isolated profiles |
| `/switch <file.dat>`, `/vault <file.dat>` | change the active vault file |
| `/stats` | file size, document count, measured RAM, whether the vault is encrypted |
| `/inspect` | the sources and tags stored in this vault |
| `/dump` or `memory` | print every stored fact |
| `/cite` | toggle source citations |
| `/forget <topic>` | find a memory and erase it, after you confirm |
| `/ingest <file>` | read a text file into memory |
| `/import <file.dat>` | merge another vault into this one |
| `/model`, `/provider` | switch model or backend |
| `/help` | full list |
| `/quit`, `exit` | close cleanly |

### Forgetting

```text
You > /forget phone number

Found matching memory:
   • "My mobile number is 555-0143"

Permanently erase this memory? [y/N]: y
Erased.
```

The model is never allowed to delete anything on its own; you type `y`.

One thing to know: there are no tombstones in this release, so erasing one
memory rewrites the whole vault file. That takes about 7.5 ms for 1,000 memories
and 70 ms for 10,000 (`scratch/refound/rewrite_cost_v3r4.json`) — fast enough
that you will not notice it at personal scale, but it is a rewrite, not a flag.

---

## One vault or several

The default is one file, `personal_memory.dat`, for everything. That is the right
choice for daily use: you never have to think about which brain you are talking
to, and one ranking over one corpus is what the engine is measured on.

Use a separate vault when you want a hard boundary — client work, a specific
project, finances:

```bash
python chat.py                          # everyday
python chat.py --vault work.dat         # work
python chat.py --vault finances.dat     # finances
python chat.py --import old_laptop.dat  # absorb an old vault into the active one
```

### Why chat binds to exactly one vault at a time

Conversational memory is two-way: it reads facts and it also writes new ones. If
two vaults were open at once, a new fact would have no principled home — write it
to both and they drift apart; guess, and work memory gets personal details; ask
every time and the conversation stops being a conversation. So chat binds to one
vault. Switch with `--vault`, or absorb with `/import`.

Two further consequences worth knowing, both mechanical rather than measured:

* **Similarity scores are relative to one corpus.** A 0.78 in a 50-item vault and
  a 0.78 in a 10,000-item vault do not mean the same thing, so naively merging
  ranked lists from separate files can put a tangent above the fact you wanted.
* **Separate files keep separate revision clocks.** An old address in one file and
  a new one in another are both "revision 1" and neither knows about the other.
  Merged into one file, they reconcile chronologically into revision 1 and
  revision 2.

An earlier version of this manual quoted a benchmark comparing one vault against
four (586 ms vs 1,221 ms, 95.8 % factuality, 160 KB RAM). Those numbers have been
removed: they came from a benchmark whose fixture vocabulary had leaked into the
engine's own rules, and no results file in this repository supports them.

---

## Profiles from the command line

```bash
nanomem user list
nanomem user create work
python chat.py --user work
nanomem search "phone number" --user personal
nanomem user delete work
```

Deleting a profile removes its vault file and its `.v2.bak` / `.tmp-*` siblings.

---

## Using a graphical app (LM Studio, Open-WebUI, Jan)

Start the proxy:

```bash
python -m nanomem.proxy --port 5000 --upstream http://localhost:11434
```

Then point the app's API base URL at `http://localhost:5000/v1`.

The proxy listens on `127.0.0.1` by default. Its endpoints have **no
authentication**, so anyone who can reach the port can read and write your
memories. Do not bind it to `0.0.0.0` on a shared network.

---

## Privacy and security

**Nothing leaves your machine.** The vault is a local file; the only network
traffic is to the local model and embedding daemon you configured.

**Your vault is plaintext unless you give it a passphrase.** Open a plaintext
`.dat` in a text editor and you will find your sentences. `/stats` reports
`encrypted_at_rest` — believe it rather than this manual.

To turn encryption on:

```bash
nanomem rekey --vault personal_memory.dat --new-password-stdin
```

What a passphrase gets you: your text, ids, metadata and vectors become
unreadable without it, and any tampering with a stored block is detected. The
construction is scrypt plus a SHAKE256 keystream plus an HMAC-SHA256 tag, all
from the Python standard library. **It is not AES, not "256-bit encryption", and
it has not been audited.** Earlier versions of this manual said it was; that was
wrong.

What a passphrase does not get you:

* It does not hide how much you have stored. An encrypted vault is byte-for-byte
  the same size as a plaintext one (measured at 1,190 / 5,000 / 40,000 records,
  `scratch/refound/crypto_overhead_v3r3.json`), and each block states its record
  count and write time in the clear.
* It does not stop someone who can write to the file from truncating it or
  rolling it back to an older copy. Both open silently.
* It does not protect the passphrase while the program is running.
* It does not make a weak passphrase safe. One guess costs 95.67 ms, so about
  10.5 guesses per second per core.
* **If you lose it, the vault is gone.** There is no recovery key.

Cost of turning it on: about 100 ms extra when the vault opens, and nothing
measurable per search (`crypto_overhead_v3r3.json`).

## RAM

An earlier version of this manual said nanomem uses 160 KB of RAM. It does not.
Measured in a process that only opens a vault
(`scratch/refound/rss_v3r4.json`): 81.6 MB for 10,000 memories and 612.2 MB for
71,433 — about **8–9 KB per memory**. A personal vault of a few thousand
memories costs tens of megabytes, which is fine on a laptop; a 100,000-document
corpus is not a laptop-background-process workload.
