# Orchestrator decisions on the open questions in core_spec.md, prime_spec.md, fourd_spec.md

These are binding. Where a spec and this file differ, this file wins. Where this file is silent, follow the spec.

## Core engine v3 (core_spec.md)

1. **Sub-linearity target renegotiated.** Ship the measured T(N) statement: exhaustive is exact and linear (≈0.4 ms @10k, ≈1.6 ms @50k, ≈3.3 ms @100k on this machine); routing above `n_exhaustive` is a constant-factor win, not asymptotic. No level-2 router in 3.0. Docs must print the measured table and the fitted growth exponent, never the word "sub-linear".
2. **`n_exhaustive` default = 50,000.** Exact recall is preferred; 1.6 ms sits in front of a ~9 ms embedding call.
3. **Router above `n_exhaustive` = global spherical k-means cells (IVF-style), opt-in `router="auto"`.** `beam_frac` chosen by the sweep with tolerance "95% CI lower bound ≥ −1.0 pt vs exhaustive". The page-landmark power-mean router is kept in `nanomem/routing.py` as a measured ablation only. Gate: the router engages by default above `n_exhaustive` only if the sweep shows it within 1.0 pt of exhaustive at its default budget on the 71k corpus in **random insertion order after `compact(recluster=True)`**; otherwise 3.0 ships exhaustive-only and prints the numbers.
4. **No layout gate. Auto-recluster instead.** When N first crosses `n_exhaustive` (and after any `replace_all`), the engine runs `compact(recluster=True)` once (documented one-time cost, measured and logged), because random-order blocks are unroutable (measured: 5–24% recall vs 60–70% exhaustive). Routing is only ever used on reclustered layouts.
5. **fp16 vectors on disk by default**, fp32 arena in RAM; test 28 flips the default to fp32 if the 4×-distractor set changes any top-4.
6. **scrypt n=2^16, r=8, p=1** (≈100 ms per open), parameters stored in the header.
7. **SHAKE256 keyed-XOF keystream + HMAC-SHA256 encrypt-then-MAC** confirmed. Header reserves a cipher id for HMAC-CTR; not implemented. stats()/docs must say exactly this and never "AES"/"256-bit".
8. **Score scale: cosine, now (3.0).** `score` = cosine + explicit documented boosts; `cosine` key kept; thresholds retuned at every call site: 0.25→0.42, 0.32→0.53, 0.35→0.58 (cli.py, chat.py, vault.py forget, proxy.py, server.py, ingest_and_test_book.py, benchmark_multi_questions_in_one.py, query_book.py wherever present). Provide `legacy_score_to_cosine()`.
9. **Migration rewrites junk v2 entities** (e.g. "Update") using the generic tagger, preserving the original as `metadata["entity_v2"]`; the original file survives as `<path>.v2.bak`.
10. **Revision groups scoped by (user_id, project, entity)** to match `Vault.merge`.
11. **Tombstones: 3.1.** Delete/update/prune go through `Vault._rebuild` → `engine.replace_all()` (atomic temp file + `os.replace`).
12. **Migration sidecar = `<path>.v2.bak`**; `users.py` list/delete must ignore and, on delete, remove `.v2.bak` and `.tmp-*` siblings.
13. **Compat container shim: keep a minimal deprecated `_ContainerView` (`.toc`, `.read_payload`, `.close`) for 3.0**, remove in 3.1. `vault.py` itself uses `iter_records()`/`replace_all()`.
14. **`add_fact()` / `Vault.add()` return the doc id (str).** `cli` prints id + its own timing.
15. **`stats()`**: keep the 8 legacy keys with honest values; `active_heap_ram_kb` = measured (tracemalloc current or RSS delta, say which); add `engine_version`, `vector_dtype`, `router`, `n_exhaustive`, `resident_arena_mb`.
16. **VAL-4760 reproducibility**: the 4× distractor set is `D4 = vstack([D] + [D + N(0, 0.35²) noise × 3 seeds])`, rng `np.random.default_rng(0)`, texts prefixed `[distractor-k]`; put that in the head-to-head script so the 45.8% target is reproducible.

## Prime router (prime_spec.md)

1. **Router constraint rewritten as decided above**: the shipped opt-in router is global k-means cells; page-landmark power-mean is the ablation. Both are reported in the doc with the deficit printed wherever the page router trails IVF.
2. **Sub-linearity**: accept the restatement (constant-factor speed-up with the growth exponent printed).
3. **Naming**: importable router = `nanomem_standalone/nanomem/routing.py`; research reference store = `prime_4d_unified_engine_2026_09_13/prime_memory.py` (name kept for continuity, honest docstring); the doc stays `PRIME_NONLINEAR_MEMORY.md` with a new title "Prime Nonlinear Memory — what survives measurement" and a withdrawn-claims box at the top. No renames of folders.
4. **`exhaustive_max`**: 50,000 in the nanomem core; the research store may default to 100,000 but must say so.
5. **cell_target = 25 provisional**; the T2 sweep may override it.
6. **Crypto**: as core decision 6–7.
7. **Corpora**: 1,190 validation + 71,433 train paragraphs; layouts insertion / random / reclustered; 500–1,000 train questions for the 71k tables; FAISS IVFFlat at matched scan fraction as the external baseline; paired-bootstrap CIs.

## Latent bridge / "4D" (fourd_spec.md)

1. **Primary test set**: the orchestrator's test-B (1,600 gold-disjoint held-out train questions, 3,200 pairs, 71k corpus) is already run and is the primary result **for this release**; the full validation split (7,285 questions) is being fetched and embedded in the background as `scratch/refound/hotpot_val_full.json` / `hotpot_val_full_embeds.npz`; if it exists when the 4D implementer runs, evaluate every arm on it too (pair-only leakage rule) and report both; if not, state it as follow-up.
2. **Leakage rule**: pair-only (drop a train question only if both its golds are near-duplicates (cos > 0.98 or exact) of a test question's gold pair); log counts.
3. **Effect bar**: CI lower bound > 0 keeps a mechanism opt-in; CI lower bound ≥ +1.0 pt end-to-end makes it default.
4. **ODE arm kept** (already run; 3 seeds).
5. **Loss**: full-corpus InfoNCE is allowed as the primary in a re-run if time permits; the orchestrator's own-8 + in-batch run (`experiment_4d_bridge_results.json`) is the reference and must be reported either way.
6. **No hyper-parameter tuning** beyond the pre-registered dev choices.
7. **Library changes in scope**: `search_multihop` default bridge = text-hop (see implement.js core task); alpha-steering removed (measured significantly worse than query-only); latent field NOT shipped; `bridge` hook kept.
8. **Prefix convention**: raw text (no `search_query:`/`search_document:`); report the prefixed row if measured.
9. **LLM answer metrics**: not required.
10. **Verdict (already determined by the pre-registered rule)**: no trained latent bridge beats query-only on test-B (all CIs ≤ 0 or straddling 0); RK4 ≡ residual MLP within noise; text-hop is the only bridge with a significant equal-budget end-to-end gain (+5.0 @1,190 docs, +3.1 @71k) but hurts on wrong-anchor questions; widening `top_k` is the larger lever (+9.4 @71k). The doc says exactly this.

## Classifier v2
As in implement.js: trained numpy logistic head over [nomic embedding ; generic surface features]; trained on the 3-persona set, tested once on the 2-persona held-out set (target ≥ 85% acc; shipped classifier 76.2% / F1 69.6%); surface-only fallback without Ollama; all benchmark-copied phrases removed from rule layers.
