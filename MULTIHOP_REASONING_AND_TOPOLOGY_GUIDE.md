# nanomem — Multi-Hop Retrieval and Vault Topology

What nanomem does with bridge questions ("which university did the creator of X
attend?"), what was measured, what was tried and discarded, and what happens to
any of it when vaults are merged or split.

Engine 3.0.3. Every number names the results JSON it comes from.

---

## 1. Summary

* The shipped bridge is a **text hop**: re-embed the question together with the
  text of the hop-1 winner, search again, and merge the two result lists under
  the same `top_k` budget. It is **opt-in** (`search(multihop=True)` or
  `search_multihop(...)`).
* Measured, equal budget, 1,190 documents, evidence recall@4
  (`scratch/refound/multihop_texthop_v3r2_1190.json`): **68.3 % → 79.2 %** at
  `top_k=4`, and 90.0 % → 92.5 % at `top_k=8`.
* **Simply raising `top_k` is the larger lever**: single-pass `top_k=8` scores
  90.0 % against 79.2 % for the bridge at `top_k=4`.
* The bridge **hurts when the hop-1 anchor is wrong**, which is why it stays
  opt-in rather than becoming the default.
* **No trained latent bridge beat the query alone.** Linear residual, residual
  MLP and an RK4 neural ODE, three seeds each, all had confidence intervals at or
  below zero on the primary held-out set. None ships.
* **Alpha-steering (`q + α·v_anchor`) is removed.** It measured significantly
  worse than the query alone, 95 % CI [−0.050, −0.033] on MRR. `alpha`,
  `num_hops` and `beam_width` are still accepted as arguments for compatibility
  and no longer change the result.

---

## 2. The problem

> *"Which university did the creator of Python graduate from?"*
>
> * Bridge document D₁: "Guido van Rossum released Python in 1991."
> * Target document D₂: "Guido van Rossum received a master's degree from the
>   University of Amsterdam in 1982."

A single query vector is pulled toward "creator of Python". D₁ scores well; D₂
mentions neither "creator" nor "Python" and is never retrieved. This is distinct
from a **composite** question ("what is A, and how does B work?"), where the
sub-questions are independent — those are handled by query decomposition
(`decompose=True`, on by default), not by hopping.

---

## 3. What ships: the text hop

```python
hits = vault.search_multihop("Which university did the creator of Python attend?",
                             top_k=4)
# equivalently: vault.search(..., multihop=True)
```

Pass 1 is an ordinary search. Pass 2 embeds `f"{question}\n{hop1_text}"` and
searches with that vector. The two lists are merged as the top `ceil(k/2)`
single-pass hits plus the top `floor(k/2)` bridge hits, ranked by cosine and
de-duplicated by id — so the caller's budget is still `k` documents and the
comparison against single-pass is fair.

Pass 2 costs one extra embedding call (roughly 9–16 ms to a local
`nomic-embed-text`) and one extra scan (0.052 ms at 1,190 documents, 1.762 ms at
71,433 — `scratch/refound/scale_results_v3r4.json`). The embedding call dominates.
It burns no LLM tokens, but it is not free and it is not "under 2 ms": earlier
revisions of this guide said both.

### Measured, in the shipped engine

`scratch/refound/multihop_texthop_v3r2_1190.json` — 1,190 HotpotQA paragraphs,
120 real questions, re-opened vault, equal budget, evidence recall@4 (every gold
paragraph must appear):

| Budget | single pass | text hop |
| :--- | ---: | ---: |
| `top_k=4` | 68.3 % | **79.2 %** |
| `top_k=8` | 90.0 % | 92.5 % |

### Measured in the research harness, on the same corpus

`scratch/refound/experiment_4d_bridge_results.json`, arm `text_hop_reembed`, with
its own merge rule (top-2 query ∪ top-2 bridge against single-pass top-4):

| Corpus | single pass R@4 | text hop R@4 | wider lever: single pass R@8 |
| :--- | ---: | ---: | ---: |
| 1,190 docs (test-A, n=120) | 68.3 % | 73.3 % (+5.0) | 89.2 % |
| 71,433 docs (test-B, n=1,600) | 60.4 % | 63.5 % (+3.1) | 69.8 % |

McNemar discordant pairs at 71,433: 167 wins for the bridge against 117 losses.

**The two runs disagree on the size of the gain at 1,190 documents: +10.9 points
in the engine, +5.0 in the research harness.** Both used the same corpus and the
same 120 questions; the merge and de-duplication details differ. Treat +5 points
as the conservative figure and the engine's +10.9 as specific to its merge rule.

### Where it hurts

On the questions whose hop-1 anchor is wrong, the bridge makes things worse, by
construction — it steers toward a document that is not on the path:

| | single pass R@4 | text hop R@4 |
| :--- | ---: | ---: |
| wrong-anchor subset, 1,190 docs (n=12) | 25.0 % | **0.0 %** |
| wrong-anchor subset, 71,433 docs (n=218) | 25.2 % | **14.2 %** |

`scratch/refound/experiment_4d_bridge_results.json`. There is also a metric on
which the bridge is plainly worse: scored as a *retriever of the gold pair*, the
re-embedded query has MRR 0.694 against 0.727 for the query alone, 95 % CI
[−0.062, −0.009] — significantly worse. The end-to-end gain comes from the union
of two different result lists, not from the bridge query being a better query.

That is the whole case for keeping it opt-in.

---

## 4. What was tried and did not work

The pre-registered rule (`scratch/refound/design/DECISIONS.md`, 4D §3): a
mechanism stays opt-in only if the 95 % CI lower bound is above zero, and becomes
the default only if the end-to-end CI lower bound is at least +1.0 point.

Protocol: 5,776 training questions, 300 dev, **1,600 gold-disjoint held-out
questions over the 71,433-document corpus (test-B)** as the primary set, plus the
120-question 1,190-document set (test-A). Leakage rule: a training question is
dropped only if both of its gold paragraphs are near-duplicates (cosine > 0.98 or
exact) of a test question's gold pair. Paired bootstrap CIs on MRR against the
query-only baseline.

| Arm | test-B MRR | 95 % CI vs query-only | Verdict |
| :--- | ---: | :--- | :--- |
| query only (baseline) | 0.7272 | — | **this is what to beat** |
| bridge document only | 0.1951 | [−0.550, −0.515] | far worse |
| mean of query and bridge | 0.5244 | [−0.217, −0.188] | far worse |
| α-steering, α = 0.35 (0.1.x shipped default) | 0.6861 | [−0.050, −0.033] | **significantly worse — removed** |
| α-steering, α = 0.10 (best on dev) | 0.7252 | [−0.006, +0.002] | no effect |
| ridge map q,D₁ → q_D (λ = 10) | 0.7248 | [−0.008, +0.004] | no effect |
| trained linear residual, 3 seeds | 0.6905 | [−0.048, −0.026] | worse |
| trained residual MLP, 3 seeds | 0.7094 | [−0.032, −0.006] | worse |
| RK4 neural ODE, 3 seeds | 0.7217 | [−0.021, +0.008] | no effect |

`scratch/refound/experiment_4d_bridge_results.json`.

Three conclusions, all of them negative and all of them stated as such:

1. **No trained latent bridge beat the query alone.** Every interval is at or
   below zero. Under the decision rule none of them ships, not even opt-in.
2. **The RK4 neural ODE is indistinguishable from a residual MLP,** and both are
   indistinguishable from doing nothing. The extra machinery bought nothing on
   this task.
3. **α-steering was actively harmful at its old default** and is gone from the
   code. The 0.1.x manuals presented α = 0.35 as the "balanced optimal default";
   it was the worst non-degenerate arm measured here.

The `bridge` hook remains: any object with
`bridge(q_vec, d1_vec, question, d1_text) -> unit vector` can be passed to
`search_multihop(bridge=...)`, so a future latent field can be attached without
editing the engine. The default is `TextHopBridge(self.embedder)`.

Earlier documentation cited "+16.7 % EM on a 400-document synthetic corpus" from
an MLX research prototype. That measurement was of a different system on
synthetic data and says nothing about `nanomem.Vault`; it is withdrawn from these
docs.

---

## 5. Retrieval order is the same as an exhaustive scan

Neither hop uses an approximate index below `n_exhaustive` (50,000). Both are
exact fp32 scans, and that is measured: 0 of 120 top-4 order differences and 0 of
120 rank-1 differences against exhaustive fp32 cosine on a re-opened 1,190-document
vault with the real question strings (`scratch/refound/exactness_v3r2.json`). So
any difference you see between single-pass and multi-hop is the bridge, not
index noise.

---

## 6. Merge, split and unmerge

These are mechanical consequences of one corpus versus several. None of them is a
performance claim.

| Operation | Effect on bridge questions |
| :--- | :--- |
| `merge()` | Preserved. Every text, metadata dict and vector transfers exactly. Facts that were in separate files are now in one ranking, so a chain that spanned them can now be followed. |
| `split()`, `split_by_doc()` | Preserved **if** the bridge document and the target land in the same file. Severed if the partition separates them — hop 2 then searches a file that does not contain the answer. |
| `unmerge()` | Reverts to the pre-merge scope: chains inside one constituent vault still work, chains that crossed files do not. |

A merged vault also reconciles revisions chronologically: two files that each
held "revision 1" of one fact become revisions 1 and 2 of one group, scoped by
`(user_id, project, entity)`.

If files must stay separate — role-based access, tenant isolation — do the hop in
application code:

```python
def federated_bridge(query, public_vault, restricted_vault):
    hop1 = public_vault.search(query, top_k=1)
    if not hop1:
        return []
    bridge = hop1[0]
    hop2 = restricted_vault.search(f"{query}\n{bridge['text']}", top_k=1)
    return [bridge] + hop2
```

That is the same mechanism as `search_multihop`, split across two files, and it
inherits the same caveat: if `hop1` is wrong, hop 2 is worse than not hopping.

Note also that `Vault.search_multi()` sorts hits from separate files naively.
Cosine is relative to one corpus, so a 0.78 in a 50-document file is not
comparable to a 0.78 in a 10,000-document file.

---

## 7. Recipes

```python
from nanomem import Vault

with Vault("space_program.dat") as vault:
    vault.add("The Apollo 11 flight director was Gene Kranz.",
              metadata={"category": "history"}, source="archive")
    vault.add("Gene Kranz received the Presidential Medal of Freedom.",
              metadata={"category": "honors"}, source="records")

    # single pass: likely returns only the first record
    vault.search("What medal was awarded to the Apollo 11 flight director?", top_k=2)

    # text hop: pass 2 re-embeds the question with the first record's text
    vault.search_multihop("What medal was awarded to the Apollo 11 flight director?",
                          top_k=4)
```

Grounded answer:

```python
res = vault.ask("What medal was awarded to the Apollo 11 flight director?",
                llm="llama3.2:3b", multihop=True)
print(res["answer"], res["citations"])
```

---

## 8. When to use which

1. **Try `top_k=8` before you try hopping.** It is the larger measured lever
   (90.0 % vs 79.2 % at 1,190 documents) and it costs no extra embedding call.
2. **Use the text hop for genuine bridge questions** where the answer document
   does not mention the terms in the question.
3. **Do not use it when hop 1 is unreliable.** On wrong-anchor questions it cost
   11 to 25 points.
4. **Keep related documents in one vault.** Splitting partitions the search
   space; a chain cannot cross a file boundary inside one call.
5. **Do not expect a latent shortcut.** Everything trained to replace the second
   embedding call measured at or below the query alone.
