# The relevance floor deletes the current value from its own group

Status: PRE-REGISTERED 2026-09-17, before the experiment was written and before
any number was seen. Binding order: DECISIONS.md > this file.

## The defect

`finding_floor_drops_current_value.json`, severity major, reproduced identically
on 0.5.0 and still live in 0.6.4:

    search("where do I work")  ->  "I moved jobs, I now work at Initech."
    the current value, "I switched again, I work at Globex now.", ranks THIRD

Mechanism, from that file: `GROUP_COS_DELTA = 0.06`. The group's best member is
Initech at cosine 0.6348; the current value, Globex, sits at 0.5317 -- a gap of
0.1031. `group_relevance_floor` removes it from the group BEFORE
`apply_revision_lead` runs, so the lead goes to the most recent SURVIVING
member. The newest value was never a candidate for the lead.

The trigger is ordinary: the newest restatement carries more narrative before
the value ("I switched again, I work at Globex now.") than an older one, so it
sits further from the question than a superseded statement does.

## Why the existing knob is not the answer

`floor_retune_results.json` swept `group_floor_sim`. At 0.45 it cleared every
clause of `floor_retune_spec.md` -- A +19.0, B +1.0, C +0.7 -- and still could
not ship, because the same value costs **-13.9 points** on the 3-persona chat
set. That number was measured but sat OUTSIDE the pre-registered bar, so a
change that passed its bar was rejected on evidence the bar never asked for.

That is a defect in the earlier spec, not in the earlier measurement. This spec
puts the chat arm INSIDE the bar (clause 4 below).

## Candidate fixes

**F1 -- the floor may not remove the group's newest revision.** Primary.
`group_relevance_floor` is a PRECISION guard: it exists to drop records that
merely carry the tag without being restatements (a commute note tagged
`location` is not an address). The maximum-revision member of a tagged group is
not such a record. `add_fact` assigned that number as `group_max + 1` at write
time, which is the engine's own assertion that this record restates this fact.
If including it is wrong, the GROUPING is wrong and the floor is the wrong place
to correct it. Applies only when `_revisions_comparable` holds, since a revision
number orders nothing across group keys.

Costs no new state: `rev` is already resident and already read on this path.

**F2 -- provenance-gated `group_floor_sim`.** Fallback, only measured if F1
fails. `_apply_group_floor`'s own docstring records that the floor compensates
for TAGGER PRECISION and that a caller declaring its own schema has no
imprecision to compensate for. So apply `group_floor_sim = 0.45` automatically
to groups whose entity the CALLER declared, and 0.0 to groups the tagger
inferred. This needs per-record tag provenance that survives a reload, which the
arena does not carry today -- a sidecar format change. It is second for that
reason, not because it is less principled.

## Arms

  A  DRIFTING phrasing, EXPLICIT correct tags. The failure case. Primary.
  B  CANONICAL phrasing, EXPLICIT correct tags. Must not regress.
  C  SIBLING / adjacent-attribute probes. What a looser floor risks, and the
     reason three earlier retunes were rejected.
  D  3-persona clean chat set, tags INFERRED by the lexical tagger. The arm that
     killed `group_floor_sim=0.45`.

A, B and C reuse `exp_floor_retune.py`'s harness unchanged. D reuses
`exp_floor_chatcheck.py`'s, against `clean_chat_benchmark.json`.

Metric: top-1 accuracy on `current_value` questions per arm; for D, the set's
own top-1.

## Pre-registered decision rule

Change the DEFAULT only if F1 satisfies ALL FIVE:

  1. Arm A improves by >= 10.0 points over shipped 0.6.4;
  2. Arm B regresses by <= 1.0 point;
  3. Arm C regresses by <= 1.0 point;
  4. Arm D regresses by <= 1.0 point;
  5. the named repro answers "I switched again, I work at Globex now."

Clause 5 is not redundant with clause 1: a change could lift arm A by moving
cases unrelated to the one that prompted this. If arm A rises and clause 5 still
fails, that is a different fix to a different problem and it does not ship under
this spec.

If F1 fails any clause, report the negative, restore the default, and measure F2
under this same bar. If both fail, the defect stays documented and unfixed and
this file records why -- a knob that cannot be defaulted is a worse outcome than
an honest "we could not fix this without breaking something else", but it is
better than a fix that trades the chat path for the schema path.

The bar is fixed here and is not restated afterwards.

## What this change is allowed to move

DECISIONS gate G1 (search output bitwise unchanged) does NOT apply: this
deliberately alters ranking on revision groups. What must hold instead is that
the 520-query exactness baseline moves ONLY on queries that resolve a revision
group, and the existing suite stays green.

## Quarantine

`quarantine_guard.enforce()` is the first executable statement.
`bench_temporal.audit_vocabulary` is NOT called -- it reads a quarantined
fixture from its FIXTURES list. Arm D reads `clean_chat_benchmark.json`, the
3-persona set, never `clean_chat_benchmark_persona4.json` or any `_heldout`
file.

## Amendment 1 -- arms E and F

Added 2026-09-17, AFTER F1's numbers on arms A-D were seen and BEFORE any
number on the new arms was seen. Recorded rather than folded in silently,
because amending a pre-registered bar after a partial result is exactly the move
this method exists to prevent. Two things make it admissible:

  * it can only make the bar HARDER -- two more clauses, no clause relaxed, no
    threshold moved;
  * it does not touch the arms F1 has already been scored on.

The gap. Both candidates work by protecting a tagged group's NEWEST revision
from the floor. `historical_value` ("what was my original address?") and
`previous_value` ("where did I work before?") are the two question types whose
correct answer is explicitly NOT the newest revision. The canonical set carries
100 and 56 of them. `score_schema_arm` filters to `q["type"] ==
"current_value"`, so arms A, B and C never scored a single one, and arm D's
chat queries are current-value in the main. A change that lifts current-value
top-1 by breaking historical recall would have passed every clause of this spec
as written.

  E  `historical_value`, canonical phrasing, explicit correct tags.
  F  `previous_value`, canonical phrasing, explicit correct tags.

  6. Arm E regresses by <= 1.0 point;
  7. Arm F regresses by <= 1.0 point.

Note for F2. Provenance is perfectly confounded with arm: A, B, C, E and F
declare `metadata["entity"]`, arm D never does. So F2's scores on these six arms
are determined in advance -- it is F1 wherever the entity is declared and
shipped behaviour where it is inferred. Running F2 on them measures nothing.
F2 is therefore judged on whether F1 clears clauses 1, 2, 3, 5, 6 and 7 (the
declared arms plus the repro) while clause 4 is satisfied by construction. If F1
fails any DECLARED-arm clause, F2 inherits that failure and neither ships.

## Amendment 2 -- candidate F3, and why F2 is deferred

Added 2026-09-17, before F3 was implemented and before any F3 number was seen.

F2 costs more than the spec assumed. Provenance has to be readable on the block
ingest path, and that path (`arena._ingest`, line ~2216) sees only
`rec.groups[i]` -- the group key bytes. The record metadata that carries
declaredness sits in `rec.docs`, whose per-record JSON would have to be parsed
on every block load. So F2 needs either a CONTAINER format bump to add a column,
which is migration work and not a patch release, or the marker folded into the
group key, which splits a declared chain from an inferred one for the same
attribute and breaks `entity_id` interning (the entity is recovered as
`gkey.rsplit("\x1f", 1)[-1]`). Neither belongs in 0.6.5.

F3 needs no new state and is not confounded with the arms.

**F3 -- the newest revision survives the floor IF it resembles the group.**
F1 protects the newest member unconditionally, which is why it helps where the
group is real (A +19.0) and hurts where the tagger built a bad one (D -19.4).
The difference between those two cases is observable without knowing who
assigned the tag: in a real chain the newest statement resembles the others
whatever the question was worded like, and in a bad group it does not. That is
the same evidence `_cosine_window` already uses to group untagged records, and
the same evidence `group_floor_sim` applies to every member -- F3 applies it to
ONE member, the one the revision counter already identifies as current.

Vectors for the group are already fetched on this path, so the cost is one dot
product against the group lead.

Sweep, fixed here: `current_keep_sim` in {0.40, 0.50, 0.60, 0.70, 0.80}.

  8. F3 is judged on clauses 1-7 exactly as written above, with no clause
     relaxed and arms E and F included from the start.
  9. If more than one threshold clears all seven, ship the one with the highest
     arm A. If arm A ties, the LOWER threshold, because a lower threshold keeps
     more true revisions and the failure it risks is already bounded by clause 4.
 10. If no threshold clears all seven, report the negative. The defect then
     stays documented and unfixed in 0.6.5, F2 is scheduled behind a container
     format change, and this file records that the cheap fix was tried and
     failed rather than that it was never attempted.
