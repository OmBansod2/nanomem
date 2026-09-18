# The question's wording outranks the data — pre-registration

Written BEFORE any fix exists.

## The defect

Third finding of the second black-box review, reproduced. In a vault holding
four DECLARED attributes, three of four "where…" questions return the wrong
attribute's chain, on the default path and with `decompose=False`:

```
where do I train  -> home_address  score 0.9480  cos 0.4338   (gym    won cosine at 0.7015)
where do I study  -> home_address  score 0.9517  cos 0.4009   (school won cosine at 0.6890)
where is my desk  -> home_address  score 0.9674  cos 0.3929   (office won cosine at 0.6732)
where do I live   -> home_address  score 1.0950  cos 0.4569   correct
```

The correct chain won on RAW COSINE by 0.23–0.28 in every failing case. This is
not a retrieval failure; a boost applied to the wrong group overrode it.

## Mechanism, established by instrumenting

```
query_intents("where do I train")  -> ('location',)
query_intents("where is my desk")  -> ('desk', 'location')
matching_ids("location", [...])    -> home_address
```

The word **"where"** alone drives the intent to `location`, which is an alias
for `home_address`; what is being asked about is never consulted. For "where is
my desk" the better candidate `desk` IS proposed first, but it matches no entity
in this vault, so `_resolve_intent` falls through to `location`. `intent_boost`
(+0.25) plus `group_hoist` (+0.25) then makes that choice decisive.

`history` inherits it: 0.7.14 made the chain the right LENGTH, and it is still
the wrong attribute's chain.

## The proposed rule

A generalisation of the rule shipped in 0.7.14, which was "an intent that names
nothing in this vault is not evidence about it":

> An intent whose best-matching record loses to the overall best hit by more
> than `INTENT_MARGIN` cosine is not evidence either. Fall back to the best
> hit's own tag.

`INTENT_MARGIN` is swept, not chosen. It must be large enough not to fire on
ordinary phrasing drift and small enough to catch 0.23.

## Bars

- **B1 (the defect).** The four-chain probe reaches 4/4 on the default path and
  with `decompose=False`, and `history` returns the right attribute's chain for
  each.
- **B2 (the arm this risks, and the reason to expect failure).** Arm A of
  `exp_floor_current_value.py` — drifting phrasing, declared — is the case where
  the correct record IS worded far from the question. That is precisely what
  this rule treats as evidence the intent is wrong. **A must not drop.** If no
  margin satisfies B1 and B2 together, the rule is wrong and the finding needs a
  different fix.
- **B3.** Arms B, C, D, E, F must not drop by more than 1.0 point, the bar those
  arms were originally judged on.
- **B4.** `adjacent_attributes_v3r4` unchanged — the shipped 40/40 and the
  tag-check cost of 40 vs 30.
- **B5.** Clean-chat v3r2 ≥ 97.2% in-session top-1 / 100.0% top-3 / 94.4%
  end-to-end.
- **B6.** 520-result ranking baseline: every difference explained, not tolerated.
- **B7.** Fuzzer clean at 12 seeds; full suite green.

## Predictions, recorded before measuring

1. **B1 is reachable.** The failing margins are 0.23–0.28 and "where do I live"
   is correct with home_address winning cosine outright, so a threshold exists
   that separates them on this probe.
2. **B2 is where this dies, and I expect it to.** Arm A went 52.0 → 71.0 at
   0.6.5 precisely by trusting a declared group over query-relative similarity.
   This rule does the opposite when the gap is wide. I expect a direct conflict
   and a sweep with no jointly satisfying value.
3. If 2 holds, the honest fix is narrower: act only where the intent matches an
   entity that NO high-scoring record carries, rather than on a cosine margin.

## What each outcome licenses

- B1 and B2 both hold at some margin, B3–B7 clean: ship that margin.
- No margin satisfies both: record the sweep as a measured dead end and try
  prediction 3's narrower rule. Do NOT widen the bar to make a margin pass.
- B1 unreachable: the boost is not the whole cause and the spec is wrong.

A negative result is a deliverable. Two of my last three diagnoses on this
review's findings were wrong, so the mechanism above is stated as what
instrumenting showed, and the sweep is what decides.

---

## Amendment 1 — recorded after measuring, before shipping

**Prediction 1 was wrong, then right.** I predicted B1 was reachable. The first
implementation corrected `resolve_top_entity` only, and B1 topped out at 2/4 on
`search` while reaching 4/4 on `history`. The cause: `intent_boost` is applied
from `intent_ids` before `_resolve_revisions` runs, so a +0.25 still landed on
the records of an intent the margin had already rejected. Correcting the group
without withdrawing the boost is half a fix. The test now runs once, before the
boost, and drops a failing intent for everything downstream. B1 then passes 4/4
at margins 0.05–0.20.

**Prediction 2 was wrong.** I expected arm A — drifting phrasing, declared — to
fall, and said so as the reason this rule would probably die. It does not move
at all (71.0 at every margin from 0.10 to 0.20), and four other arms IMPROVE:

```
            A_drift  B_canon  C_sib  D_chat  E_hist  F_prev
margin 0.00    71.0     90.0   69.3    97.2    63.0    92.9
margin 0.15    71.0    100.0   91.4    97.2    73.0   100.0
```

I did not expect this and I do not fully trust a +22 on C from a change nobody
had made before, so it is corroborated three ways rather than taken: the
harness reproduces the shipped baseline EXACTLY at margin 0.00 (A 71.0 B 90.0
C 69.3 D 97.2 E 63.0 F 92.9, matching `exp_floor_current_value.py`'s F2 row);
the adjacent-attribute corpus, the clean-chat set and the 520-result ranking
baseline are all unchanged; and C is the sibling-probe arm, which is precisely
where wording selects the wrong attribute, so it is the arm the mechanism
predicts should move.

**Why arm A is unaffected.** Arm A's difficulty is that the correct record is
worded far from the QUESTION. It is still the best-scoring record of its own
declared group, so the margin — which compares the intent's best record against
the overall best — never fires on it. My prediction conflated "worded far from
the question" with "loses to another attribute", which are different things.

**Shipped:** `intent_margin = 0.15`, the midpoint of the range clearing every
bar (0.10–0.20 on B2/B3, 0.05–0.20 on B1), so it is furthest from both edges.
