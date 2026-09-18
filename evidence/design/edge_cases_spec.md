# Edge cases — pre-registration

Written BEFORE `exp_edge_cases.py` exists. Same rules as
`new_usecases_spec.md`: negatives are deliverables, no bar moves after a number
is seen, every reported number appears in `edge_cases_results.json`.

## What is being hunted

The established failure class in this project is **a confident wrong answer, not
an error**. So the sweep targets inputs where nanomem could plausibly answer
rather than refuse, ranked by (likelihood in real use x damage if wrong).

| # | Edge case | Why it could bite |
|---|-----------|-------------------|
| 1 | Backfilled revision (older row written AFTER a newer one) | history import, migration, replay. If `revision` is assigned by INSERTION order but ranking reads timestamps, the two disagree |
| 2 | Two revisions at the SAME timestamp | batch import; ties have to break somehow |
| 3 | Timestamp in the future | clock skew, bad parsing |
| 4 | Zero / negative timestamp | a missing field defaulting to 0 |
| 5 | Empty and whitespace-only text | user input |
| 6 | Unicode: emoji, CJK, RTL, combining marks | any real corpus |
| 7 | One enormous token, no spaces | minified blobs, base64 |
| 8 | Re-using an explicit `id` | idempotent ingestion |
| 9 | `as_of` before the first write / far future | a UI date picker |
| 10 | Odd metadata types (nested, list, None, bool, numpy) | JSON round-trip |
| 11 | Degenerate `entity` (empty string, spaces, very long) | app-supplied schema |
| 12 | Reopen after `prune` | the arena cache is rebuilt from a rewritten file |
| 13 | Two `Vault` objects on one file in one process | a web app with no singleton |
| 14 | `top_k` of 0, negative, or larger than the corpus | pagination maths |

## Bars

Each case is scored PASS if nanomem either answers correctly or refuses with a
clear error naming the problem. It is scored FAIL if it returns a plausible
wrong answer, loses data, or raises something that does not name the cause
(`TypeError: float() argument...` is a FAIL, as it was in 0.7.2).

Case 1 additionally requires: after backfilling an OLDER revision, the current
value must still be the newest BY TIME, and `history` must report all of them in
time order.

## Predictions (recorded before measurement)

- **1 is the most likely real defect.** I expect `revision` to be assigned by
  insertion order, so a backfilled older row gets the HIGHEST revision number and
  is then treated as current. If so it is the same class as 0.7.1 and 0.7.3.
- 5, 9, 14 pass — these look like paths with existing tests.
- 6, 7 pass; embedding and storage are byte-oriented.
- 12 is the second most likely: `prune` rewrites the file, and a stale arena
  cache after a rewrite has been a bug in this project before.
- 8 I genuinely cannot predict; `add(id=...)` may overwrite, duplicate or raise.
- 13 I expect to be at best undefined; I expect no explicit guard.
