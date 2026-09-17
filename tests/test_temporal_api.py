"""The temporal primitives: `history`, `changes`, and `search(as_of=...)`.

These promote machinery the ranker already had (`_resolve_revisions` assembles a
revision group on every personal-memory query and then discards it) into calls a
caller can make. The gates they are held to are in
``scratch/refound/design/temporal_api_spec.md``; the measured evidence that
`as_of` equals a physically truncated vault is
``scratch/refound/temporal_as_of_results.json`` (G2a exact over 705 admitted-set
checks; G2b 5,670 comparisons, 0 id/order differences, 0 leaks).
"""

import os

import numpy as np
import pytest

from conftest import D, unit_rows                                  # noqa: E402
from nanomem.engine import VaultEngine

T0 = 1_700_000_000.0
DAY = 86400.0


def _corr(q, target_cos, seed):
    """A unit vector at exactly ``target_cos`` from ``q``."""
    r = unit_rows(1, seed=seed)[0]
    r = r - float(r @ q) * q
    r /= np.linalg.norm(r)
    v = target_cos * q + np.sqrt(max(0.0, 1.0 - target_cos ** 2)) * r
    return (v / np.linalg.norm(v)).astype(np.float32)


def _chain(vault_path, n=3, entity="locker_code", flush=True):
    """A vault holding one attribute restated ``n`` times, oldest first."""
    q = unit_rows(1, seed=3)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    ids = []
    for i in range(n):
        v = _corr(q, 0.80 - 0.03 * i, seed=200 + i)
        ids.append(e.add_fact(f"My {entity.replace('_', ' ')} is value-{i}.", v,
                              source="chat", metadata={"entity": entity},
                              timestamp=T0 + i * DAY))
    if flush:
        e.flush()
    return e, q, ids


# --- as_of ------------------------------------------------------------------
def test_as_of_none_is_the_untouched_path(vault_path):
    e, q, _ = _chain(vault_path)
    a = e.search("what is my locker code?", q, top_k=3)
    b = e.search("what is my locker code?", q, top_k=3, as_of=None)
    assert [(h["id"], h["score"], h["cosine"]) for h in a] == \
           [(h["id"], h["score"], h["cosine"]) for h in b]


def test_as_of_never_returns_a_record_written_later(vault_path):
    e, q, _ = _chain(vault_path, n=4)
    for k in range(4):
        cut = T0 + k * DAY
        for h in e.search("what is my locker code?", q, top_k=10, as_of=cut):
            assert h["timestamp"] <= cut


def test_as_of_returns_the_value_that_was_current_then(vault_path):
    """The point of the call: an older answer, not a filtered newer one."""
    e, q, ids = _chain(vault_path, n=3)
    now = e.search("what is my locker code?", q, top_k=1)
    assert now[0]["id"] == ids[-1]
    then = e.search("what is my locker code?", q, top_k=1, as_of=T0 + 0.5 * DAY)
    assert then[0]["id"] == ids[0]
    mid = e.search("what is my locker code?", q, top_k=1, as_of=T0 + 1.5 * DAY)
    assert mid[0]["id"] == ids[1]


def test_as_of_before_everything_is_empty(vault_path):
    e, q, _ = _chain(vault_path)
    assert e.search("what is my locker code?", q, top_k=5, as_of=T0 - 1.0) == []


def test_as_of_sees_unflushed_records(vault_path):
    """The memtable is part of the vault's state, so it is part of its history."""
    e, q, ids = _chain(vault_path, n=2, flush=True)
    v = _corr(q, 0.71, seed=77)
    new_id = e.add_fact("My locker code is value-late.", v, source="chat",
                        metadata={"entity": "locker_code"}, timestamp=T0 + 9 * DAY)
    got = {h["id"] for h in e.search("what is my locker code?", q, top_k=10,
                                     as_of=T0 + 10 * DAY)}
    assert new_id in got
    assert new_id not in {h["id"] for h in e.search(
        "what is my locker code?", q, top_k=10, as_of=T0 + 3 * DAY)}


def test_as_of_turns_the_screen_off(vault_path):
    """A screened shortlist could drop the very record an as-of query wants."""
    e, q, _ = _chain(vault_path)
    assert e._mode(None, force_exact=True) == "exhaustive"


# --- history ----------------------------------------------------------------
def test_history_returns_the_whole_chain_oldest_first(vault_path):
    e, q, ids = _chain(vault_path, n=3)
    chain = e.history("what is my locker code?", q)
    assert [h["id"] for h in chain] == ids
    assert [h["timestamp"] for h in chain] == sorted(h["timestamp"] for h in chain)


def test_history_marks_exactly_the_last_entry_current(vault_path):
    e, q, ids = _chain(vault_path, n=3)
    chain = e.history("what is my locker code?", q)
    assert [h["superseded"] for h in chain] == [True, True, False]


def test_history_of_an_unchanged_fact_has_one_entry(vault_path):
    """Not an empty result: it says the value has never changed."""
    e, q, ids = _chain(vault_path, n=1)
    chain = e.history("what is my locker code?", q)
    assert len(chain) == 1
    assert chain[0]["id"] == ids[0] and chain[0]["superseded"] is False


def test_history_max_len_keeps_the_most_recent(vault_path):
    e, q, ids = _chain(vault_path, n=4)
    chain = e.history("what is my locker code?", q, max_len=2)
    assert [h["id"] for h in chain] == ids[-2:]
    assert chain[-1]["superseded"] is False


def test_history_applies_no_boost(vault_path):
    """`cosine` here is the raw dot product, not a boosted score.

    Checked against the vectors themselves rather than against `search`, so the
    assertion does not depend on the ranker it is meant to be independent of.
    """
    e, q, ids = _chain(vault_path, n=3)
    by_id = {h["id"]: h for h in e.history("what is my locker code?", q)}
    for row in range(e.arena.n_rows):
        doc = e.arena.ids[row]
        if doc in by_id:
            truth = float(np.asarray(e.arena.vector(row), dtype=np.float32) @ q)
            assert by_id[doc]["cosine"] == pytest.approx(truth, abs=1e-6)


def test_history_keeps_a_weak_old_value_and_does_not_mislabel_current(vault_path):
    """The ranker's relevance floor must not truncate an audit surface.

    Four restatements at 0.80 / 0.77 / 0.74 / 0.71 to their own question: the
    fourth is 0.09 below the best and `GROUP_COS_DELTA` is 0.06, so the ranker's
    floor drops it. Applied to `history` that produced a chain whose LAST entry
    was the third value, reported with `superseded=False` -- a superseded value
    presented as the current one. `history` therefore does not apply the floor.
    """
    e, q, ids = _chain(vault_path, n=4)
    chain = e.history("what is my locker code?", q)
    assert [h["id"] for h in chain] == ids, "history dropped a restatement"
    assert chain[-1]["id"] == ids[-1]
    assert [h["superseded"] for h in chain] == [True, True, True, False]


def test_the_ranker_still_applies_the_floor(vault_path):
    """The opt-out is history's alone; search still floors what is not a value.

    0.6.5 exempts a DECLARED group's newest revision from the floor, so this no
    longer holds for the last member. It holds for every other one: an old
    statement phrased far from the question is still dropped, which is the
    precision the floor exists for.
    """
    q = unit_rows(1, seed=5)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    for i, c in enumerate((0.80, 0.40, 0.78)):      # the MIDDLE one is far off
        e.add_fact(f"statement {i}", _corr(q, c, seed=600 + i), source="chat",
                   metadata={"entity": "locker_code"}, timestamp=T0 + i * DAY)
    e.flush()
    rows = np.arange(e.arena.n_rows, dtype=np.int64)
    cos = e.arena.scores(q / np.linalg.norm(q))
    ent, _rv, _ts = e._columns(rows)
    mask = np.ones(rows.size, dtype=bool)
    floored = e._tagged_group(ent, mask, "locker_code", rows, cos).tolist()
    full = e._tagged_group(ent, mask, "locker_code", rows, cos,
                           apply_floor=False).tolist()
    assert full == [0, 1, 2]
    assert 1 not in floored, "a distant OLD statement is still floored"
    assert 2 in floored, "the newest revision of a declared group is not"


def test_history_on_an_empty_vault(vault_path):
    e = VaultEngine(vault_path, embed_dim=D)
    assert e.history("anything", unit_rows(1, seed=1)[0]) == []


def test_history_rejects_a_wrong_width_query(vault_path):
    e, q, _ = _chain(vault_path)
    with pytest.raises(ValueError):
        e.history("what is my locker code?", np.zeros(D + 1, dtype=np.float32))


# --- changes ----------------------------------------------------------------
def test_changes_is_half_open_on_since(vault_path):
    """`changes(t)` reports exactly what `search(as_of=t)` could not see."""
    e, q, ids = _chain(vault_path, n=3)
    got = [c["id"] for c in e.changes(T0)]
    assert got == ids[1:]
    assert ids[0] not in got


def test_changes_is_inclusive_on_until(vault_path):
    e, q, ids = _chain(vault_path, n=3)
    assert [c["id"] for c in e.changes(T0, until=T0 + DAY)] == [ids[1]]


def test_changes_is_ordered_oldest_first(vault_path):
    e, q, ids = _chain(vault_path, n=4)
    ts = [c["timestamp"] for c in e.changes(T0 - 1.0)]
    assert ts == sorted(ts)


def test_changes_reports_the_attribute_and_revision(vault_path):
    e, q, ids = _chain(vault_path, n=3)
    rows = e.changes(T0 - 1.0)
    assert {r["entity"] for r in rows} == {"locker_code"}
    assert [r["revision"] for r in rows] == [1, 2, 3]


def test_changes_limit_takes_the_oldest(vault_path):
    e, q, ids = _chain(vault_path, n=4)
    assert [c["id"] for c in e.changes(T0 - 1.0, limit=2)] == ids[:2]


def test_changes_sees_unflushed_records(vault_path):
    e, q, _ = _chain(vault_path, n=2, flush=True)
    v = _corr(q, 0.70, seed=91)
    new_id = e.add_fact("My locker code is value-late.", v, source="chat",
                        metadata={"entity": "locker_code"}, timestamp=T0 + 9 * DAY)
    assert new_id in {c["id"] for c in e.changes(T0 + 5 * DAY)}


def test_changes_rejects_a_reversed_window(vault_path):
    e, q, _ = _chain(vault_path)
    with pytest.raises(ValueError):
        e.changes(T0 + DAY, until=T0)


def test_changes_outside_the_data_is_empty(vault_path):
    e, q, _ = _chain(vault_path, n=2)
    assert e.changes(T0 + 500 * DAY) == []


def test_history_is_right_where_the_ranker_is_wrong(vault_path):
    """The natural case that found the floor defect, in controlled vectors.

    On a real nomic-embed-text vault, three `employer` statements sat at cosine
    0.5926 / 0.6348 / 0.5317 to "where do I work", newest LAST and furthest
    away. The ranker's relevance floor (GROUP_COS_DELTA 0.06) dropped the newest
    from the group, and shipped 0.5.0 answered with the middle one, ranking the
    current employer THIRD. See
    ``scratch/refound/finding_floor_drops_current_value.json``.

    Fixed in 0.6.5: the floor may not drop a DECLARED group's newest revision
    (``design/floor_current_value_spec.md``, ``floor_current_value_results.json``).
    This test used to assert the defect and carried a tripwire saying "if this
    now passes, the floor was retuned"; it did, and it was. Both halves are
    asserted now -- history was always right, and the ranker agrees with it.
    """
    q = unit_rows(1, seed=5)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    ids = []
    for i, c in enumerate((0.5926, 0.6348, 0.5317)):
        ids.append(e.add_fact(f"employer statement {i}", _corr(q, c, seed=400 + i),
                              source="chat", metadata={"entity": "employer"},
                              timestamp=T0 + i * DAY))
    e.flush()

    chain = e.history("where do I work", q)
    assert [h["id"] for h in chain] == ids, "history must not drop a restatement"
    assert chain[-1]["id"] == ids[-1], "the newest value is the current one"
    assert [h["superseded"] for h in chain] == [True, True, False]

    ent, rv, tsv = e._columns(np.arange(e.arena.n_rows, dtype=np.int64))
    rows = np.arange(e.arena.n_rows, dtype=np.int64)
    cos = e.arena.scores(q / np.linalg.norm(q))
    mask = np.ones(rows.size, dtype=bool)
    floored = e._tagged_group(ent, mask, "employer", rows, cos)
    full = e._tagged_group(ent, mask, "employer", rows, cos, apply_floor=False)
    assert full.tolist() == [0, 1, 2]
    assert 2 in floored.tolist(), "the current value survives its own floor"
    assert e.search("where do I work", q, top_k=1)[0]["id"] == ids[-1]


def _declared_chain(path, declared=True, n=3):
    """Three `employer` statements, newest phrased FURTHEST from the question."""
    q = unit_rows(1, seed=5)[0]
    e = VaultEngine(path, embed_dim=D)
    ids = []
    for i, c in enumerate((0.5926, 0.6348, 0.5317)[:n]):
        meta = {"entity": "employer"}
        if not declared:
            meta["entity_declared"] = False
        ids.append(e.add_fact(f"employer statement {i}", _corr(q, c, seed=400 + i),
                              source="chat", metadata=meta, timestamp=T0 + i * DAY))
    e.flush()
    return e, q, ids


def test_provenance_survives_a_reopen(vault_path):
    """The whole fix rests on a bit that has to still be there after a restart.

    Benchmarks cannot see this: they build a vault and query it in one process.
    A caller declares its schema once, at write time, and asks the question
    weeks later against a reopened file.
    """
    e, q, ids = _declared_chain(vault_path)
    assert e.arena.group_declared[e.arena.group_id[0]] is True
    e.close()

    e2 = VaultEngine(vault_path, embed_dim=D)
    assert e2.arena.group_declared[e2.arena.group_id[0]] is True
    assert e2.search("where do I work", q, top_k=1)[0]["id"] == ids[-1]
    e2.close()


def test_provenance_survives_a_rebuild_without_the_sidecar(vault_path):
    """Both load paths must agree, or the answer depends on a cache being warm.

    The sidecar is a CACHE; deleting it forces the scan path, which reads
    provenance out of each record's own metadata instead of the cached column.
    A disagreement here is the shape of bug where a vault ranks one way on a
    warm machine and another way on a cold one.
    """
    import glob
    e, q, ids = _declared_chain(vault_path)
    e.close()
    for side in glob.glob(vault_path + "*"):
        if side != vault_path:
            os.remove(side)

    e2 = VaultEngine(vault_path, embed_dim=D)
    assert e2.arena.group_declared[e2.arena.group_id[0]] is True
    assert e2.search("where do I work", q, top_k=1)[0]["id"] == ids[-1]
    e2.close()


def test_a_record_written_before_the_marker_reads_as_inferred(vault_path):
    """An old vault must rank exactly as it did, not be silently re-judged.

    Records written before 0.6.5 carry an entity with no provenance marker. The
    conservative reading is `inferred`, because that is the behaviour those rows
    were written under; guessing `declared` would change the answers an existing
    vault gives on upgrade, without asking.
    """
    e, q, _ids = _declared_chain(vault_path, declared=False)
    assert e.arena.group_declared[e.arena.group_id[0]] is False
    rows = np.arange(e.arena.n_rows, dtype=np.int64)
    cos = e.arena.scores(q / np.linalg.norm(q))
    ent, _rv, _ts = e._columns(rows)
    mask = np.ones(rows.size, dtype=bool)
    assert 2 not in e._tagged_group(ent, mask, "employer", rows, cos).tolist()
    e.close()


def test_one_declaration_makes_the_whole_group_declared(vault_path):
    """DECLARED WINS. A caller naming the attribute is evidence; a tagger's
    guess about the same key is not, so a later inferred write must not
    downgrade a group somebody declared."""
    q = unit_rows(1, seed=5)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    e.add_fact("employer a", _corr(q, 0.60, seed=401), source="chat",
               metadata={"entity": "employer"}, timestamp=T0)
    e.add_fact("employer b", _corr(q, 0.55, seed=402), source="chat",
               metadata={"entity": "employer", "entity_declared": False},
               timestamp=T0 + DAY)
    e.flush()
    assert e.arena.group_declared[e.arena.group_id[0]] is True
    e.close()


def test_an_inferred_group_keeps_the_old_floor(vault_path):
    """The other half of the fix, and the reason it is gated at all.

    The same chain with the entity INFERRED rather than declared must rank
    exactly as 0.6.4 did. Protecting the newest member of a group the lexical
    tagger assembled costs -19.4 points on the 3-persona chat set, concentrated
    entirely on `phone number` and `address` -- the two attributes with
    siblings, which is a tagger-merged pair being promoted, not a revision.
    """
    q = unit_rows(1, seed=5)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    for i, c in enumerate((0.5926, 0.6348, 0.5317)):
        e.add_fact(f"employer statement {i}", _corr(q, c, seed=400 + i),
                   source="chat", timestamp=T0 + i * DAY,
                   metadata={"entity": "employer", "entity_declared": False})
    e.flush()
    rows = np.arange(e.arena.n_rows, dtype=np.int64)
    cos = e.arena.scores(q / np.linalg.norm(q))
    ent, _rv, _ts = e._columns(rows)
    mask = np.ones(rows.size, dtype=bool)
    assert 2 not in e._tagged_group(ent, mask, "employer", rows, cos).tolist()


# --- volatility / staleness -------------------------------------------------
def test_volatility_measures_intervals(vault_path):
    e, q, ids = _chain(vault_path, n=4)
    v = e.volatility(now=T0 + 10 * DAY)
    assert len(v) == 1
    f = v[0]
    assert f["entity"] == "locker_code" and f["n_revisions"] == 4
    assert f["intervals"] == [DAY, DAY, DAY]
    assert f["median_interval"] == DAY
    assert f["age"] == pytest.approx(7 * DAY)


def test_volatility_excludes_untagged_records(vault_path):
    """Untagged rows share the empty group key; pooling them invents one fact."""
    q = unit_rows(1, seed=9)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    for i in range(5):
        e.add_fact(f"unrelated note {i}", _corr(q, 0.4, seed=800 + i),
                   source="document", timestamp=T0 + i * DAY)
    e.flush()
    assert e.volatility() == []


def test_volatility_needs_a_restatement(vault_path):
    """A fact asserted once has no interval, so it has no rate."""
    e, q, _ = _chain(vault_path, n=1)
    assert e.volatility() == []


def test_volatility_sees_unflushed_records(vault_path):
    """0.6.3 read `arena` alone, so an unsealed block was invisible HERE while
    `search`, `history` and `changes` all scored it -- each of those has had a
    "sees unflushed records" test since the day it shipped and this did not.
    Under one block of writes, which is the entire life of a small vault,
    `volatility()` returned [] and `staleness()` with it.
    """
    e, _q, _ids = _chain(vault_path, n=3, flush=False)
    assert e.stats()["memtable_pending"] == 3
    rows = e.volatility(now=T0 + 10 * DAY)
    assert len(rows) == 1
    assert rows[0]["n_revisions"] == 3
    assert rows[0]["intervals"] == [DAY, DAY]


def test_volatility_is_the_same_answer_before_and_after_a_flush(vault_path):
    """Durability is not visibility: flushing decides what survives a crash,
    never what has been written."""
    e, _q, _ids = _chain(vault_path, n=3, flush=False)
    before = e.volatility(now=T0 + 10 * DAY)
    e.flush()
    assert e.volatility(now=T0 + 10 * DAY) == before


def test_volatility_counts_a_group_split_across_the_block_boundary(vault_path):
    """The bug at its sharpest -- not an empty answer, a STALE one.

    Measured on the published 0.6.3 wheel: 120 writes whose newest was made
    `now` reported ``n_revisions=100`` and ``age=20 days``. A fact restated
    today, called three weeks unconfirmed, by the method whose whole purpose is
    to say which facts need re-confirming.
    """
    q = unit_rows(1, seed=3)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    n = 120
    for i in range(n):
        e.add_fact(f"My locker code is value-{i}.",
                   _corr(q, 0.80, seed=200 + i % 40), source="chat",
                   metadata={"entity": "locker_code"}, timestamp=T0 + i * DAY)
    assert 0 < e.stats()["memtable_pending"] < n      # a partial block is open
    rows = e.volatility(now=T0 + (n - 1) * DAY)
    assert len(rows) == 1                             # counted once, not twice
    assert rows[0]["n_revisions"] == n
    assert rows[0]["age"] == 0.0


def test_volatility_excludes_untagged_pending_records(vault_path):
    """The same exclusion on the pending path: rows with no entity share the
    empty group key, and pooling them reports one enormous fake fact."""
    q = unit_rows(1, seed=9)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    for i in range(5):
        e.add_fact(f"unrelated note {i}", _corr(q, 0.4, seed=800 + i),
                   source="document", timestamp=T0 + i * DAY)
    assert e.stats()["memtable_pending"] == 5
    assert e.volatility() == []


def test_staleness_sees_unflushed_records(vault_path):
    """It is built on `volatility`, so it inherited the blind spot and the fix."""
    e, _q, _ids = _chain(vault_path, n=3, flush=False)
    st = e.staleness(now=T0 + 10 * DAY, assume_memoryless=True)
    assert len(st) == 1 and st[0]["n_revisions"] == 3
    assert 0.0 < st[0]["p_superseded"] < 1.0


def test_staleness_suppresses_the_probability_by_default(vault_path):
    """Pre-registered consequence of the calibration gate, not caution."""
    e, q, _ = _chain(vault_path, n=4)
    rows = e.staleness(now=T0 + 400 * DAY)
    assert rows and rows[0]["p_superseded"] is None
    assert "suppressed" in rows[0]["model"]
    assert rows[0]["rate"] is not None          # the measured rate still shows


def test_staleness_reports_a_probability_when_asked(vault_path):
    e, q, _ = _chain(vault_path, n=4)
    rows = e.staleness(now=T0 + 400 * DAY, assume_memoryless=True)
    p = rows[0]["p_superseded"]
    assert p is not None and 0.0 <= p <= 1.0
    assert p > 0.9, "400 days on a 1-day rate should read as very likely stale"


def test_staleness_confidence_tracks_how_much_was_observed(vault_path):
    e, q, _ = _chain(vault_path, n=2)
    assert e.staleness(now=T0 + DAY)[0]["confidence"] == "prior"
    e2, q2, _ = _chain(vault_path + "2", n=5)
    assert e2.staleness(now=T0 + DAY)[0]["confidence"] == "observed"


def test_group_floor_sim_recommendation_holds_in_miniature(vault_path):
    """The documented schema-aware recommendation, pinned on a small case.

    `_apply_group_floor` now tells a caller that declares its own entities to
    set group_floor_sim=0.45 (+19.0 pt measured, floor_retune_results.json) and
    tells a caller relying on the tagger not to (-13.9 pt,
    floor_chatcheck_results.json). This pins the mechanism that difference rests
    on: at 0.0 the newest revision is dropped from its own group, at 0.45 it
    survives. If this flips, the recommendation in that docstring is wrong.
    """
    q = unit_rows(1, seed=11)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    for i, c in enumerate((0.80, 0.77, 0.74, 0.71)):
        e.add_fact(f"statement {i}", _corr(q, c, seed=500 + i), source="chat",
                   metadata={"entity": "locker_code"}, timestamp=T0 + i * DAY)
    e.flush()
    rows = np.arange(e.arena.n_rows, dtype=np.int64)
    cos = e.arena.scores(q / np.linalg.norm(q))
    ent, _rv, _ts = e._columns(rows)
    mask = np.ones(rows.size, dtype=bool)

    # 0.6.5 does automatically, for a DECLARED group, what 0.45 did by hand.
    e.group_floor_sim = 0.0
    assert 3 in e._tagged_group(ent, mask, "locker_code", rows, cos).tolist()
    e.group_floor_sim = 0.45
    assert 3 in e._tagged_group(ent, mask, "locker_code", rows, cos).tolist()


def test_the_boost_bound_is_documented_as_cosine_mode_only(vault_path):
    """`legacy_score(c) < c` for |c|<1, so score - cosine goes NEGATIVE.

    The search docstring promised `0 <= score - cosine <= max_boost` with no
    exception through 3.3.0; an interaction audit measured 43 violations in a
    50-hit sample under score_mode="legacy". The bound is real in the default
    cosine mode and is now documented as cosine-mode-only. This pins both halves
    so the docstring cannot drift back.

    UNBOOSTED hits, deliberately. The violation is not universal: `score` is
    `legacy_score(c) + boosts`, so a large enough entity boost lifts a legacy
    score back over its own cosine -- which is why that audit found 43 in 50 and
    not 50. Until 0.6.5 this fixture tagged its records, and it passed only
    because the relevance floor happened to shrink the group enough to keep the
    boosts small. Fixing the floor grew the group, the boosts went up, and the
    assertion flipped on a change that has nothing to do with what it tests.
    With no tag there is no boost and `legacy_score(c) < c` is the whole story.
    """
    q = unit_rows(1, seed=13)[0]
    for mode, expect_within in (("cosine", True), ("legacy", False)):
        e = VaultEngine(vault_path + mode, embed_dim=D, score_mode=mode)
        for i in range(3):
            e.add_fact(f"note {i}", _corr(q, 0.8 - 0.05 * i, seed=600 + i),
                       source="document", timestamp=T0 + i * DAY)
        e.flush()
        cap = e.stats()["max_boost"]
        deltas = [h["score"] - h["cosine"]
                  for h in e.search("what is my locker code?", q, top_k=3)]
        assert deltas
        within = all(-1e-9 <= d <= cap + 1e-9 for d in deltas)
        assert within is expect_within, (mode, deltas, cap)
        e.close()


def test_vault_exposes_volatility_and_staleness(vault_path, offline_embedder):
    """The headline capability must be on the API users actually touch.

    0.6.0 first shipped `history` and `changes` as Vault wrappers but left
    `volatility` and `staleness` reachable only through `Vault.engine`, so the
    README had to tell people to reach past the public object.
    """
    from nanomem.vault import Vault
    v = Vault(vault_path)
    for i in range(3):
        v.engine.add_fact(f"My locker code is value-{i}.", v.embedder.embed(f"c{i}"),
                          source="chat", metadata={"entity": "locker_code"},
                          timestamp=T0 + i * DAY)
    v.engine.flush()
    rows = v.volatility(now=T0 + 10 * DAY)
    assert rows and rows[0]["entity"] == "locker_code"
    assert rows[0]["n_revisions"] == 3
    st = v.staleness(now=T0 + 10 * DAY)
    assert st[0]["p_superseded"] is None
    assert v.staleness(now=T0 + 10 * DAY, assume_memoryless=True)[0]["p_superseded"] is not None
    v.close()
