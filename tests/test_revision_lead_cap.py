"""The lead's cap must be large enough to lead a group the floor kept wide.

`floor_keeps_current` keeps the newest member of a DECLARED group however far
below the group's best it sits. That deliberately removed the bound
(`window_delta`) that justified `REVISION_LEAD` being 0.20, and the cap was not
revisited -- so a declared chain whose newest revision is worded furthest from
the question was kept in the group and then not lifted far enough to lead it.
`search[0]` returned a value two revisions old while `history()` returned the
current one, with the entity declared, which is exactly what the README promises
declaring the entity prevents.

Found by the third independent black-box review of published 0.7.15.
Pre-registered in design/revision_lead_cap_spec.md; swept in
evidence/revision_lead_cap_results.json.
"""
import numpy as np
import pytest

from nanomem.vault import Vault
from nanomem import entities as ent

DAY = 86400.0
NOW = 1_700_000_000.0

# Revision 1 NAMES the attribute, later revisions refer to it implicitly. That
# is what puts the question's keyword in the oldest member and not the newest.
QUESTIONS = {"office": "where is my desk",
             "gym": "where do I train",
             "project": "what project am I on"}

CHAINS = {
    "office": ["My desk is on the third floor of Kestrel House.",
               "Moved down to the annexe at Larkfield.",
               "Now parked in the Maple Wharf building."],
    "gym": ["I train at the gym on Perrin Street.",
            "Switched over to the leisure centre by the canal.",
            "These days it is the climbing wall under the arches."],
    "project": ["I am on the billing migration project.",
                "That wrapped in March and I moved onto the search rewrite.",
                "Since the reorg I have been on the mobile client."],
}


def _chain_vault(tmp_path, entity, name="lead.dat"):
    v = Vault(str(tmp_path / name))
    vals = CHAINS[entity]
    for i, text in enumerate(vals):
        v.add(text, metadata={"entity": entity},
              timestamp=NOW - (len(vals) - 1 - i) * 150 * DAY)
    v.flush()
    return v


def test_the_cap_is_above_every_lift_the_ranker_requests(tmp_path, offline_embedder):
    """THE GUARD. `REVISION_LEAD` is a cap, not a tuned value, and the way it
    fails is silent: the lift is clamped and a superseded value keeps rank 1.

    0.7.15's docstring said the cap was "reached 0 times" over 72 calls. That was
    true of the corpus it was measured on and was read as a property of the code.
    So this records the largest lift actually requested and fails if the cap is
    ever reached -- the next time a corpus is wide enough, this says so.
    """
    seen = []
    orig = ent.apply_revision_lead

    def spy(final, rows, rev, ts, historical, marks=None,
            historical_mode="previous", weight=ent.REVISION_LEAD):
        r = np.asarray(rows, dtype=np.int64)
        if r.size >= 2 and weight:
            order = ent.temporal_order(rev, ts, r, historical, marks, historical_mode)
            need = float(np.max(final[r])) - float(final[int(order[0])]) + ent.LEAD_EPS
            seen.append(need)
        return orig(final, rows, rev, ts, historical, marks, historical_mode, weight)

    import nanomem.engine as engine
    engine._ent.apply_revision_lead = spy
    try:
        # NATURAL QUESTIONS ONLY, and the scope is deliberate. Searching with
        # text copied from an old revision drives its cosine to ~1.0 and needs a
        # lift of 0.80 that no cap should grant -- a near-exact text match
        # arguably SHOULD lead, and promoting the newest revision over it would
        # be a different bug. What this guard watches is the case the defect was
        # about: someone asking a question in their own words.
        for entity in CHAINS:
            v = _chain_vault(tmp_path, entity, name="%s.dat" % entity)
            v.search(QUESTIONS[entity], top_k=4, decompose=False)
            v.close()
    finally:
        engine._ent.apply_revision_lead = orig

    assert seen, "the lead never ran -- this guard would pass vacuously"
    worst = max(seen)
    assert worst < ent.REVISION_LEAD, (
        "a lift of %.4f reached the cap of %.2f: some group is wider than the "
        "lead can close, which is the 0.7.15 defect returning. Re-run "
        "scratch/refound/exp_revision_lead_cap.py and raise the cap on evidence."
        % (worst, ent.REVISION_LEAD))


def test_a_declared_chain_does_not_lose_rank_one_to_its_own_past(
        tmp_path, offline_embedder):
    """`search` top-1 and `history`'s current value must agree on a declared
    chain. This is the README's bolded promise, and the shape that broke it."""
    v = _chain_vault(tmp_path, "office")
    hits = v.search("where is my desk", top_k=3, decompose=False)
    hist = v.history("where is my desk")
    assert hist, "the declared chain produced no history"
    assert hist[-1]["superseded"] is False
    assert hits[0]["text"] == hist[-1]["text"], (
        "search[0]=%r but history's current value is %r"
        % (hits[0]["text"], hist[-1]["text"]))
    v.close()


def test_the_published_bound_is_the_sum_of_the_terms(tmp_path, offline_embedder):
    """`max_boost` must stay the sum of the terms it is documented to be.

    `_apply_filter` proves the exactness of FILTERED search by assuming this
    constant bounds every boost, so a term that can exceed it silently breaks a
    proven property. That is why the lead is capped at all rather than left to
    close whatever gap it finds (design/revision_lead_cap_spec.md, amendment 1).
    """
    v = _chain_vault(tmp_path, "office")
    mb = v.stats()["max_boost"]
    assert mb == pytest.approx(ent.INTENT_BOOST + ent.GROUP_HOIST + ent.REVISION_LEAD)
    for q in ("where is my desk", "where was my desk before", "my desk"):
        for h in v.search(q, top_k=5, decompose=False):
            excess = float(h["score"]) - float(h["cosine"])
            assert -1e-6 <= excess <= mb + 1e-6, (q, h["text"], excess)
    v.close()


# ---------------------------------------------------------------------------
# M2, same review: `delete` advertised a mode that could not apply to a document
# `add()` had split. These belong with the other "the docs advertise it, so
# something must check it" tests rather than in a file about ranking, but the
# release that found them is this one and splitting them across files makes the
# finding harder to follow than keeping them together.
# ---------------------------------------------------------------------------
def _big(words=2000):
    """Long enough that `add()` splits it; uniquely numbered so any missing or
    duplicated chunk is visible rather than plausible."""
    return " ".join("w%04d" % i for i in range(words))


def test_delete_by_exact_text_reaches_a_document_that_was_split(tmp_path):
    """`Vault.delete`'s docstring advertises targeting by exact text. On a split
    document no single stored row holds the text the caller passed to `add()`,
    so this matched nothing and returned 0 -- silently, which is how a caller
    deleting a document could believe it had."""
    v = Vault(str(tmp_path / "m2.dat"))
    text = _big()
    v.add(text)
    v.flush()
    assert len(v.get_all_records()) > 1, "this test needs a SPLIT document"
    assert v.delete(text_exact=text) > 1
    assert v.get_all_records() == []
    assert b"w1999" not in open(str(tmp_path / "m2.dat"), "rb").read()
    v.close()


def test_delete_by_exact_text_still_refuses_text_that_was_never_stored(tmp_path):
    """The control. Resolving to a parent must not turn `text_exact` into a
    fuzzy match -- a near-miss has to delete nothing."""
    v = Vault(str(tmp_path / "m2b.dat"))
    text = _big()
    v.add(text)
    v.flush()
    n = len(v.get_all_records())
    assert v.delete(text_exact=text + " and one word more") == 0
    assert v.delete(text_exact=text[:-40]) == 0
    assert len(v.get_all_records()) == n
    v.close()


def test_delete_by_exact_text_on_an_unsplit_record_is_unchanged(tmp_path):
    """The other control: the single-row path must behave exactly as before."""
    v = Vault(str(tmp_path / "m2c.dat"))
    v.add("a short single-row fact.")
    v.add("another short fact.")
    v.flush()
    assert v.delete(text_exact="a short single-row fact.") == 1
    assert [r["text"] for r in v.get_all_records()] == ["another short fact."]
    v.close()
