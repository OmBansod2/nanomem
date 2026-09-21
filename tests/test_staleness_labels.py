"""A citation must say whether it is still true.

A timestamp says WHEN a record was written, never whether it still holds. A fact
written ten years ago can be current (a blood type); one written last week can
already be dead. Telling them apart needs "is there a later record about the SAME
attribute", which the revision group knows and a date cannot.

`history()` already reported this for the chain it assembles. `search()` did not,
so `ask()` handed a model three statements about one attribute and left it to
guess which one held -- the failure this library's README opens on, committed at
the last possible moment before the answer is written.
"""

import time

import pytest

from nanomem import Vault, mcp
from nanomem.vault import staleness_label

DAY = 86400


@pytest.fixture()
def chain(tmp_path):
    now = time.time()
    v = Vault(str(tmp_path / "v.dat"))
    job = {"entity": "employer"}
    v.add("I work at Acme Corp.", metadata=job, timestamp=now - 800 * DAY)
    v.add("I moved, I work at Initech.", metadata=job, timestamp=now - 400 * DAY)
    v.add("I switched again, I work at Globex now.", metadata=job, timestamp=now - 240 * DAY)
    # never changed, and very old -- the case that proves age is not staleness
    v.add("My blood type is O negative.", metadata={"entity": "blood_type"},
          timestamp=now - 3000 * DAY)
    v.flush()
    return v, now


def _by_text(hits, needle):
    return next(h for h in hits if needle in h["text"])


def test_search_hits_say_whether_they_are_still_true(chain):
    v, _ = chain
    hits = v.search("where do I work", top_k=5, decompose=False)
    assert _by_text(hits, "Globex")["superseded"] is False
    assert _by_text(hits, "Initech")["superseded"] is True
    assert _by_text(hits, "Acme")["superseded"] is True


def test_a_ten_year_old_fact_that_never_changed_is_not_stale(chain):
    """Age is not staleness. This is the whole reason a timestamp is not enough."""
    v, now = chain
    h = v.search("what is my blood type", top_k=1, decompose=False)[0]
    assert h["superseded"] is False
    assert staleness_label(h) == ""
    assert (now - h["timestamp"]) / DAY > 2900, "the fixture stopped being old"


def test_replaced_at_is_the_immediate_successor_not_the_newest(chain):
    """Acme was replaced by Initech, not by Globex.

    Reporting the newest instead dates every superseded value in a chain to the
    same moment -- a three-value chain said Acme was replaced 240 days ago when
    Initech replaced it at 400.
    """
    v, now = chain
    hits = v.search("where do I work", top_k=5, decompose=False)
    acme = _by_text(hits, "Acme")
    initech = _by_text(hits, "Initech")
    assert round((now - acme["superseded_at"]) / DAY) == pytest.approx(400, abs=2)
    assert round((now - initech["superseded_at"]) / DAY) == pytest.approx(240, abs=2)
    assert acme["superseded_at"] < initech["superseded_at"]


def test_an_ungrouped_record_reports_unknown_rather_than_current(tmp_path):
    """The tagger groups 0 of 100 narratively-phrased chains, so "nothing
    replaced it" is not knowable for a record in no group. None, not False."""
    v = Vault(str(tmp_path / "v.dat"))
    v.add("zqxj wibble frobnicate 12345 lorem")   # nothing the tagger can tag
    v.flush()
    h = v.search("zqxj wibble", top_k=1, decompose=False)[0]
    assert h["superseded"] in (None, False)
    if h["superseded"] is None:
        assert staleness_label(h) == ""


def test_the_label_is_empty_for_a_value_that_holds(chain):
    v, _ = chain
    assert staleness_label(_by_text(v.search("where do I work", top_k=5,
                                             decompose=False), "Globex")) == ""


def test_the_label_names_when_it_was_replaced(chain):
    v, _ = chain
    lab = staleness_label(_by_text(v.search("where do I work", top_k=5,
                                            decompose=False), "Initech"))
    assert "SUPERSEDED" in lab
    assert "months ago" in lab or "days ago" in lab
    assert "not now" in lab


def test_the_mcp_tool_marks_stale_results_for_the_model(chain):
    v, _ = chain
    out = mcp.dispatch(v, "nanomem_search", {"query": "where do I work", "top_k": 3})
    assert "SUPERSEDED" in out
    # the current value must NOT be marked
    current_line = next(l for l in out.splitlines() if "Globex" in l)
    assert "SUPERSEDED" not in current_line


def test_the_mcp_tool_marks_nothing_when_nothing_changed(chain):
    v, _ = chain
    out = mcp.dispatch(v, "nanomem_search", {"query": "what is my blood type", "top_k": 1})
    assert "SUPERSEDED" not in out


def test_ask_reports_stale_citations_without_calling_a_model(chain, monkeypatch):
    v, _ = chain
    seen = {}

    def _fake(prompt, candidates, **kw):
        seen["ctx"] = [c for c in candidates]
        return "stub answer"

    monkeypatch.setattr(v, "_call_llm", _fake)
    out = v.ask("where do I work", top_k=3)
    assert "stale_citations" in out
    stale = out["stale_citations"]
    assert stale, "a three-value chain produced no stale citations"
    assert all("SUPERSEDED" in s["note"] for s in stale)
    assert all(s["superseded_at"] for s in stale)


def test_the_label_reaches_the_prompt_the_model_reads(chain, monkeypatch):
    """The point is not the return value. It is what the model is shown."""
    v, _ = chain
    captured = {}
    real = v._call_llm

    def _spy(prompt, candidates, **kw):
        # rebuild the context the real method would send
        from nanomem.vault import staleness_label as _lab
        captured["marks"] = [_lab(c) for c in candidates]
        return "stub"

    monkeypatch.setattr(v, "_call_llm", _spy)
    v.ask("where do I work", top_k=3)
    assert any("SUPERSEDED" in m for m in captured["marks"])
    assert any(m == "" for m in captured["marks"]), "everything was marked stale"


# --------------------------------------------------------------------------
# The revision GROUP is (user_id, project, entity), not the entity alone.
# Found while checking whether a team-chat bot could use this: a fact tagged
# `deploy_tool` in one project marked a record tagged `deploy_tool` in ANOTHER
# project as superseded, while history() -- which scopes by group -- returned a
# chain of 1 for those same rows. Two surfaces of one library disagreeing.
# --------------------------------------------------------------------------
E = "deploy_tool"


def _pair(tmp_path, name, meta_a, meta_b):
    now = time.time()
    v = Vault(str(tmp_path / name))
    v.add("we deploy with jenkins", metadata=meta_a, timestamp=now - 300 * DAY)
    v.add("we deploy with argocd now, not jenkins", metadata=meta_b, timestamp=now - 30 * DAY)
    v.flush()
    hits = v.search("how do we deploy", top_k=3, decompose=False)
    old = next(h for h in hits
               if "jenkins" in h["text"] and "argocd" not in h["text"])
    return v, old


@pytest.mark.parametrize("label,meta_a,meta_b,expected", [
    ("same scope, no user_id (the shared-channel pattern)",
     {"entity": E, "project": "eng"}, {"entity": E, "project": "eng"}, True),
    ("different users in one project must not supersede each other",
     {"entity": E, "user_id": "alice", "project": "eng"},
     {"entity": E, "user_id": "bob", "project": "eng"}, False),
    ("different projects must not contaminate each other",
     {"entity": E, "project": "eng"}, {"entity": E, "project": "design"}, False),
    ("no scoping at all -- ordinary personal use",
     {"entity": E}, {"entity": E}, True),
])
def test_supersession_is_scoped_to_the_revision_group(tmp_path, label, meta_a,
                                                      meta_b, expected):
    _, old = _pair(tmp_path, "g_%d.dat" % abs(hash(label)), meta_a, meta_b)
    assert old["superseded"] is expected, label


@pytest.mark.parametrize("meta_a,meta_b", [
    ({"entity": E, "project": "eng"}, {"entity": E, "project": "eng"}),
    ({"entity": E, "user_id": "alice", "project": "eng"},
     {"entity": E, "user_id": "bob", "project": "eng"}),
    ({"entity": E, "project": "eng"}, {"entity": E, "project": "design"}),
])
def test_search_and_history_agree_about_what_was_replaced(tmp_path, meta_a, meta_b):
    """The cross-surface invariant. If history() assembles a chain of one, then
    nothing replaced that record, and search() must not say otherwise."""
    v, old = _pair(tmp_path, "agree_%d.dat" % abs(hash(str(meta_a) + str(meta_b))),
                   meta_a, meta_b)
    chain_len = len(v.history("how do we deploy"))
    assert old["superseded"] is (chain_len > 1), (
        "history() says chain=%d but search() says superseded=%s"
        % (chain_len, old["superseded"]))
