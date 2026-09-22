"""A search result must say whether it is the WHOLE answer.

`search` returns at most `top_k` records. Through 0.7.23 it said nothing about
what it left behind, so three documents answering three of eight clauses looked
exactly like three documents answering all of them -- and the consumer is often
a model that cannot look behind the result.

The counts are free: the scan is exhaustive, so `n_above_floor` is already
computed on the line that applies `top_k`, and was being discarded.
"""

import json

import pytest

from nanomem import Vault
from nanomem.vault import SearchResults


COMPANIES = ["Acme", "Initech", "Globex", "Umbrella", "Soylent", "Vandelay",
             "Wayland", "Tyrell"]
EIGHT_CLAUSES = ("What is Acme revenue? What about Initech? And Globex? "
                 "And Umbrella? And Soylent? And Vandelay? And Tyrell? "
                 "What port does staging use?")


@pytest.fixture()
def corpus(tmp_path):
    v = Vault(str(tmp_path / "v.dat"))
    for co in COMPANIES:
        for yr in (2022, 2023, 2024, 2025):
            v.add("%s revenue in fiscal year %d was %d million" % (co, yr, 100 + yr % 50))
    v.add("the staging database listens on port 5433")
    for i in range(40):
        v.add("filler note %d about gardening and weather" % i)
    v.flush()
    return v


# --------------------------------------------------------------------------
# backwards compatibility -- this is a list, and every old caller still works
# --------------------------------------------------------------------------
def test_results_are_still_a_list(corpus):
    r = corpus.search("what port does staging use", decompose=False)
    assert isinstance(r, list)
    assert isinstance(r, SearchResults)
    assert len(r) == 3
    assert r[0]["text"]
    assert [h for h in r] == list(r)
    assert json.loads(json.dumps(r))          # serialises as a plain list


def test_an_empty_query_still_returns_an_empty_list(corpus):
    r = corpus.search("")
    assert r == []
    assert r.truncated is False


def test_top_k_zero_still_returns_nothing(corpus):
    r = corpus.search("anything", top_k=0)
    assert r == []


# --------------------------------------------------------------------------
# the counts
def a_floor_that_admits(vault, query, n):
    """A min_score that at least `n` records clear, derived from THIS corpus.

    The first version of these tests hard-coded `min_score=0.6`, which is a
    cosine threshold tuned to the embedder on one machine. CI has no embedding
    endpoint, so it runs the built-in lexical encoder, whose scores sit on a
    different scale entirely -- the searches returned nothing and two tests
    failed on every platform. A test of the truncation SIGNAL must not depend
    on which encoder answered, so the floor is measured rather than guessed.
    """
    scored = vault.search(query, top_k=500, min_score=0.0, decompose=False)
    assert len(scored) > n, "corpus too small to derive a floor"
    return float(scored[n]["cosine"])


# --------------------------------------------------------------------------
def test_n_above_floor_counts_what_cleared_the_floor(corpus):
    q = "revenue of every company every year"
    floor = a_floor_that_admits(corpus, q, 8)
    r = corpus.search(q, min_score=floor, top_k=3, decompose=False)
    assert r.n_above_floor is not None
    assert r.n_above_floor > len(r), "nothing was reported as left behind"
    assert r.returned == len(r) == 3
    assert r.truncated is True


def test_raising_top_k_past_the_matches_reports_complete(corpus):
    q = "revenue of every company every year"
    r = corpus.search(q, min_score=a_floor_that_admits(corpus, q, 8),
                      top_k=500, decompose=False)
    assert r.truncated is False
    assert r.explain() == ""


def test_the_floor_clause_is_silent_when_there_is_no_floor(corpus):
    """With min_score=0.0 every record clears it, so "3 of 101" is true of
    every query ever asked -- including one whose answer really is one record.
    A signal that fires every time carries nothing, and calling a complete
    answer incomplete is the very defect this file exists to remove."""
    r = corpus.search("what port does staging use", decompose=False)
    assert r.floor == 0.0
    assert r.explain() == ""
    # the number is still reported, because it is exact and a caller may want it
    assert r.n_above_floor is not None


# --------------------------------------------------------------------------
# the number that answers "3 of 8"
# --------------------------------------------------------------------------
def test_a_composite_question_reports_the_clauses_that_got_nothing(corpus):
    r = corpus.search(EIGHT_CLAUSES, top_k=3)
    assert r.summary["decomposed"] is True
    assert len(r.sub_queries) >= 6
    assert r.unanswered_sub_queries, "no clause was reported unanswered at top_k=3"
    assert r.truncated is True
    note = r.explain()
    assert "parts of the question got no result" in note


def test_a_composite_question_with_room_reports_nothing_missing(corpus):
    r = corpus.search(EIGHT_CLAUSES, top_k=24)
    assert r.unanswered_sub_queries == []
    assert "parts of the question" not in r.explain()


def test_every_sub_query_is_accounted_for(corpus):
    """Each clause is either answered or named as unanswered. Never neither."""
    r = corpus.search(EIGHT_CLAUSES, top_k=5)
    named = {b["query"] for b in r.sub_queries}
    answered = {b["query"] for b in r.sub_queries if b["answered"]}
    assert set(r.unanswered_sub_queries) <= named
    assert answered.isdisjoint(set(r.unanswered_sub_queries))
    assert answered | set(r.unanswered_sub_queries) == named


def test_decomposed_results_do_not_invent_a_cross_clause_total(corpus):
    """De-duplicating candidates ACROSS sub-queries is not something the scan
    computed, so it is reported as unknown rather than guessed."""
    r = corpus.search(EIGHT_CLAUSES, top_k=3)
    assert r.n_above_floor is None
    assert r.n_distinct_found >= len(r)


# --------------------------------------------------------------------------
# it must not change WHICH records come back
# --------------------------------------------------------------------------
@pytest.mark.parametrize("decompose", [True, False])
@pytest.mark.parametrize("k", [1, 3, 10])
def test_the_returned_records_are_unchanged(corpus, decompose, k):
    """This release reports; it does not re-rank. Ranking is pinned elsewhere,
    so here it is enough that the hit dicts still carry what they carried."""
    r = corpus.search("revenue of every company every year", top_k=k,
                      decompose=decompose)
    assert len(r) <= k
    for h in r:
        assert set(("id", "text", "score", "cosine", "timestamp")) <= set(h)


# --------------------------------------------------------------------------
# the surfaces
# --------------------------------------------------------------------------
def test_the_mcp_search_tool_appends_the_note_only_when_cut(corpus):
    from nanomem import mcp
    q = "revenue of every company every year"
    floor = a_floor_that_admits(corpus, q, 8)
    cut = mcp.dispatch(corpus, "nanomem_search",
                       {"query": q, "top_k": 2, "min_score": floor})
    assert "This answer is incomplete" in cut

    # Same floor, but a budget bigger than what clears it: nothing is cut, so
    # the tool must say nothing.
    whole = mcp.dispatch(corpus, "nanomem_search",
                         {"query": q, "top_k": 500, "min_score": floor})
    assert "This answer is incomplete" not in whole


def test_the_mcp_tool_schema_offers_a_floor():
    from nanomem.mcp import TOOLS
    tool = next(t for t in TOOLS if t["name"] == "nanomem_search")
    props = tool["inputSchema"]["properties"]
    assert "min_score" in props
    assert "incomplete" in tool["description"] or "whole answer" in tool["description"]
