"""Round-2 regressions for the ranking layer.

Three claims that 3.0.0's docstrings made and its code did not keep:

1. an ordinary document query ranks on pure cosine;
2. a ``metadata_filter`` returns exactly the filtered exhaustive result;
3. the boosts are bounded and cannot re-rank across a large cosine gap.
"""

import numpy as np
import pytest

from conftest import D, unit_rows

from nanomem import entities as ent
from nanomem.engine import VaultEngine

# Question shapes that are NOT about the user. Every one of these carries a word
# 3.0.0 treated as a temporal cue ("first", "new", "before", "still", "again",
# "original", "recent", "old", "current") or a third-person possessive, which is
# why 18 of 120 real HotpotQA questions had their top-4 re-ordered.
DOC_QUESTIONS = [
    "Which was the first album released before the band changed its name?",
    "What is the current population of the old town?",
    "Who directed his most recent film, and was it a new adaptation?",
    "Which of their two stadiums is still in use today?",
    "What was the original name of the company that was later renamed?",
    "Where is the university located and when was it founded?",
    "How many US states border the river again mentioned in the article?",
    "What year did she move to the city that now hosts the festival?",
    "Which novel came first, and which one was updated for the new edition?",
    "What is the latest recorded height of the previously measured peak?",
]


def _doc_vault(tmp_path, n=400, seed=41, **kw):
    """A document corpus: the source is a document name, as `ingest_file` sets it."""
    e = VaultEngine(filepath=str(tmp_path / "docs.dat"), embed_dim=D, **kw)
    V = unit_rows(n, seed=seed)
    for i in range(n):
        e.add_fact(f"paragraph {i} about a subject with several sentences.", V[i],
                   source="encyclopedia.txt")
    e.flush()
    return e, V


def _exhaustive(e, q, k):
    A = e.arena.vec[:e.arena.n_rows]
    cos = A @ q
    idx = np.argsort(-cos, kind="stable")[:k]
    return [e.arena.ids[int(i)] for i in idx]


def test_near_duplicate_documents_are_not_treated_as_revisions(tmp_path):
    """The decisive guard is the DATA: two paragraphs of a document are not
    revisions of each other even when they are near-identical and the question is
    worded like a temporal one."""
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    base = unit_rows(1, seed=91)[0]
    near = base + 0.03 * unit_rows(1, seed=92)[0]
    near = near / np.linalg.norm(near)
    e.add_fact("The tower was originally 120 metres tall.", base, source="guide.pdf")
    e.add_fact("The tower is now 150 metres tall after the 1998 extension.", near,
               source="guide.pdf")
    e.flush()
    A = e.arena.vec[:e.arena.n_rows]
    for q_text in ("What is the current height of the tower now?",
                   "What was the original height of the tower before the extension?"):
        hits = e.search(q_text, base, top_k=2, min_score=-1.0)
        assert [h["id"] for h in hits] == [e.arena.ids[int(i)]
                                           for i in np.argsort(-(A @ base))[:2]], q_text
        for h in hits:
            assert h["score"] == pytest.approx(h["cosine"], abs=1e-6)
    e.close()


def test_cli_written_facts_are_revisable(tmp_path):
    """`nanomem add` and `Vault.add` write a personal log, not a document.

    Their sources are in ``entities.PERSONAL_SOURCES``, so the revision layer
    applies to them -- restricting it to ``source == "chat_session"`` would have
    made `nanomem search` return the superseded value.
    """
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    base = unit_rows(1, seed=93)[0]
    near = base + 0.04 * unit_rows(1, seed=94)[0]
    near = near / np.linalg.norm(near)
    e.add_fact("my phone number is 555 000 1111", base, source="cli")
    e.add_fact("my phone number is 555 000 2222", near, source="cli")
    e.flush()
    hits = e.search("what is my phone number", base, top_k=2, min_score=-1.0)
    assert "2222" in hits[0]["text"], [h["text"] for h in hits]
    e.close()


def test_document_corpus_ranks_on_pure_cosine(tmp_path):
    """With real question text, the engine must equal an exhaustive cosine scan.

    Measured on the 1,190-paragraph validation corpus with its 120 real
    questions: 3.0.0 differed on 18/120 top-4 orderings and 14/120 rank-1
    documents (``scratch/refound/exactness_v3r2.json`` records the round-3 re-run,
    which is 0/120).
    """
    e, V = _doc_vault(tmp_path)
    assert not e._has_personal_records()
    rng = np.random.default_rng(7)
    diffs = 0
    for qi, question in enumerate(DOC_QUESTIONS):
        q = V[rng.integers(0, V.shape[0])] + 0.30 * rng.normal(size=D).astype(np.float32)
        q = q / np.linalg.norm(q)
        got = [h["id"] for h in e.search(question, q, top_k=4, min_score=-1.0)]
        if got != _exhaustive(e, q, 4):
            diffs += 1
        for h in e.search(question, q, top_k=4, min_score=-1.0):
            assert h["score"] == pytest.approx(h["cosine"], abs=1e-6)
    assert diffs == 0, f"{diffs}/{len(DOC_QUESTIONS)} document queries were re-ranked"
    e.close()


def test_document_query_is_unaffected_by_personal_records_in_the_same_vault(tmp_path):
    """A mixed vault: personal records exist, but a factoid query is still exact."""
    e, V = _doc_vault(tmp_path, n=200, seed=43)
    extra = unit_rows(3, seed=44)  # noqa: E501
    e.add_fact("my phone number is 555 000 1111", extra[0],
               source="chat_session", metadata={"user_id": "u"})
    e.add_fact("my phone number is 555 000 2222", extra[1],
               source="chat_session", metadata={"user_id": "u"})
    e.flush()
    assert e._has_personal_records()
    rng = np.random.default_rng(11)
    for question in DOC_QUESTIONS:
        q = V[rng.integers(0, V.shape[0])] + 0.30 * rng.normal(size=D).astype(np.float32)
        q = q / np.linalg.norm(q)
        got = [h["id"] for h in e.search(question, q, top_k=4, min_score=-1.0)]
        assert got == _exhaustive(e, q, 4), question
    e.close()


def test_personal_question_still_resolves_revisions(tmp_path):
    """The gates must not switch the feature off: a first-person question works."""
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    base = unit_rows(1, seed=51)[0]
    near = base + 0.05 * unit_rows(1, seed=52)[0]
    near = near / np.linalg.norm(near)
    e.add_fact("my phone number is 555 000 1111", base,
               source="chat_session", metadata={"user_id": "u"})
    e.add_fact("my phone number is 555 000 2222", near,
               source="chat_session", metadata={"user_id": "u"})
    e.flush()
    hits = e.search("what is my phone number now?", base, top_k=2, min_score=-1.0)
    assert "2222" in hits[0]["text"], [h["text"] for h in hits]
    old = e.search("what was my previous phone number?", base, top_k=2,
                   min_score=-1.0, temporal_direction="historical")
    assert "1111" in old[0]["text"], [h["text"] for h in old]
    e.close()


def test_boosts_are_bounded_and_documented(tmp_path):
    """An exact match cannot be displaced by a tagged record far below it.

    3.0.0's ``group_hoist`` was a flat +1.0 applied before the relevance floor,
    so a record at cosine 0.685 outranked an exact match at cosine 1.000 -- the
    very behaviour the module docstring said had been removed.
    """
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(6, seed=61)
    q = V[0]
    e.add_fact("what is my phone number", q)                       # cosine 1.000
    e.add_fact("my phone number is 555 000 1111", V[1],
               source="chat_session", metadata={"user_id": "u"})
    e.add_fact("my phone number is 555 000 2222", V[2],
               source="chat_session", metadata={"user_id": "u"})
    e.flush()
    hits = e.search("what is my phone number", q, top_k=3, min_score=-1.0)
    assert hits[0]["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert hits[0]["id"] == hits[0]["doc_id"]
    # THE SCORE CONTRACT (DECISIONS #8): every term is an ADDITION, so no hit's
    # score may sit below its own cosine or above it by more than the published
    # cap. 3.0.1 PERMUTED a group's scores instead and broke both halves --
    # measured +0.6402 excess against a cap of 0.50, and scores below cosine.
    mb = e.stats()["max_boost"]
    for h in hits:
        assert h["score"] >= h["cosine"] - 1e-6, h
        assert h["score"] - h["cosine"] <= mb + 1e-6, h
    top_cos = max(h["cosine"] for h in hits)
    assert max(h["score"] for h in hits) <= top_cos + mb + 1e-6
    assert mb == pytest.approx(0.70)
    assert (ent.INTENT_BOOST + ent.GROUP_HOIST + ent.REVISION_LEAD) == pytest.approx(0.70)
    e.close()


def test_metadata_filter_is_exact(tmp_path):
    """A filter that matches everything must not change the result.

    3.0.0 stopped the metadata walk after ``max(64, 8*top_k)`` matches in COSINE
    order while ranking on cosine PLUS boosts, so a record that should have led
    was dropped outright by a no-op filter.
    """
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(220, seed=71)
    q = V[0] + 0.9 * unit_rows(1, seed=72)[0]
    q = q / np.linalg.norm(q)
    for i in range(200):
        e.add_fact(f"filler note {i}", V[i + 10], metadata={"tag": "keep"})
    e.add_fact("my phone number is 555 123 4567", V[0], source="chat_session",
               metadata={"tag": "keep", "user_id": "u"})
    e.flush()
    plain = e.search("what is my phone number", q, top_k=3, min_score=-1.0)
    filtered = e.search("what is my phone number", q, top_k=3, min_score=-1.0,
                        metadata_filter={"tag": "keep"})
    assert [h["id"] for h in plain] == [h["id"] for h in filtered]
    assert [h["score"] for h in plain] == pytest.approx([h["score"] for h in filtered])
    assert any("555 123 4567" in h["text"] for h in filtered)

    # a selective filter equals the brute-force filtered scan
    e.add_fact("tagged differently", unit_rows(1, seed=73)[0], metadata={"tag": "other"})
    e.flush()
    sel = e.search("", q, top_k=5, min_score=-1.0, metadata_filter={"tag": "other"})
    assert [h["text"] for h in sel] == ["tagged differently"]
    e.close()


def test_filtered_search_does_not_decode_the_whole_corpus(tmp_path):
    """The exactness bound must still stop early when no boost is in play."""
    e = VaultEngine(filepath=str(tmp_path / "v.dat"), embed_dim=D)
    V = unit_rows(2000, seed=81)
    for i in range(2000):
        e.add_fact(f"note {i}", V[i], source="notes.txt", metadata={"tag": "keep"})
    e.flush()
    seen = {"n": 0}
    real = e.arena.record

    def counting(row):
        seen["n"] += 1
        return real(row)

    e.arena.record = counting
    e.search("an impersonal document query", V[3], top_k=4, min_score=-1.0,
             metadata_filter={"tag": "keep"})
    e.arena.record = real
    print(f"\nrecords decoded for a filtered top-4 over 2,000 docs: {seen['n']}")
    assert seen["n"] < 60
    e.close()
