"""Defects found by a fourteen-dimension audit of 0.7.17, each pinned here.

Every one of these was reproduced against the shipped release before it was
fixed, and every test below fails if its fix is reverted.
"""
import re
import warnings

import pytest

from nanomem import vault as vault_mod
from nanomem.vault import Vault
from nanomem.embed import EmbeddingProvider

TOKENS = ["tok%05d" % i for i in range(3000)]


def _doc(insert=None, at=22):
    w = list(TOKENS)
    if insert:
        w.insert(at, insert)
    return " ".join(w)


def _stored_tokens(v):
    blob = " ".join(r["text"] for r in v.get_all_records())
    return {m for m in re.findall(r"tok\d{5}", blob)}


# ---------------------------------------------------------------------------
# add() must not write into the caller's dict
# ---------------------------------------------------------------------------
def test_add_does_not_mutate_the_caller_s_metadata_dict(tmp_path):
    """`add()` set `metadata["id"]` on the object the caller passed in. The next
    call reads `meta.get("id")` BEFORE hashing the text, so reusing one dict fed
    the first document's id back in as the second's and every later document
    collapsed onto it: three distinct facts, one id."""
    v = Vault(str(tmp_path / "m.dat"))
    meta = {"user_id": "alice"}
    ids = [v.add(t, metadata=meta) for t in
           ("I work at Acme.", "I live in Bristol.", "My car is blue.")]
    v.flush()

    assert "id" not in meta, "add() wrote into the caller's dict: %r" % meta
    assert len(set(ids)) == 3, "distinct documents collapsed onto one id: %r" % ids
    for i, expected in zip(ids, ("Acme", "Bristol", "blue")):
        assert expected in v.get(i)["text"]
    v.close()


def test_an_explicit_id_is_still_honoured(tmp_path):
    """The control: not writing back must not stop a caller SUPPLYING an id,
    by argument or in the metadata."""
    v = Vault(str(tmp_path / "e.dat"))
    assert v.add("a fact.", id="chosen_id") == "chosen_id"
    assert v.add("another fact.", metadata={"id": "meta_id"}) == "meta_id"
    v.flush()
    assert v.get("chosen_id")["text"] == "a fact."
    assert v.get("meta_id")["text"] == "another fact."
    v.close()


def test_the_documented_id_rule_holds_for_identical_text(tmp_path):
    """`add()` documents the id as `doc_<md5(text)[:10]>`, "a function of the TEXT
    alone". That has to stay true across calls."""
    a = Vault(str(tmp_path / "a.dat"))
    b = Vault(str(tmp_path / "b.dat"))
    assert a.add("the same sentence.") == b.add("the same sentence.")
    a.close(); b.close()


# ---------------------------------------------------------------------------
# one emoji must not reroute a Latin document into the character splitter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("marker,label", [
    (None, "plain ASCII"),
    ("🎉", "one emoji"),
    ("中", "one CJK character"),
    ("café naïve العربية", "accents and RTL"),
], ids=["plain", "emoji", "cjk-char", "accents"])
def test_no_token_is_lost_to_the_script_heuristic(tmp_path, marker, label):
    """`any(ord(c) > 0x2E80 ...)` called every emoji a dense non-spaced script, so
    ONE emoji split an English document every MAX_FACT_CHARS_CJK characters with
    no regard for word boundaries. Measured before the fix: 2982 of 3000 tokens
    survived, the rest truncated mid-word."""
    doc = _doc(marker)
    v = Vault(str(tmp_path / ("s_%s.dat" % label.split()[0])))
    v.add(doc)
    v.flush()
    kept = _stored_tokens(v)
    assert len(kept) == 3000, "%s: lost %d tokens" % (label, 3000 - len(kept))
    v.close()


def test_a_genuinely_cjk_document_still_splits_by_characters(tmp_path):
    """The control that must NOT flip: CJK has no spaces, so it still has to be
    cut by character count rather than by words."""
    doc = "这是一个很长的中文文档。" * 400
    v = Vault(str(tmp_path / "cjk.dat"))
    v.add(doc)
    v.flush()
    rows = v.get_all_records()
    assert len(rows) > 1, "a long CJK document must still be split"
    assert "".join(r["text"] for r in rows).count("这") >= 400
    v.close()


def test_the_script_test_is_a_proportion_not_an_existence_check(tmp_path):
    """One quoted Chinese character in an English paragraph must not change how
    the paragraph is split; a document that really is CJK must be detected."""
    assert vault_mod._is_dense_script("this is an english sentence with 中 in it") is False
    assert vault_mod._is_dense_script("这是一个中文句子") is True
    assert vault_mod._is_dense_script("🎉🚀✨ emoji only") is False
    assert vault_mod._is_dense_script("") is False


# ---------------------------------------------------------------------------
# the fallback encoder must announce itself
# ---------------------------------------------------------------------------
def test_falling_back_to_the_lexical_encoder_warns_and_is_recorded(tmp_path, monkeypatch):
    """A record written while the endpoint was unreachable is encoded lexically,
    is not comparable with rows the model embedded, and was unfindable
    afterwards -- with no warning at all. Measured: the fact did not appear in
    the results for its own question."""
    # Simulate the ENDPOINT being unreachable rather than replacing
    # `embed_batch`: replacing it skips the fallback branch entirely, so the test
    # would pass without exercising the thing it is named for.
    monkeypatch.setattr(EmbeddingProvider, "_fallback_announced", False, raising=False)
    monkeypatch.setattr(EmbeddingProvider, "_remote_embed_batch",
                        lambda self, texts: (_ for _ in ()).throw(OSError("connection refused")),
                        raising=True)
    v = Vault(str(tmp_path / "f.dat"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v.add("The API key rotates on Fridays.")
        v.flush()
    assert any(issubclass(w.category, RuntimeWarning) and "LEXICAL" in str(w.message)
               for w in caught), "the fallback was silent: %r" % [str(w.message) for w in caught]
    row = v.get_all_records()[0]
    assert (row.get("metadata") or {}).get("embed_backend") == "offline-lexical"
    v.close()


def test_a_model_embedded_row_carries_no_backend_stamp(tmp_path, offline_embedder):
    """The control: the stamp marks the exception, so an ordinary row must not
    carry it. `offline_embedder` patches `embed_batch` directly, which is the
    model path as far as the provider is concerned."""
    v = Vault(str(tmp_path / "n.dat"))
    v.add("an ordinary fact.")
    v.flush()
    assert "embed_backend" not in (v.get_all_records()[0].get("metadata") or {})
    v.close()


# ---------------------------------------------------------------------------
# ingest_file really does preserve formatting
# ---------------------------------------------------------------------------
LONG_LINES = "\n".join(
    "    x%d = compute(%s)" % (i, ", ".join("arg%d" % j for j in range(30)))
    for i in range(200))
SHORT_LINES = "\n".join("    x%d = %d" % (i, i) for i in range(200))


@pytest.mark.parametrize("src,label", [(SHORT_LINES, "short lines"), (LONG_LINES, "long lines")],
                         ids=["short", "long"])
def test_ingest_file_stores_every_line_verbatim(tmp_path, src, label):
    """`ingest_file` promises "exact code formatting, indentation, and line
    numbers". Two things broke it: `add_batch` stripped the leading indentation
    off each chunk's first line, and a 50-line slice of LONG lines exceeded
    MAX_FACT_WORDS and was re-split by the word-based splitter. Measured before
    the fix: 0 of 200 long lines survived verbatim."""
    p = tmp_path / "mod.py"
    p.write_text(src, encoding="utf-8")
    v = Vault(str(tmp_path / ("i_%s.dat" % label.split()[0])))
    v.ingest_file(str(p))
    v.flush()
    blob = "\n".join(r["text"] for r in v.get_all_records())
    missing = [ln for ln in src.split("\n") if ln.strip() and ln not in blob]
    assert not missing, "%s: %d line(s) not stored verbatim, e.g. %r" % (
        label, len(missing), missing[:2])
    v.close()


# ---------------------------------------------------------------------------
# the document fingerprint is computed once, not once per chunk
# ---------------------------------------------------------------------------
def test_the_document_fingerprint_is_computed_once_per_document(tmp_path, monkeypatch):
    """It sat inside the chunk loop, so the whole document was re-hashed once per
    chunk: O(chunks x bytes). A 3.3 MB document hashed 1,980 MB."""
    calls = []
    real = vault_mod._doc_fingerprint
    monkeypatch.setattr(vault_mod, "_doc_fingerprint",
                        lambda t: (calls.append(len(t)), real(t))[1])
    v = Vault(str(tmp_path / "fp.dat"))
    v.add(" ".join("word%05d" % i for i in range(20000)))
    v.flush()
    assert len(v.get_all_records()) > 10, "this test needs many chunks"
    assert len(calls) == 1, "fingerprint computed %d times for one document" % len(calls)
    v.close()


# ---------------------------------------------------------------------------
# a filtered search must return exactly what a filtered scan returns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_neighbour", [False, True], ids=["alone", "with-a-plain-tagged-row"])
def test_a_filtered_search_agrees_with_a_filtered_scan_on_list_tags(
        tmp_path, offline_embedder, with_neighbour):
    """`_apply_filter`'s cheap superset restricted to the interned id of the
    filter value. That is sound for a SCALAR tag, where `matches_filter` tests
    equality, and wrong for a LIST: `{"entity": ["work", "job"]}` interns as
    `work_job`, membership accepts it, and the shortcut dropped it -- but only
    once another row carried the plain tag, so the failure depended on the rest
    of the corpus."""
    v = Vault(str(tmp_path / ("lf_%s.dat" % with_neighbour)))
    v.add("fox one about work", metadata={"entity": ["work", "job"]}, id="listent")
    if with_neighbour:
        v.add("fox two about work", metadata={"entity": "work"}, id="strent")
    v.flush()
    got = sorted(h["id"] for h in v.search("fox", top_k=10, filter={"entity": "work"},
                                           min_score=-1.0, decompose=False))
    scan = sorted(r["id"] for r in v.get_all_records(where={"entity": "work"}))
    assert got == scan, "filtered search %r != filtered scan %r" % (got, scan)
    v.close()


def test_a_filter_that_matches_nothing_still_returns_nothing(tmp_path, offline_embedder):
    """The control: widening the superset must not start returning rows that do
    not match."""
    v = Vault(str(tmp_path / "nf.dat"))
    v.add("fox one about work", metadata={"entity": ["work", "job"]})
    v.add("fox two about cars", metadata={"entity": "car"})
    v.flush()
    assert v.search("fox", top_k=10, filter={"entity": "boat"},
                    min_score=-1.0, decompose=False) == []
    got = sorted(h["text"] for h in v.search("fox", top_k=10, filter={"entity": "car"},
                                             min_score=-1.0, decompose=False))
    assert got == ["fox two about cars"]
    v.close()


# ---------------------------------------------------------------------------
# internal metadata keys are not silently overwritable
# ---------------------------------------------------------------------------
def test_setting_an_internal_metadata_key_is_refused(tmp_path):
    """`parent_id` is an ordinary thing to want (tickets, threads). Two rows
    tagged `{"parent_id": "TICKET-42"}` read back as ONE chunked document:
    `get("TICKET-42")` returned both texts joined and `exists` said True for a
    document nobody added."""
    v = Vault(str(tmp_path / "r.dat"))
    with pytest.raises(ValueError) as e:
        v.add("A support ticket.", metadata={"parent_id": "TICKET-42"})
    assert "parent_id" in str(e.value)
    assert "my_parent_id" in str(e.value), "the error should suggest a way out"
    v.close()


def test_filtering_on_an_internal_key_still_works(tmp_path):
    """The control: these keys are part of the documented FILTER surface --
    `delete(where={"parent_id": ...})` is how you remove a chunked document."""
    v = Vault(str(tmp_path / "rf.dat"))
    pid = v.add("\n".join("Row %d." % i for i in range(900)))
    v.flush()
    assert len(v.get_all_records(where={"parent_id": pid})) > 1
    assert v.delete(where={"parent_id": pid}) > 1
    v.close()


def test_a_renamed_field_is_accepted(tmp_path):
    v = Vault(str(tmp_path / "rn.dat"))
    v.add("A support ticket.", metadata={"my_parent_id": "TICKET-42"})
    v.flush()
    assert len(v.get_all_records(where={"my_parent_id": "TICKET-42"})) == 1
    v.close()


# ---------------------------------------------------------------------------
# an ingested file is ONE version, not one version per chunk
# ---------------------------------------------------------------------------
POLICY = ("Refund Policy v4\n"
          "Customers may request a refund within 30 days of purchase.\n"
          "Refunds are issued to the original payment method.\n"
          "Shipping charges are not refundable.\n"
          "Digital goods are refundable only if unopened.\n"
          "Enterprise contracts follow the MSA, not this policy.\n"
          "Partial refunds are available for annual plans.\n"
          "Chargebacks void the refund entitlement.\n")


def test_forget_superseded_does_not_dismember_one_ingested_document(tmp_path):
    """`forget_superseded` collapses chunks into versions by `parent_id`, and
    `ingest_file` never stamped one -- so each chunk of ONE document counted as a
    separate revision of the entity and `keep=1` deleted all but one of them.
    Measured: 3 of 4 chunks gone, and the query that answered correctly before
    the call answered with a DIFFERENT passage after, which this method's own
    docstring calls worse than an empty result."""
    src = tmp_path / "policy.txt"
    src.write_text(POLICY, encoding="utf-8")
    v = Vault(str(tmp_path / "p.dat"))
    v.ingest_file(str(src), lines_per_chunk=3, overlap_lines=1,
                  metadata={"entity": "refund_policy"})
    v.flush()
    n = len(v.get_all_records())
    assert n > 1, "this test needs a multi-chunk ingest"
    before = v.search("are shipping charges refundable?", top_k=1)[0]["text"]

    report = v.forget_superseded(keep=1)
    assert report["deleted"] == 0, "deleted %d chunk(s) of one document" % report["deleted"]
    assert len(v.get_all_records()) == n
    after = v.search("are shipping charges refundable?", top_k=1)
    assert after and after[0]["text"] == before
    v.close()


def test_two_ingested_versions_of_a_document_still_collapse(tmp_path):
    """The control that must NOT flip: two files written as the same entity are
    two versions, and the older one is still what `keep=1` is for."""
    a, b = tmp_path / "v1.txt", tmp_path / "v2.txt"
    a.write_text("Refund window is 30 days.\nShipping is not refundable.\n", encoding="utf-8")
    b.write_text("Refund window is 60 days.\nShipping is not refundable.\n", encoding="utf-8")
    v = Vault(str(tmp_path / "t.dat"))
    v.ingest_file(str(a), lines_per_chunk=1, metadata={"entity": "refund"})
    v.ingest_file(str(b), lines_per_chunk=1, metadata={"entity": "refund"})
    v.flush()
    dry = v.forget_superseded(keep=1, dry_run=True)
    assert dry["groups"] == 1 and dry["deleted"] > 0, dry
    v.close()


def test_an_ingested_file_reassembles_in_line_order(tmp_path):
    """Giving ingested chunks a `parent_id` makes `get(parent)` reassemble them,
    and their ids are `name:start-end`, not `_chunk_N` -- so the ordering helper
    has to read the start line or the file comes back shuffled."""
    src = tmp_path / "doc.txt"
    src.write_text("\n".join("line %03d of the document" % i for i in range(300)),
                   encoding="utf-8")
    v = Vault(str(tmp_path / "o.dat"))
    v.ingest_file(str(src))
    v.flush()
    pid = (v.get_all_records()[0].get("metadata") or {}).get("parent_id")
    assert pid, "ingested chunks carry no parent_id"
    nums = [int(x.split()[1]) for x in v.get(pid)["text"].split("\n") if x.startswith("line ")]
    assert nums == sorted(nums), "reassembled out of order"
    assert v.delete(id=pid) == len(nums) or v.get_all_records() == []
    v.close()


# ---------------------------------------------------------------------------
# delete() must apply every criterion it was given
# ---------------------------------------------------------------------------
def _three_rows(tmp_path, name):
    v = Vault(str(tmp_path / name))
    v.add("keep me", source="report.txt", metadata={"team": "a"})
    v.add("delete me", source="report.txt", metadata={"team": "b"})
    v.add("other doc", source="other.txt", metadata={"team": "b"})
    v.flush()
    return v


def test_delete_narrows_when_given_two_criteria(tmp_path, offline_embedder):
    """This was an `elif` chain, so the first criterion that applied decided and
    the rest were ignored: `delete(source="report.txt", where={"team": "b"})`
    deleted BOTH rows of that source, including the one the filter excluded."""
    v = _three_rows(tmp_path, "d1.dat")
    assert v.delete(source="report.txt", where={"team": "b"}) == 1
    assert sorted(r["text"] for r in v.get_all_records()) == ["keep me", "other doc"]
    v.close()


@pytest.mark.parametrize("kw,expected", [
    ({"source": "report.txt"}, 2),
    ({"where": {"team": "b"}}, 2),
    ({}, 0),
    ({"text_exact": "delete me", "where": {"team": "a"}}, 0),
], ids=["source-alone", "where-alone", "nothing", "contradictory"])
def test_delete_single_and_impossible_criteria(tmp_path, offline_embedder, kw, expected):
    """Controls: one criterion behaves as before, no criteria deletes nothing,
    and two that cannot both hold delete nothing."""
    v = _three_rows(tmp_path, "d2.dat")
    assert v.delete(**kw) == expected
    v.close()


# ---------------------------------------------------------------------------
# as_of must survive the multihop path
# ---------------------------------------------------------------------------
def test_as_of_is_honoured_with_multihop(tmp_path, offline_embedder):
    """`search(as_of=..., multihop=True)` delegated to `search_multihop`, which
    did not accept `as_of` at all -- so a question asked "as it stood then" was
    answered with a record written after the cutoff."""
    v = Vault(str(tmp_path / "ao.dat"))
    v.add("old fact about bay one", timestamp=1_600_000_000.0)
    v.add("new fact about bay one", timestamp=1_800_000_000.0)
    v.flush()
    got = [h["text"] for h in v.search("bay one", as_of=1_700_000_000.0, top_k=5, multihop=True)]
    assert not any("new" in t for t in got), "a post-cutoff record leaked: %r" % got
    later = [h["text"] for h in v.search("bay one", top_k=5, multihop=True)]
    assert any("new" in t for t in later), "the control failed: multihop lost the newer record entirely"
    v.close()


# ---------------------------------------------------------------------------
# add_batch reports what it stored; ignore_dirs extends; sets stay filterable
# ---------------------------------------------------------------------------
def test_add_batch_returns_what_it_stored(tmp_path, offline_embedder):
    """It returned `len(records)` -- what it was handed -- so a batch containing
    an empty record reported storing it."""
    v = Vault(str(tmp_path / "ab.dat"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = v.add_batch([{"text": "one"}, {"text": "   "}, {"text": "three"}])
    v.flush()
    assert n == len(v.get_all_records()) == 2
    assert any("no text" in str(w.message) for w in caught)
    v.close()


def test_ignore_dirs_extends_the_automatic_skip_list(tmp_path, offline_embedder):
    """`set(ignore_dirs or [defaults])` meant naming ONE directory silently
    re-enabled every default, so `ignore_dirs=["src"]` indexed `node_modules`."""
    proj = tmp_path / "proj"
    (proj / "node_modules").mkdir(parents=True)
    (proj / "src").mkdir()
    (proj / "src" / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (proj / "node_modules" / "junk.py").write_text("def junk(): pass\n", encoding="utf-8")
    (proj / "keep.py").write_text("def keep(): pass\n", encoding="utf-8")
    v = Vault(str(tmp_path / "id.dat"))
    v.ingest_directory(str(proj), ignore_dirs=["src"])
    v.flush()
    blob = " ".join(r["text"] for r in v.get_all_records())
    assert "junk" not in blob, "a default skip was re-enabled by naming another directory"
    assert "def a" not in blob, "the caller's own skip was ignored"
    assert "keep" in blob, "nothing was indexed at all"
    v.close()


def test_a_set_valued_tag_is_stored_as_a_list_and_stays_filterable(tmp_path, offline_embedder):
    """Metadata is serialised as JSON and a `set` is not serialisable, so it was
    stored as its Python repr, `"{'x', 'y'}"`, and a filter for `"x"` matched
    nothing."""
    v = Vault(str(tmp_path / "sv.dat"))
    v.add("tagged", metadata={"tags": {"x", "y"}, "pair": ("p", "q")})
    v.flush()
    meta = v.get_all_records()[0].get("metadata") or {}
    assert meta["tags"] == ["x", "y"]
    assert meta["pair"] == ["p", "q"]
    assert len(v.get_all_records(where={"tags": "x"})) == 1
    v.close()


# ---------------------------------------------------------------------------
# a revision chain belongs to one group key, not one entity name
# ---------------------------------------------------------------------------
def test_history_does_not_merge_two_tenants_chains(tmp_path, offline_embedder):
    """`history` masked on the interned ENTITY id, which is the entity name
    alone -- but a revision group is keyed on (user_id, project, entity), which
    is why two tenants' first records are both revision 1. So one tenant's
    history came back containing another tenant's value, interleaved by
    timestamp and numbered as if it were one person's chain."""
    v = Vault(str(tmp_path / "h.dat"))
    v.add("I work at Acme.", metadata={"entity": "employer", "user_id": "alice"},
          timestamp=1_700_000_000.0)
    v.add("I work at Zenith.", metadata={"entity": "employer", "user_id": "bob"},
          timestamp=1_700_100_000.0)
    v.add("I moved to Globex.", metadata={"entity": "employer", "user_id": "alice"},
          timestamp=1_700_200_000.0)
    v.flush()
    chain = v.history("where do I work")
    users = {(h.get("metadata") or {}).get("user_id") for h in chain}
    assert len(users) == 1, "history merged %r into one chain" % (users,)
    assert not any("Zenith" in h["text"] for h in chain)
    v.close()


def test_history_still_returns_the_whole_chain_for_one_tenant(tmp_path, offline_embedder):
    """The control that must NOT flip: narrowing to the group key must not start
    truncating an ordinary chain."""
    v = Vault(str(tmp_path / "h2.dat"))
    for i, t in enumerate(("I work at Acme.", "I moved, now Initech.", "Now at Globex.")):
        v.add(t, metadata={"entity": "employer", "user_id": "alice"},
              timestamp=1_700_000_000.0 + i * 1e6)
    v.flush()
    assert len(v.history("where do I work")) == 3
    v.close()


def test_split_by_key_keeps_distinct_values_in_distinct_vaults(tmp_path, offline_embedder):
    """It grouped by the SANITISED filename, so `"a/b"` and `"a_b"` both became
    `a_b.dat` and two teams' records were written into one vault."""
    v = Vault(str(tmp_path / "s.dat"))
    v.add("row one", metadata={"team": "a/b"})
    v.add("row two", metadata={"team": "a_b"})
    v.add("row three", metadata={"team": "c"})
    v.flush()
    out = tmp_path / "split"
    out.mkdir()
    result = v.split_by_key("team", str(out))
    assert len(result) == 3, "distinct values merged: %r" % (result,)
    assert all(n == 1 for n in result.values()), result
    v.close()
