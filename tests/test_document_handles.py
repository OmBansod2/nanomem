"""A document id must survive an update, and the delete fix must reach data that
already exists.

Both defects were found by probing the 0.7.17 release rather than by a review:

* `update()` on a chunked document deleted the chunks and re-added the text
  WITHOUT passing the id, so `add()` minted a new one. `update` returned True
  while the id the caller was holding stopped resolving. The README tells callers
  to keep that id ("`add()` returns the document id, so you can `get`, `update` or
  `delete` by it later") and `update`'s own docstring calls itself "in-place".
* 0.7.17 fixed `delete(text_exact=...)` on split documents with a fingerprint
  stamped at write time -- which reaches only documents written AFTER upgrading.
  Every vault that already existed kept the defect on everything already in it.
"""
import pytest

from nanomem import vault as vault_mod
from nanomem.vault import Vault

LINES = "\n".join("Row %d holds value %d." % (i, i) for i in range(900))
CSV = "\n".join("%d,depot-%d,ok" % (i, i) for i in range(1200))
CODE = "\n".join("def f%d():\n    return %d" % (i, i) for i in range(400))
REPEATED = " ".join(["The seals were intact."] * 400)


def _legacy_vault(tmp_path, monkeypatch, name, doc):
    """A vault whose chunks carry no fingerprint, as every pre-0.7.18 vault does.

    Simulated by neutering the stamp at write time rather than by shipping a
    binary fixture, so the test still describes what it is testing.
    """
    monkeypatch.setattr(vault_mod, "_doc_fingerprint", lambda _t: None)
    v = Vault(str(tmp_path / name))
    v.add(doc)
    v.flush()
    assert not any((r.get("metadata") or {}).get("parent_sha256")
                   for r in v.get_all_records()), "fixture is not legacy-shaped"
    monkeypatch.undo()
    return v


# --------------------------------------------------------------------------
# update() keeps the caller's handle
# --------------------------------------------------------------------------
@pytest.mark.parametrize("replacement,label", [
    ("a much shorter replacement fact.", "shrink to one row"),
    ("\n".join("New row %d." % i for i in range(1200)), "grow to more chunks"),
], ids=["shrink", "grow"])
def test_update_on_a_chunked_document_keeps_its_id(tmp_path, replacement, label):
    v = Vault(str(tmp_path / "u.dat"))
    pid = v.add(LINES, metadata={"entity": "doc"}, timestamp=1_700_000_000.0)
    v.flush()
    assert len(v.get_all_records()) > 1, "this test needs a SPLIT document"

    assert v.update(pid, replacement) is True
    v.flush()

    got = v.get(pid)
    assert got is not None, "update() returned True but the id no longer resolves (%s)" % label
    assert v.exists(pid) is True
    # Compared with whitespace collapsed: when the replacement is itself long
    # enough to split, `get(parent)` returns the documented lossy reconstruction
    # (newlines become spaces), so a byte-exact comparison would be asserting the
    # opposite of what this package documents.
    assert " ".join(got["text"].split()) == " ".join(replacement.split())
    v.close()


def test_update_on_a_chunked_document_keeps_its_timestamp(tmp_path):
    """The parent path re-added the text with no timestamp, so a document silently
    became "written now" -- which moves it in `as_of`, `changes` and the temporal
    prior."""
    v = Vault(str(tmp_path / "t.dat"))
    pid = v.add(LINES, timestamp=1_700_000_000.0)
    v.flush()
    v.update(pid, "a much shorter replacement fact.")
    v.flush()
    assert abs(float(v.get(pid)["timestamp"]) - 1_700_000_000.0) < 1.0
    v.close()


def test_update_still_reports_false_for_an_id_that_is_not_there(tmp_path):
    """The control. Keeping the id must not turn `update` into an upsert."""
    v = Vault(str(tmp_path / "n.dat"))
    v.add("something else entirely.")
    v.flush()
    assert v.update("doc_does_not_exist", "replacement") is False
    assert v.get("doc_does_not_exist") is None
    v.close()


# --------------------------------------------------------------------------
# the delete fix reaches vaults that already exist
# --------------------------------------------------------------------------
@pytest.mark.parametrize("doc,label", [(LINES, "newline joined"), (CSV, "csv"), (CODE, "indented code")])
def test_delete_by_exact_text_reaches_a_legacy_document(tmp_path, monkeypatch, doc, label):
    v = _legacy_vault(tmp_path, monkeypatch, "legacy_%s.dat" % label.split()[0], doc)
    n = len(v.get_all_records())
    assert n > 1
    assert v.delete(text_exact=doc) == n, "legacy document not reachable (%s)" % label
    assert v.get_all_records() == []
    v.close()


def test_a_legacy_repeated_passage_is_honestly_unmatchable(tmp_path, monkeypatch):
    """PINS THE LIMIT of the legacy path. Whitespace normalisation rescues a
    document whose rebuild differs only in whitespace. A repeated passage loses
    real content to the overlap detector, so there is nothing to match against and
    it must NOT be matched by accident."""
    v = _legacy_vault(tmp_path, monkeypatch, "rep.dat", REPEATED)
    n = len(v.get_all_records())
    assert v.delete(text_exact=REPEATED) == 0
    assert len(v.get_all_records()) == n
    v.close()


@pytest.mark.parametrize("mutate,label", [
    (lambda d: d + "\nRow 900 holds value 900.", "one row longer"),
    (lambda d: d[:-40], "truncated"),
    (lambda d: "\n".join("Other %d." % i for i in range(900)), "a different document"),
])
def test_the_legacy_path_never_deletes_a_near_miss(tmp_path, monkeypatch, mutate, label):
    """The control that must flip. Whitespace normalisation is looser than exact
    matching, so it has to be shown it is not loose enough to hit the wrong
    document."""
    v = _legacy_vault(tmp_path, monkeypatch, "near.dat", LINES)
    n = len(v.get_all_records())
    assert v.delete(text_exact=mutate(LINES)) == 0, "near miss deleted something (%s)" % label
    assert len(v.get_all_records()) == n
    v.close()


def test_a_current_vault_is_matched_exactly_not_by_whitespace(tmp_path):
    """The looser rule is scoped to rows with no fingerprint. A document written by
    this version must still be matched on its exact bytes."""
    v = Vault(str(tmp_path / "cur.dat"))
    v.add(CSV)
    v.flush()
    assert all((r.get("metadata") or {}).get("parent_sha256") for r in v.get_all_records())
    n = len(v.get_all_records())
    assert v.delete(text_exact=CSV) == n
    v.close()


# --------------------------------------------------------------------------
# an ingested file is a document too
# --------------------------------------------------------------------------
def test_delete_by_exact_text_reaches_an_ingested_file(tmp_path):
    """`delete` advertises exact-text targeting. An ingested file was the last
    multi-row document it could not match: it returned 0 and removed nothing."""
    p = tmp_path / "mod.py"
    p.write_text(CODE, encoding="utf-8")
    v = Vault(str(tmp_path / "ing.dat"))
    n = v.ingest_file(str(p))
    v.flush()
    assert n > 1
    assert v.delete(text_exact=CODE) == n
    assert v.get_all_records() == []
    v.close()


def test_ingesting_a_file_does_not_make_delete_fuzzy(tmp_path):
    """The control: a file that was not ingested must not be deletable by text."""
    p = tmp_path / "mod.py"
    p.write_text(CODE, encoding="utf-8")
    v = Vault(str(tmp_path / "ing2.dat"))
    v.ingest_file(str(p))
    v.flush()
    n = len(v.get_all_records())
    assert v.delete(text_exact=CODE + "\ndef extra():\n    return 0") == 0
    assert len(v.get_all_records()) == n
    v.close()
