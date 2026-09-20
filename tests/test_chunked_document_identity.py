"""What `add()` split must still be gettable, and deletable, as one document.

`split_large_text` REBUILDS text rather than slicing it: paragraphs are stripped,
units re-joined with " " and "\n\n". So the chunks cannot be concatenated back
into the original and `_join_chunks` is a reconstruction, not an inverse.

0.7.16 fixed `delete(text_exact=...)` on a split document by comparing against
that reconstruction, and the test written for it used a document of unique
space-joined tokens -- the ONE shape where the reconstruction round-trips. Every
real document over 500 words has newlines in it, so the fix covered almost
nothing. Found by the fourth black-box review (F1).

The shapes below are the ones that failed. Each is a document a person would
actually store.
"""
import pytest

from nanomem.vault import Vault

SENTENCES = ["Paragraph %d records that the inspection of bay %d found the seals intact."
             % (i, i) for i in range(160)]

SHAPES = {
    "space_joined_unique": " ".join(SENTENCES),
    "newline_joined": "\n".join(SENTENCES),
    "repeated_passage": " ".join(["The seals were intact."] * 400),
    "csv_export": "\n".join("%d,depot-%d,ok,2026-01-%02d" % (i, i, (i % 28) + 1)
                            for i in range(1200)),
    "indented_code": "\n".join("def f%d():\n    x = %d\n    return x\n" % (i, i)
                               for i in range(220)),
    "application_log": "\n".join("2026-01-01 12:00:%02d INFO bay %d ok" % (i % 60, i)
                                 for i in range(900)),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_delete_by_exact_text_removes_a_split_document_of_any_shape(tmp_path, shape):
    doc = SHAPES[shape]
    v = Vault(str(tmp_path / ("%s.dat" % shape)))
    v.add(doc)
    v.flush()
    n = len(v.get_all_records())
    assert n > 1, "this test needs a SPLIT document; %s produced %d row(s)" % (shape, n)
    assert v.delete(text_exact=doc) == n
    assert v.get_all_records() == []
    v.close()


@pytest.mark.parametrize("shape", ["newline_joined", "csv_export"])
def test_a_near_miss_still_deletes_nothing(tmp_path, shape):
    """The control that must flip. Matching by fingerprint must not become a
    fuzzy match: text that is nearly the document is not the document."""
    doc = SHAPES[shape]
    v = Vault(str(tmp_path / ("near_%s.dat" % shape)))
    v.add(doc)
    v.flush()
    n = len(v.get_all_records())
    assert v.delete(text_exact=doc + " and one more row") == 0
    assert v.delete(text_exact=doc[:-40]) == 0
    assert len(v.get_all_records()) == n
    v.close()


def test_delete_by_the_text_get_returns_keeps_working(tmp_path):
    """0.7.16's changelog told callers to pass `get(parent_id)["text"]`, so that
    has to keep working even though it is the lossy reconstruction and not the
    document. Both forms are documented on `delete`."""
    doc = SHAPES["csv_export"]
    v = Vault(str(tmp_path / "recon.dat"))
    pid = v.add(doc)
    v.flush()
    n = len(v.get_all_records())
    assert v.delete(text_exact=(v.get(pid) or {})["text"]) == n
    assert v.get_all_records() == []
    v.close()


def test_get_on_a_parent_says_it_is_a_reconstruction_and_whether_it_survived(tmp_path):
    """`get`'s docstring said "exact" through 0.7.16 while returning a rebuild.
    A caller could not tell this from a stored row -- for a coding agent, source
    with its indentation silently removed."""
    v = Vault(str(tmp_path / "say.dat"))
    lossless = " ".join("Sentence %d is unique here." % i for i in range(300))
    lossy = SHAPES["csv_export"]
    a, b = v.add(lossless), v.add(lossy)
    v.flush()

    ra = v.get(a)
    assert (ra["metadata"] or {}).get("reconstructed") is True
    assert (ra["metadata"] or {}).get("reconstruction_exact") is True
    assert ra["text"] == lossless

    rb = v.get(b)
    assert (rb["metadata"] or {}).get("reconstructed") is True
    assert (rb["metadata"] or {}).get("reconstruction_exact") is False
    assert rb["text"] != lossy

    short = v.add("a single short fact.")
    v.flush()
    assert "reconstructed" not in (v.get(short)["metadata"] or {})
    v.close()


def test_the_reconstruction_is_lossy_in_the_three_documented_ways(tmp_path):
    """PINS THE LIMITATION rather than pretending it is gone. If the splitter is
    ever changed to slice the original instead of rebuilding it, this fails and
    the docs that describe the loss must be rewritten in the same commit."""
    v = Vault(str(tmp_path / "lossy.dat"))
    nl = v.add(SHAPES["newline_joined"])
    rep = v.add(SHAPES["repeated_passage"])
    code = v.add(SHAPES["indented_code"])
    v.flush()

    assert "\n" not in v.get(nl)["text"], "newlines survived: update the docs"
    assert len(v.get(rep)["text"]) < len(SHAPES["repeated_passage"])
    assert "\n    " not in v.get(code)["text"], "indentation survived: update the docs"
    v.close()


def test_every_chunk_still_carries_the_whole_document(tmp_path):
    """Whatever the join does, NOTHING may be lost from storage. The union of the
    stored chunks must still contain every unique token of the input."""
    words = ["tok%05d" % i for i in range(20000)]
    v = Vault(str(tmp_path / "whole.dat"))
    v.add(" ".join(words))
    v.flush()
    seen = set()
    for r in v.get_all_records():
        seen.update(r["text"].split())
    assert len(seen & set(words)) == len(words)
    v.close()


def test_ingest_file_preserves_formatting_that_add_would_rebuild(tmp_path):
    """`get`'s docstring points callers at `ingest_file` when formatting matters,
    and `ingest_file` claims it "preserves exact code formatting, indentation, and
    line numbers". Both are claims about the same thing, and the second was never
    asserted anywhere -- so this checks it against the one document shape that
    proves the difference: a file with NO blank lines, which `add()` collapses
    because `split_large_text` falls back to `" ".join(words)` for a paragraph
    over `MAX_FACT_WORDS`.
    """
    src = "\n".join("def f%d():\n    x = %d\n    return x" % (i, i) for i in range(400))
    p = tmp_path / "mod.py"
    p.write_text(src, encoding="utf-8")

    v = Vault(str(tmp_path / "ing.dat"))
    assert v.ingest_file(str(p)) > 1, "this test needs a file that chunks"
    v.flush()
    stored = "".join(r["text"] for r in v.get_all_records())
    assert "\n    x = 0" in stored, "ingest_file lost indentation"
    assert stored.count("\n") >= src.count("\n") - 2 * len(v.get_all_records())

    # ...and the contrast that makes the docstring's advice worth following.
    v2 = Vault(str(tmp_path / "added.dat"))
    v2.add(src)
    v2.flush()
    collapsed = "".join(r["text"] for r in v2.get_all_records())
    assert "\n    x = 0" not in collapsed, (
        "add() preserved indentation here -- if the splitter was changed, "
        "`get`'s advice to use ingest_file and the docs describing the rebuild "
        "must be rewritten in the same commit")
    v.close(); v2.close()
