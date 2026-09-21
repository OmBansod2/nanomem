"""Write paths that lose data the caller did not ask to lose.

All three were found in one black-box round against 0.7.19, and they share a
shape: a write path that is looser than the caller's instruction. For a library
whose job is holding facts people cannot afford to lose, looseness on a delete is
the worst direction to be wrong in.
"""
import time

import pytest

from nanomem.vault import Vault

DAY = 86400.0


def _three(tmp_path, name):
    v = Vault(str(tmp_path / name))
    for t in ("fact one", "fact two", "fact three"):
        v.add(t)
    v.flush()
    return v


# ---------------------------------------------------------------------------
def test_delete_with_an_empty_filter_is_refused(tmp_path, offline_embedder):
    """`_matches_filter(meta, {})` is vacuously true, so `delete(where={})`
    erased the whole vault -- and an empty dict is exactly what you get when the
    filter was BUILT and every condition dropped out."""
    v = _three(tmp_path, "e.dat")
    with pytest.raises(ValueError) as e:
        v.delete(where={})
    assert "erase" in str(e.value) or "every record" in str(e.value)
    assert len(v.get_all_records()) == 3, "the vault was modified anyway"
    v.close()


def test_a_filter_that_matches_nothing_is_still_a_no_op(tmp_path, offline_embedder):
    """The control: refusing the EMPTY filter must not break a real one that
    happens to match no rows."""
    v = _three(tmp_path, "n.dat")
    assert v.delete(where={"nonexistent": "x"}) == 0
    assert len(v.get_all_records()) == 3
    v.close()


# ---------------------------------------------------------------------------
def test_delete_by_source_matches_a_name_not_a_substring(tmp_path, offline_embedder):
    """`clean_src in src_str` meant `delete(source="notes.txt")` also deleted
    everything from `meeting_notes.txt`. The caller named one file and lost two,
    with nothing to say so."""
    v = Vault(str(tmp_path / "s.dat"))
    v.add("from notes", source="notes.txt")
    v.add("from meeting notes", source="meeting_notes.txt")
    v.add("from other", source="other.txt")
    v.flush()
    assert v.delete(source="notes.txt") == 1
    left = sorted(r["source"] for r in v.get_all_records())
    assert left == ["meeting_notes.txt", "other.txt"], left
    v.close()


def test_delete_by_source_still_reaches_an_ingested_file(tmp_path, offline_embedder):
    """The control that must NOT flip: `ingest_file` writes its source as
    `"{basename}:{start}-{end}"`, and naming the file has to keep removing it."""
    p = tmp_path / "mod.py"
    p.write_text("\n".join("line %d" % i for i in range(120)), encoding="utf-8")
    v = Vault(str(tmp_path / "i.dat"))
    n = v.ingest_file(str(p))
    v.flush()
    assert n > 1
    assert v.delete(source="mod.py") == n
    assert v.get_all_records() == []
    v.close()


# ---------------------------------------------------------------------------
def test_update_does_not_re_date_the_record(tmp_path, offline_embedder):
    """The worst of the three, because it is this library's own reason to exist
    failing. `update` stamped `time.time()` unconditionally, so relabelling a
    300-day-old row's `source` moved it 300 days forward, made it the answer to
    "where do I work" ahead of two newer values, and then
    `forget_superseded(keep=1)` permanently deleted both of those and reported
    success -- an assistant confidently repeating an address you left two years
    ago, which is the sentence the README opens with."""
    now = time.time()
    v = Vault(str(tmp_path / "u.dat"))
    old_id = v.add("I work at Acme Corp.", metadata={"entity": "employer"},
                   timestamp=now - 300 * DAY)
    v.add("I moved, now Initech.", metadata={"entity": "employer"},
          timestamp=now - 150 * DAY)
    v.add("Now at Globex.", metadata={"entity": "employer"},
          timestamp=now - 10 * DAY)
    v.flush()
    before = float(v.get(old_id)["timestamp"])
    top_before = v.search("where do I work", top_k=1)[0]["text"]

    assert v.update(old_id, source="relabelled") is True
    v.flush()

    after = float(v.get(old_id)["timestamp"])
    assert abs(after - before) < 1.0, "update moved the record %.1f days" % ((after - before) / DAY)
    assert v.get(old_id)["source"] == "relabelled", "the update did not take"
    assert v.search("where do I work", top_k=1)[0]["text"] == top_before

    report = v.forget_superseded(keep=1)
    left = [r["text"] for r in v.get_all_records()]
    assert "Now at Globex." in left, "retention deleted the current value: %r" % left
    assert report["deleted"] == 2
    v.close()


def test_update_still_increments_the_revision(tmp_path, offline_embedder):
    """The control: keeping the timestamp must not stop `update` doing its job."""
    v = Vault(str(tmp_path / "r.dat"))
    i = v.add("a fact.", timestamp=1_700_000_000.0)
    v.flush()
    before = int(v.get(i)["revision"])
    v.update(i, text="a corrected fact.")
    v.flush()
    assert v.get(i)["text"] == "a corrected fact."
    assert int(v.get(i)["revision"]) == before + 1
    assert abs(float(v.get(i)["timestamp"]) - 1_700_000_000.0) < 1.0
    v.close()


# ---------------------------------------------------------------------------
# the guard belongs to the CLASS of write paths, not to one of them
# ---------------------------------------------------------------------------
def test_every_write_surface_refuses_a_reserved_metadata_key(tmp_path, offline_embedder):
    """0.7.19 put this guard on `add()` and nowhere else, so the same defect
    walked in through `add_batch` -- three records carrying
    `{"id": "ticket-4711"}` collapsed onto one handle exactly as they had
    through `add()`. A library whose job is not losing data cannot have a
    validation rule that holds on one of four doors.

    This test is deliberately written over the SURFACES rather than one call, so
    a fifth write path added later fails here until it is listed.
    """
    v = Vault(str(tmp_path / "all.dat"))
    seed = v.add("a seed fact.")
    v.flush()
    src = tmp_path / "f.txt"
    src.write_text("hello\n", encoding="utf-8")

    surfaces = {
        "add": lambda: v.add("x.", metadata={"id": "ticket-4711"}),
        "add_batch": lambda: v.add_batch(
            [{"text": t, "metadata": {"id": "ticket-4711"}} for t in ("a.", "b.")]),
        "update": lambda: v.update(seed, metadata={"parent_id": "X"}),
        "ingest_file": lambda: v.ingest_file(str(src), metadata={"id": "ticket-4711"}),
    }
    for name, call in surfaces.items():
        with pytest.raises(ValueError, match="reserved|cannot be set|read by nanomem"):
            call()
    v.close()


def test_the_round_trip_paths_still_replay_stored_metadata(tmp_path, offline_embedder):
    """The control that must NOT flip. merge, split and export replay records
    that came out of `get_all_records()`, whose metadata legitimately carries the
    reserved keys -- including a split document's `parent_id`. Guarding the
    public surfaces must not break moving a vault."""
    donor = Vault(str(tmp_path / "donor.dat"))
    donor.add("a donor fact.")
    donor.add("\n".join("row %d" % i for i in range(900)))   # a SPLIT document
    donor.flush()
    donor.close()

    v = Vault(str(tmp_path / "host.dat"))
    v.add("a local fact.")
    v.add("tagged", metadata={"team": "x"})
    v.flush()
    report = v.merge(str(tmp_path / "donor.dat"))
    assert report["added"] >= 2, report

    out = tmp_path / "split"
    out.mkdir()
    assert v.split_by_key("team", str(out)) == {"x": 1}
    assert v.export(str(tmp_path / "exported.dat")) >= 1
    v.close()
