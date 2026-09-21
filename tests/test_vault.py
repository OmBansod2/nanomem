"""Vault-level contract: the public API every caller in the repo uses."""

import json
import os
import re
import warnings
import sys
import time

import numpy as np
import pytest

from conftest import D, unit_rows
from nanomem.vault import Vault, TextHopBridge, MAX_SUB_QUERIES


@pytest.fixture
def vault(tmp_path, offline_embedder):
    v = Vault(str(tmp_path / "v.dat"))
    yield v
    v.close()


def test_add_returns_id(vault):
    doc_id = vault.add("a fact about gardening tools")
    assert isinstance(doc_id, str) and doc_id.startswith("doc_")
    assert vault.last_id == doc_id
    assert vault.add("") == ""
    big = " ".join(f"word{i}" for i in range(2000))
    parent = vault.add(big)
    assert isinstance(parent, str)
    vault.flush()
    ids = {r["id"] for r in vault.get_all_records()}
    assert f"{parent}_chunk_1" in ids


def test_eager_creation_and_stats(tmp_path, offline_embedder):
    p = str(tmp_path / "u.dat")
    with Vault(p):
        pass
    assert os.path.exists(p)
    with Vault(p) as v:
        s = v.stats()
    assert s["total_documents"] == 0
    json.dumps(s)


def test_crud_cycle(vault):
    ids = [vault.add(f"fact number {i}", metadata={"k": i % 3}) for i in range(10)]
    vault.flush()
    assert len(vault) == 10
    assert vault.get(ids[3])["text"] == "fact number 3"
    assert ids[3] in vault
    assert vault.update(ids[3], text="fact number three, revised")
    got = vault.get(ids[3])
    assert got["text"] == "fact number three, revised" and got["revision"] == 2
    assert len(vault.get_all_records()) == 10
    assert vault.delete(id=ids[0]) == 1
    assert len(vault.get_all_records()) == 9
    assert vault.delete(where={"k": 1}) == 3
    assert len(vault.get_all_records()) == 6
    assert vault.delete(text_contains="nothing matches this") == 0


def test_get_all_records_order_is_insertion_order(vault):
    texts = [f"ordered record {i}" for i in range(30)]
    for t in texts:
        vault.add(t)
    assert [r["text"] for r in vault.get_all_records()] == texts


def test_records_survive_rebuild(vault):
    for i in range(20):
        vault.add(f"durable {i}", metadata={"keep": i < 10})
    vault.flush()
    before = vault.get_all_records(include_embeddings=True)
    vault.delete(where={"keep": False})
    after = vault.get_all_records(include_embeddings=True)
    assert len(after) == 10
    for b, a in zip(before[:10], after):
        assert b["id"] == a["id"] and b["text"] == a["text"]
        assert float(np.dot(b["embedding"], a["embedding"])) > 0.9999


def test_merge_without_dedup_loses_nothing(tmp_path, offline_embedder):
    a, b = str(tmp_path / "a.dat"), str(tmp_path / "b.dat")
    with Vault(a) as va:
        for i in range(5):
            va.add(f"shared text {i}")
    with Vault(b) as vb:
        for i in range(5):
            vb.add(f"shared text {i}")
        vb.add("unique to b")
    with Vault(a) as va:
        res = va.merge(b, deduplicate=False)
        assert res["added"] == 6 and res["duplicates_skipped"] == 0
        assert len(va.get_all_records()) == 11
        res2 = va.merge(b, deduplicate=True)
        assert res2["duplicates_skipped"] == 6


def test_export_split_and_unmerge(tmp_path, offline_embedder):
    src = str(tmp_path / "src.dat")
    with Vault(src) as v:
        for i in range(12):
            v.add(f"row {i}", metadata={"bucket": "x" if i < 6 else "y"})
        n = v.export(str(tmp_path / "x.dat"), where={"bucket": "x"}, purge=True)
        assert n == 6 and len(v.get_all_records()) == 6
        groups = v.split_by_key("bucket", str(tmp_path / "parts"))
        assert groups == {"y": 6}
    with Vault(str(tmp_path / "x.dat")) as t:
        assert len(t.get_all_records()) == 6


def test_prune_by_age(tmp_path, offline_embedder):
    p = str(tmp_path / "p.dat")
    with Vault(p) as v:
        import time
        now = time.time()
        for i in range(10):
            v.add(f"aged {i}", timestamp=now - i * 86400 * 10)
        v.flush()
        freed = v.prune(older_than_days=25)
        assert freed >= 0
        assert len(v.get_all_records()) == 3


def test_forget_threshold_is_on_the_cosine_scale(vault):
    import inspect
    sig = inspect.signature(Vault.forget)
    assert sig.parameters["min_score"].default == 0.42      # legacy 0.25 re-tuned


def test_detect_entity_has_no_inline_regex():
    import inspect
    src = inspect.getsource(Vault._detect_entity)
    assert "re.search" not in src and "re.compile" not in src
    assert Vault._detect_entity("My phone number is 020-4455-7788.") == "phone_number"
    assert Vault._detect_entity("anything", {"entity": "explicit"}) == "explicit"


def test_password_reaches_the_engine(tmp_path, offline_embedder):
    p = str(tmp_path / "enc.dat")
    with Vault(p, password="a long enough passphrase") as v:
        v.add("a protected note")
        v.flush()
        assert v.stats()["encrypted_at_rest"] is True
    from nanomem.errors import PasswordRequiredError
    with pytest.raises(PasswordRequiredError):
        Vault(p)


def test_search_multihop_uses_a_text_bridge(vault, monkeypatch):
    for i in range(20):
        vault.add(f"bridge document {i} about topic {i % 4}")
    vault.flush()
    calls = []

    class Recording(TextHopBridge):
        def bridge(self, q_vec, d1_vec, question, d1_text):
            calls.append((question, d1_text))
            return super().bridge(q_vec, d1_vec, question, d1_text)

    hits = vault.search_multihop("bridge document 3", top_k=4,
                                 bridge=Recording(vault.embedder))
    assert 0 < len(hits) <= 4
    assert len({h["id"] for h in hits}) == len(hits)          # deduped by id
    assert calls and calls[0][0] == "bridge document 3"
    assert {h.get("hop") for h in hits} <= {1, 2}


def test_search_multihop_falls_back_without_a_bridge(vault):
    class Broken:
        def bridge(self, *a, **k):
            return None
    for i in range(10):
        vault.add(f"plain document {i}")
    vault.flush()
    hits = vault.search_multihop("plain document 1", top_k=3, bridge=Broken())
    assert len(hits) == 3


def test_compact_is_exposed(vault):
    for i in range(120):
        vault.add(f"compactable {i}")
    vault.flush()
    info = vault.compact(recluster=True)
    assert info["records"] == 120 and info["reclustered"] is True
    assert len(vault.get_all_records()) == 120


# ---------------------------------------------------------------------------
# bulk ingest: the row-count hint, end to end
# ---------------------------------------------------------------------------
def _copies(v):
    """How many times this vault's arena has been copied to make room."""
    return v.engine.arena._store.copies


def test_add_batch_sizes_the_arena_for_the_batch_it_was_given(tmp_path, offline_embedder):
    """A bulk caller knows its length; the engine only knows it if told.

    Nothing in the library used to say so, and the arena grew into the count
    instead of being sized for it: 820.5 MB of peak ru_maxrss to build and then
    serve 71,433 documents on the committed engine, against 313.7 MB through
    this method today -- medians of four runs each
    (scratch/refound/ingest_ram_results.json, summary and replicate).
    """
    with Vault(str(tmp_path / "b.dat")) as v:
        V = unit_rows(500, D, seed=7)
        recs = [{"id": f"r{i}", "text": f"batch record {i}",
                 "embedding": V[i], "source": "batch",
                 "metadata": {"idx": i}} for i in range(500)]
        assert v.add_batch(recs) == 500
        assert v.engine.arena.n_rows == 500
        assert v.engine.arena.reserved_rows >= 500
        assert _copies(v) == 0
        assert len(v) == 500
        # The vectors here are seeded noise, so identity is checked by id, not
        # by asking the offline encoder to rediscover them.
        assert v.get("r250")["text"] == "batch record 250"
        assert v.get("r499")["metadata"]["idx"] == 499


def test_add_batch_hint_is_cumulative_across_batches(tmp_path, offline_embedder):
    """Ten batches in a row must not mean ten reallocations."""
    with Vault(str(tmp_path / "c.dat")) as v:
        V = unit_rows(1000, D, seed=8)
        for b in range(10):
            recs = [{"id": f"r{b}_{i}", "text": f"record {b} {i}",
                     "embedding": V[b * 100 + i]} for i in range(100)]
            v.add_batch(recs)
        assert v.engine.arena.n_rows == 1000
        assert v.engine.arena.reserved_rows >= 1000
        assert _copies(v) <= 1


def test_ingest_file_sizes_the_arena_for_its_own_chunks(tmp_path, offline_embedder):
    src = tmp_path / "long.txt"
    src.write_text("\n".join(f"line {i} of a plain text document" for i in range(4000)))
    with Vault(str(tmp_path / "f.dat")) as v:
        n = v.ingest_file(str(src))
        assert n > 50
        assert v.engine.arena.reserved_rows >= n
        assert _copies(v) == 0


def test_ingest_directory_hints_from_bytes_before_reading_anything(tmp_path, offline_embedder):
    """The chunk count is only known after the work; the byte count is free.

    ``ingest_directory`` therefore hints ``total bytes //
    BYTES_PER_CHUNK_ESTIMATE`` before it opens a file. The constant is measured,
    not asserted: scratch/refound/ingest_ram_results.json -> directory_estimate
    carries the bytes-per-chunk of four real trees at the shipped split.
    """
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    for f in range(6):
        d = tree if f % 2 else tree / "sub"
        (d / f"file{f}.txt").write_text(
            "\n".join(f"file {f} line {i} with a little text on it" for i in range(600)))

    hints = []
    with Vault(str(tmp_path / "d.dat")) as v:
        real = v.engine.reserve_additional_rows

        def spy(n):
            hints.append((int(n), int(v.engine.arena.n_rows)))
            return real(n)

        v.engine.reserve_additional_rows = spy
        out = v.ingest_directory(str(tree))

    assert out["files_indexed"] == 6
    first_hint, rows_at_first_hint = hints[0]
    assert rows_at_first_hint == 0            # hinted before anything was read
    assert first_hint > 0
    # A hint is allowed to be wrong; it is not allowed to be absurd.
    assert 0.2 <= first_hint / out["chunks_indexed"] <= 5.0
    assert Vault.BYTES_PER_CHUNK_ESTIMATE == 2048


def test_cli_ingest_accepts_an_explicit_row_count(tmp_path, offline_embedder, monkeypatch, capsys):
    """``nanomem ingest --expect-docs N`` is the hint a loader can type."""
    from nanomem import cli
    from nanomem.engine import VaultEngine
    src = tmp_path / "doc.txt"
    src.write_text("\n".join(f"line {i}" for i in range(300)))
    vault_path = str(tmp_path / "cli.dat")
    seen = []
    real = VaultEngine.reserve_additional_rows

    def spy(self, n):
        seen.append(int(n))
        return real(self, n)

    monkeypatch.setattr(VaultEngine, "reserve_additional_rows", spy)
    monkeypatch.setattr(sys, "argv",
                        ["nanomem", "ingest", str(src), "--vault", vault_path,
                         "--expect-docs", "5000"])
    cli.main()
    out = capsys.readouterr().out
    assert "Ingested" in out
    assert 5000 in seen                        # the typed count reached the arena
    with Vault(vault_path) as v:
        assert len(v) > 0


# --- retention: dropping old revisions without losing current facts ---------

DAY = 86400.0


def _dump(tmp_path, offline_embedder, name="dump.dat"):
    """A vault used as a dump: one fact restated a lot, one restated once, one
    never restated and very old, and one record that is not a fact at all."""
    v = Vault(str(tmp_path / name))
    t = 1_700_000_000.0
    for i, ago in enumerate((400, 300, 200, 100, 5)):
        v.add(f"My locker code is {i}.", metadata={"entity": "locker_code"},
              timestamp=t - ago * DAY)
    for i, ago in enumerate((500, 20)):
        v.add(f"My home address is place-{i}.", metadata={"entity": "home_address"},
              timestamp=t - ago * DAY)
    v.add("My blood type is O negative.", metadata={"entity": "blood_type"},
          timestamp=t - 700 * DAY)
    v.add("A note that is not a fact about anything.", source="document",
          timestamp=t - 900 * DAY)
    return v, t


def _texts(v):
    return [r["text"] for r in v.engine.iter_records()]


def test_forget_superseded_keeps_the_current_value_of_every_fact(tmp_path, offline_embedder):
    v, _t = _dump(tmp_path, offline_embedder)
    out = v.forget_superseded(keep=1)
    assert out["deleted"] == 5                      # 4 locker + 1 address
    for q, want in (("what is my locker code", "locker code is 4"),
                    ("what is my home address", "place-1"),
                    ("what is my blood type", "O negative")):
        assert want in v.search(q, top_k=1)[0]["text"]
    v.close()


def test_forget_superseded_never_touches_a_fact_that_never_changed(tmp_path, offline_embedder):
    """The blood type is 700 days old and still true. Age is not the rule."""
    v, _t = _dump(tmp_path, offline_embedder)
    v.forget_superseded(keep=1)
    assert any("O negative" in x for x in _texts(v))
    v.close()


def test_forget_superseded_never_touches_an_untagged_record(tmp_path, offline_embedder):
    """A record with no entity is in no chain, so nothing supersedes it."""
    v, _t = _dump(tmp_path, offline_embedder)
    v.forget_superseded(keep=1)
    assert any("not a fact about anything" in x for x in _texts(v))
    v.close()


def test_forget_superseded_keep_n_leaves_n(tmp_path, offline_embedder):
    v, _t = _dump(tmp_path, offline_embedder)
    v.forget_superseded(keep=2)
    assert len(v.history("locker code")) == 2
    v.close()


def test_forget_superseded_dry_run_writes_nothing(tmp_path, offline_embedder):
    v, _t = _dump(tmp_path, offline_embedder)
    before = len(_texts(v))
    out = v.forget_superseded(keep=1, dry_run=True)
    assert out["ids"] and out["dry_run"] is True
    assert len(_texts(v)) == before, "a dry run must not write"
    v.close()


def test_forget_superseded_dry_run_predicts_the_real_run_exactly(
        tmp_path, offline_embedder):
    """The preview's numbers are the run's numbers.

    0.7.1 hard-coded ``deleted`` to 0 on a dry run while ``ids`` held the three
    doomed records. The obvious caller --

        plan = v.forget_superseded(keep=2, dry_run=True)
        if plan["deleted"]:
            v.forget_superseded(keep=2)

    -- therefore never cleaned anything, and nothing said so. For a full-rewrite
    delete with no undo, a preview that disagrees with the run is the one thing
    it must never do.
    """
    v, _t = _dump(tmp_path, offline_embedder, name="a.dat")
    plan = v.forget_superseded(keep=2, dry_run=True)
    real = v.forget_superseded(keep=2)
    assert plan["deleted"] == real["deleted"] > 0
    assert plan["ids"] == real["ids"]
    assert plan["groups"] == real["groups"]
    assert plan["kept"] == real["kept"]
    assert plan["dry_run"] is True and real["dry_run"] is False
    v.close()


def test_forget_superseded_kept_counts_records_not_the_knob(
        tmp_path, offline_embedder):
    """``kept`` is an outcome. It used to echo the ``keep`` argument back."""
    v, _t = _dump(tmp_path, offline_embedder)
    out = v.forget_superseded(keep=1)
    # 5 locker + 2 address + 1 blood type are the tagged revisions; keep=1
    # leaves one of each, and the untagged note is not in any group.
    assert out["kept"] == 3, out
    assert out["kept"] != 1, "kept must not be the keep= knob"
    assert out["deleted"] + out["kept"] == 8
    v.close()


def test_forget_superseded_respects_older_than_days(tmp_path, offline_embedder):
    """Only revisions past the age are eligible; recent ones stay even when
    superseded."""
    v, t = _dump(tmp_path, offline_embedder)
    import time as _time
    # the fixture's timestamps are anchored at T0, so age is measured from now;
    # 365 days before NOW is far in the future of T0 -- use the vault's own clock
    out = v.forget_superseded(keep=1, older_than_days=(_time.time() - t) / DAY + 365)
    assert out["deleted"] == 2, out            # the 400-day locker and 500-day address
    assert len(v.history("locker code")) == 4
    v.close()


def test_prune_keeps_the_current_value_of_a_fact_it_would_have_deleted(
        tmp_path, offline_embedder):
    """The case this was found on, pinned so it cannot come back.

    `prune(older_than_days=365)` used to select on age alone, so a blood type
    unchanged for 700 days was deleted and "what is my blood type" then returned
    a LOCKER CODE -- a wrong answer where there had been a right one, with
    nothing to signal it. 0.7.0 documented that and shipped an alternative,
    which was not good enough: a documented trap is still a trap.
    """
    v, _t = _dump(tmp_path, offline_embedder)
    assert "O negative" in v.search("what is my blood type", top_k=1)[0]["text"]
    v.prune(older_than_days=365)
    assert "O negative" in v.search("what is my blood type", top_k=1)[0]["text"]
    v.close()


def test_prune_still_removes_superseded_revisions(tmp_path, offline_embedder):
    """Protecting current values must not turn prune into a no-op: the old
    values of a fact that DID change are still the oldest records there are."""
    v, _t = _dump(tmp_path, offline_embedder, name="still_prunes.dat")
    before = len(v.history("locker code"))
    v.prune(older_than_days=365)
    assert len(v.history("locker code")) < before
    v.close()


def test_prune_still_ages_out_a_document_corpus(tmp_path, offline_embedder):
    """The use case prune was built for is unaffected, because an ingested
    document carries no entity and so is never the current value of a chain.
    If this ever fails, the new default has started costing the case it was
    supposed to leave alone."""
    v = Vault(str(tmp_path / "corpus.dat"))
    t = 1_700_000_000.0
    for i in range(6):
        v.add(f"An ingested paragraph number {i}.", source="corpus.txt",
              timestamp=t - (800 + i) * DAY)
    import time as _time
    v.prune(older_than_days=(_time.time() - t) / DAY + 365)
    assert not [r for r in v.engine.iter_records() if "ingested paragraph" in r["text"]]
    v.close()


def test_prune_keep_current_false_restores_age_alone(tmp_path, offline_embedder):
    """The escape hatch, for a caller who means it -- and the proof that the
    protection is what changed the outcome rather than something else."""
    v, _t = _dump(tmp_path, offline_embedder, name="age_alone.dat")
    v.prune(older_than_days=365, keep_current=False)
    hits = v.search("what is my blood type", top_k=1)
    assert not hits or "O negative" not in hits[0]["text"]
    v.close()


# --- a value long enough to CHUNK is still one value -----------------------

_FILLER = (" Customers are advised that the terms described herein apply to all "
           "orders placed through any channel operated by the company. ")


def _policy_vault(tmp_path, name="pol.dat", anchor=1_700_000_000.0):
    """One policy, three versions, each long enough that `add()` chunks it."""
    v = Vault(str(tmp_path / name))
    for vi, window in enumerate(("14 days", "30 days", "90 days")):
        v.add(f"The refund policy states that the eligibility window is {window} "
              f"from the date of delivery." + _FILLER * 60,
              metadata={"entity": "refund_policy"}, source="policy_doc",
              timestamp=anchor + vi * 100 * DAY)
    v.flush()
    return v


def _answer_present(v):
    return any("eligibility window is 90 days" in t for t in _texts(v))


def test_a_chunked_value_is_chunked_at_all(tmp_path, offline_embedder):
    """If this stops chunking, the three tests below stop testing anything."""
    v = _policy_vault(tmp_path)
    recs = list(v.engine.iter_records())
    assert len(recs) > 3, "expected several chunk records per version"
    assert any((r.get("metadata") or {}).get("is_chunked") for r in recs)
    v.close()


def test_forget_superseded_keeps_every_chunk_of_the_current_version(
        tmp_path, offline_embedder):
    """keep=1 means one VERSION, not one RECORD.

    Chunks inherit `entity`, so a three-version policy stored as four chunks
    each looked like twelve revisions. keep=1 kept a single chunk -- and since
    chunks tie on revision and timestamp, the survivor was a paragraph of
    boilerplate, not the one holding the answer. Measured before the fix:
    12 records -> 1, and `eligibility window is 90 days` was gone.
    """
    v = _policy_vault(tmp_path)
    assert _answer_present(v)
    n_before = len(_texts(v))
    v.forget_superseded(keep=1)
    left = len(_texts(v))
    assert 1 < left < n_before, (n_before, left)
    assert _answer_present(v), "the current version must survive INTACT"
    v.close()


def test_prune_keep_current_protects_every_chunk_of_the_current_version(
        tmp_path, offline_embedder):
    """0.7.1's `keep_current` protected ONE id per group, so a chunked current
    value lost every chunk but one. 9 records -> 1, answer gone."""
    import time as _time
    now = _time.time()
    v = _policy_vault(tmp_path, name="p2.dat", anchor=now - 900 * DAY)
    assert _answer_present(v)
    v.prune(older_than_days=365)
    assert _answer_present(v), "keep_current must protect the whole value"
    v.close()


def test_min_revisions_counts_versions_not_chunk_records(tmp_path, offline_embedder):
    """A policy written ONCE has one revision, however many chunks it became."""
    v = Vault(str(tmp_path / "one.dat"))
    v.add("The sla policy states that the response window is 4 hours."
          + _FILLER * 60, metadata={"entity": "sla_policy"},
          source="policy_doc", timestamp=1_700_000_000.0)
    v.flush()
    n = len(_texts(v))
    assert n > 1, "precondition: it chunked"
    out = v.forget_superseded(keep=1, min_revisions=2)
    assert out["deleted"] == 0, "one version is not two revisions"
    assert len(_texts(v)) == n
    v.close()


# --- backfill: a timestamp beats the arrival counter ------------------------

def test_a_backfilled_older_revision_does_not_become_the_current_value(
        tmp_path, offline_embedder):
    """Writing an OLDER record after a newer one must not make it current.

    `add_fact` numbers a new record `group_max + 1`, i.e. arrival order, which
    is right for a chat log and wrong for an import, a replay or an out-of-order
    sync. Measured before the fix, on three addresses written newest-middle-
    oldest: the current value was the OLDEST of the three, by 350 days, and
    `history` reported them in arrival order rather than time order.
    """
    v = Vault(str(tmp_path / "bf.dat"))
    t = 1_700_000_000.0
    v.add("My home address is 10 Oak Street.",
          metadata={"entity": "home_address"}, timestamp=t + 200 * DAY)
    v.add("My home address is 22 Pine Road.",
          metadata={"entity": "home_address"}, timestamp=t + 400 * DAY)
    v.add("My home address is 3 Elm Lane.",          # backfilled, oldest
          metadata={"entity": "home_address"}, timestamp=t + 50 * DAY)
    v.flush()
    got = v.search("what is my current home address", top_k=1)
    assert got and "22 Pine Road" in got[0]["text"], got
    chain = [h["text"] for h in v.history("my home address")]
    assert len(chain) == 3
    assert "3 Elm Lane" in chain[0], "history must be in TIME order"
    assert "22 Pine Road" in chain[-1]
    v.close()


def test_the_arrival_counter_still_breaks_a_timestamp_tie(tmp_path, offline_embedder):
    """The counter exists for this and must keep doing it.

    Equal timestamps are not an inversion, so concordance must still hold and
    the last write must still win.
    """
    v = Vault(str(tmp_path / "tie.dat"))
    t = 1_700_000_000.0
    for val in ("AAA", "BBB", "CCC"):
        v.add(f"My locker code is {val}.",
              metadata={"entity": "locker_code"}, timestamp=t)
    v.flush()
    got = v.search("what is my current locker code", top_k=1)
    assert got and "CCC" in got[0]["text"], got
    v.close()


# --- top_k means what it says ----------------------------------------------

def test_top_k_zero_or_negative_returns_nothing(tmp_path, offline_embedder):
    """`max(1, top_k)` turned "give me none" into one phantom row."""
    v = Vault(str(tmp_path / "k0.dat"))
    for i in range(5):
        v.add(f"gardening fact number {i} about soil and compost")
    v.flush()
    assert v.search("gardening soil", top_k=0) == []
    assert v.search("gardening soil", top_k=-5) == []
    assert len(v.search("gardening soil", top_k=2)) == 2
    v.close()


def test_top_k_above_fifty_is_not_silently_capped(tmp_path, offline_embedder):
    """A request over 50 was truncated with nothing said, so a caller could not
    tell "only 50 matched" from "we capped you"."""
    v = Vault(str(tmp_path / "k50.dat"))
    for i in range(60):
        v.add(f"gardening fact number {i} about soil and compost")
    v.flush()
    assert len(v.search("gardening soil compost", top_k=60)) == 60
    # bounded by the corpus, which is the only honest bound
    assert len(v.search("gardening soil compost", top_k=500)) == 60
    v.close()


# --- a declared schema does not have to phrase its questions personally -----

def _declared_chain(tmp_path, name="decl.dat"):
    v = Vault(str(tmp_path / name))
    t = 1_700_000_000.0
    for i, val in enumerate(("10 seconds", "30 seconds", "60 seconds", "90 seconds")):
        v.add(f"The checkout service request timeout is {val}.",
              metadata={"entity": "checkout_timeout"}, source="config_audit",
              timestamp=t + i * 90 * DAY)
    v.flush()
    return v, t


def test_a_third_person_question_resolves_a_declared_revision_chain(
        tmp_path, offline_embedder):
    """`temporal_question` gated the entity layer on the QUESTION's wording.

    An application with its own attribute schema asks "what is the checkout
    timeout", not "what is MY checkout timeout", so the layer it adopted nanomem
    for never ran and search fell back to raw cosine over the chain. Measured
    before: the correct current value had the LOWEST cosine of the four and
    ranked last; `as_of` was 2/4. A declared group is now exempt from the
    wording test, because that test compensates for tagger imprecision and a
    declared group has none.
    """
    v, _t = _declared_chain(tmp_path)
    for q in ("what is the checkout service request timeout",
              "what is the checkout timeout",
              "checkout timeout"):
        got = v.search(q, top_k=1)
        assert got and "90 seconds" in got[0]["text"], (q, got)
    v.close()


def test_as_of_works_for_a_third_person_question(tmp_path, offline_embedder):
    v, t = _declared_chain(tmp_path, name="decl2.dat")
    q = "what is the checkout service request timeout"
    for i, val in enumerate(("10 seconds", "30 seconds", "60 seconds", "90 seconds")):
        got = v.search(q, top_k=1, as_of=t + (i * 90 + 10) * DAY)
        assert got and val in got[0]["text"], (i, val, got)
    v.close()


def test_the_exemption_does_not_promote_an_irrelevant_newest_revision(
        tmp_path, offline_embedder):
    """The regression the first version of this fix caused, pinned.

    Reaching the layer by DECLARATION is weaker evidence than reaching it by
    wording: the question never said it was about this fact. So the newest
    revision keeps NO exemption from the relevance floor on that path. Without
    the split, a question about something else promoted a declared group's
    newest revision to rank 1 at cosine 0.016 over a leader at 0.110.
    """
    v, _t = _declared_chain(tmp_path, name="decl3.dat")
    for i in range(20):
        v.add(f"Unrelated archived report paragraph {i} discussing marine biology "
              f"and tidal patterns in coastal estuaries.", source="document",
              timestamp=1_600_000_000.0 + i)
    v.flush()
    got = v.search("marine biology tidal patterns in coastal estuaries", top_k=1)
    assert got, "expected a hit"
    assert "checkout" not in got[0]["text"], (
        "a declared group's newest revision must not win a question that is "
        f"not about it: {got[0]['text']!r}")
    v.close()


def test_an_inferred_group_still_needs_the_wording_gate(tmp_path, offline_embedder):
    """The exemption is for DECLARED entities only.

    A chat write whose entity the tagger guessed keeps the gate exactly as it
    was measured, because that is the imprecision the gate exists for. This
    asserts the exemption cannot fire there, which is why the 3-persona chat set
    scored identically before and after.
    """
    v = Vault(str(tmp_path / "inf.dat"))
    t = 1_700_000_000.0
    for i, val in enumerate(("111", "222")):
        v.add(f"My phone number is {val}.", source="chat_session",
              timestamp=t + i * 90 * DAY)
    v.flush()
    recs = list(v.engine.iter_records())
    assert recs and all(not (r.get("metadata") or {}).get("entity_declared")
                        for r in recs), "precondition: the tagger inferred these"
    v.close()


# --- surface that had no test, pinned after a sweep found it correct -------

def test_export_from_an_encrypted_vault_stays_encrypted(tmp_path, offline_embedder):
    """A disclosure, not a defect, so it is checked in raw BYTES.

    `export` defaults to inheriting the source vault's password. The failure
    this guards against is silent: a readable file containing secrets, with an
    API that reports success either way.
    """
    src = str(tmp_path / "src.dat")
    v = Vault(src, password="correct horse battery staple")
    v.add("My swiss bank account number is ZXQ-4417-SECRET.",
          metadata={"entity": "bank"})
    v.flush()
    dst = str(tmp_path / "out.dat")
    v.export(dst)
    v.close()
    with open(dst, "rb") as fh:
        assert b"ZXQ-4417-SECRET" not in fh.read(), "export leaked plaintext"
    with pytest.raises(Exception):
        Vault(dst, password=None).close()
    # the explicit opt-out still works, because a caller may mean it
    v = Vault(src, password="correct horse battery staple")
    plain = str(tmp_path / "plain.dat")
    v.export(plain, target_password=None)
    v.close()
    with open(plain, "rb") as fh:
        assert b"ZXQ-4417-SECRET" in fh.read()


def test_compact_changes_no_answer(tmp_path, offline_embedder):
    """`compact` is a full rewrite, and full rewrites lost chunked values twice
    today. This pins that this one does not."""
    v = _policy_vault(tmp_path, name="comp.dat")
    v.add("My home address is 22 Pine Road.",
          metadata={"entity": "home_address"}, timestamp=1_700_000_000.0)
    v.flush()
    before = len(_texts(v)), _answer_present(v)
    v.compact()
    assert (len(_texts(v)), _answer_present(v)) == before
    got = v.search("what is my current home address", top_k=1)
    assert got and "22 Pine Road" in got[0]["text"]
    v.close()


def test_deleting_the_current_revision_promotes_the_previous_one(
        tmp_path, offline_embedder):
    """`delete` is older than the revision layer, so this pairing had no test."""
    v = Vault(str(tmp_path / "del.dat"))
    t = 1_700_000_000.0
    ids = [v.add(f"My phone number is {val}.", metadata={"entity": "phone"},
                 timestamp=t + i * 100 * DAY)
           for i, val in enumerate(("111", "222", "333"))]
    v.flush()
    assert "333" in v.search("what is my current phone number", top_k=1)[0]["text"]
    assert v.delete(id=ids[-1]) == 1
    v.flush()
    got = v.search("what is my current phone number", top_k=1)
    assert got and "222" in got[0]["text"], got
    v.close()


def test_migrated_v2_records_are_never_read_as_declared(tmp_path):
    """0.7.5 exempts DECLARED groups from the query-wording gate.

    A v2 vault predates the provenance marker, so its entities were INFERRED by
    the old tagger. If migration marked them declared, every migrated vault
    would silently change its ranking on upgrade. It must not.
    """
    import shutil
    from conftest import golden
    from nanomem.legacy_v2 import migrate_v2_to_v3
    src = golden("chat_v2.dat")                 # skips when goldens are absent
    dst = str(tmp_path / "chat_v2.dat")
    shutil.copy(src, dst)
    migrate_v2_to_v3(dst, backup=None)
    v = Vault(dst)
    recs = list(v.engine.iter_records())
    assert recs, "migration produced nothing"
    tagged = [r for r in recs if (r.get("metadata") or {}).get("entity")]
    assert tagged, "precondition: the v2 vault carried entity tags"
    declared = [r for r in recs if (r.get("metadata") or {}).get("entity_declared")]
    assert declared == [], f"{len(declared)} migrated records read as declared"
    v.close()


# --- add_batch is documented for corpora, so it must behave like add() -----

_BULK_FILLER = ("The operations team reviews logistics throughput and warehouse "
                "staffing levels across all regional distribution centres. ")
_NEEDLE = "The emergency shutdown code for the Reykjavik plant is PUFFIN-77."


def test_add_batch_splits_long_text_like_add_does(tmp_path, offline_embedder):
    """`add_batch` stored whatever it was given verbatim.

    One embedding then had to stand for a whole document, so the vector
    represented its BULK and not its details and a fact stated once near the end
    became unreachable. Measured on a 48,466-character manual against 40
    competing documents: `add()` returned the needle at rank 1 and `add_batch`
    could not place it in the top 5.
    """
    long_doc = _BULK_FILLER * 400 + " " + _NEEDLE
    q = "what is the emergency shutdown code for the Reykjavik plant"

    a = Vault(str(tmp_path / "single.dat"))
    b = Vault(str(tmp_path / "bulk.dat"))
    for v in (a, b):
        for i in range(40):
            v.add(f"Safety bulletin {i}: emergency procedures, shutdown protocols "
                  f"and plant access codes are reviewed quarterly.", source="document")
    a.add(long_doc, metadata={"entity": "manual"})
    b.add_batch([{"text": long_doc, "metadata": {"entity": "manual"}}])
    a.flush(); b.flush()

    # The invariant is PARITY between the two paths, which does not depend on
    # which embedder is in use. (Rank 1 for the needle does: measured with
    # nomic-embed-text, `add` returned it at rank 1 and `add_batch` could not
    # place it in the top 5. The suite runs the offline lexical encoder, where
    # neither path ranks it, so asserting rank 1 here would assert the encoder.)
    assert len(_texts(b)) == len(_texts(a)), "bulk must chunk the same way"
    assert len(_texts(a)) > 41, "precondition: the long document chunked"
    assert [h["text"] for h in a.search(q, top_k=5)] == \
           [h["text"] for h in b.search(q, top_k=5)], \
           "add_batch must rank identically to add"
    chunked = [r for r in b.engine.iter_records()
               if (r.get("metadata") or {}).get("is_chunked")]
    assert chunked, "add_batch must mark its chunks"
    assert any("PUFFIN-77" in r["text"] for r in chunked), \
        "the tail must land in a chunk of its own, not be diluted into one vector"
    a.close(); b.close()


def test_add_batch_does_not_re_split_a_transfer(tmp_path, offline_embedder):
    """A caller that supplies embeddings is TRANSFERRING whole rows.

    `merge`, `split` and every rebuild go through this path. Re-splitting there
    would invent records the source never had, so the chunking above must not
    apply when vectors come with the records.
    """
    src = Vault(str(tmp_path / "src.dat"))
    src.add("The refund policy eligibility window is 90 days."
            + _BULK_FILLER * 200, metadata={"entity": "refund_policy"})
    src.add("My phone number is 222.", metadata={"entity": "phone"})
    src.flush()
    n_src = len(_texts(src))
    assert n_src > 2, "precondition: the long record chunked"
    dst_path = str(tmp_path / "dst.dat")
    src.export(dst_path)
    src.close()

    dst = Vault(dst_path)
    assert len(_texts(dst)) == n_src, "a transfer must not re-chunk"
    got = dst.search("what is my current phone number", top_k=1)
    assert got and "222" in got[0]["text"]
    dst.close()


def test_add_batch_agrees_with_add_on_a_declared_chain(tmp_path, offline_embedder):
    """Bulk ingestion is how an application with a schema actually writes."""
    t = 1_700_000_000.0
    vals = ("10 seconds", "30 seconds", "60 seconds", "90 seconds")
    rows = [{"text": f"The checkout service request timeout is {v}.",
             "metadata": {"entity": "checkout_timeout"}, "source": "config_audit",
             "timestamp": t + i * 90 * DAY}
            for i, v in enumerate(vals)]
    v = Vault(str(tmp_path / "bulk2.dat"))
    v.add_batch([rows[1], rows[3], rows[0], rows[2]])      # out of order on purpose
    v.flush()
    recs = list(v.engine.iter_records())
    assert all((r.get("metadata") or {}).get("entity_declared") for r in recs)
    q = "what is the checkout service request timeout"
    got = v.search(q, top_k=1)
    assert got and "90 seconds" in got[0]["text"], got
    for i, val in enumerate(vals):
        h = v.search(q, top_k=1, as_of=t + (i * 90 + 10) * DAY)
        assert h and val in h[0]["text"], (i, val, h)
    v.close()


# --- retention must agree with the ranker about which record is current ----

def _out_of_order_chain(tmp_path, name):
    """Arrival order A, B, C; A is newest BY TIME, C is newest BY ARRIVAL."""
    v = Vault(str(tmp_path / name))
    t = 1_700_000_000.0
    for val, off in (("A", 400), ("B", 100), ("C", 200)):
        v.add(f"My phone number is value-{val}.",
              metadata={"entity": "phone"}, timestamp=t + off * DAY)
    v.flush()
    return v


def test_forget_superseded_keeps_the_record_search_calls_current(
        tmp_path, offline_embedder):
    """Found by the operation fuzzer (scratch/refound/exp_fuzz_ops.py).

    0.7.4 made the RANKER prefer the timestamp when it disagrees with the
    arrival counter. Retention still sorted by `(revision, timestamp)`, so on a
    vault written out of order the two halves disagreed about which record was
    current: `search` returned the newest BY TIME and `forget_superseded` kept
    the newest BY ARRIVAL, deleting the record search had just called the
    answer. Measured before the fix: search said value-A, keep=1 kept value-C.
    """
    v = _out_of_order_chain(tmp_path, "fs.dat")
    before = v.search("what is my current phone number", top_k=1,
                      filter={"entity": "phone"})
    assert before and "value-A" in before[0]["text"], before
    v.forget_superseded(keep=1)
    left = _texts(v)
    assert len(left) == 1 and "value-A" in left[0], left
    after = v.search("what is my current phone number", top_k=1,
                     filter={"entity": "phone"})
    assert after and "value-A" in after[0]["text"], after
    v.close()


def test_prune_protects_the_record_search_calls_current(tmp_path, offline_embedder):
    """`_current_value_ids` ranked by `(revision, timestamp)` too, so
    `keep_current` protected the newest by ARRIVAL and left the real current
    value exposed to the age cutoff."""
    import time as _time
    now = _time.time()
    v = Vault(str(tmp_path / "pr.dat"))
    for val, off in (("A", -400), ("B", -900), ("C", -800)):
        v.add(f"My phone number is value-{val}.",
              metadata={"entity": "phone"}, timestamp=now + off * DAY)
    v.flush()
    v.prune(older_than_days=365)
    left = _texts(v)
    assert any("value-A" in t for t in left), (
        f"the newest by time must survive keep_current: {left}")
    v.close()


def test_the_arrival_counter_still_breaks_a_retention_tie(tmp_path, offline_embedder):
    """Time first, arrival only to break a tie -- the tie must still break."""
    v = Vault(str(tmp_path / "tie2.dat"))
    t = 1_700_000_000.0
    for val in ("A", "B", "C"):
        v.add(f"My phone number is value-{val}.",
              metadata={"entity": "phone"}, timestamp=t)      # identical stamps
    v.flush()
    v.forget_superseded(keep=1)
    left = _texts(v)
    assert len(left) == 1 and "value-C" in left[0], (
        f"with equal timestamps the last write is current: {left}")
    v.close()


def test_the_floor_protects_the_current_value_of_an_out_of_order_chain(
        tmp_path, offline_embedder):
    """A regression 0.7.4 introduced and the operation fuzzer caught.

    The relevance floor exempts a declared group's current value from being
    dropped. That exemption asked `_revisions_comparable`, and 0.7.4 added a
    CONCORDANCE condition to that predicate -- so the protection switched off on
    exactly the chains 0.7.4 existed for, the ones whose arrival order and
    timestamps disagree. The floor then dropped the current value and a
    superseded record answered.

    Found on a chain whose revisions read 5, 3, 4 in time order: `search`
    returned the MIDDLE record of three. The one-key test and the concordance
    test are now separate predicates, and the protected record is chosen by
    timestamp with the counter only breaking a tie.
    """
    v = Vault(str(tmp_path / "ooo.dat"))
    # arrival order 11, 12, 38 -> revisions 1, 2, 3
    # time order    38, 11, 12 -> the counter and the clock disagree
    for val, ts in (("value-11", 1_736_771_200.0),
                    ("value-12", 1_753_273_600.0),
                    ("value-38", 1_633_868_800.0)):
        v.add(f"My phone is {val}.", metadata={"entity": "phone"}, timestamp=ts)
    v.flush()
    got = v.search("what is the current phone", top_k=1, filter={"entity": "phone"})
    assert got and "value-12" in got[0]["text"], (
        f"the newest by time must survive the floor: {got}")
    v.close()


def test_the_entity_prefilter_changes_no_result(tmp_path, offline_embedder):
    """A filtered search must return exactly what it returned before.

    `_apply_filter` walks records in score order decoding one per step, and its
    early-stop bound cannot fire until `top_k` matches are in hand -- so a
    selective filter walked the WHOLE corpus: 15,001 decodes and 33.70 ms
    against 0.70 ms unfiltered. The walk is now narrowed by the interned entity
    column first.

    That column holds `normalize_entity(...)` while `matches_filter` compares
    the RAW metadata, so this test exists to prove the narrowing is a SUPERSET
    and not a semantic change. Normalization is a function, so raw == filter
    implies normalize(raw) == normalize(filter); every surviving row still goes
    through `matches_filter` unchanged. The values below are deliberately
    awkward -- case, spaces, unicode, ints, lists, absent keys.
    """
    import random
    rng = random.Random(7)
    ents = ["home_address", "Home_Address", "home address", "日本", "cfg_1", None]
    rows = []
    for i in range(600):
        e = rng.choice(ents)
        m = {}
        if e is not None:
            m["entity"] = e
        if rng.random() < 0.5:
            m["user_id"] = rng.choice(["a", "b", "c"])
        if rng.random() < 0.3:
            m["tags"] = rng.sample(["x", "y", "z"], 2)
        rows.append({"text": f"record {i} about soil compost drainage beds {e}",
                     "metadata": m, "timestamp": 1_700_000_000.0 + i})
    v = Vault(str(tmp_path / "pf.dat"))
    v.add_batch(rows)
    v.flush()

    filters = []
    for e in [x for x in ents if x is not None]:
        filters += [{"entity": e}, {"entity": e, "user_id": "a"},
                    {"entity": e, "tags": ["x"]}, {"entity": e.upper()}]
    filters += [{"user_id": "b"}, {"entity": ["cfg_1"]}, {"entity": "missing"},
                {"entity": 123}]

    def run():
        return [[h["id"] for h in v.search("soil compost drainage", top_k=k,
                                           filter=f)]
                for f in filters for k in (1, 3, 10)]

    fast = run()
    arena_cls = type(v.engine.arena)
    original = arena_cls.entity_index
    arena_cls.entity_index = property(lambda self: {})   # forces the old walk
    try:
        slow = run()
    finally:
        arena_cls.entity_index = original
    assert fast == slow, "the prefilter changed a filtered result"
    assert sum(1 for r in fast if r) > len(fast) // 3, \
        "too many empty result sets for this to be testing anything"
    v.close()


# --- surface with no test until now; all of it already worked --------------

def test_unmerge_is_the_inverse_of_merge(tmp_path, offline_embedder):
    """`merge` stamps `source_vault`; `unmerge` matches on it.

    Neither half had a test, and `unmerge` keys on `source_vault` or `project`
    and NOT on `user_id` -- so the pair is only an inverse if merge actually
    stamps the records. It does, and this pins the round trip.
    """
    main = Vault(str(tmp_path / "main.dat"))
    main.add("My phone number is 111.", metadata={"entity": "phone"})
    main.add("My phone number is 222.", metadata={"entity": "phone"})
    main.flush()
    n0 = len(_texts(main))

    side_path = str(tmp_path / "partner.dat")
    side = Vault(side_path)
    for t in ("Partner fact one about shipping.", "Partner fact two about shipping."):
        side.add(t, metadata={"entity": "shipping"})
    side.flush(); side.close()

    main.merge(side_path)
    main.flush()
    assert len(_texts(main)) == n0 + 2
    stamped = [r for r in main.engine.iter_records()
               if (r.get("metadata") or {}).get("source_vault")]
    assert len(stamped) == 2, "merge must stamp what it brought in"

    out_path = str(tmp_path / "extracted.dat")
    assert main.unmerge("partner.dat", target_vault_path=out_path) == 2
    main.flush()
    assert len(_texts(main)) == n0, "unmerge must restore the original size"
    got = main.search("what is my current phone number", top_k=1)
    assert got and "222" in got[0]["text"], "the host's own facts must survive"
    main.close()

    extracted = Vault(out_path)
    assert len(_texts(extracted)) == 2, "the detached records must land in the target"
    extracted.close()


def test_ingest_file_chunks_and_stays_retrievable(tmp_path, offline_embedder):
    """A fact stated once at the top of a long file must still be findable."""
    path = tmp_path / "doc.txt"
    path.write_text("The shutdown code is PUFFIN-77.\n"
                    + "Filler sentence about logistics throughput. " * 400)
    v = Vault(str(tmp_path / "ing.dat"))
    v.ingest_file(str(path))
    v.flush()
    assert len(_texts(v)) > 1, "a long file must chunk"
    got = v.search("what is the shutdown code", top_k=1)
    assert got and "PUFFIN-77" in got[0]["text"], got
    v.close()


def test_staleness_reports_a_chain_it_can_act_on(tmp_path, offline_embedder):
    v = Vault(str(tmp_path / "st.dat"))
    t = 1_700_000_000.0
    for i, val in enumerate(("111", "222", "333")):
        v.add(f"My phone number is {val}.",
              metadata={"entity": "phone", "user_id": "alice"},
              timestamp=t + i * 100 * DAY)
    v.flush()
    rows = v.staleness()
    assert rows, "a three-deep chain must produce a staleness row"
    row = next(r for r in rows if r.get("entity") == "phone")
    assert row["n_revisions"] == 3 and row["user_id"] == "alice"
    json.dumps(rows, default=str)          # must survive the API boundary
    v.close()


def test_split_by_key_counts_every_record(tmp_path, offline_embedder):
    v = Vault(str(tmp_path / "sp.dat"))
    for i in range(3):
        v.add(f"My phone number is {i}.",
              metadata={"entity": "phone", "user_id": "alice"})
    for i in range(6):
        v.add(f"Report paragraph {i} on logistics.", source="document",
              metadata={"user_id": "bob"})
    v.flush()
    out = v.split_by_key("user_id", str(tmp_path / "parts"))
    assert out == {"alice": 3, "bob": 6}, out
    v.close()



# --- a question asked after word 100 -----------------------------------------

def test_a_question_past_word_100_still_reaches_the_query(tmp_path, offline_embedder):
    """`" ".join(words[:100])` discarded everything after the hundredth word.

    The shape it ruined is the one people actually send: context pasted first,
    the real question LAST. The assertion is on the TEXT the engine is asked
    about, not on a rank, because the offline encoder here is lexical and the
    measurement that motivated this is in query_truncation_results.json.

    `decompose=False` is here to isolate the truncation, and that isolation is
    exactly how 0.7.10 shipped an incomplete fix: on the DEFAULT path the
    question was still discarded, by the merge ordering instead of by the cap,
    and neither this test nor the release check could see it. The default path
    is covered by
    `test_the_question_beats_the_preamble_on_the_default_path`; do not let this
    test stand alone for that shape again.
    """
    v = Vault(str(tmp_path / "trunc.dat"))
    v.add("The production database runs PostgreSQL 16 on the primary cluster.")
    v.flush()

    preamble = " ".join(["meeting notes and scheduling filler"] * 40)   # 200 words
    question = "which database engine does production run"
    seen = []
    real_search = v.engine.search

    def spy(*a, **kw):
        seen.append(kw.get("query_text", ""))
        return real_search(*a, **kw)

    v.engine.search = spy
    v.search(preamble + " " + question, decompose=False)
    v.engine.search = real_search

    assert seen, "engine.search was never called"
    assert question in seen[0], (
        f"the question was dropped before it reached the engine: {seen[0][-80:]!r}")
    v.close()


def test_search_multihop_does_not_truncate_the_query_either(tmp_path, offline_embedder):
    """The same two lines existed in `search_multihop`. Fixing one is half a fix."""
    v = Vault(str(tmp_path / "trunc_mh.dat"))
    v.add("Session state is held in Redis with a 30 minute expiry.")
    v.flush()

    preamble = " ".join(["unrelated preamble about parking and catering"] * 30)
    question = "where is session state held"
    seen = []
    real_search = v.engine.search
    v.engine.search = lambda *a, **kw: (seen.append(kw.get("query_text", "")),
                                        real_search(*a, **kw))[1]
    v.search_multihop(preamble + " " + question)
    v.engine.search = real_search

    assert seen, "engine.search was never called"
    assert question in seen[0], "search_multihop still truncates at 100 words"
    v.close()


def test_a_pasted_document_is_not_decomposed_into_hundreds_of_scans(
        tmp_path, offline_embedder):
    """Every sub-query is its own full scan, so decomposition multiplies cost.

    A pasted FAQ came back as 400 sub-queries. Lifting the 100-word cap would
    have made that reachable on any paste, so the fan-out is bounded directly.
    """
    v = Vault(str(tmp_path / "fanout.dat"))
    for i in range(5):
        v.add(f"service number {i} runs on its own dedicated host")
    v.flush()

    faq = " ".join(f"What is item {i} and why does it matter?" for i in range(200))
    assert len(Vault.decompose_query(faq)) > MAX_SUB_QUERIES, "corpus no longer provokes it"

    calls = []
    real_search = v.engine.search
    v.engine.search = lambda *a, **kw: (calls.append(1), real_search(*a, **kw))[1]
    v.search(faq, top_k=3)
    v.engine.search = real_search

    assert len(calls) == 1, (
        f"a pasted document ran {len(calls)} scans; it should collapse to one query")
    v.close()


def test_a_genuine_composite_question_is_still_decomposed(tmp_path, offline_embedder):
    """The bound must not cost the feature it is protecting.

    99.95% of 145,051 real queries decompose to 8 or fewer, so anything in that
    range must still fan out.
    """
    v = Vault(str(tmp_path / "composite.dat"))
    v.add("BPE is a subword tokenisation algorithm.")
    v.add("KV caching stores past attention keys and values.")
    v.flush()

    q = "What is BPE, and how does KV caching accelerate attention?"
    assert 1 < len(Vault.decompose_query(q)) <= MAX_SUB_QUERIES

    calls = []
    real_search = v.engine.search
    v.engine.search = lambda *a, **kw: (calls.append(1), real_search(*a, **kw))[1]
    v.search(q, top_k=3)
    v.engine.search = real_search

    assert len(calls) > 1, "a real composite question stopped being decomposed"
    v.close()


# --- the id add() hands back must work on a chunked document -----------------

def _long_doc(marker="zqxjkv", words=1200):
    w = ["filler"] * words
    w[470] = marker                     # inside the 40-word overlap window
    w[900] = marker + "2"               # in a later chunk only
    return " ".join(w)


def test_the_id_add_returns_is_usable_on_a_chunked_document(tmp_path, offline_embedder):
    """`add()` documents its return as "the parent id ... for a chunked document"
    and the README says you can `get`, `update` or `delete` by it later.

    Every one of those was a silent no-op past 500 words: the parent id is not a
    stored row, so `get` returned None, `exists` False, `update` False and
    `delete` 0 -- no exception anywhere, and the document stayed on disk.
    """
    v = Vault(str(tmp_path / "chunked_id.dat"))
    doc_id = v.add(_long_doc())
    v.flush()
    rows = v.get_all_records()
    assert len(rows) > 1, "corpus no longer chunks; the test proves nothing"
    assert all(r["id"].startswith(doc_id + "_chunk_") for r in rows)

    assert v.exists(doc_id), "the id add() returned does not exist"
    got = v.get(doc_id)
    assert got is not None, "get(parent_id) returned None"
    assert "zqxjkv" in got["text"] and "zqxjkv2" in got["text"], \
        "get() must return the whole document, not one chunk"

    n = v.delete(id=doc_id)
    v.flush()
    assert n == len(rows), f"delete(id=parent) removed {n} of {len(rows)} chunks"
    assert v.get_all_records() == [], "chunks survived delete(id=parent)"
    v.close()


def test_get_on_a_chunked_document_does_not_duplicate_the_overlap(
        tmp_path, offline_embedder):
    """Chunks share a 40-word overlap bridge, so naive concatenation would repeat
    text. The marker sits inside that bridge and must appear once."""
    v = Vault(str(tmp_path / "overlap.dat"))
    doc_id = v.add(_long_doc())
    v.flush()
    carriers = [r for r in v.get_all_records() if "zqxjkv " in r["text"] + " "]
    assert len(carriers) >= 2, "marker is not in the overlap; test proves nothing"
    text = v.get(doc_id)["text"]
    assert text.split().count("zqxjkv") == 1, \
        f"overlap duplicated on reassembly: {text.split().count('zqxjkv')} copies"
    v.close()


def test_delete_ids_list_and_update_also_accept_a_parent_id(tmp_path, offline_embedder):
    """`delete(ids=[...])` and `update()` share the by-id path, so they shared
    the defect."""
    v = Vault(str(tmp_path / "parent_more.dat"))
    doc_id = v.add(_long_doc())
    v.flush()
    assert v.update(doc_id, text="a short replacement document") is True
    v.flush()
    rows = v.get_all_records()
    assert len(rows) == 1 and rows[0]["text"] == "a short replacement document"

    doc2 = v.add(_long_doc(marker="qqwwee"))
    v.flush()
    n_before = len(v.get_all_records())
    assert v.delete(ids=[doc2]) == n_before - 1
    v.close()


def test_a_short_document_still_behaves_exactly_as_before(tmp_path, offline_embedder):
    """The parent-id path must not change the ordinary un-chunked case."""
    v = Vault(str(tmp_path / "short.dat"))
    doc_id = v.add("a single short fact that will never be split")
    v.flush()
    assert v.exists(doc_id)
    assert v.get(doc_id)["text"] == "a single short fact that will never be split"
    assert v.update(doc_id, text="replaced") is True
    v.flush()
    assert v.get(doc_id)["text"] == "replaced"
    assert v.delete(id=doc_id) == 1
    assert v.get_all_records() == []
    v.close()


# --- three documented arguments that did not exist, and one that lied --------

class _TinyEmbedder:
    """The minimum the README's `embedder=` line promises: .embed/.embed_batch/.dim"""
    dim = 16

    def embed(self, text):
        import hashlib
        h = hashlib.md5(str(text).encode()).digest()
        return np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:16] / 255.0

    def embed_batch(self, texts):
        return np.stack([self.embed(t) for t in texts])


def test_a_caller_supplied_embedder_is_accepted(tmp_path):
    """README: `Vault("m.dat", embedder=MyOwnEmbedder())` — which raised TypeError."""
    v = Vault(str(tmp_path / "emb.dat"), embedder=_TinyEmbedder())
    assert v.stats()["embed_dim"] == 16
    v.add("a fact stored through a caller-supplied embedder")
    v.flush()
    assert v.search("a fact")[0]["text"].startswith("a fact stored")
    v.close()


def test_a_bad_embedder_is_refused_by_name(tmp_path):
    """Duck typing that fails later is worse than a clear refusal now."""
    class Half:
        dim = 8
        def embed(self, t): return [0.0] * 8
    with pytest.raises(TypeError, match="embed_batch"):
        Vault(str(tmp_path / "bad.dat"), embedder=Half())


def test_vector_dtype_and_group_floor_sim_reach_the_engine(tmp_path, offline_embedder):
    """Both are documented — one in this class's docstring, one in the README as
    a +19.0 point tuning step — and both raised TypeError."""
    v = Vault(str(tmp_path / "dt.dat"), vector_dtype="float32", group_floor_sim=0.45)
    v.add("a fact")
    v.flush()
    assert v.engine.header.vector_dtype == "float32"
    assert v.engine.group_floor_sim == 0.45
    v.close()
    d = Vault(str(tmp_path / "def.dat"))
    assert d.engine.header.vector_dtype == "float16"     # default unchanged
    assert d.engine.group_floor_sim == 0.0
    d.close()


def test_prune_documents_that_it_returns_bytes(tmp_path, offline_embedder):
    """`prune() -> int` next to `delete() -> "number of deleted records"` reads
    as a record count. It is bytes, and now says so."""
    assert "BYTES FREED" in (Vault.prune.__doc__ or "")
    v = Vault(str(tmp_path / "pr.dat"))
    now = time.time()
    for i in range(9):
        v.add(f"fact {i} about assorted unrelated topics", timestamp=now - (400 + i) * DAY)
    v.flush()
    before = len(v.get_all_records())
    freed = v.prune(older_than_days=365)
    v.flush()
    removed = before - len(v.get_all_records())
    assert removed > 0 and freed > removed, \
        f"prune returned {freed} against {removed} records removed"
    v.close()


def test_cli_add_can_declare_an_entity(tmp_path, offline_embedder, capsys):
    """`nanomem add` had --source, --vault and --profile only, so the README's
    own mitigation for the tagger could not be applied from the CLI either."""
    from nanomem.cli import main
    vp = str(tmp_path / "cli.dat")
    chain = ["I work at Acme Corp.",
             "I moved jobs, I now work at Initech.",
             "I switched again, I work at Globex now."]
    for t in chain:
        sys.argv = ["nanomem", "add", t, "--entity", "employer", "--vault", vp]
        main()
    out = capsys.readouterr().out
    assert "as 'employer'" in out, "the CLI must confirm what it recorded"

    v = Vault(vp)
    assert all((r.get("metadata") or {}).get("entity") == "employer"
               for r in v.get_all_records())
    cur = [e["text"] for e in v.history("where do I work") if not e.get("superseded")]
    assert v.search("where do I work", top_k=1)[0]["text"] in cur
    v.close()


# --- decomposition must not let input ORDER decide rank 1 --------------------

_F2_FACTS = [
    "The staging database listens on port 5433.",
    "Backups run nightly at 0200 UTC.",
    "The Zurich office parking code is 4417.",
    "The VPN concentrator sits in rack B12.",
    "The on-call rota rolls over every Monday at 09:00.",
]
_F2_PREAMBLE = ("Hi all, following up on yesterday's thread, a few bits are still "
                "open and I wanted them written down, also the room booking has "
                "moved again, apologies for the churn ")


def _f2_vault(tmp_path, name):
    v = Vault(str(tmp_path / name))
    for f in _F2_FACTS:
        v.add(f)
    v.flush()
    return v


def test_rank_one_is_not_decided_by_which_sub_query_came_first(
        tmp_path, offline_embedder):
    """The merge interleaved sub-query hits round-robin IN INPUT ORDER, so rank 1
    was always the best hit of whatever text appeared first.

    For a pasted email, chat turn, or any question-after-context, that is the
    preamble — so the user's actual question contributed nothing to rank 1. The
    control that proves it: the preamble ALONE returns the same record as the
    preamble plus any question.
    """
    v = _f2_vault(tmp_path, "order.dat")
    preamble_only = v.search(_F2_PREAMBLE, top_k=1)[0]["text"]

    distinct = set()
    for q in ("Which port does the staging database listen on?",
              "What is the Zurich parking code?",
              "Where is the VPN concentrator?"):
        full = _F2_PREAMBLE + q
        assert len(Vault.decompose_query(full)) > 1, "no longer decomposes"
        distinct.add(v.search(full, top_k=1)[0]["text"])

    assert len(distinct) > 1, (
        "every question returned the same record; rank 1 ignored the question")
    assert distinct != {preamble_only}, (
        "rank 1 is exactly what the preamble alone returns")
    v.close()


def test_the_question_beats_the_preamble_on_the_default_path(
        tmp_path, offline_embedder):
    """0.7.10 removed the 100-word query cap for exactly this shape and verified
    it with `decompose=False` — the one path where this defect cannot appear.
    The default path is what users get."""
    v = _f2_vault(tmp_path, "default.dat")
    for q, want in (("Which port does the staging database listen on?", "5433"),
                    ("What is the Zurich parking code?", "4417"),
                    ("Where is the VPN concentrator?", "B12")):
        top = v.search(_F2_PREAMBLE + q, top_k=1)[0]["text"]
        assert want in top, f"{q!r} -> {top!r} (default path, decompose on)"
    v.close()


def test_a_composite_question_still_answers_every_part(tmp_path, offline_embedder):
    """The reason the merge interleaves at all: a real multi-part question must
    get hits for each part, not top_k hits for its strongest clause. Ordering by
    score must not cost that."""
    v = _f2_vault(tmp_path, "composite.dat")
    q = "What is the Zurich parking code, and where is the VPN concentrator?"
    assert len(Vault.decompose_query(q)) > 1
    texts = " ".join(h["text"] for h in v.search(q, top_k=4))
    assert "4417" in texts and "B12" in texts, \
        f"a composite question lost one of its parts: {texts!r}"
    v.close()


# --- history must not report a truncated chain as "never changed" ------------

_OFFICE = ["Desk was on the third floor of Kestrel House.",
           "Moved down to the annexe at Larkfield.",
           "Now parked in the Maple Wharf building."]


def _office_vault(tmp_path, name="hist.dat"):
    v = Vault(str(tmp_path / name))
    now = 1_700_000_000.0
    for i, t in enumerate(_OFFICE):
        v.add(t, metadata={"entity": "office"}, timestamp=now - (2 - i) * 150 * DAY)
    v.flush()
    return v


def test_history_returns_the_whole_declared_chain(tmp_path, offline_embedder):
    """`history` returned ONE entry flagged superseded=False on a three-revision
    declared chain — which this library defines as "this fact never changed" —
    while `changes()` and `get_all_records()` saw all three in the same vault.

    Not the relevance floor, which `history` already skips. `resolve_top_entity`
    returned the question's inferred intent even when that intent named no
    entity the vault holds, so the tagged group was EMPTY and the cosine window
    returned a single record.
    """
    v = _office_vault(tmp_path)
    h = v.history("where is my desk")
    assert len(h) == len(_OFFICE), f"chain truncated to {len(h)} of {len(_OFFICE)}"
    assert [e["text"] for e in h] == _OFFICE, "chain is not in oldest-first order"
    assert sum(not e["superseded"] for e in h) == 1, "exactly one entry is current"
    assert h[-1]["text"] == _OFFICE[-1], "the newest revision must be the current one"
    v.close()


def test_history_agrees_with_changes_about_how_many_revisions_exist(
        tmp_path, offline_embedder):
    """The tell that this was a view bug and not data loss: three unfiltered
    accessors saw the whole chain while history saw one entry."""
    v = _office_vault(tmp_path, "agree.dat")
    assert len(v.get_all_records()) == len(_OFFICE)
    assert len(v.changes(since=0)) == len(_OFFICE)
    assert len(v.history("where is my desk")) == len(_OFFICE)
    v.close()


def test_an_intent_naming_nothing_falls_back_to_the_best_hit(
        tmp_path, offline_embedder):
    """The rule, stated directly: an intent that matches no entity in this vault
    is not evidence about it. An intent that DOES match keeps priority."""
    from nanomem import entities as _ent
    import numpy as _np
    names = ["office"]
    scored = _np.array([0.9, 0.2], dtype=_np.float64)
    ent_col = _np.array([0, 0], dtype=_np.int32)
    assert _ent.resolve_top_entity(scored, ent_col, names, intent="desk") == "office"
    assert _ent.resolve_top_entity(scored, ent_col, names, intent="office") == "office"
    assert _ent.resolve_top_entity(scored, ent_col, names, intent=None) == "office"


def test_a_genuinely_unchanged_fact_still_has_a_one_element_history(
        tmp_path, offline_embedder):
    """A one-element history is a real answer. The fix must not make every chain
    look long — it must only stop SHORT chains being reported as complete."""
    v = Vault(str(tmp_path / "single.dat"))
    v.add("My blood group is B negative.", metadata={"entity": "blood_group"})
    v.flush()
    h = v.history("what is my blood group")
    assert len(h) == 1 and h[0]["superseded"] is False
    v.close()


# --- the question's wording must not outrank the data ------------------------

_F3_CHAINS = {
    "home_address": ["I live at 14 Bracken Row.",
                     "Moved, my place is 8 Wexford Lane now.",
                     "Relocated again to 22 Pallant Street."],
    "gym": ["I train at Ironworks on Mill Road.",
            "Switched gyms, I go to Crossfield now.",
            "Now training at Bellhouse Athletic."],
    "office": ["Desk was on the third floor of Kestrel House.",
               "Moved down to the annexe at Larkfield.",
               "Now parked in the Maple Wharf building."],
}


def _f3_vault(tmp_path, name="f3.dat", **kw):
    v = Vault(str(tmp_path / name), **kw)
    now = 1_700_000_000.0
    for ent, vals in _F3_CHAINS.items():
        for i, val in enumerate(vals):
            v.add(val, metadata={"entity": ent},
                  timestamp=now - (len(vals) - 1 - i) * 150 * DAY)
    v.flush()
    return v


def test_a_where_question_does_not_always_mean_home_address(
        tmp_path, offline_embedder):
    """`query_intents` reads wording alone, so "where" resolved to `location` --
    an alias for `home_address` -- whatever the question was about. With
    `intent_boost + group_hoist` behind it that overrode the correct chain by
    0.23-0.28 of raw cosine on three of four declared attributes.

    Only the case the lexical test encoder can express is asserted end to end.
    That encoder has no notion of "desk" relating to `office`, and it scores
    "where is my desk" NEARER to "Moved, my place is 8 Wexford Lane now."
    (shared words) than to any office record -- so there the intent legitimately
    agrees with the data and the rule must NOT fire. The semantic cases are
    measured against a real model in evidence/intent_margin_results.json; the
    rule itself is pinned below without an encoder at all.
    """
    v = _f3_vault(tmp_path)
    got = (v.search("where do I train", top_k=1, decompose=False)[0]
           .get("metadata") or {}).get("entity")
    assert got == "gym", f"'where do I train' -> {got!r}"
    # WHAT "not still boosting" MEANS, stated so the assertion tests it.
    # This read `score == cosine` until 0.7.16. That was a proxy, and it held
    # only because `REVISION_LEAD` was too small for any other boost to show:
    # when the cap went 0.20 -> 0.60 the gym chain's revision lead became
    # visible and the proxy failed, though the intent rule was working exactly
    # as before -- `got == "gym"` above is the claim, and it holds at every cap
    # swept (0.20 through 1.50). A rejected intent means the
    # `intent_boost + group_hoist` pair (0.50) did not fire; the revision lead
    # is a different term and is allowed. So bound the excess by the lead alone.
    from nanomem import entities as _ent
    hits = v.search("where do I train", top_k=4, decompose=False)
    for h in hits:
        excess = float(h["score"]) - float(h["cosine"])
        assert excess >= -1e-6, h
        assert excess <= _ent.REVISION_LEAD + 1e-6, (
            "a rejected intent must not still be boosting: %.4f excess on %r "
            "exceeds REVISION_LEAD alone" % (excess, h["text"]))
    v.close()


def test_history_follows_the_same_correction(tmp_path, offline_embedder):
    """0.7.14 made the chain the right LENGTH; it was still the wrong
    attribute's chain. Same encoder caveat as above."""
    v = _f3_vault(tmp_path, "f3h.dat")
    h = v.history("where do I train")
    assert h, "no history returned"
    assert (h[-1].get("metadata") or {}).get("entity") == "gym", \
        f"history of {(h[-1].get('metadata') or {}).get('entity')!r}"
    v.close()


def test_the_margin_rule_itself(tmp_path):
    """The rule with no encoder in the way: synthetic cosines, exact thresholds.

    An intent is kept when it wins, kept when it loses by less than the margin,
    and dropped when it loses by more. The boundary is asserted on both sides so
    the threshold cannot drift silently.
    """
    from nanomem import entities as _ent
    import numpy as _np
    names = ["home_address", "gym"]
    ent_col = _np.array([0, 1], dtype=_np.int32)          # [home_address, gym]

    def top(cos_home, cos_gym, margin):
        cos = _np.array([cos_home, cos_gym], dtype=_np.float64)
        return _ent.resolve_top_entity(cos, ent_col, names, intent="home_address",
                                       cos=cos, margin=margin)

    assert top(0.70, 0.40, 0.15) == "home_address"   # intent wins outright
    assert top(0.60, 0.70, 0.15) == "home_address"   # loses by 0.10, under margin
    assert top(0.40, 0.70, 0.15) == "gym"            # loses by 0.30, over margin
    assert top(0.55, 0.70, 0.15) == "home_address"   # loses by exactly 0.15: kept
    assert top(0.54, 0.70, 0.15) == "gym"            # 0.16: dropped
    assert top(0.40, 0.70, 0.0) == "home_address"    # margin off, old behaviour


def test_an_intent_that_wins_on_cosine_still_decides(tmp_path, offline_embedder):
    """The rule must not degenerate into "always believe cosine".

    `intent_margin` only fires when the intent's best record loses by more than
    the margin. Where the intent agrees with the data it must still apply, which
    is what `intent_boost` is for.
    """
    v = _f3_vault(tmp_path, "f3keep.dat")
    hits = v.search("where do I live", top_k=3, decompose=False)
    assert (hits[0].get("metadata") or {}).get("entity") == "home_address"
    # and the boost is still doing something: score exceeds raw cosine
    assert hits[0].get("score", 0.0) > hits[0].get("cosine", 0.0), \
        "the intent boost stopped applying entirely"
    v.close()


def test_intent_margin_zero_restores_the_old_behaviour(tmp_path, offline_embedder):
    """The knob is measured, not chosen, so the previous behaviour stays
    reachable and the sweep stays reproducible."""
    v = _f3_vault(tmp_path, "f3off.dat", intent_margin=0.0)
    got = (v.search("where do I train", top_k=1, decompose=False)[0]
           .get("metadata") or {}).get("entity")
    assert got == "home_address", \
        f"margin=0.0 should reproduce the defect, got {got!r}"
    v.close()


# --- two documented methods the suite never touched --------------------------

def test_forget_erases_the_match_and_returns_what_it_erased(
        tmp_path, offline_embedder):
    """`forget` is public, documented ("Returns the deleted memory records") and
    had ZERO test coverage before this — found by enumerating the claims in the
    shipped docs and asking which test would fail if each became false."""
    v = Vault(str(tmp_path / "forget.dat"))
    v.add("The staging database listens on port 5433.")
    v.add("Backups run nightly at 0200 UTC.")
    v.flush()
    before = len(v.get_all_records())

    gone = v.forget("staging database port", min_score=0.0)
    v.flush()
    assert isinstance(gone, list) and gone, "forget returned nothing"
    assert all(isinstance(r, dict) and "text" in r for r in gone), \
        "the returned records are not records"
    assert len(v.get_all_records()) == before - len(gone), \
        "the count it reported does not match what left the vault"
    for r in gone:
        assert not any(x["text"] == r["text"] for x in v.get_all_records()), \
            "forget reported a record it did not erase"
    v.close()


def test_forget_erases_nothing_when_nothing_clears_min_score(
        tmp_path, offline_embedder):
    """The documented default is a 0.42 floor, so an unrelated query must be a
    no-op rather than erasing the nearest record anyway."""
    v = Vault(str(tmp_path / "forget2.dat"))
    v.add("The staging database listens on port 5433.")
    v.flush()
    assert v.forget("zqxjkv unrelated marker text") == []
    assert len(v.get_all_records()) == 1, "a no-op forget still deleted something"
    v.close()


def test_inspect_reports_the_sources_and_metadata_actually_stored(
        tmp_path, offline_embedder):
    """`inspect()` is public and documented and had no test. The reviewer's
    black-box round exercised it; this suite did not."""
    v = Vault(str(tmp_path / "inspect.dat"))
    v.add("Port 5433 is staging.", source="runbook.md", metadata={"entity": "port"})
    v.add("Backups at 0200 UTC.", source="runbook.md", metadata={"entity": "backup"})
    v.add("A note from chat.", source="user_chat")
    v.flush()
    out = v.inspect()
    assert isinstance(out, dict) and out, "inspect returned nothing usable"
    flat = json.dumps(out, default=str)
    for expected in ("runbook.md", "user_chat", "entity"):
        assert expected in flat, f"inspect() does not report {expected!r}"
    v.close()


def test_ingest_file_preserves_indentation_and_line_numbers(
        tmp_path, offline_embedder):
    """`ingest_file` claims it "preserves exact code formatting, indentation, and
    line numbers for coding agents". Nothing asserted it: an outside reviewer
    checked it by hand and this suite never did."""
    src = tmp_path / "runbook.py"
    lines = []
    for i in range(120):
        if i % 4 == 0:
            lines.append(f"def step_{i}():")
        else:
            lines.append(f"    payload_{i} = compute({i})      # trailing note")
    src.write_text("\n".join(lines) + "\n")

    v = Vault(str(tmp_path / "ing.dat"))
    v.ingest_file(str(src))
    v.flush()
    rows = v.get_all_records()
    assert rows, "ingest_file stored nothing"

    joined = "\n".join(r["text"] for r in rows)
    assert "    payload_5 = compute(5)      # trailing note" in joined, \
        "leading indentation or inner spacing was not preserved verbatim"
    # line numbers are recorded in `source` as name:start-end
    ranges = [r.get("source", "") for r in rows]
    assert any(re.search(r"runbook\.py:\d+-\d+$", s) for s in ranges), \
        f"no line range recorded in source: {ranges[:3]}"
    v.close()


def test_a_width_mismatched_handle_raises_on_calls_that_need_a_new_vector(tmp_path, offline_embedder):
    """Named for what it actually asserts, because the old name caused a bug report.

    This was `test_a_width_mismatched_handle_raises_on_every_call`, and its body
    only ever exercised search, history and add. The seventh black-box review read
    that name and the matching warning text, ran `prune(keep_current=False)`
    through a mismatched handle, watched six rows go to zero, and filed a CRITICAL
    for silent destruction through a handle that supposedly could not touch the
    file. It reproduced 0 of 8 ops differently from a correctly-matched handle:
    prune was doing its documented job.

    So the two halves are pinned separately below -- what raises, and what does
    not raise and must not lose anything.
    """
    p = str(tmp_path / "mismatch.dat")
    v = Vault(p)
    v.add("The database listens on port 5433.")
    v.flush()
    v.close()

    # A different MODEL NAME is not enough here: the offline test encoder emits
    # the same width whatever it is called, so there would be no mismatch to
    # warn about. A caller-supplied 384-d embedder makes the conflict real.
    class _Narrow:
        dim = 384

        def embed(self, text):
            import hashlib
            h = hashlib.md5(str(text).encode()).digest()
            return np.frombuffer(h * 24, dtype=np.uint8).astype(np.float32)[:384] / 255.0

        def embed_batch(self, texts):
            return np.stack([self.embed(t) for t in texts])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v2 = Vault(p, embedder=_Narrow())
    assert any("ignored" in str(w.message) for w in caught), "no warning emitted"
    assert v2.stats()["embed_dim"] == D, "the file did not keep its own width"
    for label, call in (("search", lambda: v2.search("database port")),
                        ("history", lambda: v2.history("database port")),
                        ("add", lambda: v2.add("another fact"))):
        with pytest.raises(ValueError, match="dims"):
            call()
    v2.close()

    v3 = Vault(p)                       # reopening correctly still works
    assert len(v3.get_all_records()) == 1, "data was lost"
    v3.close()


def test_a_width_mismatched_handle_does_not_destroy_the_vault(tmp_path, offline_embedder):
    """The negative from round 7, pinned so the refuted claim stays refuted.

    delete / compact / forget_superseded run through a mismatched handle and
    REWRITE the file -- they do not raise, and the warning used to say they would.
    What matters is that they behave identically to a matched handle. Measured
    across 8 ops: 0 differed.
    """
    p = str(tmp_path / "mismatch2.dat")
    v = Vault(p)
    for i in range(6):
        v.add("fact number %d about the service" % i)
    v.flush()
    v.close()

    class _Narrow:
        dim = 384

        def embed(self, text):
            import hashlib
            h = hashlib.md5(str(text).encode()).digest()
            return np.frombuffer(h * 24, dtype=np.uint8).astype(np.float32)[:384] / 255.0

        def embed_batch(self, texts):
            return np.stack([self.embed(t) for t in texts])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v2 = Vault(p, embedder=_Narrow())
    msg = " ".join(str(w.message) for w in caught)
    assert "every call will raise" not in msg, "the warning still overstates what raises"

    assert v2.compact() is not None
    assert len(v2.get_all_records()) == 6, "compact through a mismatched handle lost rows"
    assert v2.forget_superseded(keep=1)["deleted"] == 0
    assert len(v2.get_all_records()) == 6
    v2.close()

    v3 = Vault(p)
    assert len(v3.get_all_records()) == 6, "the file was damaged"
    v3.close()
