"""Vault-level contract: the public API every caller in the repo uses."""

import json
import os
import sys

import numpy as np
import pytest

from conftest import D, unit_rows
from nanomem.vault import Vault, TextHopBridge


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

