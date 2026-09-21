"""Differential fuzzing of the write path against an independent model.

Why this exists
---------------
Rounds 4, 5, 6 and 7 of black-box review each found the same defect class -- a
write path removing data the caller did not ask it to remove -- and each found it
one instance at a time:

    0.7.19  add() collapsed documents onto one id via a reserved metadata key
    0.7.20  ...so did add_batch(), update(), ingest_file()
    0.7.20  delete(where={}) erased the vault
    0.7.21  ...so did delete(text_contains=''), delete(source=''),
            export(purge=True), split(purge=True), split_by_doc(purge=True),
            unmerge('')

Enumerating tests (test_write_path_safety.py, test_empty_criterion_safety.py)
close this by SURFACE: a new method fails until a human classifies it. They cannot
close it by SHAPE -- they only probe the spellings someone thought to list.

This file closes it by shape. It drives random operation sequences against the
vault and against a deliberately stupid reference model kept in a dict, and after
EVERY operation asserts the two agree about what still exists. It does not model
ranking, embeddings or chunk layout; it models one thing:

    NO OPERATION MAY REMOVE A RECORD THE CALLER DID NOT NAME,
    AND NO SURVIVING RECORD MAY CHANGE.

That is the invariant every one of the defects above violated, stated once,
tested over generated inputs rather than remembered ones. A failure prints the
exact seed and operation sequence needed to reproduce it.
"""

import os
import random

import pytest

from nanomem import Vault


# Short, distinct texts: kept under the split threshold so one add() is one row
# and `get(id)["text"]` round-trips exactly. Chunking has its own tests; mixing it
# in here would make a text mismatch ambiguous between "data lost" and
# "reconstruction is lossy", and an ambiguous failure is a useless one.
_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
          "hotel", "india", "juliet", "kilo", "lima", "mike", "november"]


class Model:
    """What SHOULD be in the vault. Deliberately trivial: a dict."""

    def __init__(self):
        self.live = {}  # id -> {"text":..., "metadata":..., "source":...}

    def add(self, doc_id, text, metadata, source):
        self.live[doc_id] = {"text": text, "metadata": dict(metadata), "source": source}

    def remove(self, ids):
        for i in ids:
            self.live.pop(i, None)

    def select_where(self, where):
        return [i for i, r in self.live.items()
                if all(r["metadata"].get(k) == v for k, v in where.items())]

    def select_text_exact(self, text):
        return [i for i, r in self.live.items() if r["text"].strip() == text.strip()]

    def select_source(self, source):
        return [i for i, r in self.live.items() if r["source"] == source]


class Fuzzer:
    def __init__(self, vault, seed):
        self.v = vault
        self.m = Model()
        self.rng = random.Random(seed)
        self.log = []
        self.n = 0

    def _text(self):
        self.n += 1
        return "%s %s fact %d" % (self.rng.choice(_WORDS), self.rng.choice(_WORDS), self.n)

    def _record(self, op):
        self.log.append(op)

    # -- operations ------------------------------------------------------
    def op_add(self):
        text = self._text()
        bucket = self.rng.choice(["red", "blue", "green"])
        source = self.rng.choice(["notes.txt", "meeting_notes.txt", "notes.txt.bak"])
        meta = {"bucket": bucket, "n": self.n}
        did = self.v.add(text, metadata=meta, source=source)
        self.m.add(did, text, meta, source)
        self._record("add(%r, bucket=%s, source=%s) -> %s" % (text, bucket, source, did))

    def op_add_batch(self):
        k = self.rng.randint(2, 4)
        recs = []
        for _ in range(k):
            text = self._text()
            recs.append({"text": text,
                         "metadata": {"bucket": self.rng.choice(["red", "blue", "green"]),
                                      "n": self.n},
                         "source": "batch.txt"})
        before = {r["id"] for r in self.v.get_all_records()}
        self.v.add_batch(recs)
        after = self.v.get_all_records()
        for r in after:
            if r["id"] not in before:
                self.m.add(r["id"], r.get("text", ""), (r.get("metadata") or {}),
                           r.get("source", "batch.txt"))
        self._record("add_batch(%d records)" % k)

    def op_delete_by_id(self):
        if not self.m.live:
            return
        did = self.rng.choice(sorted(self.m.live))
        self.v.delete(id=did)
        self.m.remove([did])
        self._record("delete(id=%r)" % did)

    def op_delete_by_ids(self):
        if not self.m.live:
            return
        k = min(len(self.m.live), self.rng.randint(0, 3))
        ids = self.rng.sample(sorted(self.m.live), k)
        self.v.delete(ids=ids)
        self.m.remove(ids)
        self._record("delete(ids=%r)" % (ids,))

    def op_delete_where(self):
        where = {"bucket": self.rng.choice(["red", "blue", "green"])}
        expected = self.m.select_where(where)
        self.v.delete(where=where)
        self.m.remove(expected)
        self._record("delete(where=%r)" % where)

    def op_delete_text_exact(self):
        if not self.m.live:
            return
        did = self.rng.choice(sorted(self.m.live))
        text = self.m.live[did]["text"]
        expected = self.m.select_text_exact(text)
        self.v.delete(text_exact=text)
        self.m.remove(expected)
        self._record("delete(text_exact=%r)" % text)

    def op_delete_source(self):
        source = self.rng.choice(["notes.txt", "meeting_notes.txt", "notes.txt.bak", "batch.txt"])
        expected = self.m.select_source(source)
        self.v.delete(source=source)
        self.m.remove(expected)
        self._record("delete(source=%r)" % source)

    def op_update(self):
        if not self.m.live:
            return
        did = self.rng.choice(sorted(self.m.live))
        text = self._text()
        ok = self.v.update(did, text)
        if ok:
            self.m.live[did]["text"] = text
        self._record("update(%r, %r) -> %s" % (did, text, ok))

    def op_compact(self):
        self.v.compact()
        self._record("compact()")

    def op_export_purge(self, tmpdir):
        where = {"bucket": self.rng.choice(["red", "blue", "green"])}
        expected = self.m.select_where(where)
        target = os.path.join(tmpdir, "exp_%d.dat" % self.n)
        self.v.export(target, where=where, purge=True)
        self.m.remove(expected)
        self._record("export(where=%r, purge=True)" % where)

    def op_noop_empties(self):
        """Empty criteria in every spelling: each must refuse or do nothing.

        This is the generated version of test_empty_criterion_safety.py. That file
        pins the surfaces that are known; this drives them at random points in a
        random history, where the vault is in states nobody enumerated.
        """
        for call in (lambda: self.v.delete(where={}),
                     lambda: self.v.delete(text_contains=""),
                     lambda: self.v.delete(source=""),
                     lambda: self.v.delete(ids=[]),
                     lambda: self.v.delete(id=""),
                     lambda: self.v.delete(text_exact=""),
                     lambda: self.v.unmerge("")):
            try:
                call()
            except ValueError:
                pass  # refusing is the correct outcome
        self._record("empty-criterion sweep (must be a no-op)")

    # -- the invariant ---------------------------------------------------
    def check(self):
        rows = self.v.get_all_records()
        actual = {}
        for r in rows:
            actual[str(r.get("id"))] = r

        missing = sorted(set(self.m.live) - set(actual))
        assert not missing, self._fail(
            "records vanished that no operation named: %s" % missing[:5])

        for did, want in self.m.live.items():
            got = actual[did]
            assert got.get("text", "").strip() == want["text"].strip(), self._fail(
                "text of %s changed: %r -> %r" % (did, want["text"], got.get("text")))
            gm = got.get("metadata") or {}
            for k, v in want["metadata"].items():
                assert gm.get(k) == v, self._fail(
                    "metadata %r of %s changed: %r -> %r" % (k, did, v, gm.get(k)))

        for did in self.m.live:
            assert self.v.exists(did), self._fail("exists(%r) is False but the record is live" % did)
            assert self.v.get(did) is not None, self._fail("get(%r) is None but the record is live" % did)

    def _fail(self, msg):
        return "%s\n\nREPRODUCE -- operation sequence:\n  %s" % (
            msg, "\n  ".join(self.log))


def _run_one(tmp_path, seed, steps):
    v = Vault(str(tmp_path / ("fuzz_%d.dat" % seed)))
    f = Fuzzer(v, seed)
    ops = [f.op_add, f.op_add, f.op_add_batch, f.op_delete_by_id, f.op_delete_by_ids,
           f.op_delete_where, f.op_delete_text_exact, f.op_delete_source, f.op_update,
           f.op_compact, f.op_noop_empties]
    for _ in range(steps):
        op = f.rng.choice(ops + [lambda: f.op_export_purge(str(tmp_path))])
        op()
        f.check()
    return f


# A small, fixed set of seeds runs in CI. The same engine runs at scale from
# scratch/refound/fuzz_write_paths.py, which is where the large-N numbers come
# from; keeping CI short is deliberate, and the seeds here are the ones that have
# previously produced failures plus a spread.
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13, 21, 34])
def test_no_operation_removes_a_record_the_caller_did_not_name(tmp_path, seed):
    f = _run_one(tmp_path, seed, steps=25)
    assert f.log, "the fuzzer did nothing"


def test_the_fuzzer_would_catch_a_regression(tmp_path):
    """The fuzzer must fail on a vault that loses data. Otherwise it proves nothing.

    Drives the invariant check against a model that believes in a record the vault
    never held -- the exact signature of a silent deletion.
    """
    v = Vault(str(tmp_path / "canary.dat"))
    f = Fuzzer(v, seed=99)
    f.op_add()
    f.check()
    f.m.add("doc_never_written", "a record the vault does not have", {}, "x")
    with pytest.raises(AssertionError) as e:
        f.check()
    assert "vanished" in str(e.value)
    assert "REPRODUCE" in str(e.value)
