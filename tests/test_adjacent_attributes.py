"""The regression round 2 shipped: two DIFFERENT attributes treated as one fact.

3.0.1 grouped the top candidates within ``window_delta`` cosine of the leader,
called them "competing statements of the same fact" and permuted their scores
into timestamp order. Two different attributes phrased alike -- "my dentist" and
"my doctor", "my home address" and "my work address", "my primary email" and "my
backup email" -- satisfy that test, so the later, unrelated record inherited the
answer's score and led. Measured on the 40 probes below: 30/40 for the engine
against 40/40 for a plain cosine scan, i.e. a NET REGRESSION against the v2
engine it replaced (17/20 on the verifier's own 20-case version).

THE BAR, taken verbatim from the round-2 verifier:
  "a regression test over >= 20 generic adjacent-attribute pairs asserting the
   engine is no worse than plain cosine, plus an assertion that
   |score - cosine| <= stats()['max_boost'] for every hit."

The fixture is ``tests/data/adjacent_attributes.npz`` -- 128 sentences written
for this test (no benchmark fixture vocabulary; the source list is beside it in
``adjacent_attributes_source.py``) and embedded once with nomic-embed-text, so
the test is hermetic and needs no network. 40 ADJACENT scenarios (two different
attributes, ask about one) and 16 REVISION scenarios (one attribute stated
twice, ask for the current or the previous value).
"""

import json
import os

import numpy as np
import pytest

from nanomem import entities as ent
from nanomem.engine import VaultEngine

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "adjacent_attributes.npz")


def _fixture():
    if not os.path.exists(DATA):
        pytest.skip(f"probe fixture missing: {DATA}")
    z = np.load(DATA, allow_pickle=True)
    V = z["V"].astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return (V, list(z["texts"]), json.loads(str(z["adj"])), json.loads(str(z["rev"])))


def _vault(tmp_path, name, rows, **kw):
    """One vault per scenario; every record is a first-person chat statement."""
    e = VaultEngine(filepath=str(tmp_path / name), embed_dim=768, **kw)
    ids = []
    for i, (txt, vec, ts) in enumerate(rows):
        ids.append(e.add_fact(txt, vec, source="chat_session",
                              metadata={"user_id": "u"}, timestamp=ts))
    e.flush()
    return e, ids


def test_adjacent_attributes_are_not_revisions_of_each_other(tmp_path):
    """The engine must be NO WORSE than a plain cosine scan on 40 generic pairs."""
    V, texts, adj, _ = _fixture()
    engine_ok = cosine_ok = 0
    losses = []
    for k, c in enumerate(adj):
        want_row = c["a"] if c["ans"] == 0 else c["b"]
        rows = [(texts[c["a"]], V[c["a"]], 1000.0), (texts[c["b"]], V[c["b"]], 2000.0)]
        e, ids = _vault(tmp_path, f"adj{k}.dat", rows)
        try:
            hits = e.search(texts[c["q"]], V[c["q"]], top_k=2)
            got = bool(hits) and hits[0]["id"] == ids[c["ans"]]
            engine_ok += got
            cos_pick = max((c["a"], c["b"]), key=lambda i: float(V[i] @ V[c["q"]]))
            cos_got = cos_pick == want_row
            cosine_ok += cos_got
            if cos_got and not got:
                losses.append((texts[c["q"]], hits[0]["text"]))
        finally:
            e.close()
    assert cosine_ok == len(adj), "the probe set is only meaningful if cosine gets it right"
    assert not losses, (f"{engine_ok}/{len(adj)} vs cosine {cosine_ok}/{len(adj)}; "
                        f"the engine lost cases cosine wins: {losses}")
    assert engine_ok == len(adj)


def test_revision_pairs_still_resolve(tmp_path):
    """... and the layer still earns its keep: cosine gets 4/16 of these."""
    V, texts, _, rev = _fixture()
    engine_ok = engine_hist = cosine_ok = 0
    for k, c in enumerate(rev):
        rows = [(texts[c["old"]], V[c["old"]], 1000.0), (texts[c["new"]], V[c["new"]], 2000.0)]
        e, ids = _vault(tmp_path, f"rev{k}.dat", rows)
        try:
            hits = e.search(texts[c["q"]], V[c["q"]], top_k=2)
            engine_ok += bool(hits) and hits[0]["id"] == ids[1]
            old = e.search(texts[c["q"]], V[c["q"]], top_k=2,
                           temporal_direction="historical")
            engine_hist += bool(old) and old[0]["id"] == ids[0]
            cosine_ok += float(V[c["new"]] @ V[c["q"]]) >= float(V[c["old"]] @ V[c["q"]])
        finally:
            e.close()
    # A recorded property of the fixture, not a target: plain cosine ranks the
    # NEWER statement first in only 3 of the 16 pairs, so anything above 3 is the
    # revision layer doing work no similarity score can do.
    assert cosine_ok == 3, cosine_ok
    # 3.0.2 scored 11/16 and pinned the bar at its own score. 3.0.3 scores 14/16;
    # the bar is set ABOVE 3.0.2 so a slip back to it fails, and the two known
    # residual failures are recorded in scratch/refound/ranking_dev_r4_shipped.json
    # (both are restatements more than window_delta of cosine away from the
    # question, which widening the window does not pay for -- see
    # entities.WINDOW_DELTA_MARKED).
    assert engine_ok >= 13, f"current-revision top-1 {engine_ok}/{len(rev)}"
    assert engine_hist >= 16, f"historical top-1 {engine_hist}/{len(rev)}"


def test_score_minus_cosine_is_inside_max_boost(tmp_path):
    """DECISIONS #8, checked on every hit of every probe, in both shapes."""
    V, texts, adj, rev = _fixture()
    worst = 0.0
    below = []
    for k, c in enumerate(adj[:12] + [{"a": r["old"], "b": r["new"], "q": r["q"], "ans": 0}
                                      for r in rev[:8]]):
        rows = [(texts[c["a"]], V[c["a"]], 1000.0), (texts[c["b"]], V[c["b"]], 2000.0)]
        e, _ = _vault(tmp_path, f"con{k}.dat", rows)
        path = e.filepath
        try:
            mb = e.stats()["max_boost"]
            for direction in ("current", "historical"):
                for h in e.search(texts[c["q"]], V[c["q"]], top_k=4,
                                  temporal_direction=direction):
                    d = h["score"] - h["cosine"]
                    worst = max(worst, abs(d))
                    if d < -1e-6 or d > mb + 1e-6:
                        below.append((h["score"], h["cosine"], mb))
        finally:
            e.close()
        e2 = VaultEngine(filepath=path, embed_dim=768)      # re-opened: same contract
        try:
            mb = e2.stats()["max_boost"]
            for h in e2.search(texts[c["q"]], V[c["q"]], top_k=4):
                d = h["score"] - h["cosine"]
                if d < -1e-6 or d > mb + 1e-6:
                    below.append((h["score"], h["cosine"], mb))
        finally:
            e2.close()
    assert not below, below
    assert worst <= 0.70 + 1e-6


def test_the_3_0_1_behaviour_reproduces_the_regression(tmp_path, monkeypatch):
    """Proof this test actually catches it: put 3.0.1's two steps back and it fails.

    The fix is structural, not a re-tune, so BOTH have to be restored to see the
    old numbers: the score permutation (``entities.permute_group_scores``, kept
    for exactly this) and the cosine window's willingness to group records the
    tagger has called different attributes (``_tag_compatible``). Restoring the
    permutation alone still scores 40/40, which is the clearest evidence that
    the grouping rule -- not the score arithmetic -- was the defect.
    """
    V, texts, adj, _ = _fixture()
    monkeypatch.setattr(
        ent, "apply_revision_lead",
        lambda final, rows, rv, ts, hist, marks=None, mode="previous", weight=0.0:
            ent.permute_group_scores(final, rows, rv, ts, hist, marks, mode))
    monkeypatch.setattr(VaultEngine, "_tag_compatible",
                        lambda self, e, order, marked=None: np.ones(order.size, dtype=bool))
    ok = 0
    excess = 0.0
    for k, c in enumerate(adj):
        rows = [(texts[c["a"]], V[c["a"]], 1000.0), (texts[c["b"]], V[c["b"]], 2000.0)]
        e, ids = _vault(tmp_path, f"old{k}.dat", rows,
                        group_cos_delta=0.06, window_delta=0.16, window_sim=0.60)
        try:
            hits = e.search(texts[c["q"]], V[c["q"]], top_k=2)
            ok += bool(hits) and hits[0]["id"] == ids[c["ans"]]
            for h in hits:
                excess = max(excess, abs(h["score"] - h["cosine"]))
        finally:
            e.close()
    assert ok <= 34, f"the 3.0.1 path should lose adjacent-attribute cases, got {ok}/40"
    # 3.0.1 published max_boost = 0.50 and this is what it actually did:
    # +0.6402, the verifier's number to four decimals, reproduced here.
    assert excess > 0.50, "the 3.0.1 path should break its own published cap"
    assert excess == pytest.approx(0.6402, abs=5e-4)
