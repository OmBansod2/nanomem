"""The write gate's DECISION POINT, pinned.

0.5.0 moved the full head's threshold 0.60 -> 0.05 (`DEPLOYMENT_THRESHOLD_FULL`).
That constant is worth +13.3 pt of end-to-end top-1 [CI +8.7, +18.3] on a
300-question held-out split and nothing in the suite touched the classifier
before this file, so the whole change was one unpinned float. These tests are
offline: every embedding here is a seeded numpy vector and no daemon is dialled.
"""

import numpy as np
import pytest

from nanomem.classifier import (DEPLOYMENT_THRESHOLD_FULL, WriteClassifier,
                                EMBED_DIM)


@pytest.fixture
def clf():
    # verify_embedder=False keeps a handed-in vector on the FULL head without
    # re-encoding the text to check it is not the md5 fallback.
    return WriteClassifier(verify_embedder=False)


def unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def test_the_full_head_decides_at_the_deployment_threshold_not_the_trained_one(clf):
    """The asset still carries 0.60; deployment runs at 0.05 and says which is which.

    The trained value is kept rather than discarded: it is the operating point
    the trainer's own accuracy/F1 numbers were measured at, so a caller
    reproducing them needs to be able to read it back.
    """
    assert DEPLOYMENT_THRESHOLD_FULL == 0.05
    assert clf.threshold_full == DEPLOYMENT_THRESHOLD_FULL
    if clf.threshold_trained_full is not None:          # asset installed
        assert abs(clf.threshold_trained_full - 0.60) < 1e-6
        assert clf.threshold_trained_full != clf.threshold_full
    info = clf.model_info
    assert info["threshold_full"] == DEPLOYMENT_THRESHOLD_FULL
    assert info["threshold_source"].endswith("DEPLOYMENT_THRESHOLD_FULL")


def test_the_reported_threshold_is_the_one_the_decision_used(clf):
    """`classify()["threshold"]` must be the number `should_store` compared against.

    A decision that reports a threshold it did not use is worse than one that
    reports none, so this checks the boolean against the two numbers rather than
    trusting either.
    """
    r = clf.classify("My cardiologist is at the clinic on Seventh.", embedding=unit(7))
    assert r["model"] == "logistic_v2_full"
    assert r["threshold"] == pytest.approx(DEPLOYMENT_THRESHOLD_FULL)
    assert r["should_store"] == (r["prob"] >= r["threshold"])


def test_an_explicit_threshold_still_wins_and_is_reported_as_the_callers(clf):
    """`WriteClassifier(threshold=0.60)` is the documented way back to 0.4.0."""
    old = WriteClassifier(threshold=0.60, verify_embedder=False)
    assert old.threshold_full == 0.60 and old.threshold_surf == 0.60
    assert old.model_info["threshold_source"] == "caller"
    v = unit(11)
    txt = "The spare key lives in the blue tin on the third shelf."
    p = old.classify(txt, embedding=v)["prob"]
    assert old.classify(txt, embedding=v)["should_store"] == (p >= 0.60)
    assert clf.classify(txt, embedding=v)["should_store"] == (p >= 0.05)


def test_the_surface_head_was_not_moved_by_the_full_heads_measurement(clf):
    """The study that moved 0.60 -> 0.05 handed a real embedding to every turn.

    It therefore measured the full head and only the full head.  The no-embedder
    path keeps the trained decision point until something measures IT.
    """
    assert clf.threshold_surf == pytest.approx(0.60, abs=1e-6)
    assert clf.threshold_surf != clf.threshold_full


def test_lowering_the_threshold_can_only_ADD_stores_never_remove_one(clf):
    """Monotonicity, mechanically: the gate is a threshold on one probability.

    The rule layers pre-empt the head in both directions, so this is checked on
    the head alone (`use_rules=False`), which is also the surface the sweep that
    chose 0.05 was measured on.
    """
    old = WriteClassifier(threshold=0.60, verify_embedder=False)
    texts = ["My blood type is O negative.",
             "haha that is brilliant",
             "The car is booked in for Tuesday at 8pm.",
             "sounds good to me",
             "I moved to the flat above the bakery last month."]
    for i, t in enumerate(texts):
        v = unit(100 + i)
        a = old.classify(t, embedding=v, use_rules=False)["should_store"]
        b = clf.classify(t, embedding=v, use_rules=False)["should_store"]
        assert not (a and not b), (t, a, b)
