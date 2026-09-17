"""The POSIX-only idioms, exercised on whatever platform is running this.

Windows CI found 50 failures in one run, from two root causes -- and both were
invisible here because both are things POSIX allows and Windows does not. Fixing
them by pushing to CI and reading the log is a twenty-minute round trip per
attempt, and it leaves the fix untested everywhere else.

So these force the Windows branch on any platform: `_CAN_UNLINK_OPEN` is flipped
off and `os.pwrite` is removed, which is exactly the environment the constructor
sees there. A regression on this path will now fail on the developer's machine,
which is the only place a test is cheap.
"""

import os

import numpy as np
import pytest

from conftest import D, unit_rows                                  # noqa: F401
from nanomem.arena import _Sidecar
from nanomem.container import Container
from nanomem.engine import VaultEngine


@pytest.fixture
def as_windows(monkeypatch):
    """A sidecar that behaves as it must on Windows: no unlink of an open file,
    no `os.pwrite`."""
    monkeypatch.setattr(_Sidecar, "_CAN_UNLINK_OPEN", False)
    monkeypatch.delattr(os, "pwrite", raising=False)
    return True


def test_the_sidecar_does_not_unlink_an_open_file(as_windows):
    """`os.unlink` on an open handle is PermissionError [WinError 32] there.

    It was in the constructor, so a float16 vault could not be opened at all.
    """
    sc = _Sidecar(D)
    assert os.path.exists(sc.path_hint), "the file must survive its own creation"
    sc.close()
    assert not os.path.exists(sc.path_hint), "and close() must take it away"


def test_the_sidecar_writes_without_pwrite(as_windows):
    """`os.pwrite` does not exist on Windows. The fallback must land the same
    bytes at the same offsets, including a write that is not at offset 0."""
    assert not hasattr(os, "pwrite")
    sc = _Sidecar(D)
    a = unit_rows(4, seed=1).astype(np.float16)
    b = unit_rows(3, seed=2).astype(np.float16)
    sc.write(0, a)
    sc.write(4, b)                     # a second block, at a non-zero offset
    got = np.asarray(sc.view())
    assert got.shape == (7, D)
    assert np.array_equal(got[:4], a)
    assert np.array_equal(got[4:], b), "the offset write landed in the wrong place"
    sc.close()


def test_a_float16_vault_works_with_the_windows_sidecar(as_windows, vault_path):
    """End to end: the residency that uses the sidecar, through the engine."""
    q = unit_rows(1, seed=7)[0]
    e = VaultEngine(vault_path, embed_dim=D, residency="float16")
    ids = [e.add_fact(f"row {i}", unit_rows(1, seed=100 + i)[0], source="doc")
           for i in range(60)]                       # past one 50-row block
    e.flush()
    hits = e.search("a question", q, top_k=5)
    assert len(hits) == 5
    assert {h["id"] for h in hits} <= set(ids)
    e.close()


def test_the_sidecar_is_cleaned_up_even_with_a_live_mapping(as_windows):
    """close() has to drop the mapping first, or Windows keeps the file.

    Dropping the reference is enough on CPython; this asserts the mapping is
    closed explicitly, because the collector having run is not a guarantee.
    """
    sc = _Sidecar(D)
    sc.write(0, unit_rows(8, seed=3).astype(np.float16))
    sc.view()                                        # a live mapping
    path = sc.path_hint
    sc.close()
    assert not os.path.exists(path)


def test_posix_still_unlinks_at_creation():
    """The Windows branch must not become the only branch: on POSIX the file is
    gone the moment it exists, which is what makes a crash unable to leave one.
    """
    if os.name == "nt":
        pytest.skip("this is the POSIX guarantee; Windows cannot make it")
    sc = _Sidecar(D)
    assert not os.path.exists(sc.path_hint)
    sc.write(0, unit_rows(2, seed=4).astype(np.float16))
    assert np.asarray(sc.view()).shape == (2, D)
    sc.close()


# --- os.replace onto a path something has open ------------------------------

@pytest.fixture
def replace_as_windows(monkeypatch):
    """Windows refuses `os.replace` onto a file anything holds open."""
    monkeypatch.setattr(Container, "_REPLACE_NEEDS_CLOSED_HANDLE", True)


def _vault_with(path, n=60):
    e = VaultEngine(path, embed_dim=D)
    ids = [e.add_fact(f"row {i}", unit_rows(1, seed=200 + i)[0], source="doc")
           for i in range(n)]
    e.flush()
    return e, ids


def test_replace_all_survives_the_handle_being_closed(replace_as_windows, vault_path):
    """`replace_all` called `replace_with` INSIDE `exclusive()`, which holds an
    open handle on the very file being replaced. POSIX swaps the directory entry
    and the open fd keeps the old inode; Windows raises
    `PermissionError [WinError 5]`, and did, on every rewrite -- 18 of the 50
    failures in the first Windows run.
    """
    e, ids = _vault_with(vault_path)
    recs = list(e.iter_records(include_embeddings=True))
    assert e.replace_all(recs) == len(recs)
    assert [r["id"] for r in e.iter_records()] == ids
    assert len(e.search("a question", unit_rows(1, seed=9)[0], top_k=5)) == 5
    e.close()


def test_compact_survives_it_too(replace_as_windows, vault_path):
    """`compact` goes through `replace_all`, so it failed for the same reason."""
    e, ids = _vault_with(vault_path)
    out = e.compact()
    assert out["records"] == len(ids)
    assert [r["id"] for r in e.iter_records()] == ids
    e.close()


def test_the_handle_really_is_closed_before_the_replace(replace_as_windows,
                                                        vault_path, monkeypatch):
    """The point of the fix, asserted rather than assumed.

    Without this, `replace_all` could pass the handle and `replace_with` could
    quietly not close it -- the suite would still be green here, because POSIX
    does not care, and Windows would still be broken.
    """
    seen = {}
    real = os.replace

    def spy(src, dst, *a, **kw):
        seen["open_handles"] = [f for f in Container._WATCH if not f.closed]
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(Container, "_WATCH", [], raising=False)
    real_replace_with = Container.replace_with

    def wrapped(self, tmp_path, retries=5, holding=None):
        if holding is not None:
            Container._WATCH.append(holding)
        return real_replace_with(self, tmp_path, retries=retries, holding=holding)

    monkeypatch.setattr(Container, "replace_with", wrapped)
    monkeypatch.setattr(os, "replace", spy)

    e, _ids = _vault_with(vault_path, n=20)
    e.replace_all(list(e.iter_records(include_embeddings=True)))
    e.close()

    assert Container._WATCH, "replace_with was never handed the exclusive handle"
    assert seen.get("open_handles") == [], (
        "the exclusive handle was still open when os.replace ran; Windows "
        "would refuse this")
