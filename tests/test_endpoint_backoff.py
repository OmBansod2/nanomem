"""An endpoint that did not answer must not be dialled again on every call.

`embed_batch` asked the network EVERY time and remembered nothing. Where a
refused connection is instant -- Linux, macOS -- that costs nothing measurable,
which is why it survived every review: the waste is invisible on the machines
the tests run on. It is not invisible everywhere. On the Windows CI runner the
SYN is dropped rather than refused, each attempt costs ~4 s, and one test run
made 2,267 of them: 2 h 35 m against under 4 minutes on every other platform,
and the same ~4 s on every add() and every search() for a Windows user with no
daemon running.

The fix is a per-endpoint cooldown. These tests pin the four behaviours that
make it safe: it stops dialling, a success clears it, an endpoint that ANSWERED
badly is not treated as absent, and it can be cleared on demand.
"""
import urllib.error

import numpy as np
import pytest

from nanomem.embed import EmbeddingProvider


@pytest.fixture(autouse=True)
def _clean_breaker():
    """Never let one test's outage leak into the next."""
    EmbeddingProvider.forget_unreachable_endpoints()
    yield
    EmbeddingProvider.forget_unreachable_endpoints()


def _provider(url):
    return EmbeddingProvider(model="m", base_url=url, dim=8)


def test_an_unreachable_endpoint_is_dialled_once_not_once_per_call(monkeypatch):
    dials = {"n": 0}

    def boom(*a, **k):
        dials["n"] += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("nanomem.embed.urllib.request.urlopen", boom)
    ep = _provider("http://127.0.0.1:9/a")
    for _ in range(50):
        ep.embed_batch(["x"])            # falls back every time
    assert dials["n"] == 1, (
        "dialled %d times for 50 calls; the cooldown is not holding" % dials["n"])


def test_the_cooldown_expires_so_a_daemon_that_comes_up_is_found(monkeypatch):
    """A permanent disable would be the wrong fix: daemons get started."""
    dials = {"n": 0}

    def boom(*a, **k):
        dials["n"] += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("nanomem.embed.urllib.request.urlopen", boom)
    monkeypatch.setattr(EmbeddingProvider, "DOWN_COOLDOWN", 0.0)
    ep = _provider("http://127.0.0.1:9/b")
    ep.embed_batch(["x"])
    ep.embed_batch(["x"])
    assert dials["n"] == 2, "a zero cooldown must not suppress the next dial"


def test_an_endpoint_that_answers_badly_is_not_treated_as_absent(monkeypatch):
    """HTTP 404 means the daemon is ALIVE and the request was wrong. Backing
    off from it would hide a server that is plainly there, and costs nothing to
    ask again because the connection succeeded."""
    dials = {"n": 0}

    def http_error(*a, **k):
        dials["n"] += 1
        raise urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)

    monkeypatch.setattr("nanomem.embed.urllib.request.urlopen", http_error)
    ep = _provider("http://127.0.0.1:9/c")
    for _ in range(5):
        ep.embed_batch(["x"])
    assert dials["n"] == 5, "a 404 is not an outage; it must not trip the breaker"


def test_a_success_clears_an_earlier_outage(monkeypatch):
    state = {"fail": True, "n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"embeddings": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}'

    def flaky(*a, **k):
        state["n"] += 1
        if state["fail"]:
            raise urllib.error.URLError("refused")
        return _Resp()

    monkeypatch.setattr("nanomem.embed.urllib.request.urlopen", flaky)
    monkeypatch.setattr(EmbeddingProvider, "DOWN_COOLDOWN", 0.0)
    ep = _provider("http://127.0.0.1:9/d")
    ep.embed_batch(["x"])                       # trips
    state["fail"] = False
    assert ep.embed_batch(["x"]).shape == (1, 8)
    state["fail"] = True
    before = state["n"]
    ep.embed_batch(["x"])
    assert state["n"] == before + 1, "the success should have cleared the cooldown"


def test_forget_unreachable_endpoints_redials_immediately(monkeypatch):
    dials = {"n": 0}

    def boom(*a, **k):
        dials["n"] += 1
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("nanomem.embed.urllib.request.urlopen", boom)
    ep = _provider("http://127.0.0.1:9/e")
    ep.embed_batch(["x"])
    ep.embed_batch(["x"])
    assert dials["n"] == 1
    EmbeddingProvider.forget_unreachable_endpoints()
    ep.embed_batch(["x"])
    assert dials["n"] == 2, "forget_unreachable_endpoints() must force a re-dial"
