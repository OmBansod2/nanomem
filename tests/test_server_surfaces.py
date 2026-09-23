"""The two network surfaces, exercised as a client actually drives them.

Both defects here were invisible to in-process tests: the MCP server only lost
data when it was stopped the way a real client stops it, and the proxy only read
arbitrary files when a path came in over the wire.
"""
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows(vault_path):
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys;sys.path.insert(0,%r);from nanomem.vault import Vault;"
         "print(len(Vault(%r).get_all_records()))" % (PKG_ROOT, vault_path)],
        capture_output=True, text=True, timeout=120)
    return int((out.stdout or "0").strip() or 0)


class _McpClient:
    """A stdio JSON-RPC client whose every read is bounded.

    THIS CLASS USED A BARE `self.p.stdout.readline()`. That blocks forever when
    the server dies at startup or answers nothing, and the call is made from
    `__init__`, i.e. before the caller's `try/finally` exists, so not even the
    `kill()` ran. On Windows that is not hypothetical: it is how the CI job sat
    in `Test` for two and a half hours per run and then reported nothing worth
    reading. A hang cannot be debugged; a failure carrying the server's stderr
    can. `select` cannot poll a pipe on Windows, so the bound is a reader
    thread and a queue, which behave identically on both platforms.
    """

    TIMEOUT = 60

    def __init__(self, vault_path, cwd):
        self.p = subprocess.Popen(
            [sys.executable, "-m", "nanomem.mcp", "--vault", vault_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=cwd, env=dict(os.environ, PYTHONPATH=PKG_ROOT))
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.call({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                              "clientInfo": {"name": "t", "version": "1"}}})

    def _pump(self):
        try:
            for line in self.p.stdout:
                self._lines.put(line)
        except Exception:
            pass
        finally:
            self._lines.put(None)                      # EOF sentinel

    def _stderr(self):
        """Whatever the server complained about, without blocking on the pipe."""
        got = []
        t = threading.Thread(target=lambda: got.append(self.p.stderr.read()),
                             daemon=True)
        t.start()
        t.join(30)
        return (got[0] if got else "") or ""

    def _fail(self, what):
        self.p.kill()
        try:
            self.p.wait(timeout=30)
        except Exception:
            pass
        raise AssertionError(
            "%s within %ss (returncode=%r)\n--- server stderr ---\n%s"
            % (what, self.TIMEOUT, self.p.returncode,
               self._stderr().strip() or "<empty>"))

    def call(self, obj):
        try:
            self.p.stdin.write(json.dumps(obj) + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            self._fail("the server closed stdin on %r" % obj.get("method"))
        try:
            line = self._lines.get(timeout=self.TIMEOUT)
        except queue.Empty:
            self._fail("the server never answered %r" % obj.get("method"))
        if line is None:
            self._fail("the server closed stdout without answering %r"
                       % obj.get("method"))
        return json.loads(line) if line.strip() else None

    def add(self, text, i=0):
        return self.call({"jsonrpc": "2.0", "id": 100 + i, "method": "tools/call",
                          "params": {"name": "nanomem_add", "arguments": {"text": text}}})


def test_an_mcp_write_is_on_disk_before_the_reply_is_sent(tmp_path):
    """The server buffered every write and flushed only in a `finally`, so
    `nanomem_add` answered "Stored ..." while nothing had reached disk."""
    vault = str(tmp_path / "v.dat")
    c = _McpClient(vault, str(tmp_path))
    try:
        reply = c.add("The staging port is 8443.")
        assert "Stored" in json.dumps(reply)
        assert _rows(vault) == 1, "the reply said stored, but another process sees nothing"
    finally:
        c.p.kill()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_mcp_writes_survive_the_signal_a_client_stops_servers_with(tmp_path):
    """An MCP client stops its servers with SIGTERM, which does not run
    `finally`. Measured before the fix: 20 adds, SIGTERM, 0 rows."""
    vault = str(tmp_path / "t.dat")
    c = _McpClient(vault, str(tmp_path))
    for i in range(5):
        c.add("Bay %d is clear." % i, i)
    c.p.send_signal(signal.SIGTERM)
    try:
        c.p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        c.p.kill()
    time.sleep(0.3)
    assert _rows(vault) == 5


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_mcp_writes_survive_an_unclean_kill(tmp_path):
    """kill -9 runs nothing at all. Anything already acknowledged must be there."""
    vault = str(tmp_path / "k.dat")
    c = _McpClient(vault, str(tmp_path))
    for i in range(3):
        c.add("Fact %d." % i, i)
    c.p.kill()
    time.sleep(0.3)
    assert _rows(vault) == 3


# ---------------------------------------------------------------------------
# the proxy must not read files outside its vault directory
# ---------------------------------------------------------------------------
def _start_proxy(tmp_path, port):
    code = ("import sys; sys.path.insert(0, %r);"
            "from nanomem.proxy import run_proxy;"
            "run_proxy(vault_path=%r, port=%d)"
            % (PKG_ROOT, str(tmp_path / "p.dat"), port))
    p = subprocess.Popen([sys.executable, "-c", code], cwd=str(tmp_path),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1).read()
            return p
        except Exception:
            time.sleep(0.25)
    p.kill()
    pytest.skip("proxy did not come up")


def _post(port, path, payload, origin=None):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if origin:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), dict(e.headers)


@pytest.mark.parametrize("payload,label", [
    ({"file": "/etc/hosts"}, "absolute path"),
    ({"directory": "../.."}, "parent traversal"),
], ids=["absolute", "traversal"])
def test_the_proxy_refuses_to_ingest_outside_its_vault_directory(tmp_path, payload, label):
    """`/v1/vault/init` has confined its client-supplied path since 3.0.3.
    `/v1/memory/ingest` took `file` and `directory` from the same kind of body and
    passed them straight through: `{"file": "/etc/hosts"}` answered
    `{"status": "success"}` and the contents were readable back out."""
    port = 8951 if label == "absolute path" else 8952
    p = _start_proxy(tmp_path, port)
    try:
        status, body, _ = _post(port, "/v1/memory/ingest", payload)
        assert status == 400, "%s was accepted: %s" % (label, body[:160])
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def test_the_proxy_does_not_send_a_wildcard_cors_header(tmp_path):
    """`Access-Control-Allow-Origin: *` on an UNAUTHENTICATED loopback API lets
    any page the operator visits drive it from their browser."""
    port = 8953
    p = _start_proxy(tmp_path, port)
    try:
        _, _, headers = _post(port, "/v1/memory/add", {"text": "a fact."},
                              origin="https://evil.example")
        assert headers.get("Access-Control-Allow-Origin") != "*"
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def test_the_proxy_still_ingests_a_file_inside_its_vault_directory(tmp_path):
    """The control that must NOT flip: confinement has to leave the legitimate
    case working."""
    port = 8954
    (tmp_path / "notes.md").write_text("The staging port is 8443.\n", encoding="utf-8")
    p = _start_proxy(tmp_path, port)
    try:
        status, body, _ = _post(port, "/v1/memory/ingest", {"file": "notes.md"})
        assert status == 200, body[:200]
        assert "success" in body
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
