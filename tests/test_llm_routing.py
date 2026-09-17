"""Which LLM provider a base URL resolves to, and what goes on the wire.

Through 0.6.0 the decision was a substring test:

    is_openai_compat = (base.endswith("/v1") or "/chat" in base
                        or "1234" in base or "8000" in base)

so a host named `web8000.internal` on port 11434 was POSTed to
/v1/chat/completions, llama.cpp on 8080 was POSTed to Ollama's /api/generate,
and Anthropic -- whose base ends "/v1" -- was POSTed to /v1/chat/completions,
which 404s. The caller saw "[Model not found]" for a model that exists.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nanomem.vault import Vault, _llm_endpoint


@pytest.mark.parametrize("base,flavour,url", [
    ("http://localhost:11434", "ollama", "http://localhost:11434/api/generate"),
    ("https://api.openai.com/v1", "openai", "https://api.openai.com/v1/chat/completions"),
    ("https://api.groq.com/openai/v1", "openai", "https://api.groq.com/openai/v1/chat/completions"),
    ("https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/api/v1/chat/completions"),
    ("http://localhost:1234/v1", "openai", "http://localhost:1234/v1/chat/completions"),
    ("http://localhost:8000/v1", "openai", "http://localhost:8000/v1/chat/completions"),
    ("https://api.anthropic.com/v1", "anthropic", "https://api.anthropic.com/v1/messages"),
    ("https://api.anthropic.com/v1/messages", "anthropic", "https://api.anthropic.com/v1/messages"),
])
def test_known_providers_route_correctly(base, flavour, url):
    assert _llm_endpoint(base) == (flavour, url)


def test_llama_cpp_on_8080_is_openai_compatible():
    """8080 speaks chat-completions; the old substring test sent it to Ollama."""
    assert _llm_endpoint("http://localhost:8080")[0] == "openai"


@pytest.mark.parametrize("base", ["http://web8000.internal:11434",
                                  "http://api1234.corp:11434"])
def test_a_hostname_containing_a_port_number_is_not_a_provider(base):
    """The port is parsed, never matched as a substring of the whole URL."""
    assert _llm_endpoint(base)[0] == "ollama"


def test_an_explicit_chat_completions_path_is_used_verbatim():
    """How an Azure deployment URL, with its api-version query, is supported."""
    u = "https://x.openai.azure.com/openai/deployments/gpt4/chat/completions"
    assert _llm_endpoint(u) == ("openai", u)


def test_a_malformed_port_does_not_raise():
    assert _llm_endpoint("http://host:notaport/v1")[0] == "openai"


# --- what actually goes on the wire ----------------------------------------
class _Capture(BaseHTTPRequestHandler):
    seen = None

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        # HTTP header names are case-insensitive and urllib title-cases them
        # ("x-api-key" goes out as "X-Api-Key"), so lower-case the keys rather
        # than assert on the casing a client happens to choose.
        type(self).seen = dict(
            path=self.path,
            headers={k.lower(): v for k, v in self.headers.items()},
            body=json.loads(body.decode()))
        # answer in the shape the flavour expects
        if self.path.endswith("/messages"):
            out = {"content": [{"type": "text", "text": "anthropic-answer"}]}
        else:
            out = {"choices": [{"message": {"content": "openai-answer"}}]}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def _ask(vault_path, offline, base, model="m"):
    v = Vault(vault_path)
    v.add("The staging database listens on port 5433.", source="doc")
    v.flush()
    out = v.ask("what port?", llm=model, base_url=base, api_key="k-test")
    v.close()
    return out


def test_anthropic_request_shape(vault_path, offline_embedder):
    srv, base = _serve()
    try:
        _Capture.seen = None
        r = _ask(vault_path, offline_embedder, base + "/v1/messages")
        s = _Capture.seen
        assert s["path"].endswith("/v1/messages")
        assert s["headers"].get("x-api-key") == "k-test"
        assert s["headers"].get("anthropic-version")
        assert "authorization" not in s["headers"], "Claude uses x-api-key, not a bearer token"
        assert "max_tokens" in s["body"], "max_tokens is required by the messages API"
        assert isinstance(s["body"].get("system"), str), "system is top-level, not a message"
        assert [m["role"] for m in s["body"]["messages"]] == ["user"]
        assert r["answer"] == "anthropic-answer"
    finally:
        srv.shutdown()


def test_openai_request_shape(vault_path, offline_embedder):
    srv, base = _serve()
    try:
        _Capture.seen = None
        r = _ask(vault_path, offline_embedder, base + "/v1")
        s = _Capture.seen
        assert s["path"].endswith("/v1/chat/completions")
        assert s["headers"].get("authorization") == "Bearer k-test"
        assert [m["role"] for m in s["body"]["messages"]] == ["system", "user"]
        assert r["answer"] == "openai-answer"
    finally:
        srv.shutdown()
