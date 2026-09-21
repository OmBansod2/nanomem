"""
nanomem.proxy
~~~~~~~~~~~~~
Universal OpenAI-compatible memory proxy for Ollama, LM Studio, and vLLM.
Transparently injects persistent memory into any local or self-hosted LLM runtime.
"""

import json
import os
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from .vault import Vault, resolve_vault_path


class MemoryProxyHandler(BaseHTTPRequestHandler):
    # NOTE: always test this with `is not None`. `Vault.__len__` returns the
    # record count, so a bare `if self.vault:` is FALSE for an EMPTY vault --
    # through 3.0.2 /memory/stats answered 500 "Vault not loaded" on a freshly
    # created vault, and /chat/completions skipped the write that would have
    # made it non-empty, so a proxy started on a new vault could never learn
    # anything through chat at all.
    vault: Optional[Vault] = None
    upstream_url: str = "http://localhost:11434"
    # The operator's passphrase, and the ONE directory /vault/init may name a
    # file in. Both exist because 3.0.2's unauthenticated /v1/vault/init took an
    # arbitrary filesystem path from the request body and opened it with a bare
    # `Vault(name)`: a remote client could downgrade a password-protected proxy
    # to a PLAINTEXT vault of its own choosing and every later memory write went
    # there in the clear, outside the working directory if it asked.
    password: Optional[str] = None
    vault_root: str = "."
    #: `Access-Control-Allow-Origin: *` on an UNAUTHENTICATED loopback API lets
    #: any page the operator happens to visit drive it from their browser. Off
    #: by default from 0.7.18; `run_proxy(allow_cors=True)` restores it.
    allow_cors: bool = False
    allow_vault_switch: bool = False

    def _send_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        if MemoryProxyHandler.allow_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        if MemoryProxyHandler.allow_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path in ["/health", "/"]:
            from . import __version__ as _v
            self._send_json(200, {"status": "ok", "engine": "nanomem", "version": _v})
        elif self.path in ["/v1/memory/stats", "/memory/stats"]:
            if self.vault is not None:
                self._send_json(200, self.vault.stats())
            else:
                self._send_json(500, {"error": "Vault not loaded"})
        elif self.path in ["/v1/models", "/models"]:
            self._proxy_get(self.upstream_url + self.path)
        else:
            self._proxy_get(self.upstream_url + self.path)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 10 * 1024 * 1024:
            self._send_json(413, {"error": "Payload exceeds 10 MB request limit."})
            return

        if self.path in ["/v1/vault/init", "/vault/init"]:
            self._handle_vault_init()
        elif self.path in ["/v1/memory/ingest", "/memory/ingest"]:
            self._handle_memory_ingest()
        elif self.path in ["/v1/memory/add", "/memory/add"]:
            self._handle_memory_add()
        elif self.path in ["/v1/memory/search", "/memory/search"]:
            self._handle_memory_search()
        elif "/v1/chat/completions" in self.path or "/chat/completions" in self.path:
            self._handle_chat_completions()
        else:
            self._proxy_post(self.upstream_url + self.path)

    def _handle_vault_init(self):
        """Create/open a vault BESIDE the one the operator started with.

        Three things this endpoint is not allowed to do, all of which 3.0.2 did:
        open a vault WITHOUT the operator's passphrase (silently downgrading an
        encrypted proxy to plaintext), name a path anywhere on the filesystem,
        and do either of those with no authentication at all. The listener is
        now loopback-only by default, the path is confined to ``vault_root``, the
        passphrase is always applied, and switching the ACTIVE vault has to be
        enabled explicitly (``run_proxy(allow_vault_switch=True)``).
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            try:
                path = resolve_vault_path(payload.get("vault", "memory.dat"),
                                          MemoryProxyHandler.vault_root)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            pw = MemoryProxyHandler.password
            want_active = bool(payload.get("set_active", False))
            if want_active and not MemoryProxyHandler.allow_vault_switch:
                self._send_json(403, {
                    "error": "switching the active vault is disabled; start the "
                             "proxy with allow_vault_switch=True to permit it"})
                return
            with Vault(path, password=pw) as v:
                v.flush()
            if want_active or MemoryProxyHandler.vault is None:
                old = MemoryProxyHandler.vault
                MemoryProxyHandler.vault = Vault(path, password=pw)
                if old is not None:
                    old.close()
            self._send_json(200, {
                "status": "success",
                "vault": os.path.relpath(path, MemoryProxyHandler.vault_root),
                "encrypted": bool(pw),
                "message": f"Initialized vault '{os.path.basename(path)}'."
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    @staticmethod
    def _resolve_content_path(raw, root):
        """Confine a client-supplied CONTENT path to ``root``.

        `/v1/vault/init` has confined its `vault` field since 3.0.3, and
        `resolve_vault_path`'s docstring explains why: an unauthenticated endpoint
        that takes a filesystem path from a request body is a filesystem
        primitive. `/v1/memory/ingest` took `file` and `directory` from the same
        kind of body and passed them straight through, so the confinement existed
        and was wired to one handler but not its sibling. `{"file": "/etc/hosts"}`
        answered `{"status": "success", "chunks_indexed": 1}` and the contents
        were then readable through `/v1/memory/search` -- an arbitrary file and
        directory READ, where the documented exposure is "your memories".

        This is the same rule as `resolve_vault_path` minus the suffix check,
        because the thing being named here is content rather than a vault.
        """
        root = os.path.realpath(os.path.abspath(root))
        name = str(raw or "").strip()
        if not name:
            raise ValueError("path must not be empty")
        if os.path.isabs(name) or name.startswith("~"):
            raise ValueError("path must be relative to %r, got %r" % (root, name))
        target = os.path.realpath(os.path.join(root, name))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError("path %r resolves outside %r" % (name, root))
        return target

    def _handle_memory_ingest(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            file_path = payload.get("file")
            dir_path = payload.get("directory")
            if not file_path and not dir_path:
                self._send_json(400, {"error": "Field 'file' or 'directory' is required."})
                return
            try:
                resolved = self._resolve_content_path(
                    file_path or dir_path, MemoryProxyHandler.vault_root)
            except ValueError as ve:
                self._send_json(400, {"error": str(ve)})
                return
            if file_path:
                count = self.vault.ingest_file(resolved)
                self.vault.flush()
                self._send_json(200, {"status": "success", "type": "file", "path": file_path, "chunks_indexed": count})
            else:
                # The root goes DOWN into the walk. Resolving the directory name
                # alone proved nothing about what the walk would find inside it.
                stats = self.vault.ingest_directory(
                    resolved, confine_root=MemoryProxyHandler.vault_root)
                self.vault.flush()
                self._send_json(200, {"status": "success", "type": "directory", "path": dir_path, **stats})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_memory_add(self):
        try:
            import hashlib, time
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            text = payload.get("text", "")
            if not text:
                self._send_json(400, {"error": "Field 'text' is required."})
                return
            source = payload.get("source", "rest_api")
            metadata = payload.get("metadata", {})
            # Vault.add returns the id the record is actually stored under, so the
            # response no longer invents one that nothing can look up.
            doc_id = self.vault.add(text, source=source, metadata=metadata)
            self.vault.flush()
            self._send_json(200, {"status": "success", "doc_id": doc_id, "text": text})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_memory_search(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            query = payload.get("query", "")
            if not query:
                self._send_json(400, {"error": "Field 'query' is required."})
                return
            top_k = max(1, min(50, int(payload.get("top_k", 3))))
            hits = self.vault.search(query, top_k=top_k)
            self._send_json(200, {"query": query, "top_k": top_k, "hits": hits})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_chat_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        req_data = json.loads(body.decode("utf-8"))

        messages = req_data.get("messages", [])
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # Check citation preference: request payload ("cite": false), header ("X-No-Cite: 1"), or server default
        req_cite = req_data.pop("cite", None)
        no_cite_header = self.headers.get("X-No-Cite")
        if req_cite is not None:
            cite_enabled = bool(req_cite)
        elif no_cite_header is not None:
            cite_enabled = no_cite_header.strip().lower() not in ("1", "true", "yes", "on")
        else:
            cite_enabled = getattr(self, "cite_default", True)

        # Retrieve relevant persistent memory (budgeted to max 500 words to protect LLM context window)
        if self.vault is not None and user_msg:
            candidates = self.vault.search(user_msg, top_k=3)
            if candidates:
                budgeted_snippets = []
                words_used = 0
                for c in candidates:
                    w_count = len(c["text"].split())
                    if words_used + w_count > 500 and budgeted_snippets:
                        break
                    src_tag = f"[{c['source']}] " if (cite_enabled and c.get("source") and c.get("source") not in ["unknown", "user_input", "chat_session"]) else ""
                    budgeted_snippets.append(f"• {src_tag}{c['text']}")
                    words_used += w_count

                if budgeted_snippets:
                    context_str = "\n".join(budgeted_snippets)
                    instructions = "ground answers on these facts with [source] citations where appropriate" if cite_enabled else "ground answers conversationally on these facts without source tags or citations"
                    memory_system_msg = {
                        "role": "system",
                        "content": f"[Persistent Memory Active ({instructions})]:\n{context_str}"
                    }
                    # Insert memory context after existing system prompt
                    if messages and messages[0].get("role") == "system":
                        messages.insert(1, memory_system_msg)
                    else:
                        messages.insert(0, memory_system_msg)
                    req_data["messages"] = messages

        # Forward to upstream (Ollama / LM Studio / vLLM / OpenAI)
        upstream_base = self.headers.get("X-LLM-URL") or req_data.pop("llm_url", None) or self.upstream_url
        base_clean = upstream_base.rstrip("/")
        if base_clean.endswith("/v1/chat/completions"):
            upstream_target = base_clean
        elif base_clean.endswith("/v1"):
            upstream_target = base_clean + "/chat/completions"
        else:
            upstream_target = base_clean + "/v1/chat/completions"

        fwd_headers = {"Content-Type": "application/json"}
        auth_header = self.headers.get("Authorization")
        if auth_header:
            fwd_headers["Authorization"] = auth_header

        is_stream = bool(req_data.get("stream", False))
        new_payload = json.dumps(req_data).encode("utf-8")
        req = urllib.request.Request(
            upstream_target,
            data=new_payload,
            headers=fwd_headers
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                if is_stream:
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    for chunk in resp:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    resp_body = resp.read()
                    for k, v in resp.headers.items():
                        if k.lower() not in ["content-length", "transfer-encoding"]:
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)

                # Record user interaction into memory
                if self.vault is not None and user_msg:
                    self.vault.add(user_msg, source="proxy_chat", metadata={"role": "user"})
                    self.vault.flush()

        except urllib.error.HTTPError as e:
            # Forward exact upstream status code and JSON error payload to client
            resp_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_json = json.dumps({"error": f"Upstream connection failed: {e}"}).encode("utf-8")
            self.wfile.write(err_json)

    def _proxy_get(self, target_url):
        try:
            fwd_headers = {}
            if self.headers.get("Authorization"):
                fwd_headers["Authorization"] = self.headers.get("Authorization")
            req = urllib.request.Request(target_url, headers=fwd_headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()

    def _proxy_post(self, target_url):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            fwd_headers = {"Content-Type": "application/json"}
            if self.headers.get("Authorization"):
                fwd_headers["Authorization"] = self.headers.get("Authorization")
            req = urllib.request.Request(target_url, data=body, headers=fwd_headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()


def run_proxy(vault_path: str = "memory.dat", port: int = 5000,
              upstream_url: str = "http://localhost:11434", cite: bool = True,
              password: str = None, host: str = "127.0.0.1",
              allow_vault_switch: bool = False, allow_cors: bool = False):
    """Run the memory proxy. LOOPBACK-ONLY by default.

    None of these endpoints authenticate, so 3.0.2's ``0.0.0.0`` bind exposed
    the vault -- and an arbitrary-path vault-creation primitive -- to the whole
    network. ``host="0.0.0.0"`` is still available and is now a deliberate act.
    """
    vault = Vault(vault_path, password=password)
    MemoryProxyHandler.vault = vault
    MemoryProxyHandler.password = password
    MemoryProxyHandler.vault_root = os.path.realpath(
        os.path.dirname(os.path.abspath(vault_path)) or ".")
    MemoryProxyHandler.allow_vault_switch = bool(allow_vault_switch)
    MemoryProxyHandler.allow_cors = bool(allow_cors)
    MemoryProxyHandler.upstream_url = upstream_url.rstrip("/")
    MemoryProxyHandler.cite_default = cite
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[nanomem proxy] WARNING: binding {host} exposes an UNAUTHENTICATED "
              f"memory API to the network.")
    server = HTTPServer((host, port), MemoryProxyHandler)
    print(f"[nanomem proxy] Listening on http://{host}:{port}")
    print(f"[nanomem proxy] Forwarding to {upstream_url}")
    print(f"[nanomem proxy] Memory Vault: {vault_path}")
    print(f"[nanomem proxy] Citations: {'ENABLED' if cite else 'DISABLED'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[nanomem proxy] Shutting down...")
    finally:
        vault.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="nanomem Universal Memory Proxy")
    parser.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    parser.add_argument("--port", type=int, default=5000, help="Listen port")
    parser.add_argument("--upstream", type=str, default="http://localhost:11434", help="Upstream LLM base URL")
    parser.add_argument("--no-cite", "--no-citations", dest="no_cite", action="store_true", help="Disable document and line citations in forwarded memory prompts")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (loopback by default; these endpoints do not authenticate)")
    parser.add_argument("--password-stdin", action="store_true", help="Read the vault passphrase from stdin")
    parser.add_argument("--allow-vault-switch", action="store_true", help="Let /vault/init change the ACTIVE vault (off by default)")
    args = parser.parse_args()
    pw = None
    if args.password_stdin:
        import sys as _sys
        pw = _sys.stdin.readline().rstrip("\n") or None
    run_proxy(vault_path=args.vault, port=args.port, upstream_url=args.upstream,
              cite=not args.no_cite, password=pw, host=args.host,
              allow_vault_switch=args.allow_vault_switch)


