"""
Production HTTP REST Server for NanoMem Continuous Memory Engine
===================================================================
Author: Om Yadorao Bansod
Features:
- Zero external web framework dependencies (built on standard library ThreadingHTTPServer)
- Pure Python / NumPy; resident footprint is reported by /health (active_heap_ram_kb)
- Endpoints:
  * GET  /health  : System health, indexed chunk count, RAM footprint, engine state.
  * POST /ingest  : Batched document ingestion and indexing.
  * POST /search  : Exhaustive exact cosine retrieval over the whole vault.
                    Measured engine p50 0.34 ms at 10k records and 1.81 ms at
                    71,433 (scratch/refound/scale_results_v3r3.json), plus the
                    embedding call, which dominates. Not sub-millisecond at
                    scale, and it never was.
  * POST /ask     : Grounded retrieval question answering.
  * POST /save    : Save index / flush to disk for persistence.
  * POST /load    : Restore / switch vault container on disk.
"""

import os
import sys
import json
import time
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional

# Ensure nanomem package is importable
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

from nanomem.vault import Vault, resolve_vault_path

_VAULT: Optional[Vault] = None
_VAULT_PATH: str = "production_vault.dat"
_PASSWORD: Optional[str] = None
# The one directory /load may name a vault in. /load takes its path from an
# unauthenticated request body, and Container creates the file and os.makedirs
# its parents, so 3.0.2's /load was an arbitrary-path file-creation primitive.
_VAULT_ROOT: str = os.path.abspath(".")
_LOCK = threading.RLock()


def set_password(password: Optional[str]) -> None:
    """Passphrase every vault this server opens will use.

    Without it a password-protected vault was simply unreachable over REST:
    /ingest and /search raised PasswordRequiredError and the only way to supply
    one was the inherited NANOMEM_PASSWORD environment variable, which cli.py
    itself warns against. ``run_server(..., password_stdin=True)`` reads it from
    the first line of stdin so it never appears in `ps` or shell history.
    """
    global _PASSWORD, _VAULT
    with _LOCK:
        _PASSWORD = password or None
        if _VAULT is not None:
            _VAULT.close()
            _VAULT = None


def get_vault(path: Optional[str] = None) -> Vault:
    global _VAULT, _VAULT_PATH
    with _LOCK:
        if path and path != _VAULT_PATH:
            if _VAULT is not None:
                _VAULT.flush()
                _VAULT.close()
            _VAULT_PATH = path
            _VAULT = Vault(_VAULT_PATH, password=_PASSWORD)
        elif _VAULT is None:
            _VAULT = Vault(_VAULT_PATH, password=_PASSWORD)
        return _VAULT


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class LatentMemoryAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ["/", "/health"]:
            vault = get_vault()
            stats = vault.stats()
            self._send_json(200, {
                "status": "healthy",
                "service": "NanoMem Continuous Memory Engine",
                "inventor": "Om Yadorao Bansod",
                "indexed_chunks": stats["total_documents"],
                "active_heap_ram_kb": stats["active_heap_ram_kb"],
                "file_size_mb": stats["file_size_mb"],
                "cipher": stats["cipher"],
                "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            self._send_json(404, {"error": "Endpoint not found", "path": self.path})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 10 * 1024 * 1024:
            self._send_json(413, {"error": "Payload exceeds 10 MB limit."})
            return

        post_body = self.rfile.read(content_length)
        try:
            payload = json.loads(post_body.decode("utf-8")) if post_body else {}
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {str(e)}"})
            return

        vault = get_vault()

        if self.path == "/ingest":
            docs = payload.get("documents", [])
            chunks = payload.get("chunks", [])
            items = docs if docs else chunks
            if not items:
                self._send_json(400, {"error": "Must provide 'documents' or 'chunks' list in payload."})
                return

            with _LOCK:
                t0 = time.perf_counter()
                for item in items:
                    if isinstance(item, str):
                        vault.add(item, source="rest_api")
                    elif isinstance(item, dict):
                        vault.add(item.get("text", ""), source=item.get("source", "rest_api"), metadata=item.get("metadata", {}))
                vault.flush()
                t_elapsed = time.perf_counter() - t0

            self._send_json(200, {
                "status": "success",
                "chunks_ingested": len(items),
                "total_indexed_chunks": vault.stats()["total_documents"],
                "time_seconds": round(t_elapsed, 4)
            })

        elif self.path == "/search":
            query = payload.get("query", "")
            if not query:
                self._send_json(400, {"error": "Must provide 'query' parameter."})
                return

            top_k = max(1, min(int(payload.get("top_k", 5)), 50))
            temporal_dir = payload.get("temporal_direction", "current")

            with _LOCK:
                t0 = time.perf_counter()
                hits = vault.search(query, top_k=top_k, temporal_direction=temporal_dir)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            results = [
                {
                    "rank": r + 1,
                    "id": h.get("doc_id", r + 1),
                    "score": round(float(h.get("score", 0.0)), 4),
                    "text": h.get("text", ""),
                    "revision": h.get("revision", 1),
                    "metadata": h.get("metadata", {})
                }
                for r, h in enumerate(hits)
            ]

            self._send_json(200, {
                "query": query,
                "top_k": top_k,
                "retrieval_latency_ms": round(elapsed_ms, 3),
                "hits_count": len(results),
                "hits": results
            })

        elif self.path == "/ask":
            query = payload.get("query", "")
            if not query:
                self._send_json(400, {"error": "Must provide 'query' parameter."})
                return

            top_k = max(1, min(int(payload.get("top_k", 3)), 10))
            with _LOCK:
                t0 = time.perf_counter()
                hits = vault.search(query, top_k=top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            context_texts = [h.get("text", "") for h in hits]
            answer = f"Based on retrieved memory: {' | '.join(context_texts[:2])}" if context_texts else "No matching memory found."

            self._send_json(200, {
                "query": query,
                "answer": answer,
                "sources": context_texts,
                "latency_ms": round(elapsed_ms, 2)
            })

        elif self.path == "/save":
            with _LOCK:
                vault.flush()
                stats = vault.stats()
            self._send_json(200, {
                "status": "success",
                "message": f"Index persisted to {_VAULT_PATH}",
                "total_chunks": stats["total_documents"]
            })

        elif self.path == "/load":
            requested = payload.get("save_dir") or payload.get("vault")
            if requested:
                try:
                    save_dir = resolve_vault_path(requested, _VAULT_ROOT)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
            else:
                save_dir = _VAULT_PATH
            with _LOCK:
                t0 = time.perf_counter()
                v = get_vault(save_dir)
                load_ms = (time.perf_counter() - t0) * 1000.0
                stats = v.stats()

            self._send_json(200, {
                "status": "success",
                "message": f"Index restored from {save_dir}",
                "cold_start_load_ms": round(load_ms, 2),
                "total_chunks": stats["total_documents"]
            })

        else:
            self._send_json(404, {"error": f"Unknown POST endpoint '{self.path}'"})


def run_server(host="127.0.0.1", port=8080, vault_path="production_vault.dat",
               password: Optional[str] = None, password_stdin: bool = False):
    print("=" * 80)
    print("STARTING NANOMEM CONTINUOUS MEMORY HTTP REST MICROSERVICE")
    print(f"Listening on http://{host}:{port}")
    print("Endpoints: GET /health, POST /ingest, POST /search, POST /ask, POST /save, POST /load")
    print(f"Active Vault: {vault_path}")
    print("=" * 80)

    if password_stdin:
        password = sys.stdin.readline().rstrip("\n") or None
    set_password(password)

    global _VAULT_ROOT
    _VAULT_ROOT = os.path.realpath(
        os.path.dirname(os.path.abspath(vault_path)) or ".")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding {host} exposes an UNAUTHENTICATED memory API "
              f"to the network.")

    # Pre-warm vault
    v = get_vault(vault_path)
    print(f"Encrypted at rest: {v.stats().get('encrypted_at_rest')}")

    server = ThreadedHTTPServer((host, port), LatentMemoryAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoMem REST API Microservice")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--vault", default="production_vault.dat", help="Vault container file path")
    parser.add_argument("-p", "--password", action="store_true",
                        help="Prompt for the vault passphrase (no echo)")
    parser.add_argument("--password-stdin", action="store_true",
                        help="Read the vault passphrase from the first line of stdin")
    args = parser.parse_args()
    pw = None
    if args.password:
        import getpass
        pw = getpass.getpass("Vault passphrase: ") or None
    run_server(host=args.host, port=args.port, vault_path=args.vault,
               password=pw, password_stdin=args.password_stdin)
