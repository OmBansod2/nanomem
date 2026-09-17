"""
NanoMem Personal AI Terminal (Universal Multi-Engine Support)
=============================================================
Universal personal AI terminal with continuous latent memory.
Supports:
  1. Ollama    (Default: http://localhost:11434)
  2. LM Studio (Default: http://localhost:1234)
  3. vLLM      (Default: http://localhost:8000)
  4. Custom    (Any OpenAI-compatible /v1 endpoint)

Usage:
  python chat.py                         # Auto-detects running engine & models
  python chat.py --lmstudio              # Connects to LM Studio
  python chat.py --vllm                  # Connects to vLLM
  python chat.py --model qwen3.5:9b-16k   # Chooses specific model
  python chat.py --url http://10.0.0.5:8000/v1  # Custom endpoint
  python chat.py --select                # Interactive engine & model picker
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

# Ensure local package import
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_dir)
mac_bundle_dir = os.path.join(cur_dir, "nanomem_mac_bundle")
if os.path.isdir(mac_bundle_dir):
    sys.path.insert(0, mac_bundle_dir)

from nanomem import Vault, list_users, create_user, delete_user, user_exists

VAULT_FILE = "personal_memory.dat"

STANDARD_BACKENDS = {
    "ollama": {
        "name": "Ollama",
        "url": "http://localhost:11434/v1",
        "health": "http://localhost:11434/api/tags",
        "default_port": 11434
    },
    "lmstudio": {
        "name": "LM Studio",
        "url": "http://localhost:1234/v1",
        "health": "http://localhost:1234/v1/models",
        "default_port": 1234
    },
    "vllm": {
        "name": "vLLM",
        "url": "http://localhost:8000/v1",
        "health": "http://localhost:8000/v1/models",
        "default_port": 8000
    }
}

def probe_backend(base_url: str, timeout: float = 1.5) -> Tuple[bool, List[str]]:
    """Probes an endpoint for /v1/models and returns (is_alive, model_ids)."""
    norm_url = base_url.rstrip("/")
    models_url = norm_url + "/models" if norm_url.endswith("/v1") else norm_url + "/v1/models"
    try:
        req = urllib.request.Request(models_url, headers={"User-Agent": "nanomem-client"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            if "data" in data and isinstance(data["data"], list):
                for m in data["data"]:
                    mid = m.get("id") or m.get("name")
                    if mid:
                        models.append(mid)
            elif "models" in data and isinstance(data["models"], list):
                for m in data["models"]:
                    mid = m.get("name") or m.get("id")
                    if mid:
                        models.append(mid)
            return True, models
    except Exception:
        return False, []

def detect_available_backends() -> Dict[str, Dict]:
    """Scans standard ports to find running LLM engines."""
    active = {}
    for key, info in STANDARD_BACKENDS.items():
        alive, models = probe_backend(info["url"])
        if alive:
            # Filter out non-chat models like embedding-only weights
            chat_models = [m for m in models if not any(x in m.lower() for x in ["embed", "minilm", "bge", "bert"])]
            active[key] = {
                "name": info["name"],
                "url": info["url"],
                "models": chat_models if chat_models else models
            }
    return active

class LLMServiceError(Exception):
    def __init__(self, status_code: int, message: str, is_context_overflow: bool = False, is_model_missing: bool = False, is_oom: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.is_context_overflow = is_context_overflow
        self.is_model_missing = is_model_missing
        self.is_oom = is_oom


def stream_openai_chat(api_url: str, model: str, messages: list, api_key: Optional[str] = None):
    """Universal SSE streaming for OpenAI /v1/chat/completions with structured error diagnosis."""
    chat_url = api_url.rstrip("/") + "/chat/completions" if api_url.rstrip("/").endswith("/v1") else api_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    headers = {"Content-Type": "application/json"}
    key = api_key or os.getenv("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
            err_msg = err_json.get("error", {}).get("message", err_json.get("error", err_body))
        except Exception:
            err_msg = err_body or str(e)

        err_lower = str(err_msg).lower()
        is_ctx = any(k in err_lower for k in ["context length", "context window", "maximum context", "tokens exceed", "too many tokens", "input length"])
        is_missing = e.code == 404 or any(k in err_lower for k in ["not found", "does not exist", "pull"])
        is_oom = e.code == 500 and any(k in err_lower for k in ["out of memory", "cuda", "vram", "allocat"])

        raise LLMServiceError(
            status_code=e.code,
            message=str(err_msg).strip(),
            is_context_overflow=is_ctx,
            is_model_missing=is_missing,
            is_oom=is_oom
        )

    with resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: ") and line_str != "data: [DONE]":
                try:
                    chunk = json.loads(line_str[6:])
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                except Exception:
                    continue

def parse_args():
    parser = argparse.ArgumentParser(description="NanoMem Universal AI Chat Terminal")
    parser.add_argument("--backend", choices=["ollama", "lmstudio", "vllm", "custom"], help="Force specific engine")
    parser.add_argument("--ollama", action="store_true", help="Shortcut for Ollama (port 11434)")
    parser.add_argument("--lmstudio", action="store_true", help="Shortcut for LM Studio (port 1234)")
    parser.add_argument("--vllm", action="store_true", help="Shortcut for vLLM (port 8000)")
    parser.add_argument("--url", type=str, help="Custom OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model", type=str, help="Specific model name (e.g. llama3.2:3b, qwen2.5:7b)")
    parser.add_argument("--vault", type=str, default=VAULT_FILE,
                        help="Path to the vault file (plaintext unless you set a passphrase)")
    parser.add_argument("--password-stdin", action="store_true",
                        help="Read the vault passphrase from the first line of stdin")
    parser.add_argument("-p", "--password", action="store_true",
                        help="Prompt for the vault passphrase (no echo)")
    parser.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")
    parser.add_argument("--import", dest="import_vault", type=str, help="Import and merge memories from another .dat vault file")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authenticated endpoints (or set OPENAI_API_KEY env var)")
    parser.add_argument("--no-cite", "--no-citations", dest="no_cite", action="store_true", help="Disable document and line citations in LLM responses")
    parser.add_argument("--select", action="store_true", help="Prompt to interactively pick backend & model on start")
    return parser.parse_args()

def run_terminal_chat():
    args = parse_args()

    # ── User Profile Resolution & First-Time Onboarding ──
    all_users = list_users()

    if not all_users and (not args.vault or args.vault == VAULT_FILE) and not args.profile:
        # First-time / empty state: No profiles exist!
        print("\n" + "=" * 70)
        print("  👋 Welcome to NanoMem!")
        print("=" * 70)
        print("  No user profiles found. Let's create your first profile.")
        try:
            raw_name = input("  Enter username (e.g. 'OM-personal', 'default'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
        user_name = raw_name if raw_name else "default"
        try:
            vault_path = create_user(user_name)
            print(f"  ✅ Created new user profile '{user_name}' -> '{vault_path}'\n")
        except Exception as e:
            vault_path = f"memory_{user_name}.dat"
    elif args.profile:
        # User explicitly specified --user / --profile
        clean_p = args.profile.strip()
        if not user_exists(clean_p):
            print(f"\n⚠️  User profile '{clean_p}' does not exist.")
            try:
                create_choice = input(f"   Would you like to create user '{clean_p}' now? [Y/n]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
            if create_choice in ["", "y", "yes"]:
                vault_path = create_user(clean_p)
                print(f"   ✅ Created user profile '{clean_p}' -> '{vault_path}'\n")
            else:
                print("   Available users:")
                for u in all_users:
                    print(f"     • {u['username']} ({u['documents']} docs)")
                sys.exit(0)
        else:
            vault_path = f"memory_{clean_p}.dat" if not clean_p.endswith(".dat") else clean_p
    elif args.vault != VAULT_FILE:
        vault_path = args.vault
    else:
        # No user specified and users exist
        if len(all_users) == 1:
            vault_path = all_users[0]["filepath"]
        elif len(all_users) > 1:
            print("\n👥 Multiple User Profiles Detected:")
            for idx, u in enumerate(all_users, 1):
                print(f"  [{idx}] {u['username']:<18} ({u['documents']} memories, {u['size_kb']} KB)")
            print(f"  [{len(all_users) + 1}] [Create new user profile...]")
            try:
                u_choice = input(f"\nSelect profile [1-{len(all_users) + 1}] (Default 1): ").strip()
            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
            if u_choice.isdigit() and int(u_choice) == len(all_users) + 1:
                try:
                    new_u = input("Enter new username (e.g. OM-work): ").strip()
                except (KeyboardInterrupt, EOFError):
                    sys.exit(0)
                new_u = new_u if new_u else "default"
                vault_path = create_user(new_u)
                print(f"✅ Created user profile '{new_u}' -> '{vault_path}'\n")
            elif u_choice.isdigit() and 1 <= int(u_choice) <= len(all_users):
                vault_path = all_users[int(u_choice) - 1]["filepath"]
            else:
                vault_path = all_users[0]["filepath"]
        else:
            vault_path = VAULT_FILE

    # Determine target backend URL
    target_url = None
    target_name = "LLM Engine"
    selected_model = args.model

    if args.url:
        target_url = args.url.rstrip("/")
        target_name = "Custom Endpoint"
    elif args.lmstudio or args.backend == "lmstudio":
        target_url = STANDARD_BACKENDS["lmstudio"]["url"]
        target_name = "LM Studio"
    elif args.vllm or args.backend == "vllm":
        target_url = STANDARD_BACKENDS["vllm"]["url"]
        target_name = "vLLM"
    elif args.ollama or args.backend == "ollama":
        target_url = STANDARD_BACKENDS["ollama"]["url"]
        target_name = "Ollama"

    # Auto-detection if not specified
    if not target_url:
        active_backends = detect_available_backends()
        if not active_backends:
            print("\n⚠️  No running local LLM engine detected on ports 11434 (Ollama), 1234 (LM Studio), or 8000 (vLLM).")
            print("   Please start Ollama, LM Studio, or vLLM and try again.")
            print("   Or pass an explicit URL: python chat.py --url http://your-server:port/v1\n")
            sys.exit(1)

        if args.select and len(active_backends) > 0:
            print("\n🔍 Available Local LLM Backends:")
            backend_keys = list(active_backends.keys())
            for i, k in enumerate(backend_keys, 1):
                b = active_backends[k]
                print(f"  [{i}] {b['name']} ({b['url']}) - {len(b['models'])} model(s) detected")
            choice = input(f"\nSelect backend [1-{len(backend_keys)}] (Default 1): ").strip()
            chosen_idx = int(choice) - 1 if (choice.isdigit() and 1 <= int(choice) <= len(backend_keys)) else 0
            b_key = backend_keys[chosen_idx]
            target_url = active_backends[b_key]["url"]
            target_name = active_backends[b_key]["name"]
        else:
            # Pick first active backend (prioritize Ollama -> LM Studio -> vLLM)
            for preferred in ["ollama", "lmstudio", "vllm"]:
                if preferred in active_backends:
                    target_url = active_backends[preferred]["url"]
                    target_name = active_backends[preferred]["name"]
                    break
            if not target_url:
                k = list(active_backends.keys())[0]
                target_url = active_backends[k]["url"]
                target_name = active_backends[k]["name"]

    # Verify connection and get models
    alive, available_models = probe_backend(target_url)
    if not alive:
        print(f"\n⚠️  Could not connect to {target_name} at {target_url}.")
        print("   Make sure the server is running and accepting HTTP requests.\n")
        sys.exit(1)

    # Model selection
    if not selected_model:
        if args.select and available_models:
            print(f"\n🔍 Available Models on {target_name}:")
            for i, m in enumerate(available_models, 1):
                print(f"  [{i}] {m}")
            m_choice = input(f"Select model [1-{len(available_models)}] (Default 1): ").strip()
            m_idx = int(m_choice) - 1 if (m_choice.isdigit() and 1 <= int(m_choice) <= len(available_models)) else 0
            selected_model = available_models[m_idx]
        elif available_models:
            # Pick best chat model
            chat_candidates = [m for m in available_models if not any(x in m.lower() for x in ["embed", "minilm", "bge", "bert"])]
            for preferred in ["llama3.2:3b", "llama3.2:latest", "qwen3.5:9b-16k", "gemma3:4b"]:
                if preferred in chat_candidates:
                    selected_model = preferred
                    break
            if not selected_model:
                selected_model = chat_candidates[0] if chat_candidates else available_models[0]
        else:
            selected_model = "default"

    print("\n" + "=" * 78)
    print("  🧠 NANOMEM UNIVERSAL AI TERMINAL (Continuous Latent Memory Active)")
    print("=" * 78)
    print(f"  • Engine Backend : {target_name} ({target_url})")
    print(f"  • Active Model   : {selected_model}")
    print(f"  • Memory Vault   : {vault_path}")
    print(f"  • Privacy Mode   : 100% Offline & Air-Gapped")
    print("  • Commands       : 'exit' to quit | 'memory' to view | '/ingest <file>' | '/profile <name>' | '/model'")
    print("=" * 78)

    vault_password = None
    if getattr(args, "password_stdin", False):
        vault_password = sys.stdin.readline().rstrip("\n") or None
    elif getattr(args, "password", False):
        import getpass
        vault_password = getpass.getpass("Vault passphrase: ") or None
    elif os.getenv("NANOMEM_PASSWORD"):
        vault_password = os.getenv("NANOMEM_PASSWORD")

    with Vault(vault_path, password=vault_password) as vault:
        try:
            _st = vault.stats()
            print(f"  • Vault Profile   : {_st['total_documents']} docs, "
                  f"{_st['active_heap_ram_kb']:.0f} KB resident, {_st['cipher']}")
        except Exception:
            pass
        # One-click vault import on launch
        if getattr(args, "import_vault", None):
            imp_path = args.import_vault
            if os.path.exists(imp_path):
                print(f"\n[Importing]: Merging memories from '{imp_path}' into '{vault_path}'...")
                res = vault.merge(imp_path, deduplicate=False, reconcile_revisions=True)
                print(f"✅ Successfully imported {res['added']} memories from '{imp_path}'!\n")
            else:
                print(f"\n⚠️  Import file not found: '{imp_path}'\n")

        stats = vault.stats()
        print(f"\n[Memory Active]: Currently holding {stats['total_documents']} saved facts.\n")

        conversation_history = [
            {
                "role": "system",
                "content": "You are an intelligent, helpful personal AI assistant with continuous long-term memory. When provided with [Retrieved Memory from Past Sessions], naturally ground your answers on those facts without explicitly repeating memory IDs."
            }
        ]
        cite_enabled = not getattr(args, "no_cite", False)

        while True:
            try:
                user_input = input("👤 You > ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q", "/exit", "/quit", "/q"]:
                    # Say what the vault actually is. Plaintext is the default and
                    # this script has no password path of its own beyond --password.
                    _enc = vault.stats().get("encrypted_at_rest", False)
                    print("\nSaving memory and exiting. Memories are "
                          + ("encrypted" if _enc else "stored UNENCRYPTED")
                          + " on disk. Goodbye!\n")
                    break

                # In-chat helper commands
                if user_input.lower().startswith("/ingest ") or user_input.lower().startswith("/doc "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        f_path = parts[1].strip().strip('"').strip("'")
                        if os.path.exists(f_path):
                            try:
                                chunks = vault.ingest_file(f_path)
                                print(f"\n📄 Successfully ingested {chunks} memory chunks from '{os.path.basename(f_path)}' into your brain!\n")
                            except Exception as e:
                                print(f"\n⚠️  Could not ingest file: {e}\n")
                        else:
                            print(f"\n⚠️  File not found: '{f_path}'\n")
                    else:
                        print("\nUsage: /ingest <path_to_document> (e.g. /ingest notes.txt, /ingest document.md)\n")
                    continue

                if user_input.lower().startswith("/import ") or user_input.lower().startswith("/merge "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        target_import = parts[1].strip().strip('"').strip("'")
                        if os.path.exists(target_import):
                            res = vault.merge(target_import, deduplicate=False, reconcile_revisions=True)
                            print(f"\n✅ Successfully imported {res['added']} memories from '{target_import}' into your brain!\n")
                        else:
                            print(f"\n⚠️  File not found: '{target_import}'\n")
                    else:
                        print("\nUsage: /import <path_to_vault.dat>\n")
                    continue

                if user_input.lower() in ["/users", "/user list", "users"]:
                    u_list = list_users()
                    print(f"\n👥 Registered User Profiles ({len(u_list)} total):")
                    for u in u_list:
                        curr = " [ACTIVE]" if os.path.abspath(u["filepath"]) == os.path.abspath(vault_path) else ""
                        print(f"  • {u['username']:<18} ({u['documents']} memories, {u['size_kb']} KB){curr}")
                    print("\nCommands:")
                    print("  /user create <name>  -> Create a brand new user profile")
                    print("  /user switch <name>  -> Switch to an existing user profile\n")
                    continue

                if user_input.lower().startswith("/user create ") or user_input.lower().startswith("/profile create "):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 3 or not parts[2].strip():
                        print("\nUsage: /user create <username>\n")
                        continue
                    p_name = parts[2].strip().strip('"').strip("'")
                    try:
                        new_target = create_user(p_name)
                        vault.close()
                        vault = Vault(new_target, password=vault_password)
                        vault_path = new_target
                        print(f"\n✅ Created and switched to new user profile '{p_name}' -> '{new_target}' (0 memories).\n")
                    except Exception as e:
                        print(f"\n⚠️  Could not create user profile: {e}\n")
                    continue

                if user_input.lower().startswith("/user switch ") or user_input.lower().startswith("/user ") or user_input.lower().startswith("/profile "):
                    parts = user_input.split()
                    if len(parts) >= 2:
                        p_name = parts[-1].strip().strip('"').strip("'")
                        from nanomem.users import get_vault_filename
                        if not user_exists(p_name):
                            print(f"\n⚠️  User profile '{p_name}' does not exist.")
                            print(f"   To create this user, type: /user create {p_name}")
                            print(f"   To see all available users, type: /users\n")
                            continue
                        new_vault_path = get_vault_filename(p_name)
                        vault.close()
                        vault = Vault(new_vault_path, password=vault_password)
                        vault_path = new_vault_path
                        st = vault.stats()
                        print(f"\n🔄 Switched active user profile to '{p_name}' -> '{new_vault_path}' ({st['total_documents']} memories loaded).\n")
                    else:
                        print("\nUsage: /user switch <username> or /user create <username>\n")
                    continue

                if user_input.lower().startswith("/switch ") or user_input.lower().startswith("/vault "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        new_vault_path = parts[1].strip().strip('"').strip("'")
                        vault.close()
                        vault = Vault(new_vault_path, password=vault_password)
                        vault_path = new_vault_path
                        st = vault.stats()
                        print(f"\n🔄 Switched active vault to '{new_vault_path}' ({st['total_documents']} memories loaded).\n")
                    else:
                        print(f"\nCurrent vault: '{vault_path}'. Usage: /switch <path_to_vault.dat>\n")
                    continue

                if user_input.lower() in ["/help", "help", "/h"]:
                    print("\n--- 💡 Available Commands ---")
                    print("  /users               -> View all user profiles")
                    print("  /user create <name>  -> Create and switch to new profile")
                    print("  /user switch <name>  -> Switch active profile")
                    print("  /stats               -> Show memory vault statistics")
                    print("  /dump or memory      -> Display stored personal facts")
                    print("  /model               -> Switch active LLM model")
                    print("  /provider            -> Switch LLM provider (Ollama / LM Studio / vLLM)")
                    print("  /ingest <file>       -> Ingest document into memory")
                    print("  /import <vault.dat>  -> Merge external vault into current brain")
                    print("  /cite                -> Toggle document/source citations ON/OFF")
                    print("  /forget <topic>      -> Safely erase a fact with confirmation (e.g. /forget phone number)")
                    print("  /quit or exit        -> Save and exit session\n")
                    continue

                if user_input.lower() in ["/stats", "stats"]:
                    st = vault.stats()
                    print("\n--- 📊 Memory Vault Stats ---")
                    print(f"  • Vault File     : {st['file_path']}")
                    print(f"  • Total Memories : {st['total_documents']}")
                    print(f"  • Physical Size  : {st['file_size_mb']:.3f} MB")
                    print(f"  • Active Heap RAM: {st['active_heap_ram_kb']} KB")
                    print(f"  • Encryption     : {st['cipher']}")
                    print("-----------------------------\n")
                    continue

                if user_input.lower() in ["/inspect", "inspect"]:
                    insp = vault.inspect()
                    print("\n--- 🔍 Memory Vault Inspection ---")
                    print(f"  • File: {insp['file_path']}")
                    print(f"  • Total Memories: {insp['total_documents']}")
                    print("\n  📁 Sources:")
                    if not insp["sources"]:
                        print("    (None)")
                    else:
                        for s, c in list(insp["sources"].items())[:8]:
                            print(f"    • {s:<28} ({c} memories)")
                    if insp["metadata_summary"]:
                        print("\n  🏷️  Categories & Tags:")
                        for k, vals in list(insp["metadata_summary"].items())[:6]:
                            tag_str = ", ".join([f"'{vn}' ({cnt})" for vn, cnt in list(vals.items())[:5]])
                            print(f"    • {k}: {tag_str}")
                    print("----------------------------------\n")
                    continue

                if user_input.lower() in ["memory", "/memory", "/dump", "dump"]:
                    total = vault.stats()["total_documents"]
                    print(f"\n--- 🧠 Stored Memories ({total} Total) ---")
                    hits = vault.get_all_records(include_embeddings=False)[:50]
                    for idx, h in enumerate(hits, 1):
                        print(f"  [{idx}] {h['text']}")
                    print("------------------------------------------\n")
                    continue

                if user_input.lower() in ["/provider", "provider"]:
                    print("\nAvailable Providers:")
                    providers = list(STANDARD_BACKENDS.keys())
                    for idx, p_key in enumerate(providers, 1):
                        curr = " (CURRENT)" if STANDARD_BACKENDS[p_key]["url"] == target_url else ""
                        print(f"  [{idx}] {STANDARD_BACKENDS[p_key]['name']} ({STANDARD_BACKENDS[p_key]['url']}){curr}")
                    p_choice = input("Enter provider number or name (or Enter to keep): ").strip()
                    if p_choice.isdigit() and 1 <= int(p_choice) <= len(providers):
                        chosen = providers[int(p_choice) - 1]
                        target_name = STANDARD_BACKENDS[chosen]["name"]
                        target_url = STANDARD_BACKENDS[chosen]["url"]
                        _, live_models = probe_backend(target_url)
                        selected_model = live_models[0] if live_models else "default"
                        print(f"✅ Switched provider to {target_name} ({target_url}), model: {selected_model}\n")
                    continue

                if user_input.lower() in ["/model", "model"]:
                    _, live_models = probe_backend(target_url)
                    print(f"\nActive Model: {selected_model}")
                    print("Available Models:")
                    for i, m in enumerate(live_models, 1):
                        curr = " (CURRENT)" if m == selected_model else ""
                        print(f"  [{i}] {m}{curr}")
                    m_choice = input(f"Enter model name or number to switch (or Enter to keep): ").strip()
                    if m_choice.isdigit() and 1 <= int(m_choice) <= len(live_models):
                        selected_model = live_models[int(m_choice) - 1]
                        print(f"✅ Switched to model: {selected_model}\n")
                    elif m_choice in live_models:
                        selected_model = m_choice
                        print(f"✅ Switched to model: {selected_model}\n")
                    else:
                        print("Kept current model.\n")
                    continue

                if user_input.lower() in ["/cite", "cite", "/citations", "citations"]:
                    cite_enabled = not cite_enabled
                    st = "ENABLED (document and line citations will be attached)" if cite_enabled else "DISABLED (pure conversational responses without source tags)"
                    print(f"✅ Citations are now {st}.\n")
                    continue

                if user_input.lower().startswith("/forget ") or user_input.lower().startswith("/delete "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        target_query = parts[1].strip().strip('"').strip("'")
                        hits = vault.search(target_query, top_k=1, min_score=0.42)
                        if not hits:
                            print(f"\n🔍 No memory found matching '{target_query}'. Nothing was deleted.\n")
                        else:
                            cand = hits[0]
                            print(f"\n🗑️  Found matching memory in your brain:")
                            print(f"   • \"{cand['text']}\"")
                            src = cand.get("source", "unknown")
                            if src and src != "unknown":
                                print(f"     (Source: {src})")
                            try:
                                ans = input("\n⚠️  Are you sure you want to permanently erase this memory? [y/N]: ").strip().lower()
                                if ans in ["y", "yes"]:
                                    vault.delete(text_exact=cand["text"])
                                    print(f"✅ Memory erased permanently from disk.\n")
                                else:
                                    print("❌ Deletion cancelled. Your memory is safe.\n")
                            except (KeyboardInterrupt, EOFError):
                                print("\n❌ Deletion cancelled.\n")
                    else:
                        print("\nUsage: /forget <topic or fact> (e.g. /forget phone number, /delete address)\n")
                    continue

                # 1. Background Memory Retrieval
                memories = vault.search(user_input, top_k=3, min_score=0.53)
                context_block = ""
                if memories:
                    mem_texts = []
                    for m in memories:
                        src = m.get("source")
                        src_tag = f"[{src}] " if (cite_enabled and src and src not in ["chat_session", "user_input", "rest_api", "unknown"]) else ""
                        mem_texts.append(f"• {src_tag}{m['text']}")
                    header = "[Retrieved Context & Memory]:" if cite_enabled else "[Retrieved Memory]:"
                    context_block = f"\n{header}\n" + "\n".join(mem_texts) + "\n"

                # 2. Build message payload
                current_msg_content = context_block + user_input if context_block else user_input
                conversation_history.append({"role": "user", "content": current_msg_content})

                # 3. Stream Response
                print(f"\n🤖 AI > ", end="", flush=True)
                full_reply = ""
                try:
                    for chunk in stream_openai_chat(target_url, selected_model, conversation_history, api_key=args.api_key):
                        full_reply += chunk
                        print(chunk, end="", flush=True)
                    print("\n")
                except LLMServiceError as e:
                    if e.is_context_overflow:
                        print(f"\n⚠️  [Notice: Model '{selected_model}' context length exceeded. Pruning older chat history and retrying...]\n")
                        # Aggressively prune conversation_history: keep only system message and current message
                        conversation_history = [conversation_history[0], conversation_history[-1]]
                        c_text = conversation_history[-1]["content"]
                        c_words = c_text.split()
                        if len(c_words) > 1200:
                            conversation_history[-1]["content"] = " ".join(c_words[:1200]) + "..."
                        try:
                            print(f"🤖 AI > ", end="", flush=True)
                            for chunk in stream_openai_chat(target_url, selected_model, conversation_history, api_key=args.api_key):
                                full_reply += chunk
                                print(chunk, end="", flush=True)
                            print("\n")
                        except Exception as retry_err:
                            print(f"\n⚠️  Could not complete response after context recovery: {retry_err}\n")
                            continue
                    elif e.is_model_missing:
                        print(f"\n⚠️  Model '{selected_model}' is not installed on {target_name}.")
                        print(f"   Run 'ollama pull {selected_model}' or type '/model' to switch models.\n")
                        continue
                    elif e.is_oom:
                        print(f"\n⚠️  {target_name} ran out of memory (VRAM/RAM) while running '{selected_model}'.")
                        print(f"   Try a lighter model (e.g. llama3.2:3b or qwen2.5:3b) with '/model'.\n")
                        continue
                    else:
                        print(f"\n⚠️  {target_name} Error ({e.status_code}): {e.message}\n")
                        continue
                except urllib.error.URLError as e:
                    print(f"\n⚠️  Connection Error: Could not reach {target_name} at {target_url}.")
                    print(f"   Make sure {target_name} is running (e.g. 'ollama serve' or start LM Studio).\n")
                    continue
                except Exception as e:
                    print(f"\n⚠️  Unexpected Error: {e}\n")
                    continue

                conversation_history.append({"role": "assistant", "content": full_reply})

                # 4. Background Fact Extraction & Storage
                if Vault.should_store(user_input):
                    vault.add(user_input, source="chat_session")
                    vault.flush()
                    print("   ↳ 💾 [Memory Saved: Stored and safely synced to SSD]")

                # Keep in-context history bounded (memory vault maintains long-term persistence!)
                if len(conversation_history) > 8:
                    conversation_history = [conversation_history[0]] + conversation_history[-6:]

            except (KeyboardInterrupt, EOFError):
                _enc = vault.stats().get("encrypted_at_rest", False)
                print("\n\nSession paused. Memories saved to disk"
                      + (" (encrypted)." if _enc else " (UNENCRYPTED; run "
                         "`nanomem rekey --vault <path>` to add a passphrase).") + "\n")
                break

if __name__ == "__main__":
    run_terminal_chat()
