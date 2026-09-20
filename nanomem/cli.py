"""
nanomem.cli
~~~~~~~~~~~
Command-line interface for nanomem.
"""

from datetime import datetime
import os
import sys
import time
import getpass
import argparse
from .vault import Vault
from .proxy import run_proxy
from .users import list_users, create_user, delete_user, user_exists, get_vault_filename


def _parse_when(value):
    """A CLI time: ``YYYY-MM-DD``, an ISO timestamp, or a unix time.

    Returns ``None`` for ``None`` so callers can pass it straight through.
    A bare date means midnight local time, which is what someone typing
    ``--since 2026-03-01`` means by it.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        raise SystemExit(
            f"[nanomem] could not read '{s}' as a time. Use YYYY-MM-DD, "
            f"an ISO timestamp like 2026-03-01T14:30, or a unix time.")


def _fmt_when(ts):
    if ts is None:
        return "-"
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")


def _fail(message, code=1):
    """Report a failure the way a shell expects: stderr, and a non-zero exit.

    Four paths used to `print(...)` an error to STDOUT and fall off the end of
    the function, which exits 0 -- so `nanomem ingest missing.txt && next_step`
    ran `next_step`, and `nanomem ... > out.txt` put the error in out.txt with
    nothing on stderr. Measured: `ingest` a missing file, `user delete` an
    absent user, and `search` a vault that does not exist all exited 0.
    """
    print(f"[nanomem] Error: {message}", file=sys.stderr)
    sys.exit(code)


def _require_vault(path):
    """A read command must not treat a missing vault as an empty one.

    `nanomem search x --vault /tmp/typo.dat` printed "No matching memories found"
    and exited 0, so a mistyped path was indistinguishable from an empty vault --
    to a person and to a script. Opening a vault CREATES it, so this has to be
    checked before the open. Write commands still create on demand.
    """
    if not os.path.exists(path):
        _fail("no vault at %r. Write to it first (nanomem add ... --vault %s), "
              "or check the path." % (path, path))
    return path


def main():
    """Entry point. Vault-level failures are reported, not traced."""
    from .errors import NanomemError
    try:
        _main()
    except NanomemError as exc:
        print(f"[nanomem] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[nanomem] Interrupted.")
        sys.exit(130)


def _main():
    parser = argparse.ArgumentParser(
        prog="nanomem",
        description="nanomem: single-file persistent memory engine. numpy only; `nanomem stats` reports the measured resident footprint."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Every vault-touching command accepts these. Without them the ONLY way to
    # open a password-protected vault was NANOMEM_PASSWORD, which every child
    # process inherits and which lands in shell history and CI logs.
    pw_parent = argparse.ArgumentParser(add_help=False)
    pw_parent.add_argument("-p", "--password", action="store_true",
                           help="Prompt for the vault passphrase (no echo)")
    pw_parent.add_argument("--password-stdin", action="store_true",
                           help="Read the vault passphrase from the first line of stdin")

    # init
    p_init = subparsers.add_parser("init", parents=[pw_parent], help="Initialize a new empty memory vault")
    # BOTH spellings. `init` was the only vault command without `--vault`, so the
    # obvious first thing a user types -- `nanomem init --vault m.dat` -- failed
    # with "unrecognized arguments"; the positional stays for compatibility.
    p_init.add_argument("vault_pos", metavar="vault", type=str, nargs="?", default=None,
                        help="Path for the new vault file (default: memory.dat)")
    p_init.add_argument("--vault", dest="vault_opt", type=str, default=None,
                        help="Path for the new vault file (same as the positional)")
    p_init.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal)")

    # add
    p_add = subparsers.add_parser("add", parents=[pw_parent], help="Add a fact or note to memory")
    p_add.add_argument("text", type=str, help="Text to store")
    p_add.add_argument("--source", type=str, default="cli", help="Source tag")
    p_add.add_argument("--entity", type=str, default=None,
                       help="The attribute this states (e.g. employer, home_address). "
                            "Use the SAME value each time you record a new value of the "
                            "same attribute: that is what tells nanomem which statement "
                            "supersedes which. Without it a lexical tagger guesses, and "
                            "it guesses badly on narrative phrasing.")
    p_add.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_add.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")

    # search
    p_search = subparsers.add_parser("search", parents=[pw_parent], help="Search memory for relevant facts")
    p_search.add_argument("query", type=str, help="Search query")
    p_search.add_argument("-k", "--top-k", type=int, default=3, help="Number of results")
    p_search.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_search.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")
    p_search.add_argument("--as-of", dest="as_of", type=str, default=None,
                          help="Answer as the vault stood at this moment "
                               "(YYYY-MM-DD, an ISO timestamp, or a unix time). "
                               "Records written later are not candidates, so a "
                               "value that was superseded since is returned "
                               "wherever it was still the current one.")

    # history
    p_history = subparsers.add_parser("history", parents=[pw_parent],
                                      help="Every value a fact has held, oldest first")
    p_history.add_argument("query", type=str, help="What to trace")
    p_history.add_argument("-n", "--max-len", type=int, default=None,
                           help="Keep only the N most recent entries")
    p_history.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_history.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")

    # changes
    p_changes = subparsers.add_parser("changes", parents=[pw_parent],
                                      help="What the vault learned in a time window")
    p_changes.add_argument("--since", type=str, required=True,
                           help="YYYY-MM-DD, an ISO timestamp, or a unix time (exclusive)")
    p_changes.add_argument("--until", type=str, default=None,
                           help="Upper bound, inclusive (default: now)")
    p_changes.add_argument("-n", "--limit", type=int, default=None, help="Cap the number of rows")
    p_changes.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_changes.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")

    # ingest
    p_ingest = subparsers.add_parser("ingest", parents=[pw_parent], help="Ingest a file into memory")
    p_ingest.add_argument("file", type=str, help="Path to document file")
    p_ingest.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_ingest.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")
    p_ingest.add_argument("--expect-docs", dest="expect_docs", type=int, default=0,
                          help="How many chunks this ingest is about to add, if you know. "
                               "Sizes the resident arena once instead of growing into it. "
                               "Without it a directory ingest estimates the count from the "
                               "bytes it is about to read and a file ingest counts its own "
                               "chunks; both are hints and a wrong one is harmless.")

    # merge
    p_merge = subparsers.add_parser("merge", parents=[pw_parent], help="Merge an incoming vault into a target vault (Zero Data Loss)")
    p_merge.add_argument("source", type=str, help="Source vault to import from")
    p_merge.add_argument("--into", type=str, default="memory.dat", help="Target master vault to merge into")
    p_merge.add_argument("--project", type=str, default=None, help="Incoming project tag")
    p_merge.add_argument("--user-id", type=str, default=None, help="Incoming user ID tag")
    p_merge.add_argument("--dedup", action="store_true", help="Deduplicate identical texts within the same scope")

    # split
    p_split = subparsers.add_parser("split", parents=[pw_parent], help="Divide/split a vault into smaller target vaults")
    p_split.add_argument("source", type=str, help="Source vault to split")
    p_split.add_argument("--target", type=str, default=None, help="Target vault path for single split")
    p_split.add_argument("--source-doc", type=str, default=None, help="Split by source document name")
    p_split.add_argument("--key", type=str, default=None, help="Split and partition by metadata key into multiple files")
    p_split.add_argument("--dir", type=str, default="./splits", help="Output directory when partitioning by key")
    p_split.add_argument("--before", type=float, default=None, help="Cutoff timestamp for date-based split")
    p_split.add_argument("--purge", action="store_true", help="Perform real split by deleting matching records from source vault")

    # forget
    p_forget = subparsers.add_parser("forget", parents=[pw_parent], help="Erase a memory with confirmation")
    p_forget.add_argument("query", type=str, help="Search query for memory to erase")
    p_forget.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_forget.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    # proxy
    p_proxy = subparsers.add_parser("proxy", parents=[pw_parent], help="Run universal memory proxy for Ollama / LM Studio / vLLM")
    p_proxy.add_argument("--upstream", type=str, default="http://localhost:11434", help="Upstream LLM server URL (default: http://localhost:11434)")
    p_proxy.add_argument("--lmstudio", action="store_true", help="Shortcut to connect to LM Studio (http://localhost:1234)")
    p_proxy.add_argument("--vllm", action="store_true", help="Shortcut to connect to vLLM (http://localhost:8000)")
    p_proxy.add_argument("--port", type=int, default=5000, help="Proxy listen port")
    p_proxy.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_proxy.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")
    p_proxy.add_argument("--no-cite", "--no-citations", dest="no_cite", action="store_true", help="Disable document and line citations in forwarded memory prompts")
    p_proxy.add_argument("--host", type=str, default="127.0.0.1", help="Listen address. LOOPBACK by default: these endpoints do not authenticate, so binding 0.0.0.0 exposes your memory to the network")
    p_proxy.add_argument("--allow-vault-switch", action="store_true", help="Let POST /vault/init change the ACTIVE vault (off by default; the new vault is still confined to the active vault's directory)")

    # stats
    p_stats = subparsers.add_parser("stats", parents=[pw_parent], help="Show memory statistics")
    p_stats.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_stats.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")

    # inspect
    p_inspect = subparsers.add_parser("inspect", parents=[pw_parent], help="Inspect sources and metadata tags stored in vault")
    p_inspect.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_inspect.add_argument("--profile", "--user", dest="profile", type=str, default=None, help="User profile name (e.g. work, personal, alice)")

    # rekey
    p_rekey = subparsers.add_parser(
        "rekey", parents=[pw_parent],
        help="Add, change or remove a vault passphrase (rewrites the whole file)")
    p_rekey.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    p_rekey.add_argument("--profile", "--user", dest="profile", type=str, default=None,
                         help="User profile name (e.g. work, personal, alice)")
    p_rekey.add_argument("--remove", action="store_true",
                         help="Decrypt the vault back to plaintext")
    p_rekey.add_argument("--new-password-stdin", action="store_true",
                         help="Read the NEW passphrase from the second line of stdin "
                              "(the first is the old one, as for --password-stdin)")

    # user
    p_user = subparsers.add_parser("user", help="Manage isolated user profiles (list, create, delete)")
    user_subs = p_user.add_subparsers(dest="user_action", help="User profile actions")

    # user list
    user_subs.add_parser("list", help="List all created user profiles")

    # user create
    p_u_create = user_subs.add_parser("create", help="Create a new isolated user profile")
    p_u_create.add_argument("name", type=str, help="Username (e.g. OM-personal, OM-work)")

    # user delete
    p_u_delete = user_subs.add_parser("delete", help="Delete a user profile and its memory vault")
    p_u_delete.add_argument("name", type=str, help="Username to delete")
    p_u_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    def _resolve_vault(a, must_exist=False):
        if getattr(a, "profile", None):
            uname = a.profile.strip()
            if must_exist and not user_exists(uname):
                print(f"[nanomem] Error: User profile '{uname}' does not exist.", file=sys.stderr)
                print(f"  To create this user, run: nanomem user create {uname}")
                print(f"  To see all created users, run: nanomem user list\n")
                sys.exit(1)
            return get_vault_filename(uname)
        if getattr(a, "vault_opt", None) or getattr(a, "vault_pos", None):
            return a.vault_opt or a.vault_pos          # `init`, either spelling
        return getattr(a, "vault", None) or "memory.dat"

    def _password_for(a):
        """Passphrase for this invocation: flag > stdin > NANOMEM_PASSWORD > none."""
        if getattr(a, "password_stdin", False):
            return sys.stdin.readline().rstrip("\n") or None
        if getattr(a, "password", False):
            return getpass.getpass("Vault passphrase: ") or None
        return os.getenv("NANOMEM_PASSWORD") or None

    v_path = _resolve_vault(args, must_exist=bool(getattr(args, "profile", None)))
    pw = _password_for(args)

    # A READ against a vault that is not there is an error, not an empty result.
    # Listed by command so a WRITE still creates a vault on demand, which is how
    # `nanomem add` is meant to work.
    if args.command in ("search", "history", "changes", "stats", "inspect", "forget"):
        _require_vault(v_path)

    if args.command == "init":
        v_path = _resolve_vault(args, must_exist=False)
        if os.path.exists(v_path):
            print(f"[nanomem] Vault already exists at '{v_path}'.")
        else:
            with Vault(v_path, password=pw) as v:
                v.flush()
            print(f"[nanomem] ✓ Initialized new empty vault at '{v_path}' (0 documents).")
        return

    if args.command == "add":
        with Vault(v_path, password=pw) as v:
            t0 = time.perf_counter()
            # `--entity` did not exist until 0.7.12, so the CLI -- like the MCP
            # server -- could not apply the mitigation the README prescribes for
            # the tagger, and did not say so.
            meta = {"entity": args.entity} if getattr(args, "entity", None) else None
            doc_id = v.add(args.text, source=args.source, metadata=meta)
            dt_us = (time.perf_counter() - t0) * 1e6
            as_ent = f" as '{args.entity}'" if getattr(args, "entity", None) else ""
            # `add()` returns "" for empty or whitespace-only input. This printed
            # "Stored fact  in 1.7 µs ... -> '   '" for it -- an empty id, a
            # confirmed entity, and a success exit, for a fact that was never
            # stored. Say what happened instead.
            if not doc_id:
                _fail("nothing to store: the text is empty or only whitespace.")
            print(f"[nanomem] Stored fact {doc_id}{as_ent} in {dt_us:.1f} µs "
                  f"into '{v_path}' -> '{args.text}'")

    elif args.command == "search":
        with Vault(v_path, password=pw) as v:
            as_of = _parse_when(getattr(args, "as_of", None))
            results = v.search(args.query, top_k=args.top_k, as_of=as_of)
            if not results:
                when = f" as of {_fmt_when(as_of)}" if as_of is not None else ""
                print(f"[nanomem] No matching memories found in '{v_path}'{when}.")
                return
            when = f" (as of {_fmt_when(as_of)})" if as_of is not None else ""
            print(f"\n[nanomem] Found {len(results)} matching facts in '{v_path}'{when}:")
            for i, r in enumerate(results):
                src = r.get("source", "unknown")
                score = r.get("score", 0.0)
                print(f"  [{i+1}] (Score: {score:.3f} | Source: {src})")
                print(f"      {r['text']}\n")

    elif args.command == "history":
        with Vault(v_path, password=pw) as v:
            chain = v.history(args.query, max_len=args.max_len)
            if not chain:
                print(f"[nanomem] Nothing in '{v_path}' matches '{args.query}'.")
                return
            if len(chain) == 1:
                print(f"\n[nanomem] '{args.query}' has one value and has never changed:")
            else:
                print(f"\n[nanomem] '{args.query}' has held {len(chain)} values:")
            for h in chain:
                tag = "superseded" if h["superseded"] else "current"
                print(f"  {_fmt_when(h['timestamp'])}  [{tag:^10}] {h['text']}")
            print()

    elif args.command == "changes":
        # A reversed range printed a raw Python traceback. It is a typo, not a bug
        # in the library, and the message should say which way round the arguments
        # go rather than showing the user a stack.
        def _guard_range(since_ts, until_ts):
            if since_ts is not None and until_ts is not None and until_ts < since_ts:
                _fail("--until (%s) is earlier than --since (%s); the range is "
                      "(since, until]." % (args.until, args.since))
        with Vault(v_path, password=pw) as v:
            since = _parse_when(args.since)
            until = _parse_when(args.until)
            _guard_range(since, until)
            rows = v.changes(since, until=until, limit=args.limit)
            if not rows:
                print(f"[nanomem] '{v_path}' learned nothing in that window.")
                return
            span = _fmt_when(since) + " to " + (_fmt_when(until) if until else "now")
            print(f"\n[nanomem] {len(rows)} records written between {span}:")
            for r in rows:
                what = r.get("entity") or "-"
                mark = f" (revision {r['revision']})" if r["revision"] > 1 else ""
                print(f"  {_fmt_when(r['timestamp'])}  {what}{mark}")
                print(f"      {r['text']}")
            print()

    elif args.command == "ingest":
        with Vault(v_path, password=pw) as v:
            if int(getattr(args, "expect_docs", 0) or 0) > 0:
                v.engine.reserve_additional_rows(int(args.expect_docs))
            if os.path.isdir(args.file):
                stats = v.ingest_directory(args.file)
                print(f"[nanomem] Ingested {stats['chunks_indexed']} chunks across {stats['files_indexed']} files from directory '{args.file}' into '{v_path}'")
            elif os.path.isfile(args.file):
                n_chunks = v.ingest_file(args.file)
                print(f"[nanomem] Ingested {n_chunks} chunks from '{args.file}' into '{v_path}'")
            else:
                _fail(f"Path '{args.file}' not found.")

    elif args.command == "merge":
        with Vault(args.into, password=pw) as v:
            res = v.merge(
                args.source,
                deduplicate=args.dedup,
                incoming_project=args.project,
                incoming_user_id=args.user_id
            )
            print(f"[nanomem] Merged '{args.source}' into '{args.into}': {res['added']} added, {res['duplicates_skipped']} duplicates skipped.")

    elif args.command == "split":
        with Vault(args.source, password=pw) as v:
            purge = getattr(args, "purge", False)
            act = "Moved and purged" if purge else "Copied"
            if args.source_doc:
                target = args.target or f"split_{args.source_doc}.dat"
                n = v.export(target, source_doc=args.source_doc, purge=purge)
                print(f"[nanomem] {act} {n} records with source='{args.source_doc}' into '{target}'.")
            elif args.key:
                counts = v.split_by_key(args.key, output_dir=args.dir)
                print(f"[nanomem] Partitioned into {len(counts)} vaults by key '{args.key}' in '{args.dir}': {counts}")
            elif args.before is not None:
                target = args.target or "split_archive.dat"
                n = v.split_by_date(target, before_timestamp=args.before)
                print(f"[nanomem] Split {n} records older than {args.before} into '{target}'.")
            elif args.target:
                n = v.export(args.target, purge=purge)
                print(f"[nanomem] {act} {n} records into '{args.target}'.")
            else:
                print("[nanomem] Error: Specify --source-doc, --key, --before, or --target.")

    elif args.command == "forget":
        with Vault(v_path, password=pw) as v:
            hits = v.search(args.query, top_k=1, min_score=0.42)
            if not hits:
                print(f"[nanomem] No memory found matching '{args.query}'.")
            else:
                cand = hits[0]
                print(f"[nanomem] Found memory: \"{cand['text']}\"")
                if not getattr(args, "yes", False):
                    try:
                        ans = input("Permanently delete this memory? [y/N]: ").strip().lower()
                        if ans not in ["y", "yes"]:
                            print("[nanomem] Deletion cancelled.")
                            return
                    except (KeyboardInterrupt, EOFError):
                        print("\n[nanomem] Deletion cancelled.")
                        return
                v.delete(text_exact=cand["text"])
                print("[nanomem] Memory permanently erased.")

    elif args.command == "proxy":
        upstream = args.upstream
        if getattr(args, "lmstudio", False):
            upstream = "http://localhost:1234"
        elif getattr(args, "vllm", False):
            upstream = "http://localhost:8000"
        v_path = args.vault
        if getattr(args, "profile", None):
            v_path = f"memory_{args.profile}.dat" if not args.profile.endswith(".dat") else args.profile
        run_proxy(vault_path=v_path, port=args.port, upstream_url=upstream,
                  cite=not getattr(args, "no_cite", False), password=pw,
                  host=getattr(args, "host", "127.0.0.1"),
                  allow_vault_switch=getattr(args, "allow_vault_switch", False))

    elif args.command == "stats":
        with Vault(v_path, password=pw) as v:
            import json
            print(json.dumps(v.stats(), indent=2))

    elif args.command == "inspect":
        with Vault(v_path, password=pw) as v:
            data = v.inspect()
            print(f"\n🔍 NanoMem Vault Inspection: '{data['file_path']}'")
            print(f"Total Stored Documents: {data['total_documents']}\n")
            print("📁 Document Sources:")
            if not data["sources"]:
                print("  (None)")
            else:
                for s, count in data["sources"].items():
                    print(f"  • {s:<32} ({count} memories/chunks)")
            print("\n🏷️  Metadata Tags & Categories:")
            if not data["metadata_summary"]:
                print("  (None)")
            else:
                for k, vals in data["metadata_summary"].items():
                    print(f"  • {k}:")
                    for val_name, count in vals.items():
                        print(f"      - '{val_name}': {count} records")
            print()

    elif args.command == "rekey":
        if args.remove:
            new_pw = None
        elif args.new_password_stdin:
            # Scripted / CI re-keying. 3.0.1 read the OLD passphrase from stdin
            # but always called getpass() for the NEW one, so piping one line
            # raised EOFError out of getpass and piping three only "worked"
            # because getpass degraded to stdin with a warning.
            new_pw = sys.stdin.readline().rstrip("\n")
            if not new_pw:
                print("[nanomem] Empty passphrase on stdin; use --remove to decrypt.")
                sys.exit(1)
        else:
            new_pw = getpass.getpass("New vault passphrase: ")
            if new_pw != getpass.getpass("Repeat new passphrase: "):
                print("[nanomem] Passphrases do not match; nothing was changed.")
                sys.exit(1)
            if not new_pw:
                print("[nanomem] Empty passphrase; use --remove to decrypt instead.")
                sys.exit(1)
        with Vault(v_path, password=pw) as v:
            t0 = time.perf_counter()
            v.engine.rekey(new_pw)
            dt = time.perf_counter() - t0
            st = v.stats()
        print(f"[nanomem] Rewrote '{v_path}' in {dt:.2f}s -- cipher now: {st['cipher']}")
        if new_pw:
            print("[nanomem] There is no recovery key. If you lose this passphrase the "
                  "vault is gone.")

    elif args.command == "user":
        if args.user_action in ["list", None]:
            users = list_users()
            if not users:
                print("\n[nanomem] No user profiles found. Create your first user profile with:")
                print("  nanomem user create <username>\n")
                return
            print("\n👥 Registered NanoMem User Profiles:")
            print(f"  {'Username':<18} {'Vault File':<24} {'Docs':<8} {'Size (KB)':<10}")
            print("  " + "-" * 62)
            for u in users:
                print(f"  {u['username']:<18} {u['filename']:<24} {u['documents']:<8} {u['size_kb']:<10.1f}")
            print(f"\nTotal: {len(users)} user(s). Select with '--user <name>' in any command.\n")

        elif args.user_action == "create":
            try:
                target = create_user(args.name)
                print(f"[nanomem] ✅ Created user profile '{args.name}' -> '{target}'")
            except Exception as e:
                _fail(str(e))

        elif args.user_action == "delete":
            if not user_exists(args.name):
                _fail(f"User '{args.name}' does not exist.")
                return
            if not getattr(args, "yes", False):
                try:
                    confirm = input(f"Are you sure you want to permanently delete user '{args.name}' and all their memories? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nDeletion cancelled.")
                    return
                if confirm != "y":
                    print("[nanomem] Deletion cancelled.")
                    return
            delete_user(args.name)
            print(f"[nanomem] 🗑️ Deleted user profile '{args.name}'.")


if __name__ == "__main__":
    main()
