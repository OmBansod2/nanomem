"""
nanomem.mcp
~~~~~~~~~~~
Model Context Protocol (MCP) server for nanomem, over stdio.

Lets any MCP client -- Claude, Cursor, Zed -- use a local nanomem vault as
persistent memory. Run it as ``python -m nanomem.mcp --vault memory.dat``.

WHAT THIS EXPOSES, AND WHY IT MATTERS. Through 0.5.x this server offered three
tools -- add, search, stats -- which is the surface of an ordinary vector store,
and it therefore presented nanomem to every agent as one. The capabilities that
distinguish an append-only revision log from a vector index were unreachable
from the only integration path most callers will ever use. The temporal tools
below fix that: an agent can now ask what a fact USED to be, what the memory
believed at a past moment, what changed in a window, and which of its own
beliefs have gone stale.
"""

import json
import sys
from typing import Any, Dict

from . import __version__
from .vault import Vault

TOOLS = [
    {
        "name": "nanomem_add",
        "description": "Store a fact, note, or document into persistent long-term memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The information to remember."},
                "source": {"type": "string", "description": "Source identifier (e.g. user_chat, doc.pdf)."},
                "entity": {
                    "type": "string",
                    "description": (
                        "The attribute this states, e.g. 'employer', 'home_address', "
                        "'phone'. Pass the SAME value every time you write a new value "
                        "of the same attribute -- that is what lets nanomem know which "
                        "statement supersedes which. Omit it only if you genuinely do "
                        "not know; a lexical tagger then guesses, and it guesses badly "
                        "on narrative phrasing ('I moved jobs, I now work at ...')."),
                },
                "timestamp": {
                    "type": "number",
                    "description": "Unix seconds this was true. Defaults to now.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "nanomem_search",
        "description": "Search persistent long-term memory for relevant past facts and context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "Number of results to return.", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "nanomem_history",
        "description": (
            "Every value a fact has held, oldest first, with the current one marked. Cross-check with nanomem_changes before relying on it: this view applies a relevance filter and can return a SHORTER chain than exists, including a single entry marked current when later revisions were written. nanomem_changes is unfiltered."
            "asserted. The last entry is the current value. Use this when the user asks "
            "what something used to be, when it changed, or whether it changed at all. "
            "A fact that never changed returns a single entry, which is an answer."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The fact to trace, e.g. 'where do I work'."},
                "max_len": {"type": "integer", "description": "Keep only the N most recent entries."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "nanomem_as_of",
        "description": (
            "Answer a question as the memory stood at a past moment. Records written "
            "after that moment are not considered, so a value that has since been "
            "superseded is returned wherever it was still current. Use this for "
            "'what did I think in March', or to reconstruct past state."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to ask."},
                "as_of": {"type": "string", "description": "YYYY-MM-DD, an ISO timestamp, or a unix time."},
                "top_k": {"type": "integer", "description": "Number of results.", "default": 3},
            },
            "required": ["query", "as_of"],
        },
    },
    {
        "name": "nanomem_changes",
        "description": (
            "What the memory learned in a time window, oldest first, with no query "
            "needed. Use this for 'what did I tell you since last week' or to catch up "
            "on what changed while away."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "YYYY-MM-DD, ISO timestamp, or unix time (exclusive)."},
                "until": {"type": "string", "description": "Upper bound, inclusive. Defaults to now."},
                "limit": {"type": "integer", "description": "Cap the number of rows."},
            },
            "required": ["since"],
        },
    },
    {
        "name": "nanomem_volatility",
        "description": (
            "How often each remembered fact actually changes, measured from the memory's "
            "own revision log: how many times it has been restated, the typical interval "
            "between changes, and how long the current value has stood unconfirmed. Use "
            "this to decide which facts are probably stale and worth re-confirming with "
            "the user. These are measured statistics, not a prediction."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_revisions": {"type": "integer", "description": "Only facts restated at least this often.", "default": 2},
            },
        },
    },
    {
        "name": "nanomem_stats",
        "description": "Return current memory vault size and document count.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _require_when(args, key, tool_name):
    """A time argument the tool declares as required, or a usable complaint.

    The schema marks these required, but nothing between a model and this
    function enforces that, and an LLM client omits an argument routinely.
    Through 0.7.1 the miss reached ``float(None)`` and the client was told
    "float() argument must be a string or a real number, not 'NoneType'" --
    which names neither the tool, the argument, nor the fix, so the model had
    nothing to retry with. The bad-VALUE path already answered well; only the
    missing-value path did not.
    """
    value = args.get(key)
    if value is None or not str(value).strip():
        raise ValueError(
            f"{tool_name} requires {key!r} and none was given; "
            f"use YYYY-MM-DD, an ISO timestamp, or a unix time")
    return _parse_when(value)


def _parse_when(value):
    """A tool argument time: YYYY-MM-DD, an ISO timestamp, or a unix time."""
    from datetime import datetime
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
        raise ValueError(
            f"could not read {s!r} as a time; use YYYY-MM-DD, an ISO timestamp, "
            f"or a unix time")


def _fmt_when(ts):
    """Date AND time.

    This was `%Y-%m-%d` alone, which is fine until a caller corrects the same
    fact more than once in a day -- and a fact corrected twice in a day is
    exactly the kind this store exists for. Four revisions hours apart rendered
    as four identical dates, so the agent reading them could not tell how
    recent any of them was, or that any time had passed at all. The CLI has
    always printed the time; only this surface dropped it.
    """
    from datetime import datetime
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")


def _fmt_span(seconds):
    """A duration in whatever unit a reader can act on.

    `_days()` divided by 86400 and the caller printed it with no decimals, so a
    fact restated every three hours read "~every 0d, last confirmed 0d ago" --
    identical to a fact with no measurable interval at all. That rate IS the
    answer `volatility` exists to give: something changing every few hours is
    the most volatile thing in the vault, and it was the one case rendered as
    nothing.
    """
    s = max(0.0, float(seconds))
    if s < 90:
        return f"{s:.0f}s"
    if s < 90 * 60:
        return f"{s / 60:.0f}min"
    if s < 48 * 3600:
        return f"{s / 3600:.0f}h"
    if s < 90 * 86400:
        return f"{s / 86400:.0f}d"
    if s < 730 * 86400:
        return f"{s / (30.44 * 86400):.0f}mo"
    return f"{s / (365.25 * 86400):.1f}y"


def dispatch(vault: Vault, tool_name: str, args: Dict[str, Any]) -> str:
    """Run one tool and return the text an MCP client should see."""
    if tool_name == "nanomem_add":
        # AN AGENT COULD NOT DECLARE AN ATTRIBUTE HERE UNTIL 0.7.12.
        # The README's stated mitigation for the tagger -- "if your application
        # knows its own attributes, declare them" -- was unreachable from MCP and
        # the CLI, the two surfaces this module's own docstring calls "the only
        # integration path most callers will ever use". Both were locked onto the
        # tagger, which the README measures at 0 of 100 on narrative phrasing,
        # and neither said so. `metadata` was accepted and silently dropped, so
        # an agent that reasonably tried got `Stored: ...` and no entity.
        text = args.get("text", "")
        meta = dict(args.get("metadata") or {})
        entity = args.get("entity") or meta.get("entity")
        if entity:
            meta["entity"] = str(entity)
        ts = args.get("timestamp")
        vault.add(text, source=args.get("source", "mcp_client"),
                  metadata=meta or None,
                  timestamp=float(ts) if ts is not None else None)
        return (f"Stored: {text!r}" if not entity
                else f"Stored as {entity!r}: {text!r}")

    if tool_name == "nanomem_search":
        hits = vault.search(args.get("query", ""), top_k=int(args.get("top_k", 3)))
        if not hits:
            return "No relevant memories found."
        return "\n\n".join(
            f"[{i + 1}] ({h.get('source')}, {_fmt_when(h['timestamp'])}): {h['text']}"
            for i, h in enumerate(hits))

    if tool_name == "nanomem_history":
        chain = vault.history(args.get("query", ""), max_len=args.get("max_len"))
        if not chain:
            return "Nothing in memory matches that."
        if len(chain) == 1:
            head = "This has one value and has never changed:"
        else:
            head = f"This has held {len(chain)} values:"
        lines = [
            f"  {_fmt_when(h['timestamp'])}  "
            f"[{'superseded' if h['superseded'] else 'current'}]  {h['text']}"
            for h in chain]
        return head + "\n" + "\n".join(lines)

    if tool_name == "nanomem_as_of":
        when = _require_when(args, "as_of", "nanomem_as_of")
        hits = vault.search(args.get("query", ""), top_k=int(args.get("top_k", 3)),
                            as_of=when)
        if not hits:
            return f"Memory held nothing matching that as of {_fmt_when(when)}."
        head = f"As of {_fmt_when(when)}:"
        return head + "\n" + "\n".join(
            f"  [{i + 1}] ({_fmt_when(h['timestamp'])}): {h['text']}"
            for i, h in enumerate(hits))

    if tool_name == "nanomem_changes":
        rows = vault.changes(_require_when(args, "since", "nanomem_changes"),
                             until=_parse_when(args.get("until")),
                             limit=args.get("limit"))
        if not rows:
            return "Nothing was learned in that window."
        return "\n".join(
            f"  {_fmt_when(r['timestamp'])}  "
            f"{r.get('entity') or '-'}"
            f"{' (revision %d)' % r['revision'] if r['revision'] > 1 else ''}: {r['text']}"
            for r in rows)

    if tool_name == "nanomem_volatility":
        rows = vault.engine.volatility(
            min_revisions=int(args.get("min_revisions", 2)))
        if not rows:
            return ("No fact has been restated often enough to have a rate yet. "
                    "A fact needs at least two assertions before an interval exists.")
        out = ["fact / restated / typical interval / unconfirmed for"]
        for f in rows:
            out.append(
                f"  {f['entity']}: {f['n_revisions']}x, "
                f"~every {_fmt_span(f['median_interval'])}, "
                f"last confirmed {_fmt_span(f['age'])} ago")
        return "\n".join(out)

    if tool_name == "nanomem_stats":
        return json.dumps(vault.stats(), indent=2, default=str)

    # A plain string here became a NORMAL result: the client saw a successful
    # call whose content happened to read "Unknown tool", with no isError and no
    # JSON-RPC error to notice. Raising routes it through the error path, and
    # naming the real tools gives a model something to retry with.
    raise ValueError(f"unknown tool {tool_name!r}; this server provides "
                     + ", ".join(t["name"] for t in TOOLS))


def run_mcp_server(vault_path: str = "memory.dat"):
    vault = Vault(vault_path)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req_id = None
            try:
                req = json.loads(line)
                req_id = req.get("id")
                method = req.get("method")

                # A JSON-RPC NOTIFICATION carries no id and MUST NOT be answered.
                # Through 0.5.x every unrecognised method got a reply, including
                # `notifications/initialized`, which answers a notification with
                # id null -- a protocol violation some clients reject the session
                # over.
                if req_id is None:
                    continue

                if method == "initialize":
                    result = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "nanomem-mcp", "version": __version__},
                    }
                elif method == "tools/list":
                    result = {"tools": TOOLS}
                elif method == "tools/call":
                    params = req.get("params", {})
                    text = dispatch(vault, params.get("name"),
                                    params.get("arguments", {}) or {})
                    result = {"content": [{"type": "text", "text": text}]}
                else:
                    result = {}

                sys.stdout.write(json.dumps(
                    {"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
                sys.stdout.flush()

            except Exception as e:                       # noqa: BLE001
                # Report the failure against the request that caused it. Through
                # 0.5.x the id was hard-coded to null, so a client could not tell
                # which call had failed.
                sys.stdout.write(json.dumps(
                    {"jsonrpc": "2.0", "id": req_id,
                     "error": {"code": -32603, "message": str(e)}}) + "\n")
                sys.stdout.flush()
    finally:
        vault.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="nanomem Model Context Protocol (MCP) server")
    p.add_argument("--vault", type=str, default="memory.dat", help="Path to memory vault")
    run_mcp_server(vault_path=p.parse_args().vault)
