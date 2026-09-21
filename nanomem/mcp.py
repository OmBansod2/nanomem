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
            "Every value a fact has held, oldest first, with the current one marked. "
            "The last entry is the current "
            "value. Use this when the user asks what something used to be, when it "
            "changed, or whether it changed at all. A fact that never changed "
            "returns a single entry, which is an answer, not an empty result. For a "
            "chain whose attribute was not declared with `entity`, nanomem_search "
            "can still rank a superseded value first; nanomem_changes is unfiltered "
            "and reports every write in a window."),
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
    # A group admitted by min_revisions<=1 has ONE timestamp, so np.diff is empty
    # and the engine honestly reports median_interval=None. This function had no
    # None branch, so the tool raised TypeError and the server answered JSON-RPC
    # -32603 for a value its own inputSchema declares as a plain integer.
    # Rendered, not clamped: clamping min_revisions to 2 would silently ignore
    # what the caller asked for.
    if seconds is None:
        return "n/a"
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
        # ASK FOR THE WHOLE CHAIN, THEN TRUNCATE HERE. Passing `max_len` down left
        # this code guessing at whether truncation had happened, and the guess
        # `len(chain) >= max_len` was wrong in exactly the cases where max_len
        # EQUALS the true chain length -- the most natural call an agent makes.
        # Measured over a 24-case grid: 4 wrong, every one of them announcing
        # "there are earlier ones" about a chain being shown in full. Comparing
        # against the full chain is 0 wrong over the same grid.
        full = vault.history(args.get("query", ""))
        if not full:
            return "Nothing in memory matches that."
        _max_len = args.get("max_len")
        chain = full[-int(_max_len):] if _max_len else full
        # "never changed" IS A CLAIM, AND max_len TRUNCATES. This keyed on
        # `len(chain) == 1` AFTER the caller's `max_len` had already cut the
        # chain, so `nanomem_history(max_len=1)` told the agent a fact "has never
        # changed" about a fact that had changed three times. The consumer here
        # is a model that cannot check, and the whole point of this tool is to
        # say what a value HAS BEEN.
        #
        # Truncation is now a FACT, not an inference: the rows shown against the
        # rows the vault returned.
        truncated = len(chain) < len(full)
        # "NEVER CHANGED" IS ONLY KNOWABLE FOR A DECLARED CHAIN. A one-entry
        # history can also mean the TAGGER did not group a restatement -- which
        # the README measures at 0/100 on narrative phrasing when no entity was
        # declared. The limitation is documented; asserting the opposite to a
        # model that cannot check is not. Measured: two employers in the vault,
        # `nanomem_changes` showing both, and this line saying the fact had never
        # changed.
        declared = bool((chain[0].get("metadata") or {}).get("entity_declared"))
        # A DECLARED ANCHOR IS NOT A COMPLETE CHAIN. `entity_declared` says this
        # RECORD named its entity. It cannot say that every OTHER write about the
        # same fact was grouped with it, and that is the difference that bites: a
        # vault where one write declared `entity` and the later restatements did
        # not leaves those restatements ungrouped (measured here: `changes()`
        # shows them with entity=None), so a one-entry chain with declared=True
        # was printed above two newer values the vault still held, the oldest
        # tagged [current].
        #
        # What IS knowable is narrower, and is what gets claimed now: every write
        # that DECLARED this entity is in this chain. Writes that declared nothing
        # are grouped by the tagger, which the README measures at 0 of 100 on
        # narrative phrasing -- so when any such write exists, the unqualified
        # sentence is not available. Counting them is one scan, on the only branch
        # that asserts something about what does not exist.
        undeclared = 0
        if len(chain) == 1 and not truncated and declared:
            for r in vault.get_all_records():
                if not (r.get("metadata") or {}).get("entity_declared"):
                    undeclared += 1
        if len(chain) == 1 and not truncated and declared and undeclared:
            head = (f"This has one value, and no other write that DECLARED this "
                    f"entity has changed it. {undeclared} write(s) in the vault "
                    f"declared no entity and are grouped separately, so this may "
                    f"not be every value -- use nanomem_changes to see every write:")
        elif len(chain) == 1 and not truncated and declared:
            head = "This has one value and has never changed:"
        elif len(chain) == 1 and not truncated:
            head = ("One value is grouped under this. The entity was not declared, "
                    "so a restatement worded differently may not have been grouped "
                    "with it -- use nanomem_changes to see every write:")
        elif truncated:
            head = (f"Showing the {len(chain)} most recent value(s); there are "
                    f"earlier ones (this view was limited by max_len):")
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


#: Tools that only read. Everything else is flushed before its reply is sent.
_READ_ONLY_TOOLS = frozenset({
    "nanomem_search", "nanomem_history", "nanomem_as_of",
    "nanomem_changes", "nanomem_volatility", "nanomem_stats",
})


def run_mcp_server(vault_path: str = "memory.dat"):
    vault = Vault(vault_path)

    # DURABILITY. Through 0.7.17 this server buffered every write and flushed
    # only in the `finally` below, so `nanomem_add` answered "Stored ..." while
    # nothing had reached disk: another process saw an empty vault, and a SIGTERM
    # -- which is how an MCP client stops its servers -- bypassed `finally`
    # entirely and lost the lot. Measured: add, then SIGTERM, then 0 rows.
    # An agent memory that confirms a write it did not keep is worse than one
    # that refuses the write.
    import signal as _signal

    def _shutdown(_signum, _frame):
        try:
            vault.close()
        finally:
            raise SystemExit(0)

    for _sig in (_signal.SIGTERM, _signal.SIGINT, _signal.SIGHUP):
        try:
            _signal.signal(_sig, _shutdown)
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or the platform has no such signal.
            pass

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
                    tool_name = params.get("name")
                    text = dispatch(vault, tool_name,
                                    params.get("arguments", {}) or {})
                    # Persist BEFORE answering, so "Stored ..." is true when the
                    # caller reads it rather than whenever the process happens to
                    # end. Listed by what does NOT need it, so a tool added later
                    # is durable by default rather than by someone remembering.
                    if tool_name not in _READ_ONLY_TOOLS:
                        vault.flush()
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
