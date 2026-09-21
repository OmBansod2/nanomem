"""The MCP server: tool surface, dispatch, and JSON-RPC framing.

Through 0.5.x this server existed in ONE distribution bundle, was absent from
the source of truth, and offered only add/search/stats -- so every agent that
reached nanomem through MCP saw an ordinary vector store. These tests pin the
temporal tools to that surface, and pin two protocol bugs that were fixed when
it was promoted.
"""

import io
import json

import numpy as np
import pytest

from conftest import D, unit_rows                                  # noqa: E402
from nanomem import __version__
from nanomem import mcp as nm_mcp
from nanomem.vault import Vault

T0 = 1_700_000_000.0
DAY = 86400.0


@pytest.fixture
def chat_vault(vault_path, offline_embedder):
    v = Vault(vault_path)
    rows = [("My locker code is value-0.", "locker_code", 0),
            ("My locker code is value-1.", "locker_code", 30),
            ("My locker code is value-2.", "locker_code", 60)]
    for text, ent, day in rows:
        v.engine.add_fact(text, v.embedder.embed(text), source="chat",
                          metadata={"entity": ent}, timestamp=T0 + day * DAY)
    v.engine.flush()
    yield v
    v.close()


def test_the_default_mcp_source_is_a_personal_source():
    """The write source this server uses must let the entity tagger run.

    `nanomem_add` writes with ``source="mcp_client"``. Until 0.6.6 that string
    was not in ``PERSONAL_SOURCES``, so the tagger never ran on the MCP path and
    no revision group ever formed -- which silently disabled the tools this
    server was expanded to offer. `nanomem_history` answered "This has one value
    and has never changed" for a fact that had just been revised, and
    `nanomem_volatility` could only ever return nothing, because `volatility()`
    excludes records with no entity. Both LOOKED like working answers.
    """
    from nanomem import entities as ent
    assert "mcp_client" in ent.PERSONAL_SOURCES


def test_an_mcp_write_forms_a_revision_group(vault_path, offline_embedder):
    """End to end over the tool surface, not the library underneath it."""
    v = Vault(vault_path)
    for i, text in enumerate(("My locker code is value-0.",
                              "I changed it. My locker code is value-1.")):
        nm_mcp.dispatch(v, "nanomem_add",
                         {"text": text, "source": "mcp_client"})
    chain = v.history("locker code")
    assert len(chain) == 2, "an MCP-written revision must form a chain"
    assert [h["superseded"] for h in chain] == [True, False]
    assert v.volatility(), "volatility() must see a group the MCP path wrote"
    v.close()


def test_every_tool_declares_a_usable_schema():
    names = [t["name"] for t in nm_mcp.TOOLS]
    assert len(names) == len(set(names)), "duplicate tool name"
    for t in nm_mcp.TOOLS:
        assert t["description"].strip()
        s = t["inputSchema"]
        assert s["type"] == "object"
        for req in s.get("required", []):
            assert req in s["properties"], f"{t['name']} requires undeclared {req}"


def test_the_temporal_tools_are_exposed():
    """The capabilities that distinguish nanomem from a vector index."""
    names = {t["name"] for t in nm_mcp.TOOLS}
    assert {"nanomem_history", "nanomem_as_of",
            "nanomem_changes", "nanomem_volatility"} <= names


def test_history_tool_reports_the_chain(chat_vault):
    out = nm_mcp.dispatch(chat_vault, "nanomem_history", {"query": "locker code"})
    assert "has held 3 values" in out
    assert out.count("[superseded]") == 2
    assert out.count("[current]") == 1
    assert out.rindex("[current]") > out.rindex("[superseded]")


def test_as_of_tool_excludes_later_records(chat_vault):
    out = nm_mcp.dispatch(chat_vault, "nanomem_as_of",
                          {"query": "locker code", "as_of": "2023-11-14"})
    assert "value-2" not in out


def test_changes_tool_reads_a_window(chat_vault):
    out = nm_mcp.dispatch(chat_vault, "nanomem_changes",
                          {"since": str(T0 + 15 * DAY)})
    assert "value-1" in out and "value-2" in out and "value-0" not in out


def test_volatility_tool_is_honest_when_there_is_no_rate(vault_path, offline_embedder):
    v = Vault(vault_path)
    v.engine.add_fact("My locker code is value-0.", v.embedder.embed("x"),
                      source="chat", metadata={"entity": "locker_code"},
                      timestamp=T0)
    v.engine.flush()
    out = nm_mcp.dispatch(v, "nanomem_volatility", {})
    assert "at least two assertions" in out
    v.close()


def test_parse_when_accepts_the_three_documented_forms():
    assert nm_mcp._parse_when("1700000000") == 1700000000.0
    assert nm_mcp._parse_when("2026-03-01") > 1.7e9
    assert nm_mcp._parse_when(None) is None
    with pytest.raises(ValueError):
        nm_mcp._parse_when("not a date")


def _serve(monkeypatch, vault_path, lines):
    """Drive run_mcp_server over a fake stdio pair and return parsed replies."""
    out = io.StringIO()
    monkeypatch.setattr(nm_mcp.sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    monkeypatch.setattr(nm_mcp.sys, "stdout", out)
    nm_mcp.run_mcp_server(vault_path)
    return [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


def test_a_notification_is_never_answered(monkeypatch, vault_path, offline_embedder):
    """A JSON-RPC notification carries no id and MUST NOT get a reply.

    0.5.x answered `notifications/initialized` with id null, which is a protocol
    violation some clients drop the session over.
    """
    replies = _serve(monkeypatch, vault_path, [
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    ])
    assert len(replies) == 1
    assert replies[0]["id"] == 1


def test_an_error_is_reported_against_its_own_request(monkeypatch, vault_path,
                                                      offline_embedder):
    """0.5.x hard-coded the error id to null, so a client could not tell which
    call had failed."""
    replies = _serve(monkeypatch, vault_path, [
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": "nanomem_as_of",
                               "arguments": {"query": "x", "as_of": "not a date"}}}),
    ])
    assert len(replies) == 1
    assert replies[0]["id"] == 7 and "error" in replies[0]


def test_initialize_reports_the_real_version(monkeypatch, vault_path, offline_embedder):
    replies = _serve(monkeypatch, vault_path, [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})])
    assert replies[0]["result"]["serverInfo"]["version"] == __version__


# --- same-day revisions: the case this store exists for ---------------------

def _same_day_vault(vault_path, offline_embedder):
    """One fact corrected four times inside a single day, hours apart."""
    v = Vault(vault_path)
    t = T0
    for txt, hrs in (("My locker code is 1111.", 9),
                     ("Changed it. My locker code is 2222.", 6),
                     ("Changed again. My locker code is 3333.", 2),
                     ("Final. My locker code is 4444.", 0)):
        v.add(txt, metadata={"entity": "locker_code"}, timestamp=t - hrs * 3600)
    return v, t


def test_history_distinguishes_revisions_made_the_same_day(vault_path, offline_embedder):
    """`_fmt_when` printed `%Y-%m-%d` alone, so four revisions hours apart came
    back as four identical dates -- and a fact corrected twice in one day is
    precisely the kind this store is for. The agent reading that could not tell
    how recent any of them was."""
    v, _t = _same_day_vault(vault_path, offline_embedder)
    out = nm_mcp.dispatch(v, "nanomem_history", {"query": "locker code"})
    stamps = [ln.strip().split("  ")[0] for ln in out.splitlines() if ln.startswith("  2")]
    assert len(stamps) == 4
    assert len(set(stamps)) == 4, f"same-day revisions are indistinguishable: {stamps}"
    assert all(":" in s_ for s_ in stamps), "the time is what makes them distinct"
    v.close()


def test_volatility_does_not_report_a_three_hour_rate_as_zero(vault_path, offline_embedder):
    """It rendered every duration in whole days: a fact restated every three
    hours read "~every 0d, last confirmed 0d ago" -- the most volatile fact
    there is, shown as the same thing as no measurable interval."""
    v, _t = _same_day_vault(vault_path, offline_embedder)
    out = nm_mcp.dispatch(v, "nanomem_volatility", {})
    assert "0d" not in out, f"a sub-day rate rendered as zero days: {out}"
    assert "~every 3h" in out, out
    v.close()


def test_the_duration_ladder_picks_a_unit_a_reader_can_act_on():
    f = nm_mcp._fmt_span
    assert f(45) == "45s"
    assert f(400) == "7min"
    assert f(3 * 3600) == "3h"
    assert f(5 * 86400) == "5d"
    assert f(145 * 86400) == "5mo"
    assert f(900 * 86400) == "2.5y"
    assert f(-10) == "0s", "a negative age is 0, not a negative duration"


def _call(monkeypatch, vault_path, name, arguments):
    """One tools/call over the wire, returning the reply a client would parse."""
    replies = _serve(monkeypatch, vault_path, [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "t", "version": "1"}}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments}}),
    ])
    return replies[-1]


def test_an_unknown_tool_reaches_the_client_as_an_error(monkeypatch, vault_path,
                                                        offline_embedder):
    """It used to come back as a SUCCESSFUL call whose text read "Unknown tool".

    No `error`, no `isError` -- so a client had to string-match the content to
    notice its call had not happened. The reply must fail, and must name the
    tools that do exist so a model can retry.
    """
    reply = _call(monkeypatch, vault_path, "nanomem_nope", {})
    assert "error" in reply, reply
    assert "result" not in reply
    msg = reply["error"]["message"]
    assert "nanomem_nope" in msg and "nanomem_search" in msg


@pytest.mark.parametrize("tool,args,missing", [
    ("nanomem_changes", {}, "since"),
    ("nanomem_as_of", {"query": "locker code"}, "as_of"),
])
def test_a_missing_required_time_says_which_argument(monkeypatch, vault_path,
                                                     offline_embedder,
                                                     tool, args, missing):
    """An LLM client omits an argument routinely; nothing enforces `required`.

    Through 0.7.1 the omission reached float(None) and the client was told
    "float() argument must be a string or a real number, not 'NoneType'", which
    names neither the tool nor the argument. The bad-VALUE path already answered
    well -- only this one did not.
    """
    reply = _call(monkeypatch, vault_path, tool, args)
    assert "error" in reply, reply
    msg = reply["error"]["message"]
    assert missing in msg and tool in msg
    assert "NoneType" not in msg
    assert "YYYY-MM-DD" in msg, "it must say what a usable value looks like"


def test_a_bad_time_value_still_explains_itself(monkeypatch, vault_path,
                                                offline_embedder):
    """The path that was already right stays right."""
    reply = _call(monkeypatch, vault_path, "nanomem_changes", {"since": "last week"})
    assert "error" in reply
    assert "last week" in reply["error"]["message"]



# --- an agent must be able to declare the attribute it is writing -----------

_CHAIN = ["I work at Acme Corp.",
          "I moved jobs, I now work at Initech.",
          "I switched again, I work at Globex now."]


def test_nanomem_add_advertises_entity(vault_path, offline_embedder):
    """The README's mitigation for the tagger is `metadata={"entity": ...}`, and
    an MCP client can only pass what the schema advertises. Through 0.7.11 the
    schema offered `text` and `source` alone, so the mitigation was unreachable
    from the surface this module calls the main integration path."""
    from nanomem.mcp import TOOLS
    props = [t for t in TOOLS if t["name"] == "nanomem_add"][0]["inputSchema"]["properties"]
    assert "entity" in props
    assert "narrative" in props["entity"]["description"], \
        "it must warn that the tagger is what you get without it"


def test_a_declared_chain_over_mcp_makes_search_and_history_agree(
        vault_path, offline_embedder):
    """The defect an outside reviewer found: two tools, one vault, one session,
    opposite answers about which value is current, with nothing to say which to
    believe. Declaring the attribute is what resolves it -- and until 0.7.12
    there was no way to declare it here."""
    from nanomem.mcp import dispatch
    v = Vault(vault_path)
    for t in _CHAIN:
        out = dispatch(v, "nanomem_add", {"text": t, "entity": "employer"})
        assert "employer" in out, "the reply must confirm what it recorded"
    hist = dispatch(v, "nanomem_history", {"query": "where do I work"})
    assert "This has held 3 values" in hist, hist
    current = [ln for ln in hist.splitlines() if "[current]" in ln]
    assert len(current) == 1 and "Globex" in current[0]
    top = dispatch(v, "nanomem_search", {"query": "where do I work", "top_k": 3})
    assert "Globex" in top.splitlines()[0], \
        f"search disagrees with history about the current value:\n{top}"
    v.close()


def test_metadata_is_no_longer_accepted_and_dropped(vault_path, offline_embedder):
    """`nanomem_add` took an undeclared `metadata` argument, ignored it, and
    replied `Stored: ...`. An agent that reasonably tried to declare an entity
    got a success message and no entity."""
    from nanomem.mcp import dispatch
    v = Vault(vault_path)
    dispatch(v, "nanomem_add", {"text": _CHAIN[0], "metadata": {"entity": "employer"}})
    v.flush()
    rec = v.get_all_records()[0]
    assert (rec.get("metadata") or {}).get("entity") == "employer"
    v.close()


# --------------------------------------------------------------------------
# Registries and inspectors introspect by CALLING every list method, whether or
# not the server advertises that capability. Through 0.8.0 `resources/list` and
# `prompts/list` fell through to a bare `{}` -- a reply missing the array the
# method is defined to return, which reads as a broken server rather than an
# empty one. Found while preparing the Glama listing that awesome-mcp-servers
# now requires, where a failed introspection withholds the listing entirely.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method,field", [
    ("tools/list", "tools"),
    ("resources/list", "resources"),
    ("resources/templates/list", "resourceTemplates"),
    ("prompts/list", "prompts"),
])
def test_every_list_method_returns_its_declared_array(monkeypatch, vault_path,
                                                      offline_embedder, method, field):
    replies = _serve(monkeypatch, vault_path, [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "inspector", "version": "1"}}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": method}),
    ])
    result = next(r for r in replies if r.get("id") == 2)["result"]
    assert field in result, "%s returned %r, missing %r" % (method, result, field)
    assert isinstance(result[field], list)


def test_capabilities_do_not_claim_resources_or_prompts(monkeypatch, vault_path,
                                                        offline_embedder):
    """Answering the call is not the same as advertising the capability.

    This server has no resources and no prompts. It replies correctly when asked
    -- see above -- but must not announce features it does not have.
    """
    replies = _serve(monkeypatch, vault_path, [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "c", "version": "1"}}}),
    ])
    caps = replies[0]["result"]["capabilities"]
    assert "tools" in caps
    assert "resources" not in caps and "prompts" not in caps
