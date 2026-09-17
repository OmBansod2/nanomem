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


def test_unknown_tool_does_not_raise(chat_vault):
    assert "Unknown tool" in nm_mcp.dispatch(chat_vault, "nanomem_nope", {})


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
