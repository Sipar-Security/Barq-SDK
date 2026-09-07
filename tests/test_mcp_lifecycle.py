"""MCP connection lifecycle: health, eviction, reconnect, and fault isolation.

A server that dies mid-session used to stay advertised and fail every call for the rest of
the run, with no route back short of rebuilding the handler. These cover the health/evict/
reconnect cycle and confirm the original fault-isolation guarantee still holds.
"""

from __future__ import annotations

import asyncio

import pytest

from engine.mcp import MCPHandler
from engine.mcp.handler import sanitize_tool_name


class FlakyServer:
    """A server that can be killed and revived, and counts its list_tools calls."""

    def __init__(self, server_id="fs", tools=("read",)):
        self.server_id = server_id
        self.alive = True
        self.lists = 0
        self._tools = tools

    async def list_tools(self):
        self.lists += 1
        if not self.alive:
            raise ConnectionError("broken pipe")
        return [{"name": t, "description": t, "inputSchema": {"type": "object"}}
                for t in self._tools]

    async def call_tool(self, name, args):
        if not self.alive:
            raise ConnectionError("broken pipe")
        return f"{name}:ok"


def run(coro):
    return asyncio.run(coro)


def test_dead_server_is_evicted_then_restored_by_reconnect():
    srv = FlakyServer()
    h = MCPHandler()
    h.add_server(srv)
    run(h.refresh())
    assert h.tool_names() == ["fs__read"]
    assert h.healthy() == {"fs": True}

    srv.alive = False
    with pytest.raises(ConnectionError):
        run(h.call_tool("fs__read", {}))

    # the failure is recorded and its tools withdrawn, so the model stops being offered them
    assert h.healthy() == {"fs": False}
    assert h.tool_names() == []
    assert "broken pipe" in h.failures["fs"]

    srv.alive = True
    assert run(h.reconnect("fs")) is True
    assert h.tool_names() == ["fs__read"]
    assert h.failures == {}
    assert run(h.call_tool("fs__read", {})) == "read:ok"


def test_reconnect_reports_failure_without_raising():
    srv = FlakyServer()
    h = MCPHandler()
    h.add_server(srv)
    run(h.refresh())
    srv.alive = False
    assert run(h.reconnect("fs")) is False
    assert "fs" in h.failures
    assert run(h.reconnect("nonexistent")) is False


def test_reconnect_uses_a_connection_supplied_hook():
    calls = []

    class Reconnectable(FlakyServer):
        async def reconnect(self):
            calls.append(1)
            self.alive = True

    srv = Reconnectable()
    h = MCPHandler()
    h.add_server(srv)
    run(h.refresh())
    srv.alive = False
    assert run(h.reconnect("fs")) is True
    assert calls == [1], "a connection that knows how to reconnect should be asked to"


def test_one_broken_server_does_not_blank_the_others():
    good, bad = FlakyServer("good", ("a",)), FlakyServer("bad", ("b",))
    bad.alive = False
    h = MCPHandler()
    h.add_server(good)
    h.add_server(bad)
    run(h.refresh())  # must not raise
    assert h.tool_names() == ["good__a"]
    assert "bad" in h.failures and "good" not in h.failures


def test_duplicate_server_ids_are_rejected():
    h = MCPHandler()
    h.add_server(FlakyServer("x"))
    with pytest.raises(ValueError, match="duplicate server id"):
        h.add_server(FlakyServer("x"))


def test_unknown_tool_raises_keyerror():
    h = MCPHandler()
    with pytest.raises(KeyError):
        run(h.call_tool("nope__tool", {}))


def test_list_timeout_isolates_a_hanging_server():
    class Hang:
        server_id = "slow"

        async def list_tools(self):
            await asyncio.sleep(30)

        async def call_tool(self, n, a):
            return "x"

    h = MCPHandler(list_timeout=0.2)
    h.add_server(Hang())
    run(asyncio.wait_for(h.refresh(), timeout=10))
    assert h.tool_names() == []
    assert "timed out" in h.failures["slow"]


def test_sanitised_names_do_not_collide():
    class Colliding:
        server_id = "s"

        async def list_tools(self):
            # both sanitise to "do_it": distinct tools must stay distinct
            return [{"name": "do it", "description": "", "inputSchema": {}},
                    {"name": "do/it", "description": "", "inputSchema": {}}]

        async def call_tool(self, n, a):
            return n

    h = MCPHandler()
    h.add_server(Colliding())
    run(h.refresh())
    assert len(h.tool_names()) == 2, "a sanitising collision must not drop a tool"


def test_sanitize_tool_name_never_returns_empty():
    assert sanitize_tool_name("!!!") == "tool"
    assert sanitize_tool_name("ok-name_1") == "ok-name_1"
    assert sanitize_tool_name("a b.c") == "a_b_c"


def test_evict_can_be_disabled():
    srv = FlakyServer()
    h = MCPHandler(evict_failed=False)
    h.add_server(srv)
    run(h.refresh())
    srv.alive = False
    with pytest.raises(ConnectionError):
        run(h.call_tool("fs__read", {}))
    assert h.tool_names() == ["fs__read"], "eviction was opted out of"
    assert h.healthy() == {"fs": False}
