import asyncio

from engine.audit import AuditLog
from engine.coordinator import Coordinator, ModelResponse
from engine.hooks import HookEngine
from engine.mcp import MCPHandler
from engine.permissions import HostAllowlist, Mode, PermissionEngine


# ---------- MCP handler (offline, fake connections) ----------
class FakeConn:
    def __init__(self, server_id, tools, results=None):
        self.server_id = server_id
        self._tools = tools
        self._results = results or {}
        self.calls = []

    async def list_tools(self):
        return self._tools

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self._results.get(tool_name, "ok")


def test_mcp_namespacing_and_routing():
    async def scenario():
        h = MCPHandler()
        h.add_server(FakeConn("memory", [
            {"name": "search", "description": "graph search", "inputSchema": {"type": "object"}},
        ]))
        h.add_server(FakeConn("kb", [
            {"name": "search", "description": "different tool same name", "inputSchema": {}},
        ], results={"search": "kb-result"}))
        await h.refresh()
        names = set(h.tool_names())
        assert names == {"memory__search", "kb__search"}
        specs = h.get_tool_specs()
        assert all("name" in s and "input_schema" in s for s in specs)
        out = await h.call_tool("kb__search", {"q": "x"})
        assert out == "kb-result"
    asyncio.run(scenario())


def test_mcp_one_broken_server_is_isolated():
    async def scenario():
        class Broken:
            server_id = "broken"
            async def list_tools(self):
                raise RuntimeError("server down")
            async def call_tool(self, *a):
                return "x"

        h = MCPHandler()
        h.add_server(FakeConn("ok", [{"name": "t", "description": "", "inputSchema": {}}]))
        h.add_server(Broken())
        await h.refresh()
        assert h.tool_names() == ["ok__t"]        # healthy server still loaded
        assert "broken" in h.failures             # failure surfaced, not raised
    asyncio.run(scenario())


# ---------- Coordinator loop (scripted fake model) ----------
class ScriptedModel:
    def __init__(self, responses):
        self._responses = list(responses)

    async def create(self, messages, tools):
        return self._responses.pop(0)


def build_coordinator(tmp_path, mode, elicit=None):
    policy = HostAllowlist(allow=("*.acme.com",), deny=("staging.acme.com",))
    eng = PermissionEngine(HookEngine(), mode=mode, network_policy=policy)
    audit = AuditLog(tmp_path / "audit.jsonl")
    ran = []

    async def http_get(inp):
        ran.append(inp["url"])
        return f"200 {inp['url']}"

    coord = Coordinator(
        model=None, permissions=eng, audit=audit,
        native_tools={"HttpRequest": http_get}, elicit=elicit,
    )
    return coord, ran, audit


def test_auto_allows_in_policy_blocks_out_of_policy(tmp_path):
    coord, ran, audit = build_coordinator(tmp_path, Mode.AUTO)
    coord.model = ScriptedModel([
        ModelResponse(content=[
            {"type": "tool_use", "id": "t1", "name": "HttpRequest", "input": {"url": "https://api.acme.com/"}},
            {"type": "tool_use", "id": "t2", "name": "HttpRequest", "input": {"url": "https://evil.example.com/"}},
        ], stop_reason="tool_use"),
        ModelResponse(content=[{"type": "text", "text": "done"}], stop_reason="end_turn"),
    ])
    messages = asyncio.run(coord.run("go"))
    assert ran == ["https://api.acme.com/"]           # in-policy ran, out-of-policy did not
    results = messages[2]["content"]
    by_id = {r["tool_use_id"]: r for r in results}
    assert by_id["t1"]["is_error"] is False
    assert by_id["t2"]["is_error"] is True and "DENIED" in by_id["t2"]["content"]
    kinds = [e.kind for e in audit.read_all()]
    assert "blocked" in kinds


def test_ask_mode_requires_approval(tmp_path):
    coord, ran, audit = build_coordinator(tmp_path, Mode.ASK, elicit=lambda d, c: False)
    coord.model = ScriptedModel([
        ModelResponse(content=[
            {"type": "tool_use", "id": "t1", "name": "HttpRequest", "input": {"url": "https://api.acme.com/login"}},
        ], stop_reason="tool_use"),
        ModelResponse(content=[{"type": "text", "text": "stopped"}], stop_reason="end_turn"),
    ])
    messages = asyncio.run(coord.run("go"))
    assert ran == []  # not approved -> never ran
    results = messages[2]["content"]
    assert results[0]["is_error"] and "NOT APPROVED" in results[0]["content"]


def test_ask_mode_runs_when_approved(tmp_path):
    coord, ran, audit = build_coordinator(tmp_path, Mode.ASK, elicit=lambda d, c: True)
    coord.model = ScriptedModel([
        ModelResponse(content=[
            {"type": "tool_use", "id": "t1", "name": "HttpRequest", "input": {"url": "https://api.acme.com/login"}},
        ], stop_reason="tool_use"),
        ModelResponse(content=[{"type": "text", "text": "ok"}], stop_reason="end_turn"),
    ])
    asyncio.run(coord.run("go"))
    assert ran == ["https://api.acme.com/login"]
