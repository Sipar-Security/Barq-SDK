"""The engine as a SUBSTRATE: a consumer installs their own native tool and mounts
several MCP servers, and the Agent makes all of them callable by the model, without
editing engine code. If this breaks, the engine is just a fixed app.
"""

import asyncio

import pytest

from engine import Agent
from engine.coordinator import ModelResponse


class _FakeMCP:
    """Minimal ServerConnection (server_id + list_tools + call_tool)."""

    def __init__(self, server_id, tool):
        self.server_id = server_id
        self._tool = tool
        self.calls = []

    async def list_tools(self):
        return [{"name": self._tool, "description": "x", "inputSchema": {"type": "object"}}]

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return f"{self.server_id}:{tool_name}:ok"


class _Scripted:
    def __init__(self, responses):
        self._r = list(responses)

    async def create(self, messages, tools):
        return self._r.pop(0)


def test_consumer_tools_and_multiple_mcp_servers_are_callable(tmp_path):
    calc_calls = []

    async def calc(inp):
        calc_calls.append(inp)
        return "calc:42"

    calc_spec = {
        "name": "CalcTool",
        "description": "adds numbers",
        "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
    }

    # Two MCP servers exposing a tool with the SAME bare name; proves namespacing keeps
    # multiple servers from colliding.
    svc_a = _FakeMCP("svcA", "ping")
    svc_b = _FakeMCP("svcB", "ping")

    model = _Scripted([
        ModelResponse([
            {"type": "tool_use", "id": "1", "name": "CalcTool", "input": {"x": 40}},
            {"type": "tool_use", "id": "2", "name": "svcA__ping", "input": {"n": 1}},
            {"type": "tool_use", "id": "3", "name": "svcB__ping", "input": {"n": 2}},
        ], "tool_use"),
        ModelResponse([{"type": "text", "text": "done"}], "end_turn"),
    ])

    agent = Agent(
        model=model, workdir=tmp_path / "wd",
        tools=[(calc_spec, calc)],
        mcp_servers=[svc_a, svc_b],
        trusted_tools=("CalcTool",),
        enable_file_tools=False, enable_memory_tools=False,
    )
    out = asyncio.run(agent.run("use the tools"))

    assert out == "done"
    assert calc_calls == [{"x": 40}]            # consumer tool was invoked
    assert svc_a.calls == [("ping", {"n": 1})]  # first MCP server routed
    assert svc_b.calls == [("ping", {"n": 2})]  # second MCP server routed


def test_tool_name_collision_is_rejected(tmp_path):
    bad_spec = {"name": "ReadFile", "description": "clash",
                "input_schema": {"type": "object"}}

    async def noop(inp):
        return "x"

    model = _Scripted([ModelResponse([{"type": "text", "text": "n/a"}], "end_turn")])
    agent = Agent(model=model, workdir=tmp_path / "wd", tools=[(bad_spec, noop)])
    # a tool that shadows a built-in must be refused, not silently override it
    with pytest.raises(ValueError, match="collides"):
        asyncio.run(agent.run("go"))
