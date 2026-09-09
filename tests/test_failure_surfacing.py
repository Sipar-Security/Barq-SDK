"""Failure has to reach the caller, not only the model.

Three reproduced classes of silent failure, all of which returned normally:

  * A tool that RAISED. The error was formatted into the transcript for the model and
    nowhere else, so `run()` returned the model's apology and `completed` was True.
  * A dead MCP server. `refresh()` isolated it into `handler.failures`, the facade never
    read that, and mounting three servers with zero working tools produced a plausible
    answer. The only access was `agent._mcp_handler.failures`.
  * No run identifier. The audit chain stamped one on every record and exposed it nowhere,
    so the evidence could not be joined to the caller's own logs or trace.
"""

from __future__ import annotations

import pytest

from engine import Agent, ModelResponse
from engine.coordinator import ToolFailure
from engine.permissions import Mode

SPEC = {
    "name": "T",
    "description": "a tool",
    "input_schema": {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    },
}
DONE = ModelResponse([{"type": "text", "text": "I could not do it"}], "end_turn", {})


class Script:
    """A model that plays a fixed list of responses, then ends the turn."""

    def __init__(self, responses):
        self.responses = list(responses)

    async def create(self, messages, tools):
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse([{"type": "text", "text": "end"}], "end_turn", {})


def use(call_id="1", name="T", inp=None):
    return ModelResponse(
        [{"type": "tool_use", "id": call_id, "name": name, "input": inp or {"n": 1}}],
        "tool_use",
        {},
    )


class BrokenServer:
    server_id = "broken"

    async def list_tools(self):
        raise RuntimeError("server died")

    async def call_tool(self, name, args):  # pragma: no cover - never reached
        raise RuntimeError("unreachable")


class HealthyServer:
    server_id = "healthy"

    async def list_tools(self):
        return [{"name": "ping", "description": "p", "inputSchema": {"type": "object"}}]

    async def call_tool(self, name, args):
        return "pong"


async def build(tmp_path, **kw):
    kw.setdefault("enable_memory_tools", False)
    kw.setdefault("enable_file_tools", False)
    return Agent(workdir=tmp_path, **kw)


# --- a failing tool ----------------------------------------------------------
@pytest.mark.asyncio
async def test_a_raising_tool_is_reported_to_the_caller(tmp_path):
    def boom(inp):
        raise RuntimeError("database is on fire")

    agent = await build(tmp_path, model=Script([use(), DONE]), tools=[(SPEC, boom)])
    text = await agent.run("go")

    # The model still gets to answer, and the transcript is unchanged...
    assert text == "I could not do it"
    assert agent.completed is True
    # ...but the caller can now tell that nothing actually worked.
    assert agent.ok is False
    assert len(agent.failures) == 1
    failure = agent.failures[0]
    assert isinstance(failure, ToolFailure)
    assert failure.tool_name == "T"
    assert failure.kind == "error"
    assert "database is on fire" in failure.error
    assert failure.turn == 1
    assert failure.tool_input == {"n": 1}
    await agent.aclose()


@pytest.mark.asyncio
async def test_a_denied_tool_is_reported(tmp_path):
    agent = await build(
        tmp_path, model=Script([use(), DONE]), tools=[(SPEC, lambda i: "ok")],
        mode=Mode.LOCKED,
    )
    await agent.run("go")
    assert agent.ok is False
    assert [f.kind for f in agent.failures] == ["denied"]
    await agent.aclose()


@pytest.mark.asyncio
async def test_an_unapproved_ask_is_reported(tmp_path):
    agent = await build(
        tmp_path, model=Script([use(), DONE]), tools=[(SPEC, lambda i: "ok")],
        mode=Mode.ASK, elicit=lambda decision, call: False,
    )
    await agent.run("go")
    assert [f.kind for f in agent.failures] == ["unapproved"]
    await agent.aclose()


@pytest.mark.asyncio
async def test_invalid_arguments_are_reported(tmp_path):
    agent = await build(
        tmp_path,
        model=Script([use(inp={"n": "not-an-int"}), DONE]),
        tools=[(SPEC, lambda i: "ok")],
    )
    await agent.run("go")
    assert [f.kind for f in agent.failures] == ["invalid"]
    assert "expected integer" in agent.failures[0].error
    await agent.aclose()


@pytest.mark.asyncio
async def test_a_tool_timeout_is_reported(tmp_path):
    import asyncio

    async def slow(inp):
        await asyncio.sleep(5)

    agent = await build(
        tmp_path, model=Script([use(), DONE]), tools=[(SPEC, slow)], tool_timeout=0.2
    )
    await agent.run("go")
    assert [f.kind for f in agent.failures] == ["error"]
    assert "time limit" in agent.failures[0].error
    await agent.aclose()


@pytest.mark.asyncio
async def test_every_failure_in_a_run_is_collected(tmp_path):
    def boom(inp):
        raise RuntimeError("nope")

    # Distinct arguments per turn: three IDENTICAL turns are a degenerate loop and the
    # loop guard correctly stops the run before the third executes.
    agent = await build(
        tmp_path,
        model=Script([
            use("1", inp={"n": 1}), use("2", inp={"n": 2}), use("3", inp={"n": 3}), DONE,
        ]),
        tools=[(SPEC, boom)],
    )
    await agent.run("go")
    assert len(agent.failures) == 3
    assert [f.turn for f in agent.failures] == [1, 2, 3]
    assert [f.tool_input["n"] for f in agent.failures] == [1, 2, 3]
    await agent.aclose()


@pytest.mark.asyncio
async def test_failures_reset_between_runs(tmp_path):
    """`failures` answers 'did THIS run fail', matching `completed`."""
    def boom(inp):
        raise RuntimeError("nope")

    agent = await build(tmp_path, model=Script([use(), DONE]), tools=[(SPEC, boom)])
    await agent.run("first")
    assert len(agent.failures) == 1

    agent._coord.model = Script([ModelResponse([{"type": "text", "text": "ok"}], "end_turn", {})])
    await agent.run("second")
    assert agent.failures == []
    assert agent.ok is True
    await agent.aclose()


# --- a broken MCP server -----------------------------------------------------
@pytest.mark.asyncio
async def test_a_dead_mcp_server_is_reported(tmp_path):
    agent = await build(tmp_path, model=Script([]), mcp_servers=[BrokenServer()])
    await agent.run("hi")
    assert agent.mcp_failures == {"broken": "RuntimeError: server died"}
    assert agent.ok is False
    await agent.aclose()


@pytest.mark.asyncio
async def test_one_broken_server_does_not_hide_a_healthy_one(tmp_path):
    """Fault isolation is the existing behaviour; reporting it is the new part."""
    agent = await build(
        tmp_path, model=Script([]), mcp_servers=[BrokenServer(), HealthyServer()]
    )
    await agent.run("hi")
    assert set(agent.mcp_failures) == {"broken"}
    assert any(s["name"].startswith("healthy__") for s in agent._coord.tool_specs)
    await agent.aclose()


@pytest.mark.asyncio
async def test_no_mcp_servers_means_no_mcp_failures(tmp_path):
    agent = await build(tmp_path, model=Script([]))
    await agent.run("hi")
    assert agent.mcp_failures == {}
    await agent.aclose()


# --- correlation -------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_id_is_exposed_and_matches_the_audit_records(tmp_path):
    """The evidence chain has to be joinable to the caller's own logs."""
    import json

    # A run with no tool calls writes no decision records, so it must actually call one.
    agent = await build(
        tmp_path, model=Script([use(), DONE]), tools=[(SPEC, lambda i: "fine")]
    )
    await agent.run("hi")
    run_id = agent.run_id
    assert run_id and len(run_id) >= 8
    await agent.aclose()

    audit = tmp_path / ".agent-state" / "audit.jsonl"
    stamped = {
        json.loads(line)["data"].get("run_id")
        for line in audit.read_text(encoding="utf-8").splitlines()
        if line.strip() and "run_id" in line
    }
    assert stamped == {run_id}


@pytest.mark.asyncio
async def test_run_id_is_empty_before_the_first_run(tmp_path):
    agent = Agent(model=Script([]), workdir=tmp_path)
    assert agent.run_id == ""
    assert agent.failures == []
    assert agent.mcp_failures == {}


# --- the single check a caller needs -----------------------------------------
@pytest.mark.asyncio
async def test_ok_is_true_only_for_a_genuinely_clean_run(tmp_path):
    agent = await build(
        tmp_path, model=Script([use(), DONE]), tools=[(SPEC, lambda i: "fine")]
    )
    await agent.run("go")
    assert agent.ok is True
    assert agent.failures == [] and agent.mcp_failures == {}
    await agent.aclose()


@pytest.mark.asyncio
async def test_ok_is_false_when_the_run_was_truncated(tmp_path):
    """`completed` already distinguished truncation; `ok` must not lose that."""
    agent = await build(
        tmp_path,
        model=Script([use("1"), use("2"), use("3")]),
        tools=[(SPEC, lambda i: "fine")],
        max_turns=2,
    )
    await agent.run("go")
    assert agent.completed is False
    assert agent.ok is False
    await agent.aclose()
