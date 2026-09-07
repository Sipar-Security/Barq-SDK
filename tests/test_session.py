"""Crash-resume: the coordinator snapshots its transcript each turn and a later run
continues it instead of restarting. Covers the SessionStore, coordinator resume, the
total-turn budget across resumes, exactly-once tool execution, and Agent-level resume.
"""

import asyncio

from engine import Agent
from engine.audit import AuditLog
from engine.coordinator import Coordinator, ModelResponse
from engine.hooks import HookEngine
from engine.permissions import HostAllowlist, Mode, PermissionEngine
from engine.session import SessionStore, ToolJournal


# ---------- SessionStore basics ----------
def test_session_store_roundtrip_and_atomic(tmp_path):
    s = SessionStore(tmp_path / "s.json")
    assert not s.exists()
    s.save([{"role": "user", "content": "hi"}], done=False)
    assert s.exists()
    msgs, done = s.load()
    assert msgs == [{"role": "user", "content": "hi"}] and done is False
    s.save(msgs, done=True)
    assert s.load()[1] is True
    s.clear()
    assert not s.exists()


# ---------- coordinator resume ----------
class _CrashModel:
    """Serves scripted responses, then raises on the (crash_after+1)-th call."""

    def __init__(self, responses, crash_after):
        self._r = list(responses)
        self.calls = 0
        self.crash_after = crash_after

    async def create(self, messages, tools):
        self.calls += 1
        if self.calls > self.crash_after:
            raise RuntimeError("simulated crash")
        return self._r.pop(0)


def _engine():
    return PermissionEngine(HookEngine(), mode=Mode.AUTO,
                            network_policy=HostAllowlist(allow=("*.acme.com",)))


def test_coordinator_resumes_instead_of_restarting(tmp_path):
    store = SessionStore(tmp_path / "session.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    tool_runs = []

    async def tool(inp):
        tool_runs.append(inp)
        return "ok"

    specs = [{"name": "T", "description": "t", "input_schema": {"type": "object", "properties": {}}}]

    # Run 1: one tool turn, then a crash on the next model call.
    crash = _CrashModel([ModelResponse([{"type": "tool_use", "id": "1", "name": "T", "input": {}}], "tool_use")], crash_after=1)
    coord1 = Coordinator(model=crash, permissions=_engine(), audit=audit,
                         native_tools={"T": tool}, tool_specs=specs,
                         session=store, resume=False, max_turns=5)
    try:
        asyncio.run(coord1.run("go"))
    except RuntimeError:
        pass
    assert len(tool_runs) == 1
    saved_msgs, done = store.load()
    assert done is False and saved_msgs[-1]["role"] == "user"  # snapshot at turn boundary

    # Run 2: fresh coordinator, resume=True, a model that just ends. It must NOT replay
    # turn 1 (tool runs only once total) and must finish cleanly.
    ender = _CrashModel([ModelResponse([{"type": "text", "text": "done"}], "end_turn")], crash_after=99)
    coord2 = Coordinator(model=ender, permissions=_engine(), audit=audit,
                         native_tools={"T": tool}, tool_specs=specs,
                         session=store, resume=True, max_turns=5)
    msgs = asyncio.run(coord2.run("go"))
    assert coord2.completed is True
    assert ender.calls == 1                 # continued, didn't restart
    assert len(tool_runs) == 1              # turn 1 not replayed
    assert msgs[-1]["content"][0]["text"] == "done"
    assert store.load()[1] is True          # marked done


class _Crash(BaseException):
    """A non-Exception so it slips past _handle_tool_use's `except Exception`, simulating
    the process dying mid tool-execution."""


def test_tool_execution_is_exactly_once_across_crash(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    a_runs = []

    async def tool_a(inp):
        a_runs.append(1)
        return "A-done"

    async def crash_b(inp):
        raise _Crash("died mid-turn")

    specs = [
        {"name": "A", "description": "a", "input_schema": {"type": "object", "properties": {}}},
        {"name": "B", "description": "b", "input_schema": {"type": "object", "properties": {}}},
    ]
    two_calls = ModelResponse([
        {"type": "tool_use", "id": "a1", "name": "A", "input": {}},
        {"type": "tool_use", "id": "b1", "name": "B", "input": {}},
    ], "tool_use")

    # Run 1: A executes (journaled), then B crashes the process before the turn finishes.
    coord1 = Coordinator(model=_CrashModel([two_calls], crash_after=99),
                         permissions=_engine(), audit=audit,
                         native_tools={"A": tool_a, "B": crash_b}, tool_specs=specs,
                         session=SessionStore(tmp_path / "s.json"),
                         journal=ToolJournal(tmp_path / "j.jsonl"), resume=False, max_turns=5)
    try:
        asyncio.run(coord1.run("go"))
    except _Crash:
        pass
    assert a_runs == [1]
    assert coord1._is_pending_tool_turn(SessionStore(tmp_path / "s.json").load()[0])

    # Run 2: resume. A must NOT re-run (journal hit); B now completes; then the run ends.
    async def ok_b(inp):
        return "B-done"

    coord2 = Coordinator(model=_CrashModel([ModelResponse([{"type": "text", "text": "done"}], "end_turn")], crash_after=99),
                         permissions=_engine(), audit=audit,
                         native_tools={"A": tool_a, "B": ok_b}, tool_specs=specs,
                         session=SessionStore(tmp_path / "s.json"),
                         journal=ToolJournal(tmp_path / "j.jsonl"), resume=True, max_turns=5)
    asyncio.run(coord2.run("go"))
    assert a_runs == [1]              # EXACTLY once — not replayed on resume
    assert coord2.completed is True


def test_completed_session_is_not_rerun(tmp_path):
    store = SessionStore(tmp_path / "session.json")
    store.save([{"role": "user", "content": "x"}], done=True)
    audit = AuditLog(tmp_path / "audit.jsonl")

    class _Boom:
        async def create(self, messages, tools):
            raise AssertionError("model must not be called for a completed session")

    coord = Coordinator(model=_Boom(), permissions=_engine(), audit=audit,
                        session=store, resume=True, max_turns=5)
    msgs = asyncio.run(coord.run("go"))
    assert coord.completed is True and msgs[-1]["content"] == "x"


def test_total_turn_budget_spans_resumes(tmp_path):
    # A transcript already holding 2 assistant turns leaves only 1 of a 3-turn budget.
    store = SessionStore(tmp_path / "session.json")
    store.save([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "r", "is_error": False}]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "2", "content": "r", "is_error": False}]},
    ], done=False)
    audit = AuditLog(tmp_path / "audit.jsonl")

    class _Counter:
        def __init__(self):
            self.calls = 0

        async def create(self, messages, tools):
            self.calls += 1
            return ModelResponse([{"type": "tool_use", "id": "x", "name": "T", "input": {}}], "tool_use")

    m = _Counter()
    coord = Coordinator(model=m, permissions=_engine(), audit=audit,
                        native_tools={"T": lambda inp: "ok"},
                        tool_specs=[{"name": "T", "description": "t", "input_schema": {"type": "object", "properties": {}}}],
                        session=store, resume=True, max_turns=3)
    asyncio.run(coord.run("go"))
    assert m.calls == 1  # 3 total - 2 already taken = 1 remaining


def test_context_compactor_runs_between_model_cycles_and_persists_replacement(tmp_path):
    store = SessionStore(tmp_path / "session.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    model = _CrashModel([
        ModelResponse([{"type": "tool_use", "id": "1", "name": "T", "input": {}}],
                      "tool_use"),
        ModelResponse([{"type": "text", "text": "done"}], "end_turn"),
    ], crash_after=99)
    calls = []

    async def compact(messages):
        calls.append(len(messages))
        if any(message.get("role") == "assistant" for message in messages):
            return [{"role": "user", "content": "[compacted transcript]"}]
        return messages

    coord = Coordinator(
        model=model, permissions=_engine(), audit=audit,
        native_tools={"T": lambda _inp: "ok"},
        tool_specs=[{"name": "T", "description": "t",
                     "input_schema": {"type": "object", "properties": {}}}],
        session=store, max_turns=4, context_compactor=compact,
    )

    messages = asyncio.run(coord.run("go"))

    assert calls == [1, 3]
    assert messages[0] == {"role": "user", "content": "[compacted transcript]"}
    assert messages[-1]["content"][0]["text"] == "done"
    assert store.load()[1] is True


# ---------- Agent-level resume (exactly-once tools) ----------
class _ScriptOrCrash:
    """Serves scripted responses; an extra model call past the script raises (crash)."""

    def __init__(self, responses):
        self._r = list(responses)

    async def create(self, messages, tools):
        return self._r.pop(0)  # IndexError past the script == a crash


def test_agent_resume_runs_tools_exactly_once(tmp_path):
    wd = tmp_path / "wd"
    runs = []
    spec = {"name": "Count", "description": "count",
            "input_schema": {"type": "object", "properties": {}}}

    async def count(_inp):
        runs.append(1)
        return "counted"

    # Run 1: the model calls Count, then the script runs dry -> crash mid-run.
    a1 = Agent(model=_ScriptOrCrash([
        ModelResponse([{"type": "tool_use", "id": "1", "name": "Count", "input": {}}], "tool_use"),
    ]), workdir=wd, tools=[(spec, count)], enable_memory_tools=False, enable_file_tools=False)
    try:
        asyncio.run(a1.run("go"))
    except IndexError:
        pass
    assert runs == [1]
    assert (wd / "session.json").exists()

    # Run 2: resume -> Count must NOT re-run (journal hit); the run finishes cleanly.
    a2 = Agent(model=_ScriptOrCrash([
        ModelResponse([{"type": "text", "text": "done"}], "end_turn"),
    ]), workdir=wd, tools=[(spec, count)], enable_memory_tools=False,
        enable_file_tools=False, resume=True)
    out = asyncio.run(a2.run("go"))
    assert runs == [1]                 # EXACTLY once across the crash
    assert a2.completed is True
    assert out == "done"
