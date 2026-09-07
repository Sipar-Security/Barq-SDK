import asyncio

from engine.audit import AuditLog
from engine.coordinator import Coordinator, ModelResponse
from engine.hooks import HookEngine
from engine.permissions import HostAllowlist, Mode, PermissionEngine
from engine.subagents import Subagent, make_spawn_tool


class Scripted:
    def __init__(self, responses):
        self._r = list(responses)

    async def create(self, messages, tools):
        return self._r.pop(0)


def _engine():
    policy = HostAllowlist(allow=("*.acme.com", "127.0.0.1"))
    return PermissionEngine(HookEngine(), mode=Mode.AUTO, network_policy=policy)


def test_subagent_runs_isolated_and_summarizes(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")
    ran = []

    async def probe(inp):
        ran.append(inp["url"])
        return "200 ok"

    model = Scripted([
        ModelResponse([{"type": "tool_use", "id": "1", "name": "Probe",
                        "input": {"url": "https://api.acme.com/"}}], "tool_use"),
        ModelResponse([{"type": "text", "text": "found login endpoint"}], "end_turn"),
    ])
    sub = Subagent(
        name="recon", model=model, permissions=_engine(), audit=audit,
        native_tools={"Probe": probe},
        tool_specs=[{"name": "Probe", "description": "p", "input_schema": {"type": "object"}}],
    )
    res = asyncio.run(sub.run("recon acme"))
    assert res.final_text == "found login endpoint"
    assert res.tool_calls == 1
    assert ran == ["https://api.acme.com/"]
    assert res.isolated_messages >= 3


def test_spawn_tool_returns_only_summary_to_parent(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")

    def recon_factory():
        model = Scripted([
            ModelResponse([{"type": "tool_use", "id": "1", "name": "Probe",
                            "input": {"url": "https://api.acme.com/"}}], "tool_use"),
            ModelResponse([{"type": "text", "text": "SUBAGENT_SUMMARY: 3 hosts live"}], "end_turn"),
        ])

        async def probe(inp):
            return "noise " * 100  # high-volume output that must NOT reach the parent

        return Subagent(
            name="recon", model=model, permissions=_engine(), audit=audit,
            native_tools={"Probe": probe},
            tool_specs=[{"name": "Probe", "description": "p", "input_schema": {"type": "object"}}],
        )

    spawn, spec = make_spawn_tool({"recon": recon_factory})
    assert spec["name"] == "SpawnSubagent"

    # parent coordinator: calls SpawnSubagent, then ends
    parent_model = Scripted([
        ModelResponse([{"type": "tool_use", "id": "p1", "name": "SpawnSubagent",
                        "input": {"subagent_type": "recon", "task": "map acme"}}], "tool_use"),
        ModelResponse([{"type": "text", "text": "done"}], "end_turn"),
    ])
    parent = Coordinator(
        model=parent_model, permissions=_engine(), audit=audit,
        native_tools={"SpawnSubagent": spawn}, tool_specs=[spec],
    )
    messages = asyncio.run(parent.run("orchestrate"))

    # the tool_result the parent saw is the subagent SUMMARY, not its raw "noise" output
    tool_result = messages[2]["content"][0]
    assert "SUBAGENT_SUMMARY: 3 hosts live" in tool_result["content"]
    assert "noise noise" not in tool_result["content"]  # isolation: raw output stayed inside


def test_spawn_unknown_type_errors(tmp_path):
    spawn, _ = make_spawn_tool({})
    out = asyncio.run(spawn({"subagent_type": "ghost", "task": "x"}))
    assert "unknown subagent_type" in out


# --- C1: truncation must be reported, not silently returned as an empty success ----------
class UsageScripted(Scripted):
    """A scripted model that also exposes a usage_total, like SystemPromptClient does."""
    def __init__(self, responses, usage=None):
        super().__init__(responses)
        self.usage_total = usage or {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


def _loop_model():
    class Loop:
        # Never ends: always wants another tool call -> the subagent hits max_turns.
        async def create(self, messages, tools):
            return ModelResponse(
                [{"type": "tool_use", "id": "x", "name": "Probe",
                  "input": {"url": "https://api.acme.com/"}}], "tool_use")
    return Loop()


def test_truncated_subagent_reports_incomplete_not_empty_success(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")

    async def probe(inp):
        return "NOISE " * 20

    sub = Subagent(
        name="recon", model=_loop_model(), permissions=_engine(), audit=audit,
        native_tools={"Probe": probe},
        tool_specs=[{"name": "Probe", "description": "p", "input_schema": {"type": "object"}}],
        max_turns=3,
    )
    res = asyncio.run(sub.run("map it"))
    assert res.completed is False          # the run was cut off...
    assert res.tool_calls == 3             # ...after doing real work

    # And through the spawn tool the parent is told INCOMPLETE, never a clean empty summary.
    spawn, _ = make_spawn_tool({"recon": lambda: Subagent(
        name="recon", model=_loop_model(), permissions=_engine(), audit=audit,
        native_tools={"Probe": probe},
        tool_specs=[{"name": "Probe", "description": "p", "input_schema": {"type": "object"}}],
        max_turns=3)})
    out = asyncio.run(spawn({"subagent_type": "recon", "task": "map it"}))
    assert "TRUNCATED" in out and "completed" not in out.split("]")[0]


def test_final_text_falls_back_to_last_assistant_text_when_run_ends_on_tools(tmp_path):
    """A subagent whose LAST message is a tool_result turn still surfaces its most recent
    words (regression: previously read only messages[-1] and returned '')."""
    audit = AuditLog(tmp_path / "a.jsonl")
    model = Scripted([
        ModelResponse([{"type": "text", "text": "interim: found /login"},
                       {"type": "tool_use", "id": "1", "name": "Probe",
                        "input": {"url": "https://api.acme.com/"}}], "tool_use"),
        # second turn also wants a tool -> run ends on a tool_result (max_turns=2)
        ModelResponse([{"type": "tool_use", "id": "2", "name": "Probe",
                        "input": {"url": "https://api.acme.com/x"}}], "tool_use"),
    ])

    async def probe(inp):
        return "ok"

    sub = Subagent(
        name="recon", model=model, permissions=_engine(), audit=audit,
        native_tools={"Probe": probe},
        tool_specs=[{"name": "Probe", "description": "p", "input_schema": {"type": "object"}}],
        max_turns=2,
    )
    res = asyncio.run(sub.run("go"))
    assert res.completed is False
    assert res.final_text == "interim: found /login"


# --- C2: ASK inside a subagent reaches the elicit callback (not a silent fail-closed) -----
def _ask_engine():
    policy = HostAllowlist(allow=("*.acme.com",))
    return PermissionEngine(HookEngine(), mode=Mode.ASK, network_policy=policy, ask_rules=("Repro",))


def _repro_model():
    return Scripted([
        ModelResponse([{"type": "tool_use", "id": "1", "name": "Repro", "input": {"x": 1}}],
                      "tool_use"),
        ModelResponse([{"type": "text", "text": "done"}], "end_turn"),
    ])


def test_subagent_ask_is_approved_via_elicit(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")
    ran = []

    async def repro(inp):
        ran.append(1)
        return "reproduced"

    async def approve(decision, call):
        return True

    sub = Subagent(
        name="tester", model=_repro_model(), permissions=_ask_engine(), audit=audit,
        native_tools={"Repro": repro},
        tool_specs=[{"name": "Repro", "description": "p", "input_schema": {"type": "object"}}],
        elicit=approve,
    )
    res = asyncio.run(sub.run("confirm"))
    assert ran == [1]                       # the gated PoC actually ran
    assert res.completed is True


def test_subagent_ask_denied_without_elicit(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")
    ran = []

    async def repro(inp):
        ran.append(1)
        return "reproduced"

    sub = Subagent(  # no elicit -> ASK fails closed, exactly as before the fix
        name="tester", model=_repro_model(), permissions=_ask_engine(), audit=audit,
        native_tools={"Repro": repro},
        tool_specs=[{"name": "Repro", "description": "p", "input_schema": {"type": "object"}}],
    )
    asyncio.run(sub.run("confirm"))
    assert ran == []                        # denied — the whole point of threading elicit


# --- C4: crash-resume keeps subagent tool side effects exactly-once ----------------------
def test_subagent_journal_makes_tool_calls_exactly_once(tmp_path):
    from engine.session import ToolJournal

    audit = AuditLog(tmp_path / "a.jsonl")
    calls = []

    async def probe(inp):
        calls.append(inp["url"])
        return "200"

    # Pre-record the result for tool_use id "1" as if it ran before a crash.
    journal = ToolJournal(tmp_path / "j.jsonl")
    journal.record("1", {"type": "tool_result", "tool_use_id": "1",
                         "content": "200 (from before crash)", "is_error": False})

    model = Scripted([
        ModelResponse([{"type": "tool_use", "id": "1", "name": "Probe",
                        "input": {"url": "https://api.acme.com/"}}], "tool_use"),
        ModelResponse([{"type": "text", "text": "done"}], "end_turn"),
    ])
    sub = Subagent(
        name="recon", model=model, permissions=_engine(), audit=audit,
        native_tools={"Probe": probe},
        tool_specs=[{"name": "Probe", "description": "p", "input_schema": {"type": "object"}}],
        journal=journal,
    )
    asyncio.run(sub.run("go"))
    assert calls == []                      # the journalled call was NOT re-executed


def test_spawn_wires_session_and_resume(tmp_path):
    from engine.session import SessionStore, ToolJournal

    audit = AuditLog(tmp_path / "a.jsonl")
    seen = {}

    def factory():
        return Subagent(
            name="recon", model=Scripted([
                ModelResponse([{"type": "text", "text": "mapped"}], "end_turn")]),
            permissions=_engine(), audit=audit)

    def session_for(stype, task):
        seen["key"] = (stype, task)
        return SessionStore(tmp_path / f"{stype}.json"), ToolJournal(tmp_path / f"{stype}.jsonl")

    spawn, _ = make_spawn_tool({"recon": factory}, session_for=session_for, resume=True)
    out = asyncio.run(spawn({"subagent_type": "recon", "task": "map acme"}))
    assert "mapped" in out
    assert seen["key"] == ("recon", "map acme")
    assert (tmp_path / "recon.json").exists()   # the subagent persisted its transcript


# --- O1/O2: usage metering + spawn budget ------------------------------------------------
def test_spawn_reports_usage_and_enforces_budget(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")
    results = []

    def factory():
        return Subagent(
            name="recon",
            model=UsageScripted([ModelResponse([{"type": "text", "text": "ok"}], "end_turn")]),
            permissions=_engine(), audit=audit)

    spawn, _ = make_spawn_tool(
        {"recon": factory}, on_result=results.append, max_spawns=1)

    out1 = asyncio.run(spawn({"subagent_type": "recon", "task": "a"}))
    assert "ok" in out1
    assert results[0].usage["total_tokens"] == 12      # usage captured from model.usage_total

    out2 = asyncio.run(spawn({"subagent_type": "recon", "task": "b"}))
    assert "budget exhausted" in out2                  # second spawn refused
    assert len(results) == 1                            # and never ran


# --- L2: a subagent can never hold the spawn tool (no recursion) -------------------------
def test_subagent_strips_spawn_tools(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")
    inner_spawn, inner_spec = make_spawn_tool({})
    sub = Subagent(
        name="x", model=Scripted([ModelResponse([{"type": "text", "text": "ok"}], "end_turn")]),
        permissions=_engine(), audit=audit,
        native_tools={"Task": inner_spawn, "SpawnSubagent": inner_spawn, "ReadFile": inner_spawn},
        tool_specs=[{"name": "Task"}, {"name": "SpawnSubagent"}, {"name": "ReadFile"}],
    )
    native, specs = sub._safe_toolset()
    assert "Task" not in native and "SpawnSubagent" not in native
    assert "ReadFile" in native                          # only spawn tools are stripped
    assert {s["name"] for s in specs} == {"ReadFile"}


# --- MCP handler is forwarded into the subagent's Coordinator ----------------------------
def test_subagent_forwards_mcp_handler(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")
    called = []

    class FakeMCP:
        def tool_names(self):
            return {"cbm__search"}

        async def call_tool(self, name, inp):
            called.append((name, inp))
            return "graph result"

    model = Scripted([
        ModelResponse([{"type": "tool_use", "id": "1", "name": "cbm__search",
                        "input": {"q": "x"}}], "tool_use"),
        ModelResponse([{"type": "text", "text": "used the graph"}], "end_turn"),
    ])
    sub = Subagent(
        name="recon", model=model, permissions=_engine(), audit=audit,
        mcp_handler=FakeMCP(),
    )
    res = asyncio.run(sub.run("go"))
    assert called == [("cbm__search", {"q": "x"})]
    assert res.final_text == "used the graph"


# --- input validation / timeout ----------------------------------------------------------
def test_spawn_empty_task_and_missing_type_error(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")

    def factory():
        return Subagent(
            name="recon", model=Scripted([ModelResponse([{"type": "text", "text": "ok"}], "end_turn")]),
            permissions=_engine(), audit=audit)

    spawn, _ = make_spawn_tool({"recon": factory})
    assert "task must be" in asyncio.run(spawn({"subagent_type": "recon", "task": "   "}))
    assert "subagent_type is required" in asyncio.run(spawn({"task": "x"}))


def test_subagent_timeout_returns_incomplete(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")

    class Slow:
        async def create(self, messages, tools):
            await asyncio.sleep(1.0)
            return ModelResponse([{"type": "text", "text": "late"}], "end_turn")

    sub = Subagent(
        name="recon", model=Slow(), permissions=_engine(), audit=audit, timeout=0.05)
    res = asyncio.run(sub.run("go"))
    assert res.completed is False
