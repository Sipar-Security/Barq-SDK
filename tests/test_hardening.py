"""Regression tests for the hardening pass.

Each test pins one defect that was reproduced against the running SDK. They are written to
fail loudly if the old behaviour ever returns.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from engine.audit import AuditLog
from engine.audit.log import verify_audit_file
from engine.coordinator import Coordinator, ModelResponse
from engine.hooks import CommandHook, FunctionHook, HookEngine, HookEvent
from engine.mcp import MCPHandler
from engine.memory import Memory, MemoryStore, MemoryType
from engine.permissions import HostAllowlist, Mode, PermissionEngine, ToolCall
from engine.permissions.engine import network_targets
from engine.providers.openai_compat import ProviderResponseError, parse_openai_response
from engine.session import SessionStore
from engine.tools.http import make_http_request
from engine.validation import validate_tool_input


def _perms(**kw) -> PermissionEngine:
    return PermissionEngine(HookEngine(), mode=Mode.AUTO, **kw)


class Scripted:
    """A model that plays a fixed list of turns, then ends."""

    def __init__(self, turns):
        self.turns = list(turns)

    async def create(self, messages, tools):
        if self.turns:
            return self.turns.pop(0)
        return ModelResponse([{"type": "text", "text": "done"}], "end_turn")


def _coord(tmp_path, model, **kw) -> Coordinator:
    return Coordinator(
        model=model, permissions=kw.pop("permissions", _perms()),
        audit=AuditLog(tmp_path / "audit.jsonl"), **kw
    )


# --- 1. the permission path fails CLOSED, it does not crash -----------------------------
@pytest.mark.parametrize("kind", ["hook", "danger", "network"])
def test_permission_check_fails_closed_on_exception(kind):
    def boom(*_a, **_k):
        raise RuntimeError("policy backend unreachable")

    if kind == "hook":
        hooks = HookEngine()
        hooks.register(HookEvent.PRE_TOOL_USE, FunctionHook("boom", boom))
        engine = PermissionEngine(hooks, mode=Mode.AUTO)
    elif kind == "danger":
        engine = PermissionEngine(HookEngine(), mode=Mode.AUTO, danger_check=boom)
    else:
        class BadPolicy:
            def check(self, target):
                raise ConnectionError("policy service down")

        engine = PermissionEngine(HookEngine(), mode=Mode.AUTO, network_policy=BadPolicy())

    call = ToolCall("HttpRequest", {"url": "https://example.com"})
    for decision in (engine.check(call), asyncio.run(engine.check_async(call))):
        assert decision.behavior.value == "ask", "a broken gate must never resolve to allow"


def test_broken_hook_does_not_kill_the_run(tmp_path):
    hooks = HookEngine()
    hooks.register(HookEvent.PRE_TOOL_USE,
                   FunctionHook("boom", lambda i: (_ for _ in ()).throw(RuntimeError("x"))))
    coord = _coord(
        tmp_path,
        Scripted([ModelResponse(
            [{"type": "tool_use", "id": "t1", "name": "Echo", "input": {}}], "tool_use")]),
        permissions=PermissionEngine(hooks, mode=Mode.AUTO),
        native_tools={"Echo": lambda i: "hi"},
    )
    messages = asyncio.run(coord.run("go"))  # must not raise
    assert any(m.get("role") == "assistant" for m in messages)


# --- 2. a malformed tool_use block is survivable -----------------------------------------
def test_tool_use_without_id_does_not_crash(tmp_path):
    coord = _coord(
        tmp_path,
        Scripted([ModelResponse([{"type": "tool_use", "name": "Echo", "input": None}],
                                "tool_use")]),
        native_tools={"Echo": lambda i: "hi"},
    )
    messages = asyncio.run(coord.run("go"))  # used to raise KeyError: 'id'
    tool_use = messages[1]["content"][0]
    assert tool_use["id"], "a missing id must be synthesised, not fatal"
    assert messages[2]["content"][0]["tool_use_id"] == tool_use["id"]


# --- 3. a hung tool cannot hang the agent -------------------------------------------------
def test_tool_timeout_bounds_a_hanging_tool(tmp_path):
    async def hang(_inp):
        await asyncio.sleep(30)

    coord = _coord(
        tmp_path,
        Scripted([ModelResponse(
            [{"type": "tool_use", "id": "t1", "name": "Hang", "input": {}}], "tool_use")]),
        native_tools={"Hang": hang}, tool_timeout=0.2,
    )
    messages = asyncio.run(asyncio.wait_for(coord.run("go"), timeout=10))
    result = messages[2]["content"][0]
    assert result["is_error"] and "time limit" in result["content"]


def test_mcp_call_timeout(tmp_path):
    class HangServer:
        server_id = "slow"

        async def list_tools(self):
            return [{"name": "q", "description": "", "inputSchema": {}}]

        async def call_tool(self, n, a):
            await asyncio.sleep(30)

    async def go():
        h = MCPHandler(call_timeout=0.2)
        h.add_server(HangServer())
        await h.refresh()
        with pytest.raises(TimeoutError):
            await h.call_tool("slow__q", {})
        # a failed server's tools are withdrawn so the model stops being offered them
        assert "slow__q" not in h.tool_names()
        assert h.healthy()["slow"] is False

    asyncio.run(asyncio.wait_for(go(), timeout=10))


# --- 4. network policy survives a redirect -------------------------------------------------
def test_redirect_hop_is_policy_checked():
    hops = []

    def handler(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        if request.url.host == "api.trusted.com":
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
        return httpx.Response(200, text="ACCESS_KEY_ID=AKIA_SECRET")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    tool = make_http_request(
        network_policy=HostAllowlist(allow=("api.trusted.com",)), client=client)
    out = asyncio.run(tool({"url": "https://api.trusted.com/redirect"}))

    assert out.startswith("BLOCKED"), out
    assert "169.254.169.254" not in "".join(hops[1:]) or len(hops) == 1
    assert "AKIA_SECRET" not in out


def test_redirect_to_allowed_host_still_works():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/r":
            return httpx.Response(302, headers={"location": "https://api.trusted.com/final"})
        return httpx.Response(200, text="payload")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    tool = make_http_request(
        network_policy=HostAllowlist(allow=("api.trusted.com",)), client=client)
    out = asyncio.run(tool({"url": "https://api.trusted.com/r"}))
    assert "HTTP 200" in out and "payload" in out and "redirect" in out


def test_custom_destination_field_is_seen_by_the_policy():
    for key in ("webhook_url", "callbackUri", "api_endpoint", "targetHost"):
        targets = network_targets(ToolCall("Custom", {key: "https://evil.com"}))
        assert targets, f"{key} must be treated as a network destination"


# --- 5. the audit log redacts secrets ------------------------------------------------------
def test_audit_redacts_credentials(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.log_exchange(
        "POST", "https://api/pay",
        request={"method": "POST",
                 "headers": {"Authorization": "Bearer sk-live-SUPERSECRET123",
                             "X-Api-Key": "ak_live_9f3d", "Accept": "application/json"}},
        response={"status": 200, "body": "token=ghp_abcdefghijklmnopqrst uses AKIA1234567890AB"},
    )
    log.close()
    raw = (tmp_path / "a.jsonl").read_text(encoding="utf-8")
    for secret in ("sk-live-SUPERSECRET123", "ak_live_9f3d",
                   "ghp_abcdefghijklmnopqrst", "AKIA1234567890AB"):
        assert secret not in raw, f"{secret} leaked into the audit log"
    assert "Authorization" in raw, "the header NAME should remain visible"
    assert "application/json" in raw, "non-secret headers must survive"
    assert verify_audit_file(tmp_path / "a.jsonl").ok, "redaction must not break the chain"


# --- 6. concurrent writers keep the chain intact --------------------------------------------
def test_concurrent_audit_writers_do_not_fork_the_chain(tmp_path):
    path = tmp_path / "shared.jsonl"

    def worker():
        log = AuditLog(path)
        for i in range(15):
            log.log_decision("T", "allow", f"entry-{i}")
        log.close()

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    report = verify_audit_file(path)
    assert report.ok, f"chain forked under concurrency: {report.reason}"


# --- 7. recovery paths keep the transcript wire-valid ----------------------------------------
def _unanswered_tool_calls(messages) -> list[int]:
    """Indices of assistant turns whose tool_use blocks got no matching tool_result."""
    answered = {
        b.get("tool_use_id")
        for m in messages if isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    }
    bad = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b["id"] not in answered:
                bad.append(i)
    return bad


def test_max_tokens_recovery_answers_pending_tool_calls(tmp_path):
    coord = _coord(
        tmp_path,
        Scripted([ModelResponse(
            [{"type": "tool_use", "id": "t1", "name": "Echo", "input": {}}], "max_tokens")]),
        native_tools={"Echo": lambda i: "x"},
    )
    messages = asyncio.run(coord.run("go"))
    assert _unanswered_tool_calls(messages) == [], "provider would reject this transcript"


def test_loop_guard_nudge_answers_pending_tool_calls(tmp_path):
    from engine.loopguard import LoopGuard

    repeat = ModelResponse(
        [{"type": "text", "text": "checking\n" * 12},
         {"type": "tool_use", "id": "t1", "name": "Echo", "input": {}}], "tool_use")
    coord = _coord(
        tmp_path, Scripted([repeat]), native_tools={"Echo": lambda i: "x"},
        loop_guard=LoopGuard(), max_turns=4,
    )
    messages = asyncio.run(coord.run("go"))
    assert _unanswered_tool_calls(messages) == []


# --- 8. the provider adapter survives non-standard bodies -------------------------------------
def test_provider_error_body_raises_a_typed_error():
    with pytest.raises(ProviderResponseError, match="upstream overloaded"):
        parse_openai_response({"error": {"message": "upstream overloaded"}})
    with pytest.raises(ProviderResponseError):
        parse_openai_response({"choices": []})


def test_provider_handles_list_content_and_reasoning_only():
    r = parse_openai_response(
        {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]},
                      "finish_reason": "stop"}]})
    assert r.content == [{"type": "text", "text": "hi"}]

    r = parse_openai_response(
        {"choices": [{"message": {"content": None, "reasoning_content": "thought"},
                      "finish_reason": "stop"}]})
    assert r.content and r.content[0]["text"] == "thought"


# --- 9. usage is accumulated and a budget is enforced -------------------------------------------
def test_usage_accumulates_and_budget_stops_the_run(tmp_path):
    turn = ModelResponse(
        [{"type": "tool_use", "id": "t1", "name": "Echo", "input": {}}], "tool_use",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

    class Loop:
        async def create(self, messages, tools):
            return ModelResponse(list(turn.content), "tool_use", dict(turn.usage))

    coord = _coord(tmp_path, Loop(), native_tools={"Echo": lambda i: "x"},
                   max_turns=20, token_budget=400)
    asyncio.run(coord.run("go"))
    assert coord.usage["total_tokens"] >= 400
    assert coord.usage["model_calls"] == 3          # stopped at the budget, not at max_turns
    assert coord.completed is False


# --- 10. tool arguments are validated against the declared schema ---------------------------------
def test_invalid_tool_arguments_are_rejected_before_dispatch(tmp_path):
    seen = []
    spec = {"name": "Transfer", "description": "move money",
            "input_schema": {"type": "object",
                             "properties": {"amount": {"type": "integer"},
                                            "to": {"type": "string"}},
                             "required": ["amount", "to"]}}

    coord = _coord(
        tmp_path,
        Scripted([ModelResponse(
            [{"type": "tool_use", "id": "t1", "name": "Transfer",
              "input": {"amount": "ALL THE MONEY"}}], "tool_use")]),
        native_tools={"Transfer": lambda i: seen.append(i) or "ok"}, tool_specs=[spec],
    )
    messages = asyncio.run(coord.run("go"))
    result = messages[2]["content"][0]
    assert seen == [], "the handler must never see arguments that violate its schema"
    assert result["is_error"] and "INVALID ARGUMENTS" in result["content"]
    assert "missing required field 'to'" in result["content"]


def test_validator_accepts_valid_input_and_ignores_unknown_schemas():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    assert validate_tool_input({"n": 3}, schema) == []
    assert validate_tool_input({"n": True}, schema), "bool must not satisfy integer"
    assert validate_tool_input({"anything": 1}, None) == []


# --- 11. same-turn tool calls run concurrently ------------------------------------------------------
def test_same_turn_tools_run_in_parallel(tmp_path):
    async def slow(_inp):
        await asyncio.sleep(0.25)
        return "ok"

    blocks = [{"type": "tool_use", "id": f"t{i}", "name": "Slow", "input": {}} for i in range(4)]
    coord = _coord(tmp_path, Scripted([ModelResponse(blocks, "tool_use")]),
                   native_tools={"Slow": slow})

    async def timed():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await coord.run("go")
        return loop.time() - t0

    elapsed = asyncio.run(timed())
    assert elapsed < 0.6, f"4x0.25s ran in {elapsed:.2f}s — still sequential"


def test_results_keep_the_order_of_their_calls(tmp_path):
    async def echo(inp):
        await asyncio.sleep(0.05 * (3 - inp["i"]))  # finish in reverse order
        return f"r{inp['i']}"

    blocks = [{"type": "tool_use", "id": f"t{i}", "name": "E", "input": {"i": i}}
              for i in range(4)]
    coord = _coord(tmp_path, Scripted([ModelResponse(blocks, "tool_use")]),
                   native_tools={"E": echo})
    messages = asyncio.run(coord.run("go"))
    results = messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t0", "t1", "t2", "t3"]


# --- 12. a blocking hook does not stall the event loop ------------------------------------------------
def test_command_hook_does_not_block_the_event_loop():
    hooks = HookEngine()
    hooks.register(HookEvent.PRE_TOOL_USE,
                   CommandHook("slow", 'python -c "import time; time.sleep(1)"'))
    engine = PermissionEngine(hooks, mode=Mode.AUTO)

    async def go():
        ticks = []

        async def heartbeat():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.05)

        hb = asyncio.create_task(heartbeat())
        await engine.check_async(ToolCall("Echo", {}))
        hb.cancel()
        return len(ticks)

    ticks = asyncio.run(asyncio.wait_for(go(), timeout=20))
    assert ticks > 5, f"loop stalled during the hook (only {ticks} ticks)"


# --- 13. memory: body-inclusive recall, namespaces, delete, no silent overwrite -------------------------
def test_recall_searches_the_body(tmp_path):
    store = MemoryStore(tmp_path / "m")
    store.save(Memory("deploy-runbook", "how we ship", MemoryType.PROJECT,
                      "Production deploys go through Kubernetes cluster prod-eu-west-1."))
    assert [m.name for m in store.find_relevant("kubernetes")] == ["deploy-runbook"]
    assert [m.name for m in store.find_relevant("prod-eu-west-1")] == ["deploy-runbook"]


def test_name_match_outranks_a_body_mention(tmp_path):
    store = MemoryStore(tmp_path / "m")
    store.save(Memory("postgres", "the database", MemoryType.PROJECT, "version 16"))
    store.save(Memory("misc", "odds and ends", MemoryType.PROJECT, "we once used postgres"))
    assert [m.name for m in store.find_relevant("postgres")][0] == "postgres"


def test_memory_namespaces_are_isolated(tmp_path):
    a = MemoryStore(tmp_path / "m", namespace="tenant-a")
    b = MemoryStore(tmp_path / "m", namespace="tenant-b")
    a.save(Memory("secret-plan", "tenant a only", MemoryType.PROJECT, "acquire competitor"))
    assert a.find_relevant("secret-plan")
    assert b.find_relevant("secret-plan") == [], "memory leaked across tenants"


def test_memory_delete_and_collision_safety(tmp_path):
    store = MemoryStore(tmp_path / "m")
    store.save(Memory("Deploy Process", "prod steps", MemoryType.PROJECT, "FIRST"))
    store.save(Memory("deploy-process", "dev note", MemoryType.PROJECT, "SECOND"),
               overwrite=False)
    bodies = {m.body for m in store.all()}
    assert bodies == {"FIRST", "SECOND"}, "a colliding slug silently destroyed a memory"

    assert store.delete("deploy-process") is True
    assert store.delete("deploy-process") is False
    assert "deploy-process.md" not in (tmp_path / "m" / "MEMORY.md").read_text(encoding="utf-8")


# --- 14. MCP tool names are provider-legal ---------------------------------------------------------------
def test_mcp_tool_names_are_sanitised_and_still_route():
    called = {}

    class WeirdServer:
        server_id = "my-server.v2"

        async def list_tools(self):
            return [{"name": "do it now!", "description": "", "inputSchema": {}},
                    {"name": "x" * 80, "description": "", "inputSchema": {}}]

        async def call_tool(self, name, args):
            called["name"] = name
            return "ok"

    async def go():
        import re
        h = MCPHandler()
        h.add_server(WeirdServer())
        await h.refresh()
        for spec in h.get_tool_specs():
            assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec["name"]), spec["name"]
        target = [n for n in h.tool_names() if "do_it_now" in n][0]
        await h.call_tool(target, {})
        # the server must receive the name IT published, not the sanitised one
        assert called["name"] == "do it now!"

    asyncio.run(go())


# --- 15. session persistence is incremental ----------------------------------------------------------------
def test_session_save_is_incremental(tmp_path):
    store = SessionStore(tmp_path / "s.json")
    messages = []
    written = 0
    for _ in range(40):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "x" * 2000}]})
        before = tmp_path / "s.json"
        prev = before.stat().st_size if before.exists() else 0
        store.save(messages, done=False)
        written += before.stat().st_size - prev

    final = (tmp_path / "s.json").stat().st_size
    # An append-only log writes each message once: total growth ~= final size. The old
    # full-rewrite-per-turn cost ~20x that for this transcript.
    assert written < final * 1.5, f"wrote {written} bytes for a {final}-byte transcript"

    loaded = SessionStore(tmp_path / "s.json").load()
    assert loaded is not None and len(loaded[0]) == 40


def test_session_rewrites_when_the_transcript_is_replaced(tmp_path):
    store = SessionStore(tmp_path / "s.json")
    store.save([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    store.save([{"role": "user", "content": "compacted summary"}], done=True)  # compaction
    loaded = SessionStore(tmp_path / "s.json").load()
    assert loaded == ([{"role": "user", "content": "compacted summary"}], True)


def test_session_reads_v1_files(tmp_path):
    import json
    p = tmp_path / "s.json"
    p.write_text(json.dumps(
        {"version": 1, "done": False, "messages": [{"role": "user", "content": "hi"}]}),
        encoding="utf-8")
    assert SessionStore(p).load() == ([{"role": "user", "content": "hi"}], False)


# --- 16. ReadFile marks truncation ----------------------------------------------------------------------
def test_read_file_marks_truncation(tmp_path):
    from engine.sandbox import FilesystemGuard, FilesystemPolicy
    from engine.tools.files import make_read_file

    big = tmp_path / "big.txt"
    big.write_text("y" * 25000, encoding="utf-8")
    read = make_read_file(FilesystemGuard(FilesystemPolicy.build(tmp_path)))

    out = asyncio.run(read({"path": "big.txt"}))
    assert "[TRUNCATED" in out and "offset=20000" in out
    rest = asyncio.run(read({"path": "big.txt", "offset": 20000}))
    assert "[TRUNCATED" not in rest and "end of file" in rest


# --- 17. the agent facade releases what it opened ----------------------------------------------------------
def test_agent_is_an_async_context_manager(tmp_path):
    from engine.agent import Agent

    async def go():
        async with Agent(model=Scripted([]), workdir=tmp_path / "wd") as agent:
            assert await agent.run("hi") == "done"
            return agent

    agent = asyncio.run(go())
    assert agent._audit._fh.closed, "the audit file handle was left open"
