"""End-to-end: an OpenAI-compatible model (driven via httpx MockTransport) runs the full
coordinator loop, and the permission / network-policy / audit layers gate its tool calls.
No real network; proves the whole stack works over the OpenAI-compatible wire format.
"""

import asyncio
import json

import httpx

from engine.audit import AuditLog
from engine.coordinator import Coordinator
from engine.hooks import HookEngine
from engine.permissions import HostAllowlist, Mode, PermissionEngine
from engine.providers import ModelSpec, OpenAICompatClient


def _two_turn_handler():
    """Turn 1: model asks to hit an in-scope AND an out-of-scope host.
    Turn 2 (after it sees the DENIED tool result): it stops."""
    state = {"turn": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["turn"] += 1
        if state["turn"] == 1:
            return httpx.Response(200, json={"choices": [{
                "finish_reason": "tool_calls",
                "message": {"content": None, "tool_calls": [
                    {"id": "a", "type": "function", "function": {
                        "name": "HttpRequest", "arguments": '{"url":"https://api.acme.com/"}'}},
                    {"id": "b", "type": "function", "function": {
                        "name": "HttpRequest", "arguments": '{"url":"https://evil.example.com/"}'}},
                ]},
            }]})
        # turn 2: acknowledge and end
        body = json.loads(request.content)
        # sanity: the tool results were fed back to the model
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop",
            "message": {"content": "Stopped: evil.example.com is out of policy."},
        }]})

    return handler


def test_openai_compat_model_drives_gated_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    model = OpenAICompatClient(
        ModelSpec("deepseek", "deepseek-chat"),
        transport=httpx.MockTransport(_two_turn_handler()),
    )

    eng = PermissionEngine(HookEngine(), mode=Mode.AUTO,
                           network_policy=HostAllowlist(allow=("*.acme.com",)))
    audit = AuditLog(tmp_path / "audit.jsonl")
    ran = []

    async def http_get(inp):
        ran.append(inp["url"])
        return f"200 {inp['url']}"

    coord = Coordinator(
        model=model, permissions=eng, audit=audit,
        native_tools={"HttpRequest": http_get},
        tool_specs=[{"name": "HttpRequest", "description": "http",
                     "input_schema": {"type": "object",
                                      "properties": {"url": {"type": "string"}}}}],
    )

    messages = asyncio.run(coord.run("Fetch the two URLs"))

    # in-scope call executed; out-of-scope call never dialed
    assert ran == ["https://api.acme.com/"]
    # final assistant message is the model's end-turn text
    assert messages[-1]["role"] == "assistant"
    assert "out of policy" in messages[-1]["content"][0]["text"].lower()
    # audit captured the block
    assert any(e.kind == "blocked" for e in audit.read_all())
