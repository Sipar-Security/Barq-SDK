import asyncio
import json

import httpx
import pytest

from engine.coordinator import ModelResponse
from engine.providers import (
    ModelRole,
    ModelRouter,
    ModelSpec,
    OpenAICompatClient,
    parse_openai_response,
    to_openai_messages,
    to_openai_tools,
)


def test_to_openai_tools_shape():
    specs = [{"name": "search_graph", "description": "d", "input_schema": {"type": "object"}}]
    out = to_openai_tools(specs)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "search_graph"
    assert out[0]["function"]["parameters"] == {"type": "object"}


def test_to_openai_messages_translates_blocks():
    internal = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "call_1", "name": "HttpRequest", "input": {"url": "https://api.acme.com/"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "200 OK", "is_error": False},
        ]},
    ]
    out = to_openai_messages(internal)
    assert out[0] == {"role": "user", "content": "do the task"}
    asst = out[1]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"][0]["id"] == "call_1"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"url": "https://api.acme.com/"}
    tool_msg = out[2]
    assert tool_msg == {"role": "tool", "tool_call_id": "call_1", "content": "200 OK"}


def test_parse_response_tool_calls():
    data = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_9", "type": "function",
                    "function": {"name": "Nuclei", "arguments": '{"target": "api.acme.com"}'},
                }],
            },
        }]
    }
    resp = parse_openai_response(data)
    assert isinstance(resp, ModelResponse)
    assert resp.stop_reason == "tool_use"
    assert resp.content[0] == {"type": "tool_use", "id": "call_9", "name": "Nuclei", "input": {"target": "api.acme.com"}}


def test_parse_response_plain_text_end_turn():
    data = {"choices": [{"finish_reason": "stop", "message": {"content": "no bugs found"}}]}
    resp = parse_openai_response(data)
    assert resp.stop_reason == "end_turn"
    assert resp.content[0] == {"type": "text", "text": "no bugs found"}


def test_router_requires_all_roles():
    spec = ModelSpec(provider="deepseek", model="deepseek-chat")
    with pytest.raises(ValueError):
        ModelRouter({ModelRole.FAST: spec})  # missing SMART
    r = ModelRouter({
        ModelRole.FAST: ModelSpec("deepseek", "deepseek-chat"),
        ModelRole.SMART: ModelSpec("zhipu", "glm-4.6"),
    })
    assert r.spec(ModelRole.SMART).provider == "zhipu"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ModelSpec("deepseek", "deepseek-chat").api_key()


def test_create_full_path_with_mock_transport(monkeypatch):
    # Full create() path (payload build -> HTTP -> parse) without network, via httpx
    # MockTransport. This is a transport seam, not a stubbed model response object.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "HttpRequest", "arguments": '{"url":"https://api.acme.com/"}'},
                }]},
            }]
        })

    spec = ModelSpec("deepseek", "deepseek-chat")
    client = OpenAICompatClient(spec, transport=httpx.MockTransport(handler))
    resp = asyncio.run(client.create(
        messages=[{"role": "user", "content": "go"}],
        tools=[{"name": "HttpRequest", "description": "http", "input_schema": {"type": "object"}}],
    ))
    assert resp.stop_reason == "tool_use"
    assert resp.content[0]["name"] == "HttpRequest"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "deepseek-chat"
    assert captured["body"]["tools"][0]["function"]["name"] == "HttpRequest"


def test_max_tokens_in_payload_when_set(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    from engine.providers import ModelSpec
    from engine.providers.openai_compat import OpenAICompatClient
    # default: no cap -> no max_tokens key
    c0 = OpenAICompatClient(ModelSpec("deepseek", "deepseek-chat"))
    assert "max_tokens" not in c0.last_payload([{"role": "user", "content": "hi"}], [])
    # capped: the payload carries the bound so one degenerate response can't run unbounded
    c1 = OpenAICompatClient(ModelSpec("deepseek", "deepseek-chat", max_tokens=8192))
    p = c1.last_payload([{"role": "user", "content": "hi"}], [])
    assert p["max_tokens"] == 8192


def test_length_finish_reason_is_not_a_clean_end_turn():
    from engine.providers.openai_compat import parse_openai_response

    response = parse_openai_response({
        "choices": [{"finish_reason": "length", "message": {"content": ""}}],
        "usage": {"completion_tokens": 8192},
    })

    assert response.stop_reason == "max_tokens"
