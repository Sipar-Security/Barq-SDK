"""OpenAI-compatible model adapter (DeepSeek / Kimi / GLM).

Implements the coordinator's ModelClient Protocol by translating between our internal
(Anthropic-ish) message/tool block shapes and the OpenAI /chat/completions wire format,
then calling the endpoint over httpx.

The translation functions (to_openai_tools / to_openai_messages / parse_openai_response)
are pure and unit-tested offline. Only `create()` touches the network; it is exercised
against a live provider, not mocked into looking tested.

Internal shapes (produced/consumed by engine.coordinator):
  message: {"role":"user","content": str}
           {"role":"assistant","content": [ {type:text,text} | {type:tool_use,id,name,input} ]}
           {"role":"user","content": [ {type:tool_result, tool_use_id, content, is_error} ]}
  tool spec: {"name","description","input_schema"}
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from typing import Any

import httpx

from engine.coordinator import ModelResponse

from .base import ModelSpec

# HTTP statuses worth retrying: rate-limit + transient server/gateway errors. Other 4xx are
# client bugs (bad payload/auth) and must NOT be retried.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 520, 522, 524})
_MAX_ATTEMPTS = 5


def to_openai_tools(specs: list[dict]) -> list[dict]:
    out = []
    for s in specs:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s.get("input_schema") or {"type": "object"},
                },
            }
        )
    return out


def _assistant_to_openai(content: list[dict]) -> dict:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    msg: dict[str, Any] = {"role": "assistant"}
    msg["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def to_openai_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif role == "assistant":
            out.append(_assistant_to_openai(content))
        else:
            # user content that is a list => tool_result blocks -> role:tool messages
            for block in content:
                btype = block.get("type") if isinstance(block, dict) else None
                if btype == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": str(block.get("content", "")),
                        }
                    )
                elif btype == "text":
                    out.append({"role": "user", "content": str(block.get("text", ""))})
                elif btype in ("image", "image_url", "document", "audio"):
                    # This adapter is text-only. The old fallback `str(block)` put the
                    # Python repr of the dict into a user message, so a vision model got
                    # `{'type': 'image', 'source': {...}}` as prose and no error was raised
                    # anywhere — the request "succeeded" and the answer was nonsense.
                    # Failing here names the actual problem at the actual boundary.
                    raise ValueError(
                        f"OpenAICompatClient is text-only and cannot send a {btype!r} "
                        "block. Strip multimodal content before the call, or supply a "
                        "ModelClient that implements the provider's multimodal format."
                    )
                else:
                    out.append({"role": "user", "content": str(block)})
    return out


# How structured output is requested on the wire. "OpenAI-compatible" is not one dialect:
# `{"type": "json_schema"}` is supported by OpenAI and a growing set of others, plenty of
# endpoints support only `{"type": "json_object"}`, and some support neither and reject the
# field with a 400 that kills the run. `auto` starts at the strictest and DEGRADES on the
# provider's own error, so the caller does not have to maintain a capability table that
# goes stale every time a provider ships.
STRUCTURED_MODES = ("auto", "json_schema", "json_object", "none")
_DEGRADE = {"auto": "json_object", "json_schema": "json_object", "json_object": "none"}
# Substrings in a 4xx body that mean "I do not support that response_format".
_UNSUPPORTED_FORMAT_HINTS = (
    "response_format", "json_schema", "json schema", "unsupported", "not supported",
    "invalid_request_error", "unrecognized", "unknown field", "unexpected keyword",
)


def _normalise_response_format(response_format: dict, mode: str) -> dict | None:
    """Shape a `json_schema` request for whatever dialect `mode` selects."""
    if mode == "none":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    schema_block = dict(response_format.get("json_schema") or {})
    schema_block.setdefault("name", "output")
    # `strict` requires the schema to be closed; sending it on a schema that is not makes
    # the provider reject the request rather than relax the constraint.
    if schema_block.get("schema", {}).get("additionalProperties") is False:
        schema_block.setdefault("strict", True)
    return {"type": "json_schema", "json_schema": schema_block}


def _looks_like_unsupported_format(body: str) -> bool:
    lowered = (body or "").lower()
    return any(hint in lowered for hint in _UNSUPPORTED_FORMAT_HINTS)


class ProviderResponseError(RuntimeError):
    """The endpoint returned a body we cannot read as a completion.

    Aggregators (OpenRouter, ZenMux, and gateways generally) routinely answer HTTP 200 with
    an error object instead of `choices` (so status-code retry logic never sees it). Indexing
    `data["choices"][0]` blindly turned that into a KeyError/IndexError escaping create() and
    killing the run. Raising a typed error lets the caller decide, and keeps the message.
    """


def _content_to_text(content: Any) -> str:
    """Flatten a message `content` to text. Most providers send a string, but the multimodal
    shape is a list of parts: assigning that list into a text block silently corrupted it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            elif p is not None:
                parts.append(str(p))
        return "".join(parts)
    return "" if content is None else str(content)


def parse_openai_response(data: dict) -> ModelResponse:
    """Convert a /chat/completions JSON response into our internal ModelResponse."""
    if not isinstance(data, dict):
        raise ProviderResponseError(f"expected a JSON object, got {type(data).__name__}")
    choices = data.get("choices")
    if not choices:
        # Surface the provider's own error text; it is the only diagnostic there is.
        err = data.get("error")
        if err is not None:
            detail = err.get("message") if isinstance(err, dict) else str(err)
            raise ProviderResponseError(f"provider returned an error: {detail}")
        raise ProviderResponseError(
            f"response contained no choices (keys: {sorted(data)[:8]})"
        )
    choice = choices[0] if isinstance(choices, list) else choices
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    finish = choice.get("finish_reason", "stop") if isinstance(choice, dict) else "stop"

    blocks: list[dict] = []
    text = _content_to_text(msg.get("content"))
    if text:
        blocks.append({"type": "text", "text": text})
    elif not msg.get("tool_calls"):
        # A reasoning model (DeepSeek R1 and kin) can return its chain of thought in
        # `reasoning_content` with an empty `content`. Dropping it silently produced an
        # empty assistant turn the loop could not act on or recover from.
        reasoning = _content_to_text(msg.get("reasoning_content"))
        if reasoning:
            blocks.append({"type": "text", "text": reasoning})
    used_ids: set[str] = set()
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw_arguments": fn.get("arguments", "")}
        # The coordinator's exactly-once journal is keyed by this id. Some providers return an
        # empty or repeated tool_call id; guarantee a non-empty, response-unique id so two distinct
        # tool calls never share a key (an empty tool_call_id is also ambiguous on the wire, which
        # otherwise lets one call's cached result be served for another, e.g. ReadFile returning
        # the wrong file).
        tid = str(tc.get("id") or "").strip()
        if not tid:
            tid = "call_" + uuid.uuid4().hex[:16]
        while tid in used_ids:
            tid = f"{tid}_{i}"
        used_ids.add(tid)
        blocks.append(
            {
                "type": "tool_use",
                "id": tid,
                "name": fn.get("name", ""),
                "input": args,
            }
        )

    # Preserve token exhaustion as a distinct terminal condition. Collapsing `length` into
    # `end_turn` makes a reasoning model that spent its whole allowance on hidden reasoning
    # look as though it deliberately finished the assessment.
    stop_reason = (
        "tool_use" if finish == "tool_calls"
        else "max_tokens" if finish == "length"
        else "end_turn"
    )
    u = data.get("usage") or {}
    usage = {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
        "total_tokens": int(u.get("total_tokens", 0) or 0),
    }
    # Reasoning models (Kimi K2.x, o-series) spend hidden tokens counted inside completion_tokens
    # and reported in completion_tokens_details.reasoning_tokens. Surface it so cost/latency of the
    # thinking is observable (billing + the empty-turn/budget diagnostics), not silently folded in.
    details = u.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens", 0) or 0)
    if reasoning:
        usage["reasoning_tokens"] = reasoning
    return ModelResponse(content=blocks, stop_reason=stop_reason, usage=usage)


class OpenAICompatClient:
    """ModelClient over any OpenAI-compatible endpoint (DeepSeek/Kimi/GLM)."""

    def __init__(
        self,
        spec: ModelSpec,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        *,
        system_prompt: str | None = None,
        structured_output_mode: str = "auto",
    ) -> None:
        if structured_output_mode not in STRUCTURED_MODES:
            raise ValueError(
                f"structured_output_mode must be one of {STRUCTURED_MODES}, "
                f"got {structured_output_mode!r}"
            )
        # Degrades at runtime when the endpoint rejects the dialect (see create()), and the
        # downgrade STICKS for the life of the client so one 400 is paid once per process
        # rather than once per turn.
        self.structured_output_mode = structured_output_mode
        self.spec = spec
        self._url = spec.resolved_base_url().rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {spec.api_key()}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout
        # transport is a test/ops seam: pass httpx.MockTransport in tests, or a
        # proxy-aware transport in production so model traffic also honors egress rules.
        self._transport = transport
        # A real system message, sent as the first turn of every request. Concatenating the
        # preamble onto the first USER message instead (the old behaviour) means the prefix
        # is not stable across a conversation, so prompt caching (the single largest cost
        # lever on a long agent run) can never engage.
        self.system_prompt = system_prompt
        # Cumulative usage across every call this client makes. Subagent metering reads
        # `usage_total`; before this the attribute did not exist, so it always read {}.
        self.usage_total: dict[str, int] = {}
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        """One pooled client for the life of this object. Building an AsyncClient per request
        forced a fresh TLS handshake on every model call."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _track(self, usage: dict) -> None:
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)):
                self.usage_total[k] = self.usage_total.get(k, 0) + int(v)
        self.usage_total["model_calls"] = self.usage_total.get("model_calls", 0) + 1

    def last_payload(
        self,
        messages: list[dict],
        tools: list[dict],
        response_format: dict | None = None,
    ) -> dict:
        wire = to_openai_messages(messages)
        if self.system_prompt and not (wire and wire[0].get("role") == "system"):
            wire = [{"role": "system", "content": self.system_prompt}] + wire
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": wire,
            "temperature": self.spec.temperature,
        }
        if self.spec.max_tokens:
            payload["max_tokens"] = int(self.spec.max_tokens)
        if tools:
            payload["tools"] = to_openai_tools(tools)
            payload["tool_choice"] = "auto"
        if response_format is not None:
            shaped = _normalise_response_format(
                response_format, self.structured_output_mode
            )
            # `none` means send nothing. Setting the key to null still sends the field, and
            # an endpoint that rejects `response_format` rejects it just as hard when the
            # value is null — so the degradation would never bottom out.
            if shaped is not None:
                payload["response_format"] = shaped
        return payload

    async def _backoff(self, attempt: int, resp: httpx.Response | None = None) -> None:
        """Exponential backoff with jitter; honor a Retry-After header if the server sent one."""
        delay = min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0.0, 0.3)
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                try:
                    delay = max(delay, float(ra))
                except ValueError:
                    pass
        await asyncio.sleep(delay)

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        response_format: dict | None = None,
    ):
        """Yield text deltas as they arrive, then the assembled `ModelResponse` last.

        The coordinator uses this only when something is consuming events; a caller that
        just awaits the transcript still goes through `create()`. Streaming is what makes a
        token-by-token UI and an early abort possible at all — with a single blocking
        `create()` there is nothing to show and nothing to interrupt.

        Deltas are reassembled here rather than pushed at the caller, so the loop receives
        exactly the same `ModelResponse` shape either way and no downstream code has to
        know which path was taken.
        """
        payload = {**self.last_payload(messages, tools, response_format), "stream": True,
                   "stream_options": {"include_usage": True}}
        text_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish = "stop"
        usage: dict = {}
        client = self._http()
        async with client.stream(
            "POST", self._url, headers=self._headers, json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                piece = _content_to_text(delta.get("content"))
                if piece:
                    text_parts.append(piece)
                    yield piece
                # Tool-call arguments arrive in fragments keyed by index, not by id.
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

        # Rebuild the non-streamed response shape and reuse the ONE parser, so a fix to
        # response handling cannot land in the blocking path and be missed here.
        message: dict[str, Any] = {"content": "".join(text_parts)}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["args"] or "{}"},
                }
                for _, slot in sorted(tool_calls.items())
            ]
        parsed = parse_openai_response({
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": usage,
        })
        self._track(parsed.usage)
        yield parsed

    async def create(
        self,
        messages: list[dict],
        tools: list[dict],
        response_format: dict | None = None,
    ) -> ModelResponse:
        """POST to the provider with retry on transient failures.

        Retries rate-limit (429) and transient server/gateway errors (5xx), and network
        transport errors (connection resets, read/connect timeouts, e.g. httpx.ReadError),
        with exponential backoff. A non-retryable 4xx (bad payload/auth) raises immediately;
        after the last attempt the final error is raised. This is what makes a long, tool-heavy
        long, tool-heavy run survive a blip from the model API instead of crashing mid-run.
        """
        payload = self.last_payload(messages, tools, response_format)
        client = self._http()
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = await client.post(self._url, headers=self._headers, json=payload)
                if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
                    await self._backoff(attempt, resp)
                    continue
                # A 4xx naming `response_format` means this endpoint does not speak the
                # dialect we asked for. Retry in the next dialect down rather than failing
                # the run: "OpenAI-compatible" covers endpoints that support json_schema,
                # only json_object, or neither, and a hard failure here would make
                # structured output unusable on most of them.
                if (
                    response_format is not None
                    and 400 <= resp.status_code < 500
                    and self.structured_output_mode != "none"
                    and _looks_like_unsupported_format(resp.text)
                ):
                    self.structured_output_mode = _DEGRADE.get(
                        self.structured_output_mode, "none"
                    )
                    payload = self.last_payload(messages, tools, response_format)
                    continue
                resp.raise_for_status()  # non-retryable 4xx (or final retryable) -> raise
                parsed = parse_openai_response(resp.json())
                self._track(parsed.usage)
                return parsed
            except httpx.TransportError as e:  # ReadError/ConnectError/timeouts: transient
                if attempt < _MAX_ATTEMPTS - 1:
                    await self._backoff(attempt)
                    continue
                raise
        # Unreachable: every branch above returns, raises, or continues, and `continue` only
        # happens while attempt < _MAX_ATTEMPTS - 1. Kept as an explicit assertion rather than
        # the dead "one more request" block that used to sit here.
        raise AssertionError("retry loop exited without a result")
