"""Structured output: a run returns a validated object, not prose.

`Agent.run()` returned a bare string, so every integration needing a field out of an agent
re-parsed free text, and there was no point at which the engine could say the model got the
shape wrong — the failure surfaced downstream, in the caller's code, on data nobody checked.

The three pieces tested here:
  * schema derivation from a dict / Pydantic model / dataclass / TypedDict, with NO new
    dependency (Pydantic is duck-typed, never imported);
  * JSON recovery from a model's actual output, which is routinely fenced and prefaced;
  * a bounded repair loop that hands validation errors back rather than failing the run.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Optional, TypedDict

import httpx
import pytest

from engine import Agent, ModelResponse, OutputValidationError
from engine.structured import build, extract_json, repair_prompt, schema_for, type_name


# --- fixtures ----------------------------------------------------------------
@dataclasses.dataclass
class Triage:
    severity: str
    summary: str
    confidence: float = 0.0


class Incident(TypedDict):
    title: str
    count: int


class FakePydantic:
    """Duck-typed Pydantic v2 stand-in. Pydantic is NOT a dependency of this SDK, so the
    support has to work through the protocol rather than an import."""

    def __init__(self, **data):
        self.data = data

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

    @classmethod
    def model_validate(cls, data):
        return cls(**data)


class Script:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def create(self, messages, tools):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse([{"type": "text", "text": "end"}], "end_turn", {})


def text(body: str) -> ModelResponse:
    return ModelResponse([{"type": "text", "text": body}], "end_turn", {})


TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {"severity": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["severity", "count"],
}


# --- schema derivation --------------------------------------------------------
def test_a_schema_dict_is_used_verbatim():
    assert schema_for(TRIAGE_SCHEMA) is TRIAGE_SCHEMA


def test_schema_from_a_dataclass():
    schema = schema_for(Triage)
    assert schema["type"] == "object"
    assert schema["properties"]["severity"] == {"type": "string"}
    assert schema["properties"]["confidence"] == {"type": "number"}
    # a field with a default is not required
    assert set(schema["required"]) == {"severity", "summary"}


def test_schema_from_a_typed_dict():
    schema = schema_for(Incident)
    assert schema["properties"]["count"] == {"type": "integer"}
    assert set(schema["required"]) == {"title", "count"}


def test_schema_from_a_pydantic_like_model_without_importing_pydantic():
    assert schema_for(FakePydantic)["required"] == ["name"]


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (bool, {"type": "boolean"}),
        (list[str], {"type": "array", "items": {"type": "string"}}),
    ],
)
def test_schema_from_primitives(annotation, expected):
    assert schema_for(annotation) == expected


def test_optional_becomes_a_nullable_type():
    @dataclasses.dataclass
    class WithOptional:
        note: Optional[str]

    assert schema_for(WithOptional)["properties"]["note"]["type"] == ["string", "null"]


def test_nested_dataclass_is_expanded():
    @dataclasses.dataclass
    class Outer:
        inner: Triage

    assert schema_for(Outer)["properties"]["inner"]["properties"]["severity"] == {
        "type": "string"
    }


def test_an_underivable_type_is_refused_loudly():
    with pytest.raises(TypeError, match="cannot derive a JSON Schema"):
        schema_for(object())


def test_type_name_is_provider_safe():
    assert type_name(Triage) == "Triage"
    assert type_name({"title": "My Output!"}) == "My_Output_"
    assert type_name({}) == "output"


# --- recovering the JSON from real model output --------------------------------
@pytest.mark.parametrize(
    "body",
    [
        '{"severity":"high","count":2}',
        '  {"severity":"high","count":2}  ',
        '```json\n{"severity":"high","count":2}\n```',
        '```\n{"severity":"high","count":2}\n```',
        'Here is the result:\n\n{"severity":"high","count":2}',
        'Sure!\n```json\n{"severity":"high","count":2}\n```\nLet me know if that helps.',
    ],
)
def test_json_is_recovered_from_prose_and_fences(body):
    """Models fence and preface their answers. Refusing to parse that would make the
    feature fail on its most common success case."""
    value, error = extract_json(body)
    assert error is None
    assert value == {"severity": "high", "count": 2}


def test_nested_braces_do_not_confuse_the_extractor():
    value, error = extract_json('note\n{"a": {"b": [1, 2]}, "c": "}"}\ntrailing')
    assert error is None and value == {"a": {"b": [1, 2]}, "c": "}"}


def test_a_brace_inside_a_string_is_not_a_delimiter():
    value, _ = extract_json('{"text": "a { b } c"}')
    assert value == {"text": "a { b } c"}


def test_an_escaped_quote_inside_a_string_is_handled():
    value, _ = extract_json(r'{"text": "he said \"hi\" {"}')
    assert value == {"text": 'he said "hi" {'}


def test_an_array_result_is_recovered():
    value, error = extract_json('```json\n[1, 2, 3]\n```')
    assert error is None and value == [1, 2, 3]


@pytest.mark.parametrize("body", ["", "   ", "no json here at all"])
def test_unparseable_output_reports_an_error(body):
    value, error = extract_json(body)
    assert value is None and error


# --- building the caller's own type --------------------------------------------
def test_a_dataclass_comes_back_as_an_instance():
    got = build(Triage, {"severity": "high", "summary": "s", "confidence": 0.5})
    assert isinstance(got, Triage) and got.severity == "high"


def test_extra_keys_do_not_break_dataclass_construction():
    got = build(Triage, {"severity": "high", "summary": "s", "unexpected": 1})
    assert isinstance(got, Triage)


def test_a_pydantic_like_model_is_validated_through_its_own_method():
    got = build(FakePydantic, {"name": "x"})
    assert isinstance(got, FakePydantic) and got.data == {"name": "x"}


def test_a_plain_schema_dict_yields_plain_data():
    assert build(TRIAGE_SCHEMA, {"severity": "high"}) == {"severity": "high"}


def test_repair_prompt_names_the_actual_errors():
    prompt = repair_prompt(["severity: expected string, got integer"], TRIAGE_SCHEMA)
    assert "severity: expected string" in prompt
    assert "ONLY the corrected JSON" in prompt


# --- end to end through the Agent ----------------------------------------------
@pytest.mark.asyncio
async def test_run_returns_the_validated_object(tmp_path):
    agent = Agent(
        model=Script([text('{"severity":"high","count":2}')]),
        workdir=tmp_path, output_type=TRIAGE_SCHEMA,
        enable_file_tools=False, enable_memory_tools=False,
    )
    result = await agent.run("classify")
    assert result == {"severity": "high", "count": 2}
    assert agent.output == result
    assert agent.output_error == ""
    assert agent.ok is True
    await agent.aclose()


@pytest.mark.asyncio
async def test_run_returns_a_dataclass_instance(tmp_path):
    agent = Agent(
        model=Script([text('{"severity":"high","summary":"disk full","confidence":0.9}')]),
        workdir=tmp_path, output_type=Triage,
        enable_file_tools=False, enable_memory_tools=False,
    )
    result = await agent.run("classify")
    assert isinstance(result, Triage)
    assert result.severity == "high" and result.confidence == 0.9
    await agent.aclose()


@pytest.mark.asyncio
async def test_invalid_output_is_handed_back_with_the_errors_and_repaired(tmp_path):
    """The repair loop is the whole point: a bare 'try again' converts far less than an
    error list the model can act on."""
    model = Script([
        text('{"severity":"high"}'),                    # missing `count`
        text('{"severity":"high","count":"two"}'),      # wrong type
        text('{"severity":"high","count":2}'),          # correct
    ])
    agent = Agent(
        model=model, workdir=tmp_path, output_type=TRIAGE_SCHEMA, output_retries=2,
        enable_file_tools=False, enable_memory_tools=False,
    )
    result = await agent.run("classify")
    assert result == {"severity": "high", "count": 2}
    assert model.calls == 3
    await agent.aclose()


@pytest.mark.asyncio
async def test_the_repair_message_carries_the_specific_error(tmp_path):
    seen: list[str] = []

    class Recording(Script):
        async def create(self, messages, tools):
            seen.append(json.dumps(messages))
            return await super().create(messages, tools)

    agent = Agent(
        model=Recording([text('{"severity":"high"}'), text('{"severity":"h","count":1}')]),
        workdir=tmp_path, output_type=TRIAGE_SCHEMA,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("classify")
    assert "missing required field 'count'" in seen[1] or "count" in seen[1]
    await agent.aclose()


@pytest.mark.asyncio
async def test_retries_are_bounded_and_the_failure_is_reported(tmp_path):
    # Distinct bodies: three IDENTICAL turns are a degenerate loop and the loop guard
    # correctly stops the run, which would be testing a different mechanism.
    model = Script([text("not json"), text("still not json"), text("nope, prose")])
    agent = Agent(
        model=model, workdir=tmp_path, output_type=TRIAGE_SCHEMA, output_retries=2,
        enable_file_tools=False, enable_memory_tools=False,
    )
    with pytest.raises(OutputValidationError, match="repair attempt"):
        await agent.run("classify")
    assert agent.output_error
    assert agent.ok is False
    assert model.calls == 3  # the original plus two repairs, not unbounded
    await agent.aclose()


@pytest.mark.asyncio
async def test_a_run_that_never_reaches_validation_still_reports_why(tmp_path):
    """`_drive` has several early exits — cancellation, a loop-guard abort, max_turns, the
    token budget — and none reach the validation step. Returning None with an empty
    `output_error` would be the same silent failure `failures` exists to close."""
    spec = {"name": "T", "description": "d", "input_schema": {"type": "object"}}
    tool_turn = ModelResponse(
        [{"type": "tool_use", "id": "1", "name": "T", "input": {"n": 1}}], "tool_use", {}
    )
    agent = Agent(
        model=Script([tool_turn, tool_turn, tool_turn]),
        workdir=tmp_path, output_type=TRIAGE_SCHEMA, max_turns=2,
        tools=[(spec, lambda i: "ok")], loop_guard=False,
        enable_file_tools=False, enable_memory_tools=False,
    )
    with pytest.raises(OutputValidationError, match="truncated"):
        await agent.run("classify")
    assert "max_turns" in agent.output_error
    await agent.aclose()


@pytest.mark.asyncio
async def test_output_retries_zero_means_one_attempt(tmp_path):
    model = Script([text("nope")] * 4)
    agent = Agent(
        model=model, workdir=tmp_path, output_type=TRIAGE_SCHEMA, output_retries=0,
        enable_file_tools=False, enable_memory_tools=False,
    )
    with pytest.raises(OutputValidationError):
        await agent.run("classify")
    assert model.calls == 1
    await agent.aclose()


@pytest.mark.asyncio
async def test_the_schema_reaches_the_system_prompt(tmp_path):
    """Most OpenAI-compatible endpoints support neither json_schema nor json_object, so on
    those the prompt is the ONLY thing carrying the requirement."""
    agent = Agent(
        model=Script([text('{"severity":"x","count":1}')]),
        workdir=tmp_path, output_type=TRIAGE_SCHEMA,
        enable_file_tools=False, enable_memory_tools=False,
    )
    system = agent._effective_system()
    assert "must be a single JSON value" in system
    assert '"severity"' in system
    await agent.aclose()


@pytest.mark.asyncio
async def test_an_existing_system_prompt_is_preserved(tmp_path):
    agent = Agent(
        model=Script([text("{}")]), workdir=tmp_path, output_type={"type": "object"},
        system_prompt="You are a triage bot.",
        enable_file_tools=False, enable_memory_tools=False,
    )
    system = agent._effective_system()
    assert system.startswith("You are a triage bot.")
    assert "single JSON value" in system
    await agent.aclose()


@pytest.mark.asyncio
async def test_no_output_type_is_completely_unchanged(tmp_path):
    """Backwards compatibility: a caller that did not ask for structure gets the string."""
    agent = Agent(
        model=Script([text("just some prose")]), workdir=tmp_path,
        enable_file_tools=False, enable_memory_tools=False,
    )
    assert await agent.run("hi") == "just some prose"
    assert agent.output is None and agent.output_error == ""
    assert "JSON value" not in (agent._effective_system() or "")
    await agent.aclose()


@pytest.mark.asyncio
async def test_tools_still_run_before_the_structured_answer(tmp_path):
    """Structured output must not short-circuit the tool loop."""
    spec = {"name": "T", "description": "d", "input_schema": {"type": "object"}}
    model = Script([
        ModelResponse([{"type": "tool_use", "id": "1", "name": "T", "input": {}}],
                      "tool_use", {}),
        text('{"severity":"high","count":7}'),
    ])
    calls = {"n": 0}

    def tool(inp):
        calls["n"] += 1
        return "7 incidents"

    agent = Agent(
        model=model, workdir=tmp_path, output_type=TRIAGE_SCHEMA, tools=[(spec, tool)],
        enable_file_tools=False, enable_memory_tools=False,
    )
    result = await agent.run("count them")
    assert calls["n"] == 1
    assert result == {"severity": "high", "count": 7}
    await agent.aclose()


@pytest.mark.asyncio
async def test_send_also_honours_the_output_type(tmp_path):
    agent = Agent(
        model=Script([text('{"severity":"low","count":0}')]),
        workdir=tmp_path, output_type=TRIAGE_SCHEMA,
        enable_file_tools=False, enable_memory_tools=False,
    )
    assert await agent.send("classify") == {"severity": "low", "count": 0}
    await agent.aclose()


# --- the provider wire format ---------------------------------------------------
def payloads_for(handler):
    from engine.providers import ModelSpec
    from engine.providers.openai_compat import OpenAICompatClient

    spec = ModelSpec(provider="deepseek", model="m", base_url="https://x.test")
    return OpenAICompatClient(
        spec, transport=httpx.MockTransport(handler),
    )


def _completion(body: str = '{"ok":1}') -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": body}, "finish_reason": "stop"}],
            "usage": {},
        },
    )


@pytest.mark.asyncio
async def test_response_format_is_sent_as_json_schema(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _completion()

    client = payloads_for(handler)
    await client.create(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "Out", "schema": TRIAGE_SCHEMA}},
    )
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["response_format"]["json_schema"]["name"] == "Out"
    await client.aclose()


@pytest.mark.asyncio
async def test_an_endpoint_that_rejects_json_schema_is_degraded_not_failed(monkeypatch):
    """"OpenAI-compatible" is not one dialect. A hard failure here would make structured
    output unusable on most endpoints."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        rf = payload.get("response_format", {})
        if rf.get("type") == "json_schema":
            return httpx.Response(
                400, json={"error": {"message": "response_format json_schema is not supported"}}
            )
        return _completion()

    client = payloads_for(handler)
    result = await client.create(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        response_format={"type": "json_schema", "json_schema": {"schema": TRIAGE_SCHEMA}},
    )
    assert result.content[0]["text"] == '{"ok":1}'
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[1]["response_format"]["type"] == "json_object"
    # …and the downgrade sticks, so the 400 is paid once per process, not once per turn.
    assert client.structured_output_mode == "json_object"
    await client.aclose()


@pytest.mark.asyncio
async def test_degradation_bottoms_out_at_sending_nothing(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": {"message": "unknown field response_format"}})
        return _completion()

    client = payloads_for(handler)
    await client.create(
        messages=[{"role": "user", "content": "hi"}], tools=[],
        response_format={"type": "json_schema", "json_schema": {"schema": TRIAGE_SCHEMA}},
    )
    assert "response_format" not in sent[-1]
    assert client.structured_output_mode == "none"
    await client.aclose()


@pytest.mark.asyncio
async def test_a_genuine_400_is_still_raised(monkeypatch):
    """Degradation must not swallow an unrelated client error."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "context length exceeded"}})

    client = payloads_for(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.create(
            messages=[{"role": "user", "content": "hi"}], tools=[],
            response_format={"type": "json_schema", "json_schema": {"schema": TRIAGE_SCHEMA}},
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_no_response_format_is_sent_when_none_is_requested(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _completion()

    client = payloads_for(handler)
    await client.create(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert "response_format" not in sent[0]
    await client.aclose()


@pytest.mark.asyncio
async def test_a_client_that_predates_response_format_still_works(tmp_path):
    """The ModelClient Protocol requires only (messages, tools). Every custom client
    written before this feature must keep working."""
    class Legacy:
        def __init__(self):
            self.calls = 0

        async def create(self, messages, tools):  # no response_format parameter
            self.calls += 1
            return text('{"severity":"high","count":1}')

    model = Legacy()
    agent = Agent(
        model=model, workdir=tmp_path, output_type=TRIAGE_SCHEMA,
        enable_file_tools=False, enable_memory_tools=False,
    )
    assert await agent.run("go") == {"severity": "high", "count": 1}
    assert model.calls == 1
    await agent.aclose()
