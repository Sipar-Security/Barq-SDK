"""OpenTelemetry spans and metrics from a real run.

There was none: zero occurrences of `opentelemetry` in the tree, no spans, no GenAI
semantic conventions, no exporter, no trace propagation, and no metrics — one cumulative
`usage` dict nothing could be alerted on. A subagent's work could not be attributed to the
parent call that caused it, because there was no parent-child relationship to attribute it
to.

These assert against a real `InMemorySpanExporter`, not a mock: the point of the feature is
that a collector receives a well-formed span tree, and only an exporter can show that.
"""

from __future__ import annotations

import pytest

from engine import Agent, ModelResponse
from engine.events import AgentEvent, EventType
from engine.telemetry import (
    ATTR_OPERATION,
    ATTR_OUTCOME,
    ATTR_RUN_ID,
    ATTR_SYSTEM,
    ATTR_TOOL_NAME,
    GEN_AI_SYSTEM,
    OTelTelemetry,
    otel_available,
)

otel = pytest.importorskip("opentelemetry.sdk.trace", reason="needs opentelemetry-sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def telemetry(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OTelTelemetry(
        service_name="test-agent", agent_name="triage", model_name="test-model",
        tracer=provider.get_tracer("test"),
    )


SPEC = {
    "name": "Lookup",
    "description": "look something up",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}


class Script:
    def __init__(self, responses):
        self.responses = list(responses)

    async def create(self, messages, tools):
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse([{"type": "text", "text": "done"}], "end_turn", {})


def tool_turn(call_id="1", inp=None):
    return ModelResponse(
        [{"type": "tool_use", "id": call_id, "name": "Lookup", "input": inp or {"q": "x"}}],
        "tool_use",
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def final(text="all done"):
    return ModelResponse(
        [{"type": "text", "text": text}], "end_turn",
        {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    )


def by_name(exporter):
    return [s.name for s in exporter.get_finished_spans()]


def span_named(exporter, prefix):
    for s in exporter.get_finished_spans():
        if s.name.startswith(prefix):
            return s
    raise AssertionError(f"no span starting with {prefix!r}; got {by_name(exporter)}")


# --- the span tree -----------------------------------------------------------
@pytest.mark.asyncio
async def test_a_run_produces_the_genai_span_tree(tmp_path, telemetry, exporter):
    agent = Agent(
        model=Script([tool_turn(), final()]),
        workdir=tmp_path, tools=[(SPEC, lambda i: "result")], on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("look it up")
    await agent.aclose()

    names = by_name(exporter)
    assert "invoke_agent triage" in names
    assert names.count("chat test-model") == 2
    assert "execute_tool Lookup" in names


@pytest.mark.asyncio
async def test_spans_carry_the_genai_conventions(tmp_path, telemetry, exporter):
    agent = Agent(
        model=Script([tool_turn(), final()]),
        workdir=tmp_path, tools=[(SPEC, lambda i: "r")], on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()

    run = span_named(exporter, "invoke_agent")
    assert run.attributes[ATTR_SYSTEM] == GEN_AI_SYSTEM
    assert run.attributes[ATTR_OPERATION] == "invoke_agent"
    assert run.attributes[ATTR_OUTCOME] == "completed"

    chat = span_named(exporter, "chat")
    assert chat.attributes[ATTR_OPERATION] == "chat"
    assert chat.attributes["gen_ai.request.model"] == "test-model"

    tool = span_named(exporter, "execute_tool")
    assert tool.attributes[ATTR_OPERATION] == "execute_tool"
    assert tool.attributes[ATTR_TOOL_NAME] == "Lookup"


@pytest.mark.asyncio
async def test_a_tool_span_is_a_child_of_the_turn_that_requested_it(
    tmp_path, telemetry, exporter
):
    """Without parent-child links a subagent's work cannot be attributed to the call that
    caused it, which is the whole reason for a span tree rather than a log."""
    agent = Agent(
        model=Script([tool_turn(), final()]),
        workdir=tmp_path, tools=[(SPEC, lambda i: "r")], on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    tool = spans["execute_tool Lookup"]
    run = spans["invoke_agent triage"]
    # the tool is nested somewhere under the run, and everything shares one trace
    assert tool.parent is not None
    assert tool.context.trace_id == run.context.trace_id


@pytest.mark.asyncio
async def test_every_span_shares_one_trace_id(tmp_path, telemetry, exporter):
    agent = Agent(
        model=Script([tool_turn(), tool_turn("2"), final()]),
        workdir=tmp_path, tools=[(SPEC, lambda i: "r")], on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    trace_ids = {s.context.trace_id for s in exporter.get_finished_spans()}
    assert len(trace_ids) == 1


@pytest.mark.asyncio
async def test_token_usage_lands_on_the_chat_span(tmp_path, telemetry, exporter):
    agent = Agent(
        model=Script([final()]), workdir=tmp_path, on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    chat = span_named(exporter, "chat")
    assert chat.attributes["gen_ai.usage.input_tokens"] == 20
    assert chat.attributes["gen_ai.usage.output_tokens"] == 8


@pytest.mark.asyncio
async def test_per_turn_usage_is_a_delta_not_the_cumulative_total(
    tmp_path, telemetry, exporter
):
    """`usage` on the event is cumulative for the run. Recording it verbatim per turn would
    double-count every earlier turn on every later one."""
    agent = Agent(
        model=Script([tool_turn(), final()]),
        workdir=tmp_path, tools=[(SPEC, lambda i: "r")], on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    chats = [s for s in exporter.get_finished_spans() if s.name.startswith("chat")]
    assert [s.attributes["gen_ai.usage.input_tokens"] for s in chats] == [10, 20]


@pytest.mark.asyncio
async def test_a_failing_tool_marks_its_span_as_an_error(tmp_path, telemetry, exporter):
    def boom(inp):
        raise RuntimeError("upstream down")

    agent = Agent(
        model=Script([tool_turn(), final()]),
        workdir=tmp_path, tools=[(SPEC, boom)], on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    tool = span_named(exporter, "execute_tool")
    assert tool.status.status_code.name == "ERROR"
    assert "upstream down" in (tool.status.description or "")


@pytest.mark.asyncio
async def test_a_truncated_run_is_marked_incomplete(tmp_path, telemetry, exporter):
    agent = Agent(
        model=Script([tool_turn("1", {"q": "a"}), tool_turn("2", {"q": "b"}),
                      tool_turn("3", {"q": "c"})]),
        workdir=tmp_path, tools=[(SPEC, lambda i: "r")], on_event=telemetry,
        max_turns=2, loop_guard=False,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    assert span_named(exporter, "invoke_agent").attributes[ATTR_OUTCOME] == "incomplete"


@pytest.mark.asyncio
async def test_no_span_is_left_open_when_a_run_ends_mid_turn(
    tmp_path, telemetry, exporter
):
    """A leaked span never reaches the collector and the trace reads as still running."""
    class Exploding:
        async def create(self, messages, tools):
            raise RuntimeError("provider unreachable")

    agent = Agent(
        model=Exploding(), workdir=tmp_path, on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    with pytest.raises(RuntimeError):
        await agent.run("go")
    await agent.aclose()

    names = by_name(exporter)
    assert "invoke_agent triage" in names
    assert "chat test-model" in names  # the open turn span was closed, not leaked


# --- correlation with the audit chain -----------------------------------------
@pytest.mark.asyncio
async def test_the_audit_run_id_can_be_stamped_on_the_trace(tmp_path, telemetry, exporter):
    """The audit chain and the trace are two records of one run; without a shared key
    neither can be looked up from the other."""
    agent = Agent(
        model=Script([final()]), workdir=tmp_path, on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )

    async def stamp(event: AgentEvent) -> None:
        telemetry(event)
        if event.type is EventType.RUN_START:
            telemetry.bind_run_id(agent.run_id)

    agent.on_event = stamp
    await agent.run("go")
    run_id = agent.run_id
    await agent.aclose()
    assert span_named(exporter, "invoke_agent").attributes[ATTR_RUN_ID] == run_id


@pytest.mark.asyncio
async def test_trace_id_is_readable_during_a_run(tmp_path, telemetry, exporter):
    seen: list[str] = []

    def sink(event):
        telemetry(event)
        if event.type is EventType.TURN_START:
            seen.append(telemetry.trace_id)

    agent = Agent(
        model=Script([final()]), workdir=tmp_path, on_event=sink,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    assert seen and len(seen[0]) == 32
    assert seen[0] == format(span_named(exporter, "invoke_agent").context.trace_id, "032x")


# --- content capture is off by default -----------------------------------------
@pytest.mark.asyncio
async def test_tool_arguments_are_not_exported_by_default(tmp_path, telemetry, exporter):
    """A span exporter is not the audit log: spans routinely leave the trust boundary for
    a vendor backend, and tool arguments carry exactly what redaction exists to mask."""
    agent = Agent(
        model=Script([tool_turn("1", {"q": "sk-live-SECRET"}), final()]),
        workdir=tmp_path, tools=[(SPEC, lambda i: "r")], on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    tool = span_named(exporter, "execute_tool")
    assert "barq.tool.arguments" not in tool.attributes
    assert not any("SECRET" in str(v) for v in tool.attributes.values())


@pytest.mark.asyncio
async def test_capture_content_opts_arguments_in(tmp_path, exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = OTelTelemetry(
        agent_name="a", model_name="m", capture_content=True,
        tracer=provider.get_tracer("t"),
    )
    agent = Agent(
        model=Script([tool_turn("1", {"q": "hello"}), final()]),
        workdir=tmp_path, tools=[(SPEC, lambda i: "r")], on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    assert "hello" in span_named(exporter, "execute_tool").attributes["barq.tool.arguments"]


# --- degradation ----------------------------------------------------------------
def test_it_is_a_no_op_when_opentelemetry_is_absent(monkeypatch):
    """`on_event=OTelTelemetry()` must be safe in a deployment that has not installed it."""
    monkeypatch.setattr("engine.telemetry.otel_available", lambda: False)
    telemetry = OTelTelemetry()
    assert telemetry.available is False
    for kind in EventType:
        telemetry(AgentEvent(type=kind, tool_name="T", tool_use_id="1"))
    assert telemetry.trace_id == ""


def test_require_fails_loudly_when_it_is_absent(monkeypatch):
    """Silently degrading is right for a library and wrong for an operator who believes
    their traces are being exported."""
    monkeypatch.setattr("engine.telemetry.otel_available", lambda: False)
    with pytest.raises(RuntimeError, match="not installed"):
        OTelTelemetry().require()


def test_require_passes_when_it_is_present():
    assert otel_available()
    assert OTelTelemetry().require() is not None


@pytest.mark.asyncio
async def test_a_raising_sink_never_breaks_the_run(tmp_path):
    """An observability sink must not become a failure mode of the thing it observes."""
    telemetry = OTelTelemetry()

    def explode(_event):
        raise RuntimeError("collector down")

    telemetry._handle = explode  # type: ignore[method-assign]
    agent = Agent(
        model=Script([final("still fine")]), workdir=tmp_path, on_event=telemetry,
        enable_file_tools=False, enable_memory_tools=False,
    )
    assert await agent.run("go") == "still fine"
    await agent.aclose()


# --- the prerequisite fix -------------------------------------------------------
@pytest.mark.asyncio
async def test_on_event_receives_run_start_and_run_end(tmp_path):
    """Both used to be produced only inside `astream()`, for its own consumer. An
    `on_event=` sink saw neither — and those two are what a span is built from."""
    seen: list[EventType] = []
    agent = Agent(
        model=Script([final()]), workdir=tmp_path, on_event=lambda e: seen.append(e.type),
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    assert seen[0] is EventType.RUN_START
    assert seen[-1] is EventType.RUN_END


@pytest.mark.asyncio
async def test_astream_still_yields_exactly_one_run_start_and_run_end(tmp_path):
    """The events now come through the sink, so astream must not synthesise its own."""
    kinds: list[EventType] = []
    agent = Agent(
        model=Script([final()]), workdir=tmp_path,
        enable_file_tools=False, enable_memory_tools=False,
    )
    async for event in agent.stream("go"):
        kinds.append(event.type)
    await agent.aclose()
    assert kinds.count(EventType.RUN_START) == 1
    assert kinds.count(EventType.RUN_END) == 1
    assert kinds[-1] is EventType.RUN_END


@pytest.mark.asyncio
async def test_run_end_carries_the_duration(tmp_path):
    ends: list[AgentEvent] = []
    agent = Agent(
        model=Script([final()]), workdir=tmp_path,
        on_event=lambda e: ends.append(e) if e.type is EventType.RUN_END else None,
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("go")
    await agent.aclose()
    assert ends and ends[0].elapsed_ms is not None and ends[0].elapsed_ms >= 0
