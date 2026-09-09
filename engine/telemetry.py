"""OpenTelemetry for the agent loop: spans, metrics and trace correlation.

There was none. Zero occurrences of `opentelemetry` in the tree, no spans, no GenAI
semantic conventions, no OTLP exporter, no trace-context propagation, and no metrics at
all — one cumulative `usage` dict that nothing could be alerted on. Enterprises route agent
telemetry through OTel collectors they already run, so an SDK that cannot emit a span
cannot be monitored by the tooling the customer already owns.

The event stream was documented as "the seam for one". This is that one:

    from engine import Agent
    from engine.telemetry import OTelTelemetry

    telemetry = OTelTelemetry(service_name="triage-agent")
    agent = Agent(model=model, workdir="./run", on_event=telemetry)

It produces a span tree following the OpenTelemetry **GenAI semantic conventions**:

    invoke_agent <agent>              gen_ai.operation.name=invoke_agent
      chat <model>                    gen_ai.operation.name=chat, usage attributes
      execute_tool <tool>             gen_ai.operation.name=execute_tool
      execute_tool <tool>
      chat <model>

and these metrics:

    gen_ai.client.token.usage          histogram, by token type
    gen_ai.client.operation.duration   histogram, by operation
    gen_ai.agent.runs                  counter, by outcome
    gen_ai.agent.tool.calls            counter, by tool and outcome
    gen_ai.agent.permission.decisions  counter, by tool and decision

**OpenTelemetry is not a dependency.** It is imported lazily and, when absent, every method
becomes a no-op — so `on_event=OTelTelemetry()` is safe in a deployment that has not
installed it, and adding `pip install opentelemetry-sdk` is the only step needed to turn
telemetry on. `available` says which state you are in; `require()` fails loudly for a
deployment that considers missing telemetry a misconfiguration rather than a default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from engine.events import AgentEvent, EventType

__all__ = ["OTelTelemetry", "GEN_AI_SYSTEM", "otel_available"]

# `gen_ai.system` identifies the framework producing the telemetry.
GEN_AI_SYSTEM = "barq_sdk"

# Attribute names, spelled once. The GenAI conventions are still moving, so a rename should
# be a one-line change here rather than a search across the module.
ATTR_SYSTEM = "gen_ai.system"
ATTR_OPERATION = "gen_ai.operation.name"
ATTR_AGENT_NAME = "gen_ai.agent.name"
ATTR_MODEL = "gen_ai.request.model"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_TOKEN_TYPE = "gen_ai.token.type"
# Not in the conventions, but the whole point of this SDK's audit trail is that a record
# can be joined to something outside itself.
ATTR_RUN_ID = "barq.run_id"
ATTR_DECISION = "barq.permission.decision"
ATTR_DECISION_REASON = "barq.permission.reason"
ATTR_TURN = "barq.turn"
ATTR_OUTCOME = "barq.outcome"


def otel_available() -> bool:
    """True if the OpenTelemetry API is importable in this process."""
    try:
        import opentelemetry.trace  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class _NoopSpan:
    """Stands in for a span when OpenTelemetry is not installed."""

    def set_attribute(self, *_a, **_k) -> None: ...
    def set_status(self, *_a, **_k) -> None: ...
    def record_exception(self, *_a, **_k) -> None: ...
    def end(self, *_a, **_k) -> None: ...
    def get_span_context(self) -> Any: return None


@dataclass
class OTelTelemetry:
    """An `on_event` sink that turns the run event stream into OTel spans and metrics.

    Instances are callable, so they are passed straight to `Agent(on_event=...)`. One
    instance handles one agent: spans are correlated by nesting tool spans inside the model
    span inside the run span, and a shared instance across concurrent agents would
    interleave those. Construct one per agent.
    """

    service_name: str = "barq-agent"
    agent_name: str = "agent"
    model_name: str = ""
    # Record tool arguments and results on the span. OFF by default: a span exporter is not
    # the audit log, spans routinely leave the trust boundary for a vendor backend, and
    # tool arguments carry exactly the payloads `audit/redact.py` exists to mask.
    capture_content: bool = False
    tracer: Any = None
    meter: Any = None

    _run_span: Any = field(default=None, init=False, repr=False)
    _turn_span: Any = field(default=None, init=False, repr=False)
    _tool_spans: dict = field(default_factory=dict, init=False, repr=False)
    _run_started: float = field(default=0.0, init=False, repr=False)
    _turn_started: float = field(default=0.0, init=False, repr=False)
    _last_usage: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.available = otel_available()
        self._trace = None
        self._status = None
        if self.available:
            try:
                from opentelemetry import trace as _trace
                from opentelemetry.trace import Status, StatusCode

                self._trace = _trace
                self._status = (Status, StatusCode)
                if self.tracer is None:
                    self.tracer = _trace.get_tracer(self.service_name)
            except Exception:  # noqa: BLE001
                self.available = False
        self._instruments = self._build_instruments()

    # --- setup ---------------------------------------------------------------
    def require(self) -> "OTelTelemetry":
        """Raise unless OpenTelemetry is actually installed.

        For a deployment where missing telemetry is a misconfiguration, not a default:
        silently degrading to a no-op is right for a library and wrong for an operator who
        believes their traces are being exported.
        """
        if not self.available:
            raise RuntimeError(
                "OpenTelemetry is not installed, so no spans or metrics will be exported. "
                "Install it (`pip install barq-sdk[otel]`) or drop the require() call."
            )
        return self

    def _build_instruments(self) -> dict:
        if not self.available:
            return {}
        try:
            from opentelemetry import metrics

            meter = self.meter or metrics.get_meter(self.service_name)
            return {
                "tokens": meter.create_histogram(
                    "gen_ai.client.token.usage", unit="{token}",
                    description="tokens used by the agent, by type",
                ),
                "duration": meter.create_histogram(
                    "gen_ai.client.operation.duration", unit="s",
                    description="duration of one GenAI operation",
                ),
                "runs": meter.create_counter(
                    "gen_ai.agent.runs", description="agent runs, by outcome",
                ),
                "tool_calls": meter.create_counter(
                    "gen_ai.agent.tool.calls", description="tool calls, by tool and outcome",
                ),
                "decisions": meter.create_counter(
                    "gen_ai.agent.permission.decisions",
                    description="permission decisions, by tool and verdict",
                ),
            }
        except Exception:  # noqa: BLE001 - metrics must never break the run
            return {}

    # --- span helpers --------------------------------------------------------
    def _start(self, name: str, attributes: dict, parent: Any = None) -> Any:
        if not self.available or self.tracer is None:
            return _NoopSpan()
        try:
            context = None
            if parent is not None and self._trace is not None:
                context = self._trace.set_span_in_context(parent)
            return self.tracer.start_span(name, context=context, attributes=attributes)
        except Exception:  # noqa: BLE001
            return _NoopSpan()

    def _end(self, span: Any, error: str = "") -> None:
        if span is None:
            return
        try:
            if error and self._status is not None:
                Status, StatusCode = self._status
                span.set_status(Status(StatusCode.ERROR, error[:500]))
            span.end()
        except Exception:  # noqa: BLE001
            pass

    def _record(self, instrument: str, value: Any, attributes: dict) -> None:
        inst = self._instruments.get(instrument)
        if inst is None:
            return
        try:
            if hasattr(inst, "record"):
                inst.record(value, attributes)
            else:
                inst.add(value, attributes)
        except Exception:  # noqa: BLE001
            pass

    def _base(self) -> dict:
        attrs = {ATTR_SYSTEM: GEN_AI_SYSTEM, ATTR_AGENT_NAME: self.agent_name}
        if self.model_name:
            attrs[ATTR_MODEL] = self.model_name
        return attrs

    @property
    def trace_id(self) -> str:
        """The current run's trace id as a 32-char hex string, or "".

        This is the join key between the SDK's own audit chain and everything else the
        operator runs: put it beside `Agent.run_id` in your application log and one lookup
        reaches both.
        """
        span = self._run_span
        if span is None or isinstance(span, _NoopSpan):
            return ""
        try:
            return format(span.get_span_context().trace_id, "032x")
        except Exception:  # noqa: BLE001
            return ""

    # --- the sink ------------------------------------------------------------
    def __call__(self, event: AgentEvent) -> None:
        """Consume one run event. Never raises: an observability sink must not become a
        failure mode of the thing it observes."""
        try:
            self._handle(event)
        except Exception:  # noqa: BLE001
            pass

    def _handle(self, event: AgentEvent) -> None:
        kind = event.type

        if kind is EventType.RUN_START:
            self._run_started = time.perf_counter()
            self._tool_spans.clear()
            self._run_span = self._start(
                f"invoke_agent {self.agent_name}",
                {**self._base(), ATTR_OPERATION: "invoke_agent"},
            )
            return

        if kind is EventType.TURN_START:
            # A turn IS a model round-trip, so it is the `chat` span the conventions want.
            self._turn_started = time.perf_counter()
            self._turn_span = self._start(
                f"chat {self.model_name or 'model'}",
                {**self._base(), ATTR_OPERATION: "chat", ATTR_TURN: event.turn},
                parent=self._run_span,
            )
            return

        if kind is EventType.USAGE:
            # USAGE lands immediately after the model responds, so it closes the chat span
            # and carries the token counts onto it.
            delta = {
                k: event.usage.get(k, 0) - self._last_usage.get(k, 0)
                for k in ("prompt_tokens", "completion_tokens")
            }
            self._last_usage = dict(event.usage)
            span = self._turn_span
            if span is not None:
                span.set_attribute(ATTR_INPUT_TOKENS, delta.get("prompt_tokens", 0))
                span.set_attribute(ATTR_OUTPUT_TOKENS, delta.get("completion_tokens", 0))
                self._end(span)
                self._turn_span = None
            self._record("duration", time.perf_counter() - self._turn_started,
                         {**self._base(), ATTR_OPERATION: "chat"})
            for name, key in (("input", "prompt_tokens"), ("output", "completion_tokens")):
                if delta.get(key):
                    self._record("tokens", delta[key],
                                 {**self._base(), ATTR_TOKEN_TYPE: name})
            return

        if kind is EventType.TOOL_DECISION:
            self._record("decisions", 1, {
                **self._base(), ATTR_TOOL_NAME: event.tool_name,
                ATTR_DECISION: event.decision,
            })
            return

        if kind is EventType.TOOL_START:
            attrs = {
                **self._base(), ATTR_OPERATION: "execute_tool",
                ATTR_TOOL_NAME: event.tool_name, ATTR_TURN: event.turn,
            }
            if self.capture_content and event.tool_input is not None:
                attrs["barq.tool.arguments"] = str(event.tool_input)[:2000]
            span = self._start(
                f"execute_tool {event.tool_name}", attrs,
                # Nest under the chat span when one is open, else the run span, so a tool
                # call is always attributable to the turn that requested it.
                parent=self._turn_span or self._run_span,
            )
            self._tool_spans[event.tool_use_id or event.tool_name] = (
                span, time.perf_counter()
            )
            return

        if kind is EventType.TOOL_END:
            key = event.tool_use_id or event.tool_name
            span, started = self._tool_spans.pop(key, (None, time.perf_counter()))
            if span is not None:
                if self.capture_content and event.result:
                    span.set_attribute("barq.tool.result", event.result[:2000])
                self._end(span, error=event.result if event.is_error else "")
            self._record("duration", time.perf_counter() - started,
                         {**self._base(), ATTR_OPERATION: "execute_tool",
                          ATTR_TOOL_NAME: event.tool_name})
            self._record("tool_calls", 1, {
                **self._base(), ATTR_TOOL_NAME: event.tool_name,
                ATTR_OUTCOME: "error" if event.is_error else "ok",
            })
            return

        if kind is EventType.ERROR:
            span = self._turn_span or self._run_span
            if span is not None:
                span.set_attribute("barq.error", event.text[:500])
            return

        if kind is EventType.RUN_END:
            # A turn span can still be open if the run ended mid-turn (cancellation, an
            # exception). Closing it here keeps the trace well-formed rather than leaking a
            # span that never ends.
            if self._turn_span is not None:
                self._end(self._turn_span)
                self._turn_span = None
            for span, _ in self._tool_spans.values():
                self._end(span, error="run ended before the tool returned")
            self._tool_spans.clear()
            outcome = "completed" if event.completed else "incomplete"
            if self._run_span is not None:
                self._run_span.set_attribute(ATTR_OUTCOME, outcome)
                total = event.usage.get("total_tokens")
                if total:
                    self._run_span.set_attribute("gen_ai.usage.total_tokens", total)
                self._end(self._run_span)
                self._run_span = None
            self._record("duration", time.perf_counter() - self._run_started,
                         {**self._base(), ATTR_OPERATION: "invoke_agent"})
            self._record("runs", 1, {**self._base(), ATTR_OUTCOME: outcome})
            self._last_usage = {}
            return

    def bind_run_id(self, run_id: str) -> None:
        """Stamp the SDK's audit `run_id` onto the run span.

        Called by the caller after `Agent` has built its coordinator. The audit chain and
        the trace are the two records of the same run, and without a shared key neither can
        be looked up from the other.
        """
        if self._run_span is not None and run_id:
            try:
                self._run_span.set_attribute(ATTR_RUN_ID, run_id)
            except Exception:  # noqa: BLE001
                pass
