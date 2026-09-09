"""Run events: the observable surface of an agent run.

A run used to be a function that returned a transcript. Everything a caller might want
while it is still going — the model's text as it arrives, which tool is executing, what the
permission engine decided, what it has spent, the ability to stop it — was unreachable,
because the only thing the loop handed back was the finished list of messages.

That single shape is what forecloses streaming, progress reporting, cancellation and
tracing all at once, so they are fixed together here rather than bolted on one at a time.
The loop now emits a typed event at every point where something observable happens, and
three consumers sit on the same stream:

  * `Coordinator.astream()` / `Agent.stream()` — async iteration, for a UI.
  * `on_event=` — a callback, for metrics, spans and logging.
  * nothing at all — the events are simply not consumed; `run()` behaves exactly as before.

Events are plain dataclasses, not provider payloads, so a consumer written against them
keeps working across providers and across a change of wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    RUN_START = "run_start"
    TURN_START = "turn_start"           # a model round-trip is about to be made
    TEXT_DELTA = "text_delta"           # incremental assistant text (streaming providers)
    TEXT = "text"                       # a complete assistant text block
    TOOL_DECISION = "tool_decision"     # the permission verdict for one call
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    USAGE = "usage"                     # cumulative spend after a model call
    COMPACTION = "compaction"           # the transcript was replaced by a summary
    NOTICE = "notice"                   # loop-guard trip, wrap-up nudge, budget warning
    ERROR = "error"
    RUN_END = "run_end"


@dataclass(frozen=True)
class AgentEvent:
    """One observable moment in a run.

    `type` says what happened; the remaining fields are populated when they apply. Kept as
    one flat type rather than a class per event so a consumer can switch on `type` and
    ignore what it does not care about, and so adding a field is not a breaking change.
    """

    type: EventType
    text: str = ""                       # TEXT / TEXT_DELTA / NOTICE / ERROR
    tool_name: str = ""                  # TOOL_* events
    tool_use_id: str = ""
    tool_input: Optional[dict] = None
    result: str = ""                     # TOOL_END
    is_error: bool = False               # TOOL_END / ERROR
    decision: str = ""                   # TOOL_DECISION: "allow" | "ask" | "deny"
    reason: str = ""                     # why the decision went that way
    turn: int = 0                        # 1-based index of the model round-trip
    usage: dict = field(default_factory=dict)
    completed: Optional[bool] = None     # RUN_END: did the model finish on its own?
    elapsed_ms: Optional[float] = None   # TOOL_END / RUN_END: wall-clock duration

    def __str__(self) -> str:  # readable in a log line without a custom formatter
        if self.type in (EventType.TEXT, EventType.TEXT_DELTA, EventType.NOTICE):
            return f"[{self.type.value}] {self.text}"
        if self.type is EventType.TOOL_DECISION:
            return f"[{self.type.value}] {self.tool_name}: {self.decision} ({self.reason})"
        if self.type in (EventType.TOOL_START, EventType.TOOL_END):
            suffix = f" -> {self.result[:80]}" if self.type is EventType.TOOL_END else ""
            return f"[{self.type.value}] {self.tool_name}{suffix}"
        if self.type is EventType.RUN_END:
            return f"[{self.type.value}] completed={self.completed} usage={self.usage}"
        return f"[{self.type.value}]"


class CancelledRun(RuntimeError):
    """Raised inside the loop when a caller cancels the run through a CancelToken."""


class CancelToken:
    """A cooperative stop signal for a run.

    `asyncio.Task.cancel()` tears a run down wherever it happens to be — mid tool call,
    between a side effect and its journal entry. This stops it at the next safe boundary
    instead: the loop checks the token before each model round-trip and before dispatching
    each tool, so a cancelled run leaves a consistent transcript and a resumable session.
    """

    def __init__(self) -> None:
        self._cancelled = False
        self.reason = ""

    def cancel(self, reason: str = "cancelled by the caller") -> None:
        self._cancelled = True
        self.reason = reason

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledRun(self.reason)
