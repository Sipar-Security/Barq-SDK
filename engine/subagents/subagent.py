"""Subagents — port of CC's AgentTool pattern.

A Subagent runs its OWN Coordinator with its OWN message list. High-volume tool output
(bulk search, crawl and file reads) stays inside the subagent and never enters the parent's
context — this is how context-window pressure is managed structurally, rather than by trimming.
Only the subagent's final summary crosses back to the parent (as a tool result).

Per-subagent model selection: give a broad/cheap subagent a fast model + read-only tools,
and a focused subagent a strong model + more capable tools, by handing each Subagent the
appropriate ModelClient. Callers with one provider pass the same client to every role — the
seam is here (`Subagent.model`) whether or not the wiring differentiates.

Boundary guarantees this module enforces so a delegated task behaves like the parent would:
  * ELICIT is threaded through, so a gated (ASK) call inside a subagent reaches the same
    human-approval path instead of silently failing closed (a denial the parent never sees).
  * SESSION + JOURNAL are threaded through, so a crash mid-subagent resumes the subagent
    instead of re-running it from scratch — its external side effects stay exactly-once.
  * COMPLETION status is carried back: a subagent that ran out of turns is reported as
    TRUNCATED, never as a clean empty result the parent could misread as "found nothing".
  * NO RECURSION: a subagent can never hold the spawn tool, enforced structurally here (not
    by hoping the factory left it out).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from engine.audit import AuditLog
from engine.coordinator import Coordinator
from engine.permissions import PermissionEngine

# Tool names that spawn subagents. A subagent must never be handed one of these — that is
# what prevents unbounded recursion (a tester spawning testers). Enforced in Subagent.run.
_SPAWN_TOOL_NAMES = frozenset({"Task", "SpawnSubagent"})


@dataclass
class SubagentResult:
    name: str
    final_text: str
    tool_calls: int
    turns: int
    isolated_messages: int  # size of the subagent's private context (never seen by parent)
    completed: bool = True   # False => hit max_turns/timeout with work unfinished
    usage: dict = field(default_factory=dict)  # token usage spent inside the subagent


@dataclass
class Subagent:
    name: str
    model: Any                      # ModelClient
    permissions: PermissionEngine
    audit: AuditLog
    native_tools: dict = field(default_factory=dict)
    tool_specs: list = field(default_factory=list)
    mcp_handler: Any = None
    max_turns: int = 16
    # Threaded into the subagent's Coordinator so its behaviour matches a parent-run tool call.
    elicit: Any = None              # ElicitFn — ASK decisions reach a human instead of failing closed
    session: Any = None             # SessionStore | None — crash-resume for this subagent
    journal: Any = None             # ToolJournal | None — exactly-once tool side effects on resume
    resume: bool = False
    timeout: Optional[float] = None  # wall-clock cap for the whole subagent run
    # Prompt-injection posture inside the subagent (its tool output can be attacker-influenced).
    fence_untrusted_output: bool = True
    trusted_tools: frozenset = field(default_factory=frozenset)
    context_compactor: Any = None
    # The same rails the parent loop runs under — a delegated task must not be able to hang
    # forever, skip argument validation, or spend without a ceiling just by being delegated.
    tool_timeout: Optional[float] = 120.0
    max_parallel_tools: int = 8
    token_budget: Optional[int] = None
    validate_tool_input: bool = True

    _TOKEN_KEYS = ("total_tokens", "prompt_tokens", "completion_tokens")

    def _usage_for(self, coord: Any) -> dict:
        """This subagent's own spend.

        The Coordinator accumulates usage per RUN, so it attributes correctly even when
        several subagents share one model client — prefer it. A client that only exposes a
        cumulative `usage_total` (and no per-response usage) still reports through the
        fallback, so that contract keeps working.
        """
        own = dict(getattr(coord, "usage", {}) or {})
        if any(k in own for k in self._TOKEN_KEYS):
            return own
        fallback = dict(getattr(self.model, "usage_total", {}) or {})
        return fallback or own

    def _safe_toolset(self) -> tuple[dict, list]:
        """The subagent's toolset with any spawn tool removed — a subagent may never delegate
        further, and we guarantee that here rather than trusting the factory to omit it."""
        native = {k: v for k, v in self.native_tools.items() if k not in _SPAWN_TOOL_NAMES}
        specs = [s for s in self.tool_specs if s.get("name") not in _SPAWN_TOOL_NAMES]
        return native, specs

    async def run(self, task: str) -> SubagentResult:
        native, specs = self._safe_toolset()
        coord = Coordinator(
            model=self.model,
            permissions=self.permissions,
            audit=self.audit,
            native_tools=native,
            tool_specs=specs,
            mcp_handler=self.mcp_handler,
            max_turns=self.max_turns,
            elicit=self.elicit,
            session=self.session,
            journal=self.journal,
            resume=self.resume,
            fence_untrusted_output=self.fence_untrusted_output,
            trusted_tools=self.trusted_tools,
            context_compactor=self.context_compactor,
            tool_timeout=self.tool_timeout,
            max_parallel_tools=self.max_parallel_tools,
            token_budget=self.token_budget,
            validate_tool_input=self.validate_tool_input,
        )
        try:
            if self.timeout is not None:
                messages = await asyncio.wait_for(coord.run(task), self.timeout)
            else:
                messages = await coord.run(task)  # isolated context lives entirely here
        except asyncio.TimeoutError:
            # The run is cut off; its session (if any) holds the partial transcript for a resume.
            # Report what THIS subagent spent before the timeout, not the client's lifetime
            # total — a shared client would otherwise attribute every other run's tokens here.
            return SubagentResult(
                self.name, "", 0, 0, 0, completed=False,
                usage=self._usage_for(coord),
            )

        completed = bool(getattr(coord, "completed", True))
        final_text = _last_assistant_text(messages)
        tool_calls = sum(
            1
            for m in messages
            if m.get("role") == "assistant" and isinstance(m.get("content"), list)
            for b in m["content"]
            if b.get("type") == "tool_use"
        )
        turns = sum(1 for m in messages if m.get("role") == "assistant")
        usage = self._usage_for(coord)
        return SubagentResult(
            self.name, final_text, tool_calls, turns, len(messages),
            completed=completed, usage=usage,
        )


def _last_assistant_text(messages: list[dict]) -> str:
    """The text of the LAST assistant turn that actually produced text. Scanning backwards
    (not just messages[-1]) matters when the run ends on a tool_result turn — e.g. a subagent
    truncated at max_turns — so we surface the model's most recent words instead of "" ."""
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, list):
            text = "\n".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            ).strip()
            if text:
                return text
        elif isinstance(content, str) and content.strip():
            return content.strip()
    return ""


# Factory: (subagent_type) -> a fresh Subagent. Fresh per spawn so context never carries.
SubagentFactory = Callable[[], Subagent]
# (subagent_type, task) -> (SessionStore, ToolJournal) for crash-resumable delegation.
SessionFor = Callable[[str, str], tuple]
# invoked with each SubagentResult so the parent can meter usage / surface progress.
OnResult = Callable[[SubagentResult], None]


def make_spawn_tool(
    registry: dict[str, SubagentFactory],
    *,
    session_for: Optional[SessionFor] = None,
    resume: bool = False,
    on_result: Optional[OnResult] = None,
    max_spawns: Optional[int] = None,
) -> tuple[Callable[[dict], Awaitable[str]], dict]:
    """Build the parent-facing SpawnSubagent tool.

    Returns (native_tool_fn, tool_spec). When the parent model calls it, the named subagent
    runs to completion and ONLY its final summary is returned to the parent — the subagent's
    tool-call noise stays isolated.

    Optional wiring:
      * `session_for(stype, task)` supplies a (SessionStore, ToolJournal) so a crash mid-
        subagent resumes it (with `resume=True`) instead of re-running its side effects.
      * `on_result` is called with every SubagentResult (usage metering, progress surfacing).
      * `max_spawns` caps how many subagents one orchestration may spawn — a wallet/DoS guard,
        since each spawn is its own turn budget the parent's max_turns can't see.
    """

    state = {"spawns": 0}

    async def spawn(inp: dict) -> str:
        stype = str(inp.get("subagent_type", "") or "").strip()
        task = str(inp.get("task", "") or "").strip()
        if not stype:
            return f"ERROR: subagent_type is required. Available: {sorted(registry)}"
        factory = registry.get(stype)
        if factory is None:
            return f"ERROR: unknown subagent_type {stype!r}. Available: {sorted(registry)}"
        if not task:
            return (
                f"ERROR: task must be a non-empty, fully self-contained instruction for the "
                f"'{stype}' subagent (it cannot see your conversation)."
            )
        if max_spawns is not None and state["spawns"] >= max_spawns:
            return (
                f"ERROR: subagent budget exhausted ({max_spawns} spawns already used). Stop "
                "delegating and synthesize/report from the results you already collected."
            )
        state["spawns"] += 1

        sub = factory()
        if session_for is not None:
            sess, jrnl = session_for(stype, task)
            sub.session, sub.resume = sess, resume
            # Attach the journal only when resuming: exactly-once matters on a re-run, and
            # withholding it on a fresh run avoids any chance of a stale entry from a prior
            # identical (type, task) shadowing a genuinely new call.
            sub.journal = jrnl if resume else None
        try:
            result = await sub.run(task)
        except Exception as e:  # a broken factory/model — report, don't crash the parent loop
            return f"ERROR: subagent {stype!r} failed: {type(e).__name__}: {e}"

        if on_result is not None:
            try:
                on_result(result)
            except Exception:
                pass

        if result.completed:
            status = "completed"
            body = result.final_text.strip() or "(the subagent produced no textual summary)"
        else:
            status = "TRUNCATED at max_turns — INCOMPLETE, treat as a partial result"
            body = result.final_text.strip() or (
                "(no summary — the subagent ran out of turns before it could conclude; "
                "re-delegate a narrower task or raise its budget)"
            )
        return (
            f"[subagent {result.name!r} {status}: {result.tool_calls} tool calls, "
            f"{result.turns} turns]\n{body}"
        )

    spec = {
        "name": "SpawnSubagent",
        "description": (
            "Delegate a self-contained task to an isolated subagent. The subagent has its "
            "own context; you receive only its final summary. Use for high-volume work "
            "(searching, crawling, bulk reads) so raw output does not fill your context. The task string "
            "must be fully self-contained — the subagent cannot see your conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subagent_type": {"type": "string", "description": f"one of: {sorted(registry)}"},
                "task": {"type": "string", "description": "self-contained task for the subagent"},
            },
            "required": ["subagent_type", "task"],
        },
    }
    return spawn, spec
