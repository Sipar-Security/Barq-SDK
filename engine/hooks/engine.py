"""Hook engine — lifecycle gates and side effects around tool use.

Supports the two hook types that make sense to run in-process:
  - FunctionHook: a Python callable. Best for a fast in-process policy gate.
  - CommandHook: a shell command. exit code 2 => blocking error, and stdout/stderr are
    captured for the audit log.

Each hook carries an optional `if_` condition in permission-rule syntax. A PreToolUse hook
may return a permission Decision to allow/deny/ask; the first deny wins, fail-closed.

Sync and async firing
---------------------
Hooks run inside an async agent loop, and a hook that shells out to a policy service is
routine. `CommandHook.run()` uses a blocking `subprocess.run`, so calling it directly from
a coroutine stalls the WHOLE event loop — every other agent, request and task in the
process — for as long as the hook takes (up to its timeout, 30s by default).

So the engine exposes both shapes:
  * `fire()` / `gate()`      — synchronous, for sync callers (PermissionEngine.check).
  * `fire_async()` / `gate_async()` — offload each hook to a worker thread via
    `asyncio.to_thread`, so a slow hook blocks only its own thread. The agent loop uses
    these. A hook that raises is contained and reported as a failed outcome rather than
    propagating into the permission path.

`prompt` and `agent` hook types (LLM-backed) are intentionally not implemented here so that
offline tests never silently pass on a stubbed model; see coordinator.py for where a live
classifier plugs in.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from .events import HookEvent

if TYPE_CHECKING:
    # Type-only import to avoid a package import cycle (permissions <-> hooks).
    from engine.permissions.decision import Decision


@dataclass(frozen=True)
class HookInput:
    event: HookEvent
    tool_name: str | None = None
    match_content: str = ""
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class HookOutcome:
    hook_name: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    decision: Optional[Decision] = None  # PreToolUse hooks may gate the tool
    errored: bool = False  # the hook raised: its verdict is UNKNOWN, not "no objection"

    @property
    def is_blocking(self) -> bool:
        return self.exit_code == 2 or (
            self.decision is not None and self.decision.behavior.value == "deny"
        )


# A function hook returns a Decision (to gate) or None (advisory only).
# Forward-ref string keeps Decision out of runtime import (cycle break).
FunctionHookFn = Callable[[HookInput], Optional["Decision"]]


@dataclass
class FunctionHook:
    name: str
    fn: FunctionHookFn
    if_: str | None = None

    def run(self, inp: HookInput) -> HookOutcome:
        decision = self.fn(inp)
        return HookOutcome(hook_name=self.name, decision=decision)


@dataclass
class CommandHook:
    name: str
    command: str
    if_: str | None = None
    timeout: float = 30.0

    def run(self, inp: HookInput) -> HookOutcome:
        try:
            proc = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            return HookOutcome(self.name, exit_code=124, stderr=f"timeout: {e}")
        return HookOutcome(
            self.name,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


Hook = FunctionHook | CommandHook


def _if_matches(if_: str | None, inp: HookInput) -> bool:
    if not if_:
        return True
    if inp.tool_name is None:
        return False
    from engine.permissions.rule import parse_rule, rule_matches  # lazy: cycle break

    rule = parse_rule(if_)
    return rule_matches(rule, inp.tool_name, inp.match_content)


class HookEngine:
    """Registers hooks per event and fires them. For PreToolUse it aggregates a
    gate decision: first DENY wins; else first explicit ALLOW; else None."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[Hook]] = {}

    def register(self, event: HookEvent, hook: Hook) -> None:
        self._hooks.setdefault(event, []).append(hook)

    def _selected(self, inp: HookInput) -> list[Hook]:
        return [h for h in self._hooks.get(inp.event, []) if _if_matches(h.if_, inp)]

    @staticmethod
    def _failed(hook: Hook, exc: BaseException) -> HookOutcome:
        """A hook that raises must not take the run with it: report it as a non-blocking
        failure so the permission path can still reach a decision."""
        return HookOutcome(
            getattr(hook, "name", "hook"), exit_code=1,
            stderr=f"hook raised {type(exc).__name__}: {exc}", errored=True,
        )

    def fire(self, inp: HookInput) -> list[HookOutcome]:
        outcomes: list[HookOutcome] = []
        for hook in self._selected(inp):
            try:
                outcomes.append(hook.run(inp))
            except Exception as e:
                outcomes.append(self._failed(hook, e))
        return outcomes

    async def fire_async(self, inp: HookInput) -> list[HookOutcome]:
        """Fire every matching hook off the event loop, concurrently.

        Each hook runs in a worker thread, so a blocking CommandHook cannot stall the agent
        loop or any task sharing it.
        """
        hooks = self._selected(inp)
        if not hooks:
            return []
        results = await asyncio.gather(
            *(asyncio.to_thread(h.run, inp) for h in hooks), return_exceptions=True
        )
        return [
            self._failed(h, r) if isinstance(r, BaseException) else r
            for h, r in zip(hooks, results)
        ]

    @staticmethod
    def _resolve_gate(outcomes: list[HookOutcome]) -> Optional[Decision]:
        """Resolve a PreToolUse gate. A hook that RAISED has no verdict, so the gate cannot
        conclude "nothing objected" — it resolves to ASK (fail-closed) unless another hook
        already denied. Treating a crashed policy gate as silence is a fail-OPEN bypass."""
        deciding_allow: Optional[Decision] = None
        errored: list[str] = []
        for outcome in outcomes:
            if outcome.errored:
                errored.append(f"{outcome.hook_name}: {outcome.stderr}")
                continue
            if outcome.decision is not None:
                if outcome.decision.behavior.value == "deny":
                    return outcome.decision  # first deny wins, fail-closed
                if deciding_allow is None and outcome.decision.behavior.value == "allow":
                    deciding_allow = outcome.decision
            elif outcome.exit_code == 2:
                # command hook signalled a block without a structured decision
                from engine.permissions.decision import deny

                return deny("hook", f"hook {outcome.hook_name} exited 2", outcome.stderr)
        if errored and deciding_allow is None:
            from engine.permissions.decision import ask

            return ask("hook", f"a PreToolUse hook failed; ask a human ({'; '.join(errored)})")
        return deciding_allow

    def gate(self, inp: HookInput) -> Optional[Decision]:
        """PreToolUse gate: return the deciding Decision, or None to fall through."""
        return self._resolve_gate(self.fire(inp))

    async def gate_async(self, inp: HookInput) -> Optional[Decision]:
        """PreToolUse gate, off the event loop (see fire_async)."""
        return self._resolve_gate(await self.fire_async(inp))
