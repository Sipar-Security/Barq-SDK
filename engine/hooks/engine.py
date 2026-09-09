"""Hook engine: lifecycle gates and side effects around tool use.

Supports the two hook types that make sense to run in-process:
  - FunctionHook: a Python callable. Best for a fast in-process policy gate.
  - CommandHook: a shell command, handed the call it is gating.

Each hook carries an optional `if_` condition in permission-rule syntax. A PreToolUse hook
may return a permission Decision to allow/deny/ask; the first deny wins, fail-closed.

The CommandHook contract
------------------------
A command hook receives the call on **stdin** as one JSON document, and a flattened view of
the same thing in the environment (``BARQ_HOOK_EVENT``, ``BARQ_HOOK_TOOL_NAME``,
``BARQ_HOOK_MATCH_CONTENT``, ``BARQ_HOOK_INPUT``)::

    {"version": 1,
     "event": "PreToolUse",
     "tool_name": "WriteFile",
     "match_content": "/etc/passwd",
     "payload": {"path": "/etc/passwd", "content": "..."}}

It answers on **stdout**, optionally, with::

    {"decision": "allow" | "ask" | "deny", "reason": "..."}

or by exiting 2 to block. Exit 2 outranks stdout. A hook that prints nothing JSON-shaped is
advisory, exactly as before.

None of this used to exist: `run()` took the `HookInput` and never referenced it, so a
command hook was a constant function of the tool call — it could not see which tool ran,
with what arguments, against what target. `if_` was the only rule expressible, and every
external policy integration (an OPA sidecar, a DLP service, an approval API) was impossible
to write. A hook that times out or misprints its verdict is now an ERROR, not silence, so
it escalates to ASK rather than reading as "no objection".

Sync and async firing
---------------------
Hooks run inside an async agent loop, and a hook that shells out to a policy service is
routine. `CommandHook.run()` uses a blocking `subprocess.run`, so calling it directly from
a coroutine stalls the WHOLE event loop (every other agent, request and task in the
process) for as long as the hook takes (up to its timeout, 30s by default).

So the engine exposes both shapes:
  * `fire()` / `gate()`: synchronous, for sync callers (PermissionEngine.check).
  * `fire_async()` / `gate_async()`: offload each hook to a worker thread via
    `asyncio.to_thread`, so a slow hook blocks only its own thread. The agent loop uses
    these. A hook that raises is contained and reported as a failed outcome rather than
    propagating into the permission path.

`prompt` and `agent` hook types (LLM-backed) are intentionally not implemented here so that
offline tests never silently pass on a stubbed model; see coordinator.py for where a live
classifier plugs in.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .events import HookEvent

if TYPE_CHECKING:
    # Type-only import to avoid a package import cycle (permissions <-> hooks).
    from engine.permissions.decision import Decision

# Wire-format version for the JSON handed to a CommandHook. A hook can branch on it, so it
# must change whenever a field's MEANING changes (adding a field does not).
HOOK_PAYLOAD_VERSION = 1

# Caps on the JSON handed to a command hook. A tool argument can be an entire file, and an
# unbounded write to a subprocess pipe is both a memory cost and (on the environment path)
# a hard OS limit: Windows caps the whole environment block at ~32 KB, so an oversized
# value there does not truncate, it makes CreateProcess fail and the hook never runs.
_MAX_STDIN_BYTES = 1_000_000
_MAX_ENV_VALUE = 8_000

# Environment variables a command hook can read instead of parsing stdin. The full document
# is always on stdin; these exist so a one-line shell hook is writable without a JSON parser.
ENV_PREFIX = "BARQ_HOOK_"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 14] + "…[TRUNCATED]"


@dataclass(frozen=True)
class HookInput:
    event: HookEvent
    tool_name: str | None = None
    match_content: str = ""
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The hook-facing representation of this call.

        Stable, JSON-serialisable and versioned: it is the contract a command hook is
        written against. `payload` carries the tool ARGUMENTS for a PreToolUse gate, which
        is what any real policy decision needs — a hook that can only see the tool's name
        can only express rules the `if_` string already expresses.
        """
        return {
            "version": HOOK_PAYLOAD_VERSION,
            "event": self.event.value if hasattr(self.event, "value") else str(self.event),
            "tool_name": self.tool_name,
            "match_content": self.match_content,
            "payload": self.payload,
        }

    def to_json(self, *, limit: int = _MAX_STDIN_BYTES) -> str:
        """Serialise for a subprocess. Never raises: a payload holding a non-serialisable
        object still has to reach the hook, because failing to serialise it would silently
        disarm the gate."""
        try:
            text = json.dumps(self.to_dict(), default=str, sort_keys=True)
        except (TypeError, ValueError):
            text = json.dumps(
                {
                    "version": HOOK_PAYLOAD_VERSION,
                    "event": str(self.event),
                    "tool_name": self.tool_name,
                    "match_content": self.match_content,
                    "payload": {"_unserialisable": True},
                },
                sort_keys=True,
            )
        if len(text.encode("utf-8")) > limit:
            # Truncating the JSON would hand the hook an unparseable document, so replace
            # the payload wholesale and say so — a hook must be able to tell "no arguments"
            # from "arguments too large to send".
            text = json.dumps(
                {
                    "version": HOOK_PAYLOAD_VERSION,
                    "event": self.event.value if hasattr(self.event, "value") else str(self.event),
                    "tool_name": self.tool_name,
                    "match_content": _truncate(self.match_content, 4_000),
                    "payload": {},
                    "payload_omitted": "payload exceeded the hook stdin limit",
                },
                sort_keys=True,
            )
        return text

    def to_env(self) -> dict[str, str]:
        """Hook-facing environment variables. The full document is on stdin; these are the
        convenience view for a shell one-liner."""
        return {
            f"{ENV_PREFIX}VERSION": str(HOOK_PAYLOAD_VERSION),
            f"{ENV_PREFIX}EVENT": self.event.value if hasattr(self.event, "value") else str(self.event),
            f"{ENV_PREFIX}TOOL_NAME": self.tool_name or "",
            f"{ENV_PREFIX}MATCH_CONTENT": _truncate(self.match_content, _MAX_ENV_VALUE),
            f"{ENV_PREFIX}INPUT": _truncate(self.to_json(limit=_MAX_ENV_VALUE), _MAX_ENV_VALUE),
        }


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


_VALID_DECISIONS = frozenset({"allow", "ask", "deny"})


def parse_hook_stdout(stdout: str) -> tuple[Optional["Decision"], Optional[str]]:
    """Read a structured verdict out of a command hook's stdout.

    Returns `(decision, error)`. A hook that prints nothing JSON-shaped returns
    `(None, None)` — that is the pre-existing advisory contract and it still holds, so a
    hook that only logs keeps working.

    Recognised shape (the last JSON object printed wins, so a hook may log freely first):

        {"decision": "allow" | "ask" | "deny", "reason": "..."}

    `permissionDecision` / `permissionDecisionReason` are accepted as aliases.

    A hook that emits a `decision` key with an unrecognised VALUE is an error, not a
    shrug: it tried to gate and we could not read the verdict, so the caller must escalate
    rather than continue as though nothing objected. That is the same fail-closed rule a
    hook that raises gets.
    """
    from engine.permissions.decision import allow, ask, deny  # lazy: import cycle

    text = (stdout or "").strip()
    if not text:
        return None, None

    obj: Any = None
    # Try the whole body first, then the last line — a hook that logs before printing its
    # verdict is normal, and requiring it to be silent would make the feature unusable.
    for candidate in (text, text.splitlines()[-1].strip()):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break
    if obj is None:
        return None, None

    raw = obj.get("decision", obj.get("permissionDecision"))
    if raw is None:
        return None, None
    verdict = str(raw).strip().lower()
    reason = str(
        obj.get("reason") or obj.get("permissionDecisionReason") or "hook decision"
    )[:500]
    if verdict not in _VALID_DECISIONS:
        return None, (
            f"hook returned an unrecognised decision {raw!r}; "
            f"expected one of {sorted(_VALID_DECISIONS)}"
        )
    if verdict == "deny":
        return deny("hook", reason), None
    if verdict == "ask":
        return ask("hook", reason), None
    return allow("hook", reason), None


@dataclass
class CommandHook:
    name: str
    command: str
    if_: str | None = None
    timeout: float = 30.0
    # Hand the hook the call it is gating. On by default: a gate that cannot see its
    # subject can only re-express its own `if_` string, which is what this used to be.
    # Turn it off only for a hook that must run with a pristine stdin/environment.
    pass_input: bool = True

    def _env_for(self, inp: HookInput) -> Optional[dict[str, str]]:
        if not self.pass_input:
            return None
        # Inherit the parent environment: replacing it strips PATH, and `shell=True` then
        # fails to find the interpreter the hook is written in.
        env = dict(os.environ)
        env.update(inp.to_env())
        return env

    def run(self, inp: HookInput) -> HookOutcome:
        """Execute the hook, handing it the tool call on stdin and in the environment.

        The call used to be dropped on the floor: `run()` accepted `inp` and never
        referenced it, so a command hook was a CONSTANT function of the tool call. It could
        not know which tool was running, with what arguments, against what target — which
        made every external policy integration (an OPA sidecar, a DLP service, a corporate
        approval API) impossible to write, and left `if_` as the only expressible rule.
        """
        stdin_doc = inp.to_json() if self.pass_input else None
        try:
            proc = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                input=stdin_doc,
                env=self._env_for(inp),
            )
        except subprocess.TimeoutExpired as e:
            # A gate that timed out has NO verdict. Reporting it as a clean non-blocking
            # outcome would read as "no objection", so it is an error and escalates.
            return HookOutcome(
                self.name, exit_code=124, stderr=f"timeout: {e}", errored=True
            )
        except OSError as e:
            return HookOutcome(
                self.name, exit_code=1, stderr=f"hook failed to start: {e}", errored=True
            )

        decision, parse_error = parse_hook_stdout(proc.stdout)
        stderr = proc.stderr
        if parse_error:
            stderr = f"{stderr}\n{parse_error}".strip()

        if proc.returncode == 2:
            # Exit 2 is the documented block signal and outranks anything on stdout: a hook
            # that both exits 2 and prints `{"decision":"allow"}` is contradicting itself,
            # and the strictest reading is the only safe one.
            from engine.permissions.decision import deny  # lazy: import cycle

            decision = deny(
                "hook",
                (proc.stderr or proc.stdout or "").strip()[:500]
                or f"hook {self.name} exited 2",
            )

        return HookOutcome(
            self.name,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=stderr,
            decision=decision,
            errored=bool(parse_error),
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
        """Resolve a PreToolUse gate, strictest verdict first: DENY > ASK > ALLOW.

        A hook that RAISED has no verdict, so the gate cannot conclude "nothing objected" —
        it escalates to ASK. That escalation applies even when ANOTHER hook allowed: a
        crash next to a permissive hook used to be discarded entirely, so a policy gate
        that threw was silently ignored exactly when some other hook happened to say yes.
        Treating a crashed policy gate as silence is a fail-OPEN bypass, and it does not
        stop being one because something else was optimistic.
        """
        deciding_allow: Optional[Decision] = None
        deciding_ask: Optional[Decision] = None
        errored: list[str] = []
        for outcome in outcomes:
            # A DENY is terminal whatever else went wrong in the same hook. A command hook
            # that exits 2 AND misprints its stdout still objected, and discarding that
            # objection because the same run also produced a parse error would turn the
            # strictest available verdict into an ASK.
            if outcome.decision is not None and outcome.decision.behavior.value == "deny":
                return outcome.decision
            if outcome.errored:
                errored.append(f"{outcome.hook_name}: {outcome.stderr}")
                continue
            if outcome.decision is not None:
                behavior = outcome.decision.behavior.value
                if behavior == "deny":
                    return outcome.decision  # first deny wins, fail-closed
                if behavior == "ask":
                    if deciding_ask is None:
                        deciding_ask = outcome.decision
                elif behavior == "allow" and deciding_allow is None:
                    deciding_allow = outcome.decision
            elif outcome.exit_code == 2:
                # command hook signalled a block without a structured decision
                from engine.permissions.decision import deny

                return deny("hook", f"hook {outcome.hook_name} exited 2", outcome.stderr)
        if errored:
            from engine.permissions.decision import ask

            return ask("hook", f"a PreToolUse hook failed; ask a human ({'; '.join(errored)})")
        # An ASK from one hook outranks an ALLOW from another: the stricter verdict wins.
        return deciding_ask or deciding_allow

    def gate(self, inp: HookInput) -> Optional[Decision]:
        """PreToolUse gate: return the deciding Decision, or None to fall through."""
        return self._resolve_gate(self.fire(inp))

    async def gate_async(self, inp: HookInput) -> Optional[Decision]:
        """PreToolUse gate, off the event loop (see fire_async)."""
        return self._resolve_gate(await self.fire_async(inp))
