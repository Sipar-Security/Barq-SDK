"""Degenerate-loop detection for the agent loop.

A small model can fail in two ways the turn-budget cannot catch:

  1. *Intra-response degeneration* — a single completion repeats one line until it hits the
     token cap (the classic "Let me also check X … Let me also check X …" wall). Because the
     turn ends normally, the max_turns counter never fires; the run just burns its whole
     budget in one giant useless turn.
  2. *Cross-turn stall* — the model emits the same reasoning + the same tool call, turn after
     turn, making no progress.

`LoopGuard` is a pure, deterministic detector for both. The Coordinator feeds it every
assistant turn; when it trips, the loop injects a corrective nudge once, and if the very next
turn trips again it stops the run (recorded, not silent). Default-None on the Coordinator, so
existing behaviour is unchanged unless a caller opts in.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class GuardVerdict:
    tripped: bool
    kind: str = ""      # "degenerate_text" | "stalled_repeat"
    reason: str = ""


@dataclass
class LoopGuard:
    # A response is degenerate if one line (>= min_line_len chars) repeats at least
    # repeat_line_threshold times AND those repeats are >= repeat_line_ratio of all non-empty
    # lines. Both conditions matter: a long legitimate answer can repeat a short boilerplate
    # line a few times without being degenerate.
    repeat_line_threshold: int = 6
    repeat_line_ratio: float = 0.5
    min_line_len: int = 8
    # A stall is stall_threshold consecutive turns with an identical (reasoning-head, tools)
    # signature.
    stall_threshold: int = 3
    # Stop the run after this many CONSECUTIVE tripped turns (i.e. one nudge, then stop).
    abort_after: int = 2

    _sig_history: list = field(default_factory=list)
    _consecutive_trips: int = 0
    _total_trips: int = 0

    # --- detectors -----------------------------------------------------------

    def _degenerate(self, text: str) -> str | None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < self.repeat_line_threshold:
            return None
        counts = Counter(ln for ln in lines if len(ln) >= self.min_line_len)
        if not counts:
            return None
        line, n = counts.most_common(1)[0]
        if n >= self.repeat_line_threshold and n / len(lines) >= self.repeat_line_ratio:
            pct = round(100 * n / len(lines))
            return f"one line repeated {n}x ({pct}% of the output): {line[:60]!r}"
        return None

    @staticmethod
    def _signature(text: str, tool_calls: list[dict]) -> tuple:
        head = " ".join(text.split())[:160].lower()
        tools = tuple(sorted(
            (str(c.get("name", "")), json.dumps(c.get("input", {}), sort_keys=True)[:200])
            for c in tool_calls
        ))
        return (head, tools)

    # --- the one call the loop makes each turn -------------------------------

    def inspect(self, text: str, tool_calls: list[dict]) -> GuardVerdict:
        """Record this assistant turn and report whether the run has gone degenerate."""
        text = text or ""
        tool_calls = tool_calls or []

        deg = self._degenerate(text)
        if deg:
            return self._trip("degenerate_text", deg)

        sig = self._signature(text, tool_calls)
        self._sig_history.append(sig)
        recent = self._sig_history[-self.stall_threshold:]
        # A stall needs real content — an empty (no text, no tools) signature never trips.
        if (len(recent) >= self.stall_threshold and len(set(recent)) == 1
                and (sig[0] or sig[1])):
            return self._trip("stalled_repeat",
                              f"identical turn repeated {self.stall_threshold}x in a row")

        # Progress made: clear the consecutive-trip and stall state (keep the total for audit).
        self._consecutive_trips = 0
        return GuardVerdict(False)

    def _trip(self, kind: str, reason: str) -> GuardVerdict:
        self._consecutive_trips += 1
        self._total_trips += 1
        return GuardVerdict(True, kind, reason)

    # --- caller helpers ------------------------------------------------------

    @property
    def should_abort(self) -> bool:
        """True once the model has tripped abort_after times in a row (nudge didn't help)."""
        return self._consecutive_trips >= self.abort_after

    @property
    def total_trips(self) -> int:
        return self._total_trips

    @staticmethod
    def nudge() -> str:
        return (
            "STOP — you are repeating yourself and making no progress. Do not repeat your "
            "previous message. Take exactly ONE of these actions now: (a) call a tool with "
            "NEW, different arguments that concretely advances the task, or (b) if the task is "
            "complete, give your final answer / call your finish tool. Any further repetition "
            "will end the run."
        )
