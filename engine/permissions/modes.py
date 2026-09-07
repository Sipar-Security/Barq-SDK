"""Permission modes: the default behavior a tool call resolves to when no hook, rule,
network policy, danger check, or classifier has already decided it.

  AUTO: allow by default (unattended agent). Danger checks and rules still apply.
  ASK: ask a human by default (human-in-the-loop). A provably read-only call can be
         auto-allowed via the read-only classifier.
  LOCKED: deny by default (only explicitly allow-ruled calls run).
"""

from __future__ import annotations

from enum import Enum

from .decision import Behavior


class Mode(str, Enum):
    AUTO = "auto"
    ASK = "ask"
    LOCKED = "locked"


MODE_DEFAULTS: dict[Mode, Behavior] = {
    Mode.AUTO: Behavior.ALLOW,
    Mode.ASK: Behavior.ASK,
    Mode.LOCKED: Behavior.DENY,
}
