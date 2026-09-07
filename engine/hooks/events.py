"""Hook events fired by the engine.

These are exactly the lifecycle points the coordinator/permission engine actually fire:
nothing aspirational. A hook registered on one of these runs; there are no declared-but-
never-fired events.

  PRE_TOOL_USE: before a tool runs; a PreToolUse hook may gate it (allow/ask/deny).
  POST_TOOL_USE: after a tool returns; advisory (logging, side effects).
  PERMISSION_DENIED: when a tool call is denied; advisory (alerting, metrics).
"""

from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PERMISSION_DENIED = "PermissionDenied"


HOOK_EVENTS: tuple[HookEvent, ...] = tuple(HookEvent)
