"""Permission decision types: port of CC's src/types/permissions.ts.

CC behaviors: allow | ask | deny | passthrough. We keep allow/ask/deny (passthrough
is an internal chaining detail we express by returning None from a check instead).
The `reason` mirrors CC's PermissionDecisionReason discriminated union so audit logs
can say *why* something was allowed/denied (rule/mode/hook/scope/classifier).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Behavior(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class Decision:
    behavior: Behavior
    reason_type: str  # "rule" | "mode" | "hook" | "scope" | "classifier" | "other"
    message: str
    reason_detail: str | None = None


def allow(reason_type: str, message: str, detail: str | None = None) -> Decision:
    return Decision(Behavior.ALLOW, reason_type, message, detail)


def ask(reason_type: str, message: str, detail: str | None = None) -> Decision:
    return Decision(Behavior.ASK, reason_type, message, detail)


def deny(reason_type: str, message: str, detail: str | None = None) -> Decision:
    return Decision(Behavior.DENY, reason_type, message, detail)
