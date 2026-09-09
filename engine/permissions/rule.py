"""Permission-rule syntax parser: port of CC's rule syntax used by both hook `if`
conditions (schemas/hooks.ts:19-27) and permission rules.

Syntax:  ToolName            -> matches any call to that tool
         ToolName(pattern)   -> matches that tool AND a glob match against the
                                call's "match content" (the command for Bash, the
                                url/target for network tools, the path for Read).

Examples: "Bash(git *)", "ReadFile(*.py)", "HttpRequest(https://*.example.com/*)", "MyTool".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch

# The tool-name half accepts `*` and `?` so one rule can cover a family of tools. An MCP
# server contributes N namespaced tools (`github__create_issue`, `github__merge_pr`, …) and
# without a wildcard an operator had to enumerate every one of them — `mcp__*` and `*` both
# raised ValueError, so there was no way to write "everything from this server" at all.
# `-` and `.` are allowed too: MCP tool names legitimately contain them.
_RULE = re.compile(r"^\s*([A-Za-z0-9_*?.-]+)\s*(?:\(\s*(.*?)\s*\))?\s*$", re.DOTALL)


@dataclass(frozen=True)
class PermissionRule:
    tool_name: str
    pattern: str | None  # None => match any call to this tool


def parse_rule(text: str) -> PermissionRule:
    m = _RULE.match(text)
    if not m:
        raise ValueError(f"unparseable permission rule: {text!r}")
    tool, pat = m.group(1), m.group(2)
    return PermissionRule(tool_name=tool, pattern=pat if pat else None)


def rule_matches(rule: PermissionRule, tool_name: str, match_content: str) -> bool:
    # Tool-name match is case-INSENSITIVE: an operator writing `bash(rm *)` must still
    # gate the tool the code registers as `Bash`. A case-sensitive compare here silently
    # voided the rule (and any hook `if` built on it), a fail-OPEN bypass.
    # It is also a glob, so `mcp__github__*` or `*` covers a whole family in one rule.
    if not fnmatch(tool_name.lower(), rule.tool_name.lower()):
        return False
    if rule.pattern is None:
        return True
    # glob match; also allow the pattern to be a prefix-style "git *" against a
    # multi-word command (fnmatch handles the trailing *).
    return fnmatch(match_content, rule.pattern)
