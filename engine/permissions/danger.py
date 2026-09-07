"""Deterministic dangerous-command detection.

The LLM intent classifier is non-deterministic and prompt-injectable, yet it was the only
thing standing between the agent and an obviously catastrophic *local* command. This layer
catches the well-known destructive shapes with plain regex. It returns a Decision whose
behavior distinguishes HARD danger (DENY — catastrophic, must never be allow-listable) from
SOFT danger (ASK — dangerous-but-sometimes-legit). The PermissionEngine applies the hard
DENY *before* any allow rule and the soft ASK *after* rules (see PermissionEngine._deterministic).

Scope: SHELL tools only (bash/shell/powershell). We deliberately do NOT scan HTTP payloads
or other tool inputs — a `DROP TABLE`/`rm -rf` appearing inside data the agent SENDS somewhere is a payload,
not a local action, and must not be blocked here.
"""

from __future__ import annotations

import posixpath
import re
from typing import Optional

from .decision import Decision, ask, deny
from .engine import ToolCall, match_content

_SHELL_TOOLS = {"bash", "shell", "powershell", "sh", "cmd"}

# Top-level system directories whose recursive/forced deletion wrecks the host.
_SYS_DIR = (
    r"/(?:etc|usr|var|bin|sbin|lib|lib64|libexec|boot|root|home|opt|dev|proc|sys|srv|"
    r"run|mnt|media|snap)"
)
# Split a command line into its component commands so EVERY `rm` is inspected, not just the
# first (closes `rm ok ; rm -rf /etc` and `x && rm -rf /` chaining bypasses).
_CMD_SEP = re.compile(r"[;\n|&]")


def _normalize_target(tok: str) -> str:
    """Resolve a delete target to a canonical form so obfuscated paths can't hide a wipe:
    collapse `..`/`.` (`/usr/local/../..` -> `/`, `/etc/.` -> `/etc`), unify Windows drives
    to a marker, and map home aliases. Preserves a trailing `/*` glob indicator."""
    t = tok.strip().strip("'\"")
    if not t:
        return ""
    # Windows drive roots: C:\  C:  C:\*  C:/  \  \\
    if re.fullmatch(r"[A-Za-z]:[\\/]?\*?", t) or t in ("\\", "\\\\", "//"):
        return "\x00WINROOT"
    if t in ("~", "$HOME", "${HOME}", "%USERPROFILE%", "%HOMEPATH%", "%HOMEDRIVE%"):
        return "\x00HOME"
    if re.fullmatch(r"~[\\/]?\*?|\$\{?HOME\}?[\\/]?\*?", t):
        return "\x00HOME"
    # Normalize a filesystem path (posix), keeping a trailing-glob marker.
    unified = t.replace("\\", "/")
    glob = unified.rstrip().endswith("/*") or unified.rstrip() == "*"
    if unified.startswith("/") or ".." in unified or "/." in unified or unified in (".", ".."):
        norm = posixpath.normpath(unified)
        if glob and not norm.endswith("*"):
            norm = (norm.rstrip("/") or "") + "/*"
        return norm
    return unified


def _is_catastrophic_target(tok: str) -> bool:
    """True if a delete TARGET (after normalization) is a drive/system root, home, or a
    bare/root glob. A specific deep path (rm -rf /home/me/proj, ./build, /tmp/x) is NOT caught."""
    n = _normalize_target(tok)
    if n in ("\x00WINROOT", "\x00HOME"):
        return True
    if n in ("/", "/*", "*", ".", "./*"):
        return True
    if re.fullmatch(_SYS_DIR + r"(?:/\*?)?", n, re.IGNORECASE):
        return True
    return False


def _parse_rm(rest: str) -> tuple[bool, bool, list[str]]:
    """Parse the args after `rm` (within a single command) into (recursive?, force?, targets),
    handling flags in ANY form: combined (-rf/-fr), separated (-r -f), long (--recursive)."""
    recursive = force = False
    targets: list[str] = []
    for tok in rest.split():
        tl = tok.lower()
        if tl in ("--recursive", "-r", "-R") or tl.startswith("--recursive"):
            recursive = True
        if tl in ("--force",) or tl.startswith("--force"):
            force = True
        if tl == "--no-preserve-root":
            recursive = force = True
        if tok.startswith("-") and not tok.startswith("--"):  # short flag cluster, e.g. -rf
            if re.search(r"[rR]", tok):
                recursive = True
            if "f" in tok:
                force = True
        elif not tok.startswith("-"):
            targets.append(tok)
    return (recursive, force, targets)


def _seg_catastrophic_rm(seg: str) -> bool:
    m = re.search(r"(?<![\w./-])rm(?![\w.-])", seg, re.IGNORECASE)
    if not m:
        return False
    rest = seg[m.end():]
    recursive, force, targets = _parse_rm(rest)
    if not (recursive or force):
        return False
    # A subshell/backtick/eval target can't be statically resolved (`rm -rf (echo /etc)`,
    # `rm -rf $(...)`, `rm -rf \`...\``) — fail CLOSED and treat it as catastrophic.
    if re.search(r"[`(]|\$\{|%\w+%", rest):
        return True
    return any(_is_catastrophic_target(t) for t in targets)


def _catastrophic_rm(cmd: str) -> bool:
    """Recursive/forced rm of a catastrophic target — a host/data wipe. Checks EVERY command
    in a chained line, normalizes paths, covers Windows drives, and fails closed on dynamic
    (subshell) targets. Hard DENY."""
    return any(_seg_catastrophic_rm(seg) for seg in _CMD_SEP.split(cmd))


# Catastrophic non-rm shapes: destroy the host / data. Hard DENY.
_DENY = [
    (r"--no-preserve-root", "rm with --no-preserve-root"),
    (r"\bmkfs\.[a-z0-9]+", "filesystem format"),
    (r"\bdd\b[^\n]*\bof=/dev/", "raw write to a block device"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\b(shutdown|reboot|halt|poweroff|init\s+0)\b", "host power control"),
    (r">\s*/dev/sd[a-z]", "overwrite of a block device"),
    (r"\bchmod\s+-R\s+0*777\s+/", "world-writable filesystem root"),
    (r"\bdrop\s+(table|database)\b", "destructive SQL (DROP)"),
    (r"\btruncate\s+table\b", "destructive SQL (TRUNCATE)"),
]

# Risky but sometimes legitimate. ASK (human-gated), never silent. Any recursive/forced rm of
# a NON-catastrophic path lands here (it wasn't hard-denied above).
_ASK = [
    (r"\brm\b(?:\s+\S+)*\s+(?:-[a-zA-Z]*[rRfF]|--recursive|--force)", "recursive/force delete"),
    (r"\brm\s+(?:-[a-zA-Z]*[rRfF]|--recursive|--force)", "recursive/force delete"),
    (r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(sh|bash|zsh)\b", "pipe-to-shell"),
    (r"\bgit\s+push\b[^\n]*--force", "git force push"),
    (r"\bdelete\s+from\b(?![^\n;]*\bwhere\b)", "unbounded SQL DELETE"),
    (r"\bsudo\b", "privilege escalation"),
]


def builtin_danger(call: ToolCall) -> Optional[Decision]:
    """Return a hard-DENY / soft-ASK Decision for a dangerous shell command, else None."""
    if call.name.lower() not in _SHELL_TOOLS:
        return None
    cmd = match_content(call)
    if _catastrophic_rm(cmd):
        return deny("danger", "blocked dangerous command: recursive delete of root/system/home")
    for pat, why in _DENY:
        if re.search(pat, cmd, re.IGNORECASE):
            return deny("danger", f"blocked dangerous command: {why}")
    for pat, why in _ASK:
        if re.search(pat, cmd, re.IGNORECASE):
            return ask("danger", f"dangerous command needs approval: {why}")
    return None
