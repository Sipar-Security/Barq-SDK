"""Deterministic dangerous-command detection.

The LLM intent classifier is non-deterministic and prompt-injectable, yet it was the only
thing standing between the agent and an obviously catastrophic *local* command. This layer
catches the well-known destructive shapes deterministically. It returns a Decision whose
behavior distinguishes HARD danger (DENY: catastrophic, must never be allow-listable) from
SOFT danger (ASK: dangerous-but-sometimes-legit). The PermissionEngine applies the hard
DENY *before* any allow rule and the soft ASK *after* rules.

Why this is lexed, not pattern-matched
--------------------------------------
The previous implementation ran regexes over the raw command string, and failed in both
directions at once. Both failures were reproduced against the shipped code:

  * `git commit -m "fix reboot handling in init"` was DENIED — `\\b(reboot)\\b` matched
    inside a quoted string. A hard DENY is non-overridable by design, so there was no
    operator escape hatch for a false positive on ordinary English.
  * `echo cm0gLXJmIC8K | base64 -d | sh` produced NO verdict. It deletes the filesystem
    root. A pattern that cannot see structure cannot see a decode-and-execute pipeline.

Checks now run over `engine.permissions.lexer.Segment`s and ask *what is the command word,
and what are its operands* — so a word inside a string literal or a `#` comment is data,
and `sudo env FOO=1 /bin/rm -rf /` reaches the same check as `rm -rf /`.

Scope: a call is inspected only when it EXECUTES A COMMAND LINE. A `code` parameter holds
source code in some language, not a shell command line, and scanning it for `DROP TABLE`
denied a SQL migration whose text merely contained the words. SQL checks now fire only when
the command word is a database client.

Precision budget: a hard DENY cannot be overridden, so it must not fire on anything a
competent operator would call routine. A soft ASK reaches a human, so it can be liberal.
"""

from __future__ import annotations

import posixpath
import shlex
import re
from typing import Optional

from .decision import Decision, ask, deny
from .engine import ToolCall, match_content
from .lexer import Segment, lex_command, pipeline_stages

# Names an integrator plausibly gives a command-execution tool. This list can never be
# complete, which is exactly why it is not the only test - see `is_shell_call`.
_SHELL_TOOLS = {
    "bash", "shell", "powershell", "pwsh", "sh", "zsh", "cmd", "command",
    "exec", "execute", "run", "runcommand", "run_command", "terminal", "console",
    "bash_tool", "shell_tool", "shellexec", "system", "subprocess", "localshell",
    "executecommand", "execute_command", "runshell", "run_shell", "cli",
}
# Substrings that mark a tool name as command-execution even when the exact name is new
# (`AzureRunCommand`, `k8s_exec`, `sandbox-bash`).
_SHELL_NAME_FRAGMENTS = ("shell", "bash", "powershell", "terminal", "exec", "command")
# Argument names that carry a COMMAND LINE. `code` is deliberately absent: it holds source
# code, and treating it as a shell command line is what made
# `RunCode({"code": 'sql = "DROP TABLE tmp"'})` an unblockable hard DENY, contradicting
# this module's own documented scope.
_COMMAND_KEYS = ("command", "cmd", "commandline", "command_line", "shell",
                 "bash", "powershell", "argv", "args_string", "script")
# `script`/`code` still count as a command line when the TOOL is a shell tool by name:
# `Bash({"script": "rm -rf /"})` is a shell call however the parameter is spelled.
_SHELL_TOOL_COMMAND_KEYS = _COMMAND_KEYS + ("code",)


def command_text(call: ToolCall) -> str:
    """The command line a call would execute, or "" if it does not look like one."""
    keys = _SHELL_TOOL_COMMAND_KEYS if _named_shell(call) else _COMMAND_KEYS
    for key, value in (call.input or {}).items():
        if str(key).lower().replace("-", "_") in keys and isinstance(value, str):
            return value
    return ""


def _named_shell(call: ToolCall) -> bool:
    name = (call.name or "").lower().replace("-", "_")
    return name in _SHELL_TOOLS or any(f in name for f in _SHELL_NAME_FRAGMENTS)


def is_shell_call(call: ToolCall) -> bool:
    """True if this call executes a command line.

    Keying the danger checks on an exact tool-name set was a fail-OPEN coupling: this SDK
    ships no shell tool at all, so the whole layer only engaged if the integrator happened
    to name their tool `bash`. Named `RunCommand`, `exec` or `Terminal`, `rm -rf /` sailed
    through. Two independent tests catch it - the tool's name, or a command-line-shaped
    argument - and either is enough.
    """
    return _named_shell(call) or bool(command_text(call))


# --- path taxonomy ------------------------------------------------------------
# Directories where a recursive delete of ANYTHING BENEATH them wrecks the host. Deleting
# /usr/bin leaves an unbootable machine just as surely as deleting /usr.
_FATAL_ROOTS = (
    "etc", "usr", "bin", "sbin", "lib", "lib64", "libexec", "boot", "dev", "proc",
    "sys", "root", "snap",
)
# Deeper paths that are equally fatal (data and state that cannot be reconstructed).
_FATAL_SUBPATHS = ("/var/lib", "/var/log", "/var/spool", "/var/db", "/etc")
# Directories where only the directory ITSELF is catastrophic. `rm -rf /home` destroys
# every user; `rm -rf /home/me/build` is a routine developer action and must not be denied.
_ROOT_ONLY = ("home", "opt", "mnt", "media", "srv", "run", "var", "tmp", "users")

_SYS_DIR_RE = "|".join(_FATAL_ROOTS + _ROOT_ONLY)

# Windows equivalents. The whole layer was POSIX-only, so on a Windows host — which this
# project's own CI matrix and development environment use — `del /f /s /q C:\Windows` and
# `format C: /y` produced no verdict at all.
_WIN_FATAL_ROOTS = (
    "windows", "winnt", "program files", "program files (x86)", "programdata",
    "system32", "syswow64", "boot", "perflogs", "recovery", "$recycle.bin",
)
# Directories where only the directory ITSELF is catastrophic (deleting every user's data),
# not a path beneath it (one user's project folder).
_WIN_ROOT_ONLY = ("users", "documents and settings")


def _normalize_target(tok: str) -> str:
    """Resolve a delete target to canonical form so an obfuscated path cannot hide a wipe:
    collapse `..`/`.`, unify Windows drives to a marker, map home aliases. A trailing `/*`
    glob indicator is preserved."""
    t = (tok or "").strip().strip("'\"")
    if not t:
        return ""
    if re.fullmatch(r"[A-Za-z]:[\\/]?\*?", t) or t in ("\\", "\\\\", "//"):
        return "\x00WINROOT"
    if t in ("~", "$HOME", "${HOME}", "%USERPROFILE%", "%HOMEPATH%", "%HOMEDRIVE%"):
        return "\x00HOME"
    if re.fullmatch(r"~[\\/]?\*?|\$\{?HOME\}?[\\/]?\*?", t):
        return "\x00HOME"
    unified = t.replace("\\", "/")
    glob = unified.rstrip().endswith("/*") or unified.rstrip() == "*"
    if unified.startswith("/") or ".." in unified or "/." in unified or unified in (".", ".."):
        norm = posixpath.normpath(unified)
        if glob and not norm.endswith("*"):
            norm = (norm.rstrip("/") or "") + "/*"
        return norm
    return unified


def is_catastrophic_target(tok: str) -> bool:
    """True if deleting this path recursively wrecks the host or destroys everything.

    A specific deep path under a user-owned root (`/home/me/proj`, `./build`, `/tmp/x`) is
    NOT catastrophic - denying those would make the layer unusable, and a soft ASK already
    covers any recursive delete.
    """
    n = _normalize_target(tok)
    if not n:
        return False
    if n in ("\x00WINROOT", "\x00HOME"):
        return True
    if n in ("/", "/*", "*", ".", "./*", "..", "../*"):
        return True
    stripped = n[:-2] if n.endswith("/*") else n
    stripped = stripped.rstrip("/") or "/"
    if stripped == "/":
        return True

    # Windows drive-relative paths: `C:/Windows`, `C:/Program Files/...`, `C:/Users`.
    drive = re.match(r"^([A-Za-z]):/(.*)$", stripped)
    if drive:
        rest = drive.group(2).strip().lower().rstrip("/")
        if not rest or rest == "*":
            return True
        head = rest.split("/", 1)[0]
        if head in _WIN_FATAL_ROOTS:
            return True
        if head in _WIN_ROOT_ONLY and "/" not in rest:
            return True
        return False
    # The directory itself, for every recognised system root.
    if re.fullmatch(rf"/(?:{_SYS_DIR_RE})", stripped, re.IGNORECASE):
        return True
    # Anything beneath a fatal root.
    if re.match(rf"^/(?:{'|'.join(_FATAL_ROOTS)})/", stripped + "/", re.IGNORECASE):
        return True
    for prefix in _FATAL_SUBPATHS:
        if stripped.lower() == prefix or stripped.lower().startswith(prefix + "/"):
            return True
    return False


# --- helpers over a lexed segment --------------------------------------------
_REDIRECT = re.compile(r"^>{1,2}(.*)$")


def _redirect_targets(seg: Segment) -> list[str]:
    """Files this segment writes to via `>` / `>>`, whether or not a space follows."""
    out: list[str] = []
    tokens = list(seg.tokens)
    for i, tok in enumerate(tokens):
        m = _REDIRECT.match(tok)
        if not m:
            continue
        rest = m.group(1)
        if rest:
            out.append(rest)
        elif i + 1 < len(tokens):
            out.append(tokens[i + 1])
    return out


_DELETE_CMDS = frozenset({"rm", "rmdir", "unlink", "del", "erase"})
_RECURSIVE_FLAGS = ("r", "recursive", "R")
_FORCE_FLAGS = ("f", "force")
_DECODERS = frozenset({
    "base64", "b64decode", "xxd", "uudecode", "openssl", "gunzip", "zcat", "bunzip2",
    "xz", "unxz", "atob",
})
# Programs that execute a COMMAND LINE handed to them. Their `-c` argument can be re-lexed
# as shell.
_SHELL_INTERPRETERS = frozenset({
    "sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "cmd", "cmd.exe",
    "powershell", "pwsh", "powershell.exe",
})
# Everything that can be the receiving end of a pipe-to-execute. Wider than the above,
# because `… | python` runs whatever arrives on stdin just as surely as `… | sh`.
_INTERPRETERS = _SHELL_INTERPRETERS | frozenset({
    "python", "python2", "python3", "py", "perl", "ruby", "node", "php", "lua",
})
_DB_CLIENTS = frozenset({
    "psql", "mysql", "mariadb", "sqlite3", "sqlite", "mongo", "mongosh", "cqlsh",
    "clickhouse-client", "redis-cli", "sqlcmd", "sqlplus", "cockroach", "duckdb",
})
_POWER_CMDS = frozenset({"shutdown", "reboot", "halt", "poweroff"})


_WIN_DELETE_CMDS = frozenset({"del", "erase", "rd", "rmdir", "remove-item", "ri", "rm.exe"})


def _is_recursive_delete(seg: Segment) -> bool:
    a0 = seg.argv0
    # Windows: `del /f /s /q`, `rd /s /q`, PowerShell `Remove-Item -Recurse -Force`.
    if a0 in _WIN_DELETE_CMDS:
        if a0 in ("rd", "rmdir"):
            return True
        if seg.has_win_flag("s", "q", "f"):
            return True
        if seg.has_flag("recurse", "r", "force", "f"):
            return True
        if a0 in ("del", "erase"):
            return False
        return True
    if a0 not in _DELETE_CMDS:
        return False
    return seg.has_flag(*_RECURSIVE_FLAGS) or seg.has_flag(*_FORCE_FLAGS)


def _catastrophic_delete(seg: Segment) -> Optional[str]:
    """Reason string if this segment is a host- or data-destroying delete, else None."""
    if seg.argv0 == "find":
        ops = seg.operands()
        roots = [o for o in ops if o.startswith(("/", "~")) or o in (".", "..")]
        deletes = seg.has_flag("delete") or "-delete" in seg.args or (
            "-exec" in seg.args and any(a in ("rm", "/bin/rm") for a in seg.args)
        )
        if deletes and any(is_catastrophic_target(r) for r in roots):
            return "recursive find-and-delete of a system root"
        return None
    if not _is_recursive_delete(seg):
        return None
    if seg.has_flag("no-preserve-root"):
        return "rm with --no-preserve-root"
    targets = seg.operands()
    # A target that is not statically knowable cannot be cleared. `rm -rf $(cat list)` may
    # be `/`, so an unresolved recursive delete fails closed.
    if seg.dynamic:
        return "recursive delete of a target that cannot be resolved statically"
    if not seg.parsed:
        return "recursive delete in a command line that could not be parsed"
    if any(is_catastrophic_target(t) for t in targets):
        return "recursive delete of root/system/home"
    if not targets:
        return None
    return None


def _decode_then_execute(command: str) -> Optional[str]:
    """A pipeline that DECODES data and feeds it to an interpreter.

    `echo cm0gLXJmIC8K | base64 -d | sh` deletes the filesystem root. No pattern over the
    raw string sees it, and no static analysis can see what the decoded bytes are - which
    is exactly why it is refused rather than inspected. There is no legitimate reason for
    an autonomous agent to decode a blob and execute it.
    """
    for stage in pipeline_stages(command):
        if len(stage) < 2:
            continue
        names = [s.argv0 for s in stage]
        if names[-1] not in _INTERPRETERS:
            continue
        if any(n in _DECODERS for n in names[:-1]):
            return "decoding data and piping it into a shell"
    return None


def _pipe_to_shell(command: str) -> Optional[str]:
    for stage in pipeline_stages(command):
        if len(stage) < 2:
            continue
        if stage[-1].argv0 in _INTERPRETERS and stage[0].argv0 not in _INTERPRETERS:
            return f"piping {stage[0].argv0 or 'output'} into {stage[-1].argv0}"
    return None


_INTERP_DELETE = re.compile(
    r"\b(?:rmtree|removedirs|remove_tree|unlink|rmdir|rm\s+-[a-zA-Z]*[rf])\b", re.I
)
_INTERP_UNRESOLVED = re.compile(
    r"\b(?:expanduser|expandvars|environ|getenv|chr\s*\(|decode|b64decode|argv)\b", re.I
)
_INTERP_FATAL_LITERAL = re.compile(
    rf"['\"](?:/|~|/(?:{'|'.join(_FATAL_ROOTS)})\b)", re.I
)


def _interpreter_one_liner(seg: Segment) -> Optional[str]:
    """An inline program that deletes recursively.

    Reached the same syscalls as `rm -rf /` without ever spelling `rm`. The code is only
    cleared when its delete target is a plainly safe literal: a target built from
    `expanduser`, the environment, `chr()` or a decode is unanalysable, so it fails closed.
    """
    if seg.argv0 not in ("python", "python2", "python3", "py", "perl", "ruby", "node"):
        return None
    code = ""
    args = list(seg.args)
    for i, tok in enumerate(args):
        if tok in ("-c", "-e", "--eval") and i + 1 < len(args):
            code = args[i + 1]
            break
        if tok.startswith("-c") and len(tok) > 2:
            code = tok[2:]
            break
    if not code or not _INTERP_DELETE.search(code):
        return None
    if _INTERP_FATAL_LITERAL.search(code):
        return "interpreter one-liner deleting a system path"
    if _INTERP_UNRESOLVED.search(code):
        return "interpreter one-liner deleting a target that cannot be resolved statically"
    return None


# --- the checks ---------------------------------------------------------------
_FORK_BOMB = re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")
_DEVICE = re.compile(r"^/dev/(?:sd[a-z]|nvme\d|vd[a-z]|hd[a-z]|xvd[a-z]|disk\d|mmcblk\d)",
                     re.IGNORECASE)
_CRITICAL_FILE = re.compile(
    r"^/(?:etc/(?:passwd|shadow|sudoers|fstab|hosts)|boot/)", re.IGNORECASE
)


def _hard_check(seg: Segment, command: str) -> Optional[str]:
    """Catastrophic: DENY, before any rule and past any hook allow."""
    reason = _catastrophic_delete(seg)
    if reason:
        return reason

    a0, args = seg.argv0, list(seg.args)
    joined = " ".join(args)

    if a0.startswith("mkfs"):
        return "filesystem format"
    if a0 == "dd" and any(_DEVICE.match(a[3:]) for a in args if a.lower().startswith("of=")):
        return "raw write to a block device"
    if a0 in _POWER_CMDS:
        return "host power control"
    if a0 == "init" and args and args[0] in ("0", "6"):
        return "host power control"
    if a0 == "systemctl" and args and args[0].lower() in ("poweroff", "reboot", "halt"):
        return "host power control"
    if a0 in ("iptables", "ip6tables") and (seg.has_flag("F", "flush") or seg.has_flag("X")):
        return "flushing the host firewall rules"
    if a0 == "nft" and "flush" in args and "ruleset" in args:
        return "flushing the host firewall rules"
    if a0 == "crontab" and seg.has_flag("r", "remove"):
        return "deleting every scheduled job (crontab -r)"
    if a0 in ("userdel", "deluser") and any(o in ("root", "administrator") for o in seg.operands()):
        return "deleting the root account"
    if a0 in ("chmod", "chown", "chgrp") and seg.has_flag(*_RECURSIVE_FLAGS):
        if any(is_catastrophic_target(o) for o in seg.operands()[1:] or seg.operands()):
            return f"recursive {a0} of a system root"
    if a0 == "mv" and any(is_catastrophic_target(o) for o in seg.operands()[:-1]):
        return "moving a system directory"
    if a0 == "shred" and any(is_catastrophic_target(o) for o in seg.operands()):
        return "irrecoverable wipe of a system path"
    for target in _redirect_targets(seg):
        if _DEVICE.match(target):
            return "overwrite of a block device"
        if _CRITICAL_FILE.match(target):
            return "overwrite of a critical system file"
    if seg.has_flag("no-preserve-root"):
        return "rm with --no-preserve-root"

    one_liner = _interpreter_one_liner(seg)
    if one_liner:
        return one_liner
    if a0 == "chmod" and seg.has_flag(*_RECURSIVE_FLAGS) and "777" in joined:
        if any(is_catastrophic_target(o) for o in seg.operands()):
            return "world-writable filesystem root"

    win = _windows_hard_check(seg)
    if win:
        return win
    return None


_REG_ROOT = re.compile(
    r"^(?:HKLM|HKCR|HKU|HKEY_LOCAL_MACHINE|HKEY_CLASSES_ROOT|HKEY_USERS)"
    r"(?:[\\/](?:SOFTWARE|SYSTEM|SAM|SECURITY)?[\\/]?)?$",
    re.IGNORECASE,
)


def _windows_hard_check(seg: Segment) -> Optional[str]:
    """Catastrophic Windows commands.

    The layer was POSIX-only. On a Windows host — which this project's own CI matrix and
    development environment use — `format C: /y`, `del /f /s /q C:\\Windows` and
    `vssadmin delete shadows /all` all produced no verdict whatsoever.
    """
    a0, args = seg.argv0, list(seg.args)
    lowered = [a.lower() for a in args]

    if a0 == "format" and any(
        re.fullmatch(r"[A-Za-z]:[\\/]?", a) for a in seg.win_operands()
    ):
        return "formatting a drive"
    if a0 == "vssadmin" and "delete" in lowered and "shadows" in lowered:
        # Volume shadow copies are the host's restore points. Destroying them is the
        # signature move of ransomware and has no benign automated use.
        return "deleting volume shadow copies (backup destruction)"
    if a0 == "wmic" and "shadowcopy" in lowered and "delete" in lowered:
        return "deleting volume shadow copies (backup destruction)"
    if a0 == "wbadmin" and "delete" in lowered and any(
        t in lowered for t in ("catalog", "backup", "systemstatebackup")
    ):
        return "deleting the system backup catalog"
    if a0 == "bcdedit" and seg.has_win_flag("set") and any(
        t in lowered for t in ("safeboot", "recoveryenabled", "bootstatuspolicy")
    ):
        return "tampering with the host boot configuration"
    if a0 == "reg" and lowered[:1] == ["delete"] and any(
        _REG_ROOT.match(a) for a in args[1:]
    ):
        return "deleting a registry root hive"
    if a0 == "diskpart" and "clean" in lowered:
        return "wiping a disk with diskpart"
    if a0 == "takeown" and seg.has_win_flag("r") and any(
        is_catastrophic_target(a) for a in seg.win_operands()
    ):
        return "taking ownership of a system root"
    if a0 == "net" and lowered[:1] == ["user"] and seg.has_win_flag("delete") and any(
        a.lower() in ("administrator", "admin") for a in seg.win_operands()[1:]
    ):
        return "deleting the administrator account"
    return None


_SQL_DESTRUCTIVE = re.compile(
    r"\b(?:drop\s+(?:table|database|schema)|truncate\s+table)\b", re.IGNORECASE
)
_SQL_UNBOUNDED_DELETE = re.compile(
    r"\bdelete\s+from\b(?![^;]*\bwhere\b)", re.IGNORECASE
)
_CREDENTIAL_PATH = re.compile(
    r"(?:\.ssh/|\.aws/|\.gnupg/|\.kube/|\.env\b|credentials\b|id_rsa\b|id_ed25519\b"
    r"|\.pem\b|\.p12\b|/etc/shadow|/etc/sudoers)", re.IGNORECASE
)
_READ_CMDS = frozenset({"cat", "type", "less", "more", "head", "tail", "get-content",
                        "strings", "xxd", "od"})
_NET_SEND_CMDS = frozenset({"curl", "wget", "nc", "ncat", "netcat", "scp", "rsync", "ftp"})


def _soft_check(seg: Segment, command: str) -> Optional[str]:
    """Dangerous but sometimes legitimate: ASK a human."""
    a0, args = seg.argv0, list(seg.args)
    joined = " ".join(args)

    if seg.elevated:
        return "privilege escalation"
    if _is_recursive_delete(seg):
        return "recursive/force delete"
    if a0 == "shred":
        return "irrecoverable file wipe"
    if a0 == "git":
        if args and args[0] == "push" and seg.has_flag("force", "f"):
            return "git force push"
        if args and args[0] == "reset" and seg.has_flag("hard"):
            return "discarding uncommitted work"
        if args and args[0] == "clean" and (seg.has_flag("f") or seg.has_flag("d")):
            return "discarding uncommitted work"
        if args and args[0] == "filter-branch":
            return "rewriting repository history"
    if a0 in _DB_CLIENTS:
        if _SQL_DESTRUCTIVE.search(joined):
            return "destructive SQL (DROP/TRUNCATE)"
        if _SQL_UNBOUNDED_DELETE.search(joined):
            return "unbounded SQL DELETE"
    if a0 == "kubectl" and args and args[0] == "delete":
        return "deleting Kubernetes resources"
    if a0 == "terraform" and args and args[0] == "destroy":
        return "destroying Terraform-managed infrastructure"
    if a0 == "aws" and any(
        t.startswith(("delete", "terminate", "remove")) for t in args
    ):
        return "deleting AWS resources"
    if a0 in ("gcloud", "az") and "delete" in args:
        return "deleting cloud resources"
    if a0 == "docker" and (
        ("system" in args and "prune" in args) or ("volume" in args and "rm" in args)
    ):
        return "destroying Docker state"
    if a0 == "systemctl" and args and args[0].lower() in (
        "stop", "disable", "mask", "kill"
    ):
        return "stopping or disabling a system service"
    if a0 in ("chown", "chgrp") and seg.has_flag(*_RECURSIVE_FLAGS):
        return "recursive ownership change"
    if a0 == "mv" and any(_normalize_target(o) == "\x00HOME" or
                          _normalize_target(o).startswith(("/home/", "/Users/"))
                          for o in seg.operands()[:-1]):
        return "moving a home directory"
    if a0 in ("killall", "pkill") or (a0 == "kill" and "-9" in args and "-1" in args):
        return "mass process termination"
    if a0 in ("npm", "pip", "pip3", "gem", "cargo", "go", "yarn", "pnpm") and (
        "install" in args or "add" in args
    ):
        return "installing a third-party package"
    if a0 == "history" and seg.has_flag("c"):
        return "clearing shell history"
    if a0 == "unset" and "HISTFILE" in args:
        return "clearing shell history"
    if any(t.endswith(".bash_history") or t.endswith(".zsh_history")
           for t in _redirect_targets(seg)):
        return "clearing shell history"
    # Reading a credential and sending it somewhere. Neither half is catastrophic alone.
    if a0 in _NET_SEND_CMDS and _CREDENTIAL_PATH.search(joined):
        return "sending a credential file to a network destination"
    if a0 in _READ_CMDS and _CREDENTIAL_PATH.search(joined):
        return "reading a credential file"
    if a0 == "eval" or "eval" in seg.wrappers:
        if seg.dynamic:
            return "eval of a dynamically constructed command"
    # Windows, dangerous but legitimate.
    if a0 == "cipher" and any(a.lower().startswith("/w") for a in args):
        return "secure wipe of free disk space"
    if a0 == "sc" and args and args[0].lower() in ("delete", "stop", "config"):
        return "modifying a Windows service"
    if a0 == "reg" and args and args[0].lower() in ("delete", "add"):
        return "modifying the Windows registry"
    if a0 == "icacls" and any("everyone" in a.lower() for a in args):
        return "granting Everyone access to a path"
    if a0 == "netsh" and any(a.lower() == "firewall" for a in args):
        return "changing the Windows firewall configuration"
    return None


# Commands that carry ANOTHER command line in one of their arguments. `ansible all -m shell
# -a "rm -rf /"` executes on every managed host and the destructive part never appears as a
# command word in the outer line, so the outer lexing alone cannot see it.
_REMOTE_EXEC: dict[str, tuple[str, ...]] = {
    "ansible": ("-a", "--args"),
    "ansible-playbook": (),
    "ssh": (),
    "salt": (),
    "pssh": ("-i",),
    "clusterssh": (),
}
# Sub-commands after which the remaining arguments are a command line on another host.
_NESTED_AFTER = {
    ("kubectl", "exec"), ("docker", "exec"), ("docker", "run"), ("podman", "exec"),
    ("nerdctl", "exec"), ("lxc", "exec"), ("vagrant", "ssh"),
}


# ssh flags that consume the NEXT argument. Skipping a flag without skipping its value
# leaves the value looking like the destination, so `ssh -p 22 host rm -rf /` had its
# nested command line read as `host rm -rf /` and the `rm` was never the command word.
_SSH_VALUE_FLAGS = frozenset({
    "-p", "-i", "-o", "-l", "-F", "-b", "-c", "-D", "-e", "-I", "-J", "-L", "-m",
    "-O", "-Q", "-R", "-S", "-w", "-W",
})
# `sh -c "<command line>"` — the payload is an argument, so the outer segment's command
# word is the interpreter and the destructive part is invisible without unwrapping.
_DASH_C_FLAGS = frozenset({"-c", "--command", "-Command", "/c", "/C", "-EncodedCommand"})


def _skip_flags(args: list[str], start: int, value_flags: frozenset[str] = frozenset()) -> int:
    """Index of the first non-flag argument at or after `start`."""
    i = start
    while i < len(args) and args[i].startswith("-") and args[i] != "--":
        if args[i] in value_flags:
            i += 1  # also skip the flag's value
        i += 1
    return i


def _nested_command_lines(seg: Segment) -> list[str]:
    """Command lines this segment would execute somewhere else.

    Rebuilt with `shlex.join` rather than a bare space join so a nested argument that
    contains spaces survives re-lexing as ONE token instead of splitting into several.
    """
    out: list[str] = []
    args = list(seg.args)

    if seg.argv0 in _REMOTE_EXEC:
        flags = _REMOTE_EXEC[seg.argv0]
        for i, tok in enumerate(args):
            if tok in flags and i + 1 < len(args):
                out.append(args[i + 1])
        if seg.argv0 == "ssh":
            # ssh [flags] [user@]host <command...>. Skipping flags with `operands()` also
            # stripped the nested command's OWN flags, so `ssh h rm -rf /` lost the `-rf`
            # and the nested check saw a harmless `rm /`.
            i = _skip_flags(args, 0, _SSH_VALUE_FLAGS)
            if i + 1 < len(args):
                out.append(shlex.join(args[i + 1:]))

    # A SHELL invoked with `-c` carries a whole command line as one argument.
    #
    # Restricted to real shells on purpose. Unwrapping a LANGUAGE interpreter's `-c` and
    # re-lexing its source as a command line reintroduces exactly the false-positive class
    # this module exists to remove: `python -c "reboot = True"` lexes to the command word
    # `reboot` and was denied as host power control. Python/Perl/Ruby/Node one-liners are
    # handled by `_interpreter_one_liner`, which reads them AS code.
    if seg.argv0 in _SHELL_INTERPRETERS:
        for i, tok in enumerate(args):
            if tok in _DASH_C_FLAGS and i + 1 < len(args):
                out.append(args[i + 1])
                break

    if args and (seg.argv0, args[0]) in _NESTED_AFTER:
        if "--" in args:
            # `kubectl exec pod -- rm -rf /`: everything after `--` is the remote command,
            # flags included. This is the canonical spelling and it must not be re-filtered.
            nested = args[args.index("--") + 1:]
        else:
            i = _skip_flags(args, 1)   # past the sub-command's own flags
            i += 1                     # past the pod / container name
            nested = args[i:]
        if nested:
            out.append(shlex.join(nested))

    return [c for c in out if c.strip()]


def _eval_of_decoded_blob(command: str) -> Optional[str]:
    """`eval "$(echo <base64> | base64 -d)"` is decode-and-execute wearing a different hat.

    The decode-then-execute pipeline check does not see it: the substitution body's
    pipeline does not END in a shell, `eval` is what executes the result.
    """
    from .lexer import substitution_bodies

    for seg in lex_command(command, include_substitutions=False):
        if seg.argv0 != "eval" and "eval" not in seg.wrappers:
            continue
        for body in substitution_bodies(seg.raw):
            if any(inner.argv0 in _DECODERS for inner in lex_command(body, include_substitutions=False)):
                return "eval of a decoded blob"
    return None


def builtin_danger(call: ToolCall) -> Optional[Decision]:
    """Return a hard-DENY / soft-ASK Decision for a dangerous shell command, else None."""
    if not is_shell_call(call):
        return None
    command = command_text(call) or match_content(call)
    if not command:
        return None
    return _inspect(command, depth=0)


def _inspect(command: str, depth: int) -> Optional[Decision]:
    if depth > 2:  # bound the recursion into nested command lines
        return None

    if _FORK_BOMB.search(command):
        return deny("danger", "blocked dangerous command: fork bomb")

    for reason in (_decode_then_execute(command), _eval_of_decoded_blob(command)):
        if reason:
            return deny("danger", f"blocked dangerous command: {reason}")

    segments = lex_command(command)

    # A command line executed on another host is still a command line. Inspect it with the
    # same checks and report a nested verdict at the same severity.
    for seg in segments:
        for nested in _nested_command_lines(seg):
            verdict = _inspect(nested, depth + 1)
            if verdict is not None and verdict.behavior.value == "deny":
                return deny(
                    "danger",
                    f"{verdict.message} (in a command line dispatched by {seg.argv0!r})",
                )

    for seg in segments:
        hard = _hard_check(seg, command)
        if hard:
            return deny("danger", f"blocked dangerous command: {hard}")

    for seg in segments:
        soft = _soft_check(seg, command)
        if soft:
            return ask("danger", f"dangerous command needs approval: {soft}")

    piped = _pipe_to_shell(command)
    if piped:
        return ask("danger", f"dangerous command needs approval: {piped}")
    return None
