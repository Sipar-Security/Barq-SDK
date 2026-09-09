"""Read-only command classifier: Claude Code's `isReadOnly` gate, ported.

Recovered helper: decides whether a shell command is *provably* read-only, so the
PermissionEngine can downgrade a mode-default ASK to ALLOW for it (auto-run `git status`,
`ls`, `Get-ChildItem` without a prompt) while everything else still asks. It is applied
ONLY at the mode-default step (after hook/scope/danger/rules) so it can never override a
deny; the worst it can do is *fail to* auto-allow a safe command (→ one extra prompt).

Design principle: fail CLOSED. If there is ANY doubt (metacharacters that could chain,
redirect, substitute, or execute), return False so the human is asked. Mirrors CC's
dangerous-pattern set plus per-tool read-only subcommand allowlists (git/gh/docker).
"""

from __future__ import annotations

import re

from engine.sandbox.filesystem import (
    CREDENTIAL_DIR_SEGMENTS, SENSITIVE_READ_NAMES, SENSITIVE_READ_SUFFIXES,
)

# A read-only command that nonetheless targets a secret path (`cat ~/.ssh/id_rsa`,
# `type project\.env`, `Get-Content server.pem`) must NOT auto-allow: otherwise the shell
# becomes a hole around the file-tool credential guard. Detecting it here downgrades the call
# to a normal ASK (fail-closed: at worst one extra prompt for a benign path that merely looks
# secret). Kept in sync with the FilesystemGuard read denylist (single source of truth).
_SENSITIVE_NAMES = {n for n in SENSITIVE_READ_NAMES}
_CRED_DIR_RE = re.compile(
    r"(?:^|[\\/~])(?:" + "|".join(re.escape(s) for s in CREDENTIAL_DIR_SEGMENTS) + r")(?:[\\/]|$)"
)


# Absolute paths that hold secrets but whose BASENAME is unremarkable, so the filename
# denylist above never saw them. `cat /etc/shadow` prints every password hash on the host
# and was auto-allowed with no prompt, because "shadow" is not a secret-looking filename.
_SENSITIVE_ABS_RE = re.compile(
    r"(?:^|[\s'\"=])(?:"
    r"/etc/(?:shadow|gshadow|sudoers|master\.passwd|security/|ssh/|krb5\.keytab)"
    r"|/proc/(?:self|\d+|\*)/environ"
    r"|/root(?:/|\b)"
    r"|/var/lib/(?:kubelet|rancher)/"
    r"|[A-Za-z]:[\\/]windows[\\/]system32[\\/]config[\\/](?:sam|system|security)"
    r")",
    re.IGNORECASE,
)

# The PowerShell environment drive. `Get-ChildItem Env:` is the cmdlet spelling of `env`.
_ENV_DRIVE_RE = re.compile(r"(?:^|[\s'\"])env:", re.IGNORECASE)


def _references_sensitive_path(command: str) -> bool:
    lowered = command.lower()
    if _CRED_DIR_RE.search(lowered):
        return True
    if _SENSITIVE_ABS_RE.search(command):
        return True
    if _ENV_DRIVE_RE.search(command):
        return True
    for tok in command.split():
        base = tok.strip("'\"").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
        if base in _SENSITIVE_NAMES or base.endswith(SENSITIVE_READ_SUFFIXES):
            return True
    return False


# Commands that do not mutate anything and are therefore "read-only" in the narrow sense,
# but whose OUTPUT is a secret. Auto-allowing them equates "does not write" with "safe to
# run unattended", which is exactly backwards for an exfiltration threat model — and this
# module's own threat model, per SECURITY.md, is a prompt-injected model.
#
#   env / printenv  -> every API key in the process environment, including the model
#                      provider key this SDK itself loaded from .env
#   ps / top        -> other processes' command lines, which routinely carry credentials
#                      (`mysql -pSECRET`, `curl -H "Authorization: ..."`)
#   set / export    -> the shell's own variables
#   history         -> previously typed commands, secrets included
#
# Refusing them costs one approval prompt. Allowing them costs the credential.
_SECRET_DISCLOSING_BINS = frozenset({
    "env", "printenv", "set", "export", "declare", "typeset", "history",
    "ps", "top", "htop", "pgrep", "lsof", "get-process", "gps", "get-variable",
    "systemctl-show", "journalctl", "dmesg", "last", "lastlog", "w",
})

# Patterns that make a command NOT provably read-only: command substitution, chaining,
# redirection, in-place edit, PowerShell method/expression calls, stop-parsing, UNC paths,
# and ANY parenthesis. Parentheses are load-bearing here: PowerShell evaluates `(...)` and
# `$(...)`/`@(...)`/`&(...)` subexpressions IMMEDIATELY, so `echo (Remove-Item x)` runs
# Remove-Item: it is NOT read-only. We reject any `(`/`)` (a legit read-only command that
# happens to contain a paren just falls through to a normal ASK prompt, which is safe).
# NOTE: a plain pipe `|` is allowed and handled by splitting into segments (each segment
# must itself be read-only), so `ls | grep x | wc -l` stays read-only.
_DANGEROUS = (
    "$(", "`", "${", "&&", "||", ";", "&", "\n", "\r",
    ">", "<", "::", "--%", "@(", "\\\\",
    "(", ")", " = ", "+=", "--in-place",
)

# Commands that only observe state. First token (basename) must be in here.
_READONLY_BINS = frozenset({
    # POSIX
    "ls", "pwd", "whoami", "id", "hostname", "uname", "date", "uptime",
    "echo", "cat", "head", "tail", "wc", "which", "type", "file",
    "stat", "du", "df", "free", "tree", "basename", "dirname", "realpath",
    "readlink", "grep", "egrep", "fgrep", "rg", "find", "locate", "sort", "uniq",
    "cut", "md5sum", "sha1sum", "sha256sum", "cksum", "true", "test",
    # PowerShell read-only cmdlets / aliases
    "get-childitem", "gci", "dir", "get-content", "gc", "get-item", "get-itemproperty",
    "test-path", "select-string", "sls", "measure-object", "get-location", "gl",
    "resolve-path", "split-path", "join-path", "get-command", "gcm", "get-help",
    "get-member", "gm", "get-service", "compare-object", "sort-object",
    "select-object", "format-list", "format-table", "out-string", "write-output",
    "write-host", "get-date", "get-host", "convertto-json", "convertfrom-json",
})

# Sub-command allowlists for tools that are read-only only in some modes. Membership here is
# necessary but NOT sufficient: a mutating flag/subcommand (checked below) still disqualifies
# the call, because `git branch` reads but `git branch -D` deletes, `docker system` reads but
# `docker system prune` wipes.
_GIT_RO = frozenset({
    "status", "log", "diff", "show", "branch", "tag", "remote", "rev-parse",
    "describe", "ls-files", "ls-tree", "cat-file", "blame", "shortlog", "reflog",
    "whatchanged", "name-rev", "symbolic-ref", "for-each-ref", "count-objects",
    "config",  # only a pure getter/list - see _git_config_read_only
})
# Tokens (subcommands OR flags) that make a git/docker call MUTATING: any one disqualifies.
_GIT_MUTATING = frozenset({
    "-d", "-D", "--delete", "-M", "--move", "-f", "--force", "-u", "--set-upstream",
    "--unset", "--add", "--replace-all", "--edit", "--amend", "--set",
    "remove", "rm", "prune", "add", "commit", "push", "pull", "fetch", "merge", "rebase",
    "reset", "checkout", "switch", "restore", "clean", "stash", "init", "clone", "apply",
    "cherry-pick", "revert", "gc", "mv", "am", "set-url", "set-head", "update-ref",
})
_GH_RO = frozenset({"pr", "issue", "repo", "run", "release", "workflow", "status"})
_GH_RO_SUB = frozenset({"list", "view", "status", "diff", "checks"})
_DOCKER_RO = frozenset({
    "ps", "images", "inspect", "logs", "version", "info", "top", "port", "stats",
    "history", "diff", "search", "context", "system",
})
_DOCKER_MUTATING = frozenset({
    "prune", "create", "rm", "rmi", "destroy", "kill", "stop", "start", "restart", "run",
    "exec", "build", "load", "import", "save", "use", "update", "commit", "push", "pull",
    "tag", "pause", "unpause", "rename", "export", "cp", "-f", "--force",
})


def _git_config_read_only(args: list[str]) -> bool:
    """`git config` is never auto-allowed.

    It used to auto-allow a pure getter, `--list` included. But `git config --list` prints
    `credential.helper`, and remote URLs configured with an embedded token
    (`https://x-access-token:ghp_…@github.com/…`) come back in the same output — so a
    "read-only" call handed over a live credential with no prompt. `--get <key>` discloses
    the same thing one key at a time.

    Kept as a function rather than deleted so the reason travels with the code: this is a
    deliberate refusal, not an oversight. The cost is one approval prompt on an uncommon
    command; the alternative cost is the credential.
    """
    return False


def _tokens(command: str) -> list[str]:
    return command.strip().split()


def _basename(tok: str) -> str:
    return tok.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()


def _segment_read_only(seg: str) -> bool:
    """True if a single (pipe-free) command segment is provably read-only."""
    seg = seg.strip()
    if not seg:
        return False
    # PowerShell splatting (@var) / method call (.foo()) => not provably read-only.
    if re.search(r"@\w", seg) or re.search(r"\.\w+\(", seg):
        return False
    toks = _tokens(seg)
    if not toks:
        return False
    base = _basename(toks[0])
    args = [t.lower() for t in toks[1:]]
    # Read-only, but the output IS the secret. Never auto-allowed.
    if base in _SECRET_DISCLOSING_BINS:
        return False
    if base == "git" and len(toks) >= 2:
        sub = toks[1].lower()
        if sub not in _GIT_RO:
            return False
        if any(a in _GIT_MUTATING for a in args):  # git branch -D, git remote remove, ...
            return False
        if sub == "config":
            return _git_config_read_only(toks[2:])
        return True
    if base == "gh" and len(toks) >= 3:
        return toks[1].lower() in _GH_RO and toks[2].lower() in _GH_RO_SUB
    if base == "docker" and len(toks) >= 2:
        if toks[1].lower() not in _DOCKER_RO:
            return False
        return not any(a in _DOCKER_MUTATING for a in args)  # docker system prune -f, ...
    if base in ("cargo", "npm", "pip", "pip3", "go", "mvn", "gradle", "make"):
        return False  # build tools can run scripts / fetch / execute
    if base in ("find", "locate"):
        return False  # find can mutate via -delete / -exec; never auto-allow
    return base in _READONLY_BINS


def is_read_only_command(command: str) -> bool:
    """True only if `command` is provably read-only. Fail-closed on any doubt.

    A plain pipeline is read-only iff EVERY segment is read-only (`ls | grep x | wc -l`),
    but any substitution/chaining/redirection construct disqualifies the whole command.
    """
    cmd = (command or "").strip()
    if not cmd:
        return False
    if _references_sensitive_path(cmd):
        return False  # reading a secret path is never auto-allowed: fall through to ASK
    for pat in _DANGEROUS:
        if pat in cmd:
            return False
    segments = [s for s in cmd.split("|") if s.strip()]
    if not segments:
        return False
    return all(_segment_read_only(s) for s in segments)
