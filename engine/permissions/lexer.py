"""Quote-aware lexing of a shell command line.

The danger checks used to run regexes over the raw command string. That fails in both
directions at once, and both failures were reproduced against the shipped code:

  * FALSE POSITIVES that no rule or hook can override, because a hard DENY is by design
    non-overridable. `\\b(shutdown|reboot|halt|poweroff)\\b` matched inside a quoted
    string, so `git commit -m "fix reboot handling"` and `grep -r "shutdown" ./src` were
    both denied outright, with no operator escape hatch.
  * FALSE NEGATIVES, because a pattern that does not lex cannot see structure:
    `echo cm0gLXJmIC8K | base64 -d | sh` deletes the filesystem root and produced no
    verdict at all.

The fix is to give the checks something with structure to inspect: the command line is
split into segments on the operators that separate commands (respecting quotes), each
segment is tokenised (respecting quotes and stripping comments), and a check then asks
"what is the COMMAND WORD, and what are its ARGUMENTS" rather than "does this substring
appear anywhere".

`shlex` does the tokenising where it can. Where it cannot — an unbalanced quote, a
construct it does not model — the segment is marked `parsed=False`, and the callers treat
an unparseable destructive-looking segment as suspicious rather than safe.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

__all__ = [
    "Segment",
    "lex_command",
    "split_segments",
    "substitution_bodies",
    "has_substitution",
]

# Command-list and pipeline operators. Two-character forms are matched first so `&&` is not
# read as two `&`, and `||` not as two `|`.
_OPERATORS = ("&&", "||", ";;", ";", "|", "&", "\n")

# Wrappers that stand in front of the real command without changing what it is. Stripping
# them is what lets `sudo rm -rf /` and `nohup env FOO=1 rm -rf /` reach the same check as
# a bare `rm -rf /`.
_WRAPPERS = frozenset({
    "sudo", "doas", "nohup", "time", "nice", "ionice", "stdbuf", "setsid", "env",
    "command", "builtin", "exec", "eval", "then", "else", "do", "!",
})
# Wrappers that take the real command after a flag/argument gap. `xargs -0 rm -rf /` and
# `timeout 5 rm -rf /` both hide the command one or more tokens deeper.
_ARG_WRAPPERS = frozenset({"xargs", "timeout", "watch", "flock", "chroot", "unbuffer"})

_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `/usr/bin` is a path; `/s` is a Windows switch. Anything with a further separator, or a
# name longer than a switch normally is, is treated as a path.
_LOOKS_LIKE_PATH = re.compile(r"^/(?:[^/]*/|[A-Za-z][A-Za-z0-9._-]{4,}$)")

# Command substitution. Its body is lexed as its own segment so `$(rm -rf /)` is still
# inspected; its mere presence also marks a segment as dynamic, which several checks treat
# as fail-closed because the effective command cannot be known statically.
_SUBST = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
_DYNAMIC = re.compile(r"\$\(|`|\$\{[^}]*\}|%[A-Za-z_][A-Za-z0-9_]*%")


def split_segments(command: str) -> list[str]:
    """Split a command line into individual command segments, respecting quotes.

    A naive `re.split(r"[;\\n|&]", cmd)` breaks `echo "a;b"` into two segments and
    `grep "a|b" f` into two more, so a check keyed on the first token of each segment then
    inspects fragments that were never commands.
    """
    if not command:
        return []
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(command[i + 1])
            i += 2
            continue
        matched = next((op for op in _OPERATORS if command.startswith(op, i)), None)
        if matched is not None:
            out.append("".join(buf))
            buf = []
            i += len(matched)
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [s for s in (seg.strip() for seg in out) if s]


def has_substitution(text: str) -> bool:
    """True if the text contains a construct whose value is not statically knowable."""
    return bool(_DYNAMIC.search(text or ""))


def substitution_bodies(command: str) -> list[str]:
    """The inside of every `$(...)` / backtick substitution, so it can be lexed in turn."""
    return [
        (m.group(1) if m.group(1) is not None else m.group(2)) or ""
        for m in _SUBST.finditer(command or "")
    ]


def _basename(token: str) -> str:
    """`/usr/bin/rm` -> `rm`, `C:\\Windows\\System32\\cmd.exe` -> `cmd.exe`."""
    return token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


@dataclass(frozen=True)
class Segment:
    """One command in a command list, lexed."""

    raw: str
    tokens: tuple[str, ...] = ()
    parsed: bool = True          # False => could not be lexed; callers fail closed
    argv0: str = ""              # basename of the command word, lowercased
    args: tuple[str, ...] = ()   # tokens after the command word
    wrappers: tuple[str, ...] = ()   # sudo / env / xargs / … that preceded it
    dynamic: bool = False        # contains $(...) / backticks / ${...} / %VAR%

    @property
    def elevated(self) -> bool:
        return any(w in ("sudo", "doas") for w in self.wrappers)

    def flags(self) -> set[str]:
        """Every short flag letter and long flag name in the arguments.

        `-rf`, `-r -f` and `--recursive --force` all have to answer the same question, and
        a check that only understood one spelling was a bypass by typing style.

        Short flags are recorded in BOTH cases. `-R` and `-r` mean the same thing to every
        check here, and storing only the literal character while `has_flag` lowercased its
        query meant `has_flag("r")` could not see `-R` — which silently disarmed the
        recursive-chmod, recursive-chown and iptables-flush checks entirely.
        """
        found: set[str] = set()
        for tok in self.args:
            if tok == "--":
                break
            if tok.startswith("--"):
                found.add(tok[2:].split("=", 1)[0].lower())
            elif tok.startswith("-") and len(tok) > 1:
                for ch in tok[1:]:
                    found.add(ch)
                    found.add(ch.lower())
                    found.add(ch.upper())
        return found

    def win_flags(self) -> set[str]:
        """Flags in the Windows `/x` / `/switch:value` style, lowercased.

        Kept separate from `flags()` on purpose: on POSIX `/f` is an absolute path, so
        reading every `/x` argument as a switch would misclassify operands. Only checks
        that already know they are looking at a Windows command consult this.
        """
        found: set[str] = set()
        for tok in self.args:
            if tok.startswith("/") and len(tok) > 1 and not tok.startswith("//"):
                found.add(tok[1:].split(":", 1)[0].lower())
        return found

    def has_win_flag(self, *names: str) -> bool:
        f = self.win_flags()
        return any(n.lower() in f for n in names)

    def win_operands(self) -> list[str]:
        """Arguments that are neither POSIX flags nor Windows switches."""
        return [
            t
            for t in self.args
            if not (t.startswith("-") and len(t) > 1)
            and not (t.startswith("/") and len(t) > 1 and not t.startswith("//")
                     and not _LOOKS_LIKE_PATH.match(t))
        ]

    def operands(self) -> list[str]:
        """Arguments that are not flags: the things the command acts ON."""
        out: list[str] = []
        seen_ddash = False
        for tok in self.args:
            if tok == "--" and not seen_ddash:
                seen_ddash = True
                continue
            if not seen_ddash and tok.startswith("-") and len(tok) > 1:
                continue
            out.append(tok)
        return out

    def has_flag(self, *names: str) -> bool:
        f = self.flags()
        return any(n.lower() in f for n in names)


# Backslash used as a PATH SEPARATOR rather than as a shell escape. posix lexing reads
# every backslash as an escape and deletes it, so `C:\Windows\cmd.exe` tokenised to
# `C:Windowscmd.exe` and `HKLM\SOFTWARE` to `HKLMSOFTWARE` — neither of which any path or
# registry check can recognise.
#
# The third alternative is the general case: a backslash BETWEEN two word characters. In
# POSIX a backslash before an ordinary letter is a no-op escape that yields the letter,
# so reading it as a separator instead costs nothing, while `\;`, `\|`, `\ ` and `\"` —
# the escapes that carry meaning — are excluded by the follow-set.
_WINDOWS_PATH = re.compile(
    r"[A-Za-z]:[\\/]"                          # C:\ or C:/
    r"|\\\\[^\\\s]"                            # \\server\share
    r"|[A-Za-z0-9_.$)-]\\[A-Za-z0-9_.$]"       # HKLM\SOFTWARE, .\build, dir\file
)


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _tokenize(text: str) -> tuple[tuple[str, ...], bool]:
    """Tokenise one segment. Returns (tokens, parsed_ok).

    `comments=True` drops a trailing `# …`, which is why a command carrying an explanatory
    comment no longer trips checks on words inside it.

    Windows command lines are lexed with `posix=False` so backslashes survive as path
    separators, with quotes stripped afterwards (non-posix mode keeps them on the token).
    """
    windows = bool(_WINDOWS_PATH.search(text))
    try:
        if windows:
            tokens = [_strip_quotes(t) for t in shlex.split(text, comments=True, posix=False)]
        else:
            tokens = shlex.split(text, comments=True)
        return tuple(tokens), True
    except ValueError:
        # Unbalanced quote or an unsupported construct. Fall back to a whitespace split so
        # a check still has SOMETHING to look at, and flag it so destructive-looking
        # segments are treated as unresolved rather than clean.
        return tuple(t for t in text.split() if t), False


def lex_command(command: str, *, include_substitutions: bool = True) -> list[Segment]:
    """Lex a whole command line into `Segment`s, innermost substitutions included.

    A substitution body becomes its own segment, so `eval "$(printf 'rm -rf /')"` presents
    the checks with an `rm` segment to inspect instead of one opaque string.
    """
    segments: list[Segment] = []
    for raw in split_segments(command):
        tokens, ok = _tokenize(raw)
        dynamic = has_substitution(raw)
        wrappers: list[str] = []
        idx = 0
        # Strip environment assignments and command wrappers to reach the real command.
        while idx < len(tokens):
            tok = tokens[idx]
            base = _basename(tok).lower()
            if _ENV_ASSIGN.match(tok):
                idx += 1
                continue
            if base in _WRAPPERS:
                wrappers.append(base)
                idx += 1
                continue
            if base in _ARG_WRAPPERS:
                wrappers.append(base)
                idx += 1
                # Skip this wrapper's own flags and its numeric/duration argument.
                while idx < len(tokens) and (
                    tokens[idx].startswith("-")
                    or re.fullmatch(r"\d+(\.\d+)?[smhd]?", tokens[idx])
                ):
                    idx += 1
                continue
            break
        argv0 = _basename(tokens[idx]).lower() if idx < len(tokens) else ""
        args = tokens[idx + 1:] if idx < len(tokens) else ()
        segments.append(
            Segment(
                raw=raw,
                tokens=tokens,
                parsed=ok,
                argv0=argv0,
                args=tuple(args),
                wrappers=tuple(wrappers),
                dynamic=dynamic,
            )
        )
        if include_substitutions:
            for body in substitution_bodies(raw):
                if body.strip():
                    segments.extend(lex_command(body, include_substitutions=False))
    return segments


def pipeline_stages(command: str) -> list[list[Segment]]:
    """Group segments into pipelines, so a check can ask what a pipeline ENDS in.

    `echo <base64> | base64 -d | sh` is only recognisable as decode-and-execute by looking
    at the whole pipeline; each stage on its own is innocuous.
    """
    stages: list[list[Segment]] = []
    current: list[Segment] = []
    buf: list[str] = []
    quote: str | None = None
    i, n = 0, len(command or "")

    def flush(text: str, end_pipeline: bool) -> None:
        nonlocal current
        text = text.strip()
        if text:
            current.extend(lex_command(text, include_substitutions=False))
        if end_pipeline and current:
            stages.append(current)
            current = []

    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if command.startswith("||", i):
            flush("".join(buf), True)
            buf = []
            i += 2
            continue
        if ch == "|":
            flush("".join(buf), False)
            buf = []
            i += 1
            continue
        if command.startswith("&&", i) or ch in (";", "\n", "&"):
            flush("".join(buf), True)
            buf = []
            i += 2 if command.startswith("&&", i) else 1
            continue
        buf.append(ch)
        i += 1
    flush("".join(buf), True)
    return stages
