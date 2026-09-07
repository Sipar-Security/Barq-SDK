"""PermissionEngine — the tool-permission decision flow.

Order:
  1. PreToolUse hook gate      (fast, local, can hard-deny/allow)
  2. HARD-danger DENY          (catastrophic shell commands, before any rule)
  3. network policy            (optional; each host a call would contact is checked)
  4. explicit deny/allow/ask rules
  5. SOFT-danger ASK           (dangerous-but-sometimes-legit shell commands)
  6. optional live classifier  (LLM; injected, not built in)
  7. mode default behavior

Fail-closed: if any step raises, the caller should treat it as ASK (surface to a human),
never silently allow. A configured network policy that rejects a target DENIES the call
regardless of mode (or, under `network_ask`, asks a human).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Optional

from engine.hooks.engine import HookEngine, HookInput
from engine.hooks.events import HookEvent

from .decision import Behavior, Decision, allow, ask, deny
from .modes import MODE_DEFAULTS, Mode
from .network import NetworkPolicy, extract_host, is_reserved_ip
from .rule import parse_rule, rule_matches


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict = field(default_factory=dict)


# --- tool semantics: how to read a target/content out of a tool call -----------
# Any field a tool might name a network destination with.
_NETWORK_TARGET_KEYS = (
    "url", "uri", "link", "target", "targets", "host", "hosts", "hostname",
    "domain", "domains", "endpoint", "addr", "address", "base_url", "callback",
    "redirect_uri", "proxy", "urls", "ip", "webhook", "origin", "server", "upstream",
)
# A key is also a destination if its NAME contains one of these fragments, so a custom tool
# field (`webhook_url`, `callbackUri`, `api_endpoint`, `targetHost`) is checked too. Exact-
# match-only was a fail-OPEN bypass: any destination field the list did not literally name
# was invisible to the network policy.
_NETWORK_KEY_FRAGMENTS = (
    "url", "uri", "host", "endpoint", "addr", "domain", "webhook", "callback", "proxy",
)


def _is_network_key(key: str) -> bool:
    k = key.lower()
    return k in _NETWORK_TARGET_KEYS or any(f in k for f in _NETWORK_KEY_FRAGMENTS)
_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>|\\]+")
_HOSTLIKE_RE = re.compile(r"^(?!-)(?:[A-Za-z0-9_-]+\.)+[A-Za-z0-9_-]+(?::\d+)?$")
# Shell binaries that take a host/URL as an argument, so `curl http://x` is target-checked.
_NET_BINARIES = {
    "curl", "wget", "nc", "ncat", "netcat", "telnet", "ssh", "scp", "ftp", "sftp",
    "ping", "dig", "host", "nslookup",
}


def _iter_strings(v) -> Iterable[str]:
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _iter_strings(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _iter_strings(x)


def _hosts_from_command(cmd: str) -> list[str]:
    """Destinations named in a shell command: any URL, plus the first non-flag arg to a
    known network binary (`curl example.com`, `nc host 4444`)."""
    out = [m.group(0) for m in _URL_RE.finditer(cmd)]
    toks = cmd.split()
    for i, t in enumerate(toks):
        base = t.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
        if base in _NET_BINARIES:
            for nxt in toks[i + 1:]:
                if nxt.startswith("-"):
                    continue
                if _URL_RE.match(nxt) or _HOSTLIKE_RE.match(nxt):
                    out.append(nxt)
                break
    return out


def network_targets(call: ToolCall) -> list[str]:
    """Every host/URL this call would contact. Empty for non-network calls.

    Only recognised destination fields and shell network-binary args count — a URL that
    merely appears inside a request body is a payload sent TO a target, not a host the
    engine itself dials, so it is not treated as a destination.
    """
    out: list[str] = []
    if call.name.lower() in ("bash", "shell", "powershell"):
        out.extend(_hosts_from_command(str(call.input.get("command", ""))))
    for k, v in call.input.items():
        if _is_network_key(str(k)):
            out.extend(s for s in _iter_strings(v) if s.strip())
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _target_is_reserved_ip(target: str) -> bool:
    """True only if `target`'s host is a RAW reserved/internal IP literal. A hostname is
    not reserved here — so `network_ask` can let a human approve a public host, while a
    raw internal address (SSRF/metadata) stays hard-denied."""
    host = extract_host(target)
    try:
        import ipaddress

        ipaddress.ip_address(host)
    except ValueError:
        return False
    return is_reserved_ip(host)


def network_target(call: ToolCall) -> Optional[str]:
    """The first host a tool call will contact, or None (logging convenience)."""
    ts = network_targets(call)
    return ts[0] if ts else None


def match_content(call: ToolCall) -> str:
    """String matched by permission-rule / hook `if` patterns."""
    if call.name.lower() in ("bash", "shell", "powershell"):
        return str(call.input.get("command", ""))
    tgt = network_target(call)
    return tgt if tgt is not None else " ".join(str(v) for v in call.input.values())


# Injected live classifiers: (call) -> Decision|None. None => no opinion.
Classifier = Callable[[ToolCall], Optional[Decision]]
AsyncClassifier = Callable[[ToolCall], Awaitable[Optional[Decision]]]


class PermissionEngine:
    def __init__(
        self,
        hooks: HookEngine,
        mode: Mode = Mode.AUTO,
        *,
        network_policy: Optional[NetworkPolicy] = None,
        allow_rules: tuple[str, ...] = (),
        deny_rules: tuple[str, ...] = (),
        ask_rules: tuple[str, ...] = (),
        classifier: Optional[Classifier] = None,
        async_classifier: Optional[AsyncClassifier] = None,
        read_only_classifier: Optional[Callable[["ToolCall"], bool]] = None,
        danger_check: Optional[Callable[["ToolCall"], Optional[Decision]]] = None,
        network_ask: bool = False,
    ) -> None:
        self.hooks = hooks
        self.mode = mode
        self.network_policy = network_policy
        self._allow = [parse_rule(r) for r in allow_rules]
        self._deny = [parse_rule(r) for r in deny_rules]
        self._ask = [parse_rule(r) for r in ask_rules]
        self.classifier = classifier
        self.async_classifier = async_classifier
        # Optional read-only auto-allow (Claude Code's isReadOnly gate). Applied ONLY at
        # the mode-default step and only to DOWNGRADE an ASK to ALLOW.
        self.read_only_classifier = read_only_classifier
        # Optional dangerous-shell-command detector. Defaults to the built-in one
        # (engine.permissions.danger); pass a callable to override, or `lambda c: None`
        # to disable it entirely.
        if danger_check is None:
            from .danger import builtin_danger

            danger_check = builtin_danger
        self._danger_check = danger_check
        # When True, a host rejected by the network policy becomes ASK (a human can
        # approve reaching it once) instead of a hard DENY — except a raw reserved/internal
        # IP literal, which always hard-DENYs.
        self.network_ask = network_ask

    def _gate_input(self, call: ToolCall, content: str) -> HookInput:
        return HookInput(
            event=HookEvent.PRE_TOOL_USE,
            tool_name=call.name,
            match_content=content,
            payload=call.input,
        )

    def _deterministic(
        self, call: ToolCall, content: str, gate: Optional[Decision] = None,
        gate_done: bool = False,
    ) -> Optional[Decision]:
        """Hook -> HARD-danger DENY -> network -> rules -> SOFT-danger ASK. Returns a
        terminal Decision, or None to fall through to the classifier / mode-default step.

        Order is deliberate: catastrophic commands are HARD-denied BEFORE the network
        check and any rule, so an explicit allow rule can never open a path to `rm -rf /`.
        Dangerous-but-legit commands are SOFT-asked AFTER rules, so an operator who
        explicitly allow-rules such a command still gets it.

        `gate_done` lets the async path run the PreToolUse hooks off the event loop and
        hand the result in, instead of firing them again here.
        """
        if not gate_done:
            gate = self.hooks.gate(self._gate_input(call, content))
        if gate is not None:
            return gate

        danger = self._danger_check(call)
        if danger is not None and danger.behavior is Behavior.DENY:
            return danger

        if self.network_policy is not None:
            for tgt in network_targets(call):
                verdict = self.network_policy.check(tgt)
                if not verdict.allowed:
                    if self.network_ask and not _target_is_reserved_ip(tgt):
                        return ask("network", f"host needs approval: {verdict.reason}")
                    return deny("network", f"blocked: {verdict.reason}", verdict.matched)

        for r in self._deny:
            if rule_matches(r, call.name, content):
                return deny("rule", f"matched deny rule {r.tool_name}({r.pattern})")
        for r in self._ask:
            if rule_matches(r, call.name, content):
                return ask("rule", f"matched ask rule {r.tool_name}({r.pattern})")
        for r in self._allow:
            if rule_matches(r, call.name, content):
                return allow("rule", f"matched allow rule {r.tool_name}({r.pattern})")

        if danger is not None:
            return danger
        return None

    def _mode_default(self) -> Decision:
        default = MODE_DEFAULTS[self.mode]
        if default is Behavior.ALLOW:
            return allow("mode", f"{self.mode.value} default allow")
        if default is Behavior.DENY:
            return deny("mode", f"{self.mode.value} default deny")
        return ask("mode", f"{self.mode.value} default ask")

    def _finalize(self, call: ToolCall, decision: Decision) -> Decision:
        """Downgrade a mode-default ASK to ALLOW for a provably read-only call. Never
        touches a deny/ask that came from a rule, hook, network, danger, or classifier."""
        if decision.behavior is Behavior.ASK and self.read_only_classifier is not None:
            try:
                if self.read_only_classifier(call):
                    return allow("readonly", "read-only command auto-approved")
            except Exception:
                pass
        return decision

    def check(self, call: ToolCall) -> Decision:
        """Synchronous decision path (no async LLM classifier)."""
        content = match_content(call)
        try:
            terminal = self._deterministic(call, content)
        except Exception as e:  # fail-CLOSED: a broken gate must never allow
            return ask("other", f"permission check errored ({type(e).__name__}: {e}); ask human")
        if terminal is not None:
            return terminal
        if self.classifier is not None:
            try:
                c = self.classifier(call)
            except Exception as e:  # fail-safe: a broken classifier must NOT allow
                return ask("classifier", f"classifier error ({e}); ask human")
            if c is not None and c.behavior is not Behavior.ALLOW:
                return c
        return self._finalize(call, self._mode_default())

    async def check_async(self, call: ToolCall) -> Decision:
        """Async decision path — lets a live LLM classifier run. The fast deterministic
        checks resolve most calls; the classifier only runs when they fall through."""
        content = match_content(call)
        try:
            # Fire PreToolUse hooks off the event loop: a blocking CommandHook here would
            # otherwise stall every coroutine in the process while it runs.
            gate = await self.hooks.gate_async(self._gate_input(call, content))
            terminal = self._deterministic(call, content, gate=gate, gate_done=True)
        except Exception as e:  # fail-CLOSED: a broken gate must never allow
            return ask("other", f"permission check errored ({type(e).__name__}: {e}); ask human")
        if terminal is not None:
            return terminal
        try:
            if self.async_classifier is not None:
                c = await self.async_classifier(call)
            elif self.classifier is not None:
                c = self.classifier(call)
            else:
                c = None
        except Exception as e:  # fail-safe: a broken classifier must NOT allow
            return ask("classifier", f"classifier error ({e}); ask human")
        if c is not None and c.behavior is not Behavior.ALLOW:
            return c
        return self._finalize(call, self._mode_default())
