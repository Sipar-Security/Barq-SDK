from engine.hooks import HookEngine, HookEvent, FunctionHook, HookInput
from engine.permissions import (
    Behavior,
    HostAllowlist,
    Mode,
    PermissionEngine,
    ToolCall,
    parse_rule,
    rule_matches,
)
from engine.permissions.decision import deny


def make_engine(**kw) -> PermissionEngine:
    policy = HostAllowlist(allow=("*.acme.com", "acme.com"), deny=("staging.acme.com",))
    return PermissionEngine(HookEngine(), network_policy=policy, **kw)


def test_rule_parse_and_match():
    r = parse_rule("Bash(git *)")
    assert r.tool_name == "Bash" and r.pattern == "git *"
    assert rule_matches(r, "Bash", "git status")
    assert not rule_matches(r, "Bash", "rm -rf /")
    assert not rule_matches(r, "Read", "git status")
    bare = parse_rule("Fetch")
    assert rule_matches(bare, "Fetch", "anything")
    # tool-name match is case-insensitive so `bash(...)` still gates `Bash`.
    lower = parse_rule("bash(rm *)")
    assert rule_matches(lower, "Bash", "rm -rf /")


def test_network_policy_denies_out_of_policy_call():
    eng = make_engine(mode=Mode.AUTO)
    d = eng.check(ToolCall("HttpRequest", {"url": "https://evil.example.com/"}))
    assert d.behavior is Behavior.DENY and d.reason_type == "network"


def test_network_policy_allows_in_policy_and_mode_default_allow():
    eng = make_engine(mode=Mode.AUTO)
    d = eng.check(ToolCall("HttpRequest", {"url": "https://api.acme.com/"}))
    assert d.behavior is Behavior.ALLOW


def test_deny_pattern_wins_even_under_allowed_parent():
    eng = make_engine(mode=Mode.AUTO)
    d = eng.check(ToolCall("Fetch", {"target": "staging.acme.com"}))
    assert d.behavior is Behavior.DENY and d.reason_type == "network"


def test_no_network_policy_allows_any_host():
    eng = PermissionEngine(HookEngine(), mode=Mode.AUTO)  # no policy configured
    d = eng.check(ToolCall("HttpRequest", {"url": "https://anywhere.example.com/"}))
    assert d.behavior is Behavior.ALLOW


def test_ask_mode_asks_by_default():
    eng = make_engine(mode=Mode.ASK)
    d = eng.check(ToolCall("HttpRequest", {"url": "https://api.acme.com/login"}))
    assert d.behavior is Behavior.ASK and d.reason_type == "mode"


def test_locked_mode_denies_by_default():
    eng = make_engine(mode=Mode.LOCKED)
    d = eng.check(ToolCall("HttpRequest", {"url": "https://api.acme.com/"}))
    assert d.behavior is Behavior.DENY and d.reason_type == "mode"


def test_deny_rule_beats_allow():
    eng = make_engine(mode=Mode.AUTO, deny_rules=("Bash(rm -rf *)",))
    d = eng.check(ToolCall("Bash", {"command": "rm -rf /tmp/x"}))
    # a recursive rm of a non-catastrophic path is a SOFT danger (ASK) unless a rule fires;
    # the explicit deny rule takes precedence and hard-denies.
    assert d.behavior is Behavior.DENY and d.reason_type == "rule"


def test_hard_danger_denies_before_any_rule():
    # even with an allow rule, `rm -rf /` is hard-denied before rules are consulted.
    eng = make_engine(mode=Mode.AUTO, allow_rules=("Bash(*)",))
    d = eng.check(ToolCall("Bash", {"command": "rm -rf /"}))
    assert d.behavior is Behavior.DENY and d.reason_type == "danger"


def test_reserved_ip_is_denied_by_default_policy():
    eng = make_engine(mode=Mode.AUTO)
    d = eng.check(ToolCall("HttpRequest", {"url": "http://169.254.169.254/latest/meta-data/"}))
    assert d.behavior is Behavior.DENY and d.reason_type == "network"


def test_prehook_gate_denies_first():
    hooks = HookEngine()

    def block_tool(inp: HookInput):
        if "forbidden" in inp.match_content:
            return deny("hook", "blocked by policy hook")
        return None

    hooks.register(HookEvent.PRE_TOOL_USE, FunctionHook("no-forbidden", block_tool))
    eng = PermissionEngine(hooks, mode=Mode.AUTO)
    d = eng.check(ToolCall("Bash", {"command": "run forbidden thing"}))
    assert d.behavior is Behavior.DENY and d.reason_type == "hook"


def test_hook_if_condition_scopes_the_hook():
    hooks = HookEngine()
    fired = []

    def watcher(inp: HookInput):
        fired.append(inp.match_content)
        return None

    hooks.register(
        HookEvent.PRE_TOOL_USE, FunctionHook("git-watch", watcher, if_="Bash(git *)")
    )
    hooks.gate(HookInput(HookEvent.PRE_TOOL_USE, "Bash", "git push"))
    hooks.gate(HookInput(HookEvent.PRE_TOOL_USE, "Bash", "ls -la"))
    hooks.gate(HookInput(HookEvent.PRE_TOOL_USE, "Read", "git thing"))
    assert fired == ["git push"]  # only the matching call fired the hook
