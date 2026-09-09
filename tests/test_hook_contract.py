"""The CommandHook contract: what a shell hook receives, and what it can answer.

Regression suite for the defect that `CommandHook.run()` accepted a `HookInput` and never
referenced it, so a command hook was a CONSTANT function of the tool call. It could not see
which tool ran, with what arguments, against what target — `if_` was the only expressible
rule, and every external policy integration was impossible to write.

Every hook here is a real subprocess. Hooks are written to a temp file and invoked as
`"<python>" "<script>"` so the assertions do not depend on shell quoting rules, which
differ between cmd.exe and sh.
"""

from __future__ import annotations

import json
import sys

import pytest

from engine.hooks import (
    HOOK_PAYLOAD_VERSION,
    CommandHook,
    FunctionHook,
    HookEngine,
    HookEvent,
    HookInput,
    parse_hook_stdout,
)
from engine.permissions import Behavior, Mode, PermissionEngine, ToolCall, deny


def hook_cmd(tmp_path, body: str, name: str = "hook.py") -> str:
    """Write `body` as a python script and return a shell command that runs it."""
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return f'"{sys.executable}" "{script}"'


# A hook that echoes back everything it was given, so a test can assert on it.
ECHO = """
import json, os, sys
doc = sys.stdin.read()
print(json.dumps({
    "stdin": doc,
    "env": {k: v for k, v in os.environ.items() if k.startswith("BARQ_HOOK_")},
}))
"""


def _run(tmp_path, body, inp, **kw):
    return CommandHook("h", hook_cmd(tmp_path, body), **kw).run(inp)


def pre(tool="WriteFile", content="/etc/passwd", payload=None):
    return HookInput(
        event=HookEvent.PRE_TOOL_USE,
        tool_name=tool,
        match_content=content,
        payload=payload if payload is not None else {"path": content},
    )


# --- what the hook receives -------------------------------------------------
def test_hook_receives_the_tool_call_on_stdin(tmp_path):
    """The core regression: a hook must be able to see WHAT it is gating."""
    out = _run(tmp_path, ECHO, pre(tool="WriteFile", content="/etc/passwd"))
    echoed = json.loads(out.stdout)
    doc = json.loads(echoed["stdin"])

    assert doc["version"] == HOOK_PAYLOAD_VERSION
    assert doc["event"] == "PreToolUse"
    assert doc["tool_name"] == "WriteFile"
    assert doc["match_content"] == "/etc/passwd"
    assert doc["payload"] == {"path": "/etc/passwd"}


def test_hook_receives_full_tool_arguments_not_just_the_name(tmp_path):
    """A policy decision needs the arguments. A hook that sees only the tool name can
    express nothing the `if_` string could not already express."""
    args = {"url": "https://internal.corp/admin", "method": "DELETE", "nested": {"k": [1, 2]}}
    out = _run(tmp_path, ECHO, pre(tool="HttpRequest", payload=args))
    doc = json.loads(json.loads(out.stdout)["stdin"])
    assert doc["payload"] == args


def test_hook_receives_the_call_in_the_environment_too(tmp_path):
    """A shell one-liner should not need a JSON parser to gate on the tool name."""
    out = _run(tmp_path, ECHO, pre(tool="Bash", content="rm -rf /"))
    env = json.loads(out.stdout)["env"]
    assert env["BARQ_HOOK_EVENT"] == "PreToolUse"
    assert env["BARQ_HOOK_TOOL_NAME"] == "Bash"
    assert env["BARQ_HOOK_MATCH_CONTENT"] == "rm -rf /"
    assert json.loads(env["BARQ_HOOK_INPUT"])["tool_name"] == "Bash"


def test_hook_environment_inherits_the_parent(tmp_path, monkeypatch):
    """Replacing the environment instead of extending it strips PATH, and `shell=True`
    then cannot find the interpreter the hook is written in."""
    monkeypatch.setenv("BARQ_TEST_SENTINEL", "present")
    body = 'import os; print(os.environ.get("BARQ_TEST_SENTINEL", "MISSING"))'
    assert _run(tmp_path, body, pre()).stdout.strip() == "present"


def test_pass_input_false_gives_a_pristine_stdin_and_env(tmp_path):
    out = _run(tmp_path, ECHO, pre(), pass_input=False)
    echoed = json.loads(out.stdout)
    assert echoed["stdin"] == ""
    assert echoed["env"] == {}


# --- what the hook can answer -----------------------------------------------
@pytest.mark.parametrize(
    "verdict,expected",
    [("allow", Behavior.ALLOW), ("ask", Behavior.ASK), ("deny", Behavior.DENY)],
)
def test_structured_decision_on_stdout(tmp_path, verdict, expected):
    body = f'print(\'{{"decision": "{verdict}", "reason": "policy service said so"}}\')'
    out = _run(tmp_path, body, pre())
    assert out.decision is not None
    assert out.decision.behavior is expected
    assert out.decision.message == "policy service said so"


def test_permission_decision_alias_is_accepted(tmp_path):
    body = 'print(\'{"permissionDecision": "deny", "permissionDecisionReason": "nope"}\')'
    out = _run(tmp_path, body, pre())
    assert out.decision.behavior is Behavior.DENY
    assert out.decision.message == "nope"


def test_hook_may_log_before_printing_its_verdict(tmp_path):
    """Requiring a gate to be silent to be readable would make the feature unusable."""
    body = (
        'print("checking policy...")\n'
        'print("consulted 3 rules")\n'
        'print(\'{"decision": "deny", "reason": "rule 7"}\')'
    )
    out = _run(tmp_path, body, pre())
    assert out.decision.behavior is Behavior.DENY
    assert out.decision.message == "rule 7"


def test_non_json_stdout_stays_advisory(tmp_path):
    """The pre-existing contract: a hook that only logs must keep working unchanged."""
    body = 'print("audited")'
    out = _run(tmp_path, body, pre())
    assert out.decision is None
    assert out.errored is False
    assert out.stdout.strip() == "audited"


def test_json_without_a_decision_key_is_advisory(tmp_path):
    body = 'print(\'{"note": "seen", "count": 3}\')'
    out = _run(tmp_path, body, pre())
    assert out.decision is None and out.errored is False


# --- failing closed ---------------------------------------------------------
def test_unrecognised_decision_value_is_an_error_not_a_shrug(tmp_path):
    """The hook TRIED to gate and we could not read the verdict. Treating that as
    'nothing objected' is a fail-open bypass."""
    body = 'print(\'{"decision": "maybe", "reason": "unsure"}\')'
    out = _run(tmp_path, body, pre())
    assert out.decision is None
    assert out.errored is True
    assert "maybe" in out.stderr


def test_unrecognised_decision_escalates_the_gate_to_ask(tmp_path):
    body = 'print(\'{"decision": "probably-fine"}\')'
    engine = HookEngine()
    engine.register(HookEvent.PRE_TOOL_USE, CommandHook("h", hook_cmd(tmp_path, body)))
    assert engine.gate(pre()).behavior is Behavior.ASK


def test_timeout_is_an_error_not_silence(tmp_path):
    """A gate that timed out has NO verdict. Reporting a clean non-blocking outcome
    would read as 'no objection'."""
    body = "import time; time.sleep(5)"
    out = CommandHook("slow", hook_cmd(tmp_path, body), timeout=0.4).run(pre())
    assert out.exit_code == 124
    assert out.errored is True


def test_timeout_escalates_the_gate_to_ask(tmp_path):
    body = "import time; time.sleep(5)"
    engine = HookEngine()
    engine.register(
        HookEvent.PRE_TOOL_USE, CommandHook("slow", hook_cmd(tmp_path, body), timeout=0.4)
    )
    assert engine.gate(pre()).behavior is Behavior.ASK


def test_exit_two_still_blocks(tmp_path):
    body = 'import sys; sys.stderr.write("blocked by policy"); sys.exit(2)'
    out = _run(tmp_path, body, pre())
    assert out.is_blocking
    assert out.decision.behavior is Behavior.DENY
    assert "blocked by policy" in out.decision.message


def test_exit_two_outranks_an_allow_on_stdout(tmp_path):
    """A hook that both exits 2 and prints allow is contradicting itself; the strictest
    reading is the only safe one."""
    body = 'import sys; print(\'{"decision": "allow"}\'); sys.exit(2)'
    out = _run(tmp_path, body, pre())
    assert out.decision.behavior is Behavior.DENY


def test_deny_survives_a_parse_error_in_the_same_hook(tmp_path):
    """Discarding a DENY because the same run also misprinted would turn the strictest
    available verdict into an ASK."""
    body = 'import sys; print(\'{"decision": "wat"}\'); sys.exit(2)'
    engine = HookEngine()
    engine.register(HookEvent.PRE_TOOL_USE, CommandHook("h", hook_cmd(tmp_path, body)))
    assert engine.gate(pre()).behavior is Behavior.DENY


def test_a_deny_hook_beats_an_allow_hook(tmp_path):
    engine = HookEngine()
    engine.register(
        HookEvent.PRE_TOOL_USE,
        CommandHook("yes", hook_cmd(tmp_path, 'print(\'{"decision":"allow"}\')', "a.py")),
    )
    engine.register(
        HookEvent.PRE_TOOL_USE,
        CommandHook("no", hook_cmd(tmp_path, 'print(\'{"decision":"deny"}\')', "b.py")),
    )
    assert engine.gate(pre()).behavior is Behavior.DENY


# --- payload robustness -----------------------------------------------------
def test_oversized_payload_is_replaced_not_truncated(tmp_path):
    """Truncating the JSON would hand the hook an unparseable document. A hook must be
    able to tell 'no arguments' from 'arguments too large to send'."""
    inp = pre(payload={"content": "x" * 3_000_000})
    doc = json.loads(inp.to_json())
    assert doc["payload"] == {}
    assert "payload_omitted" in doc
    # and it is still valid JSON that reaches the hook
    out = _run(tmp_path, ECHO, inp)
    assert json.loads(json.loads(out.stdout)["stdin"])["payload"] == {}


def test_unserialisable_payload_does_not_disarm_the_gate(tmp_path):
    """Failing to serialise must not mean the hook silently does not run."""
    inp = pre(payload={"handle": object(), "path": "/etc/passwd"})
    doc = json.loads(inp.to_json())
    assert doc["tool_name"] == "WriteFile"
    out = _run(tmp_path, ECHO, inp)
    assert json.loads(out.stdout)["stdin"]  # the hook still received a document


def test_env_value_is_capped_so_createprocess_cannot_fail(tmp_path):
    """Windows caps the whole environment block at ~32 KB; an oversized value there does
    not truncate, it makes the process fail to start and the hook never runs."""
    env = pre(payload={"content": "y" * 200_000}).to_env()
    assert all(len(v) <= 8_000 for v in env.values())
    out = _run(tmp_path, ECHO, pre(payload={"content": "y" * 200_000}))
    assert out.exit_code == 0  # the process started


def test_payload_with_non_ascii_and_newlines_round_trips(tmp_path):
    args = {"path": "/tmp/naïve\nfile", "note": "日本語 🎌"}
    out = _run(tmp_path, ECHO, pre(payload=args))
    assert json.loads(json.loads(out.stdout)["stdin"])["payload"] == args


# --- the async path ---------------------------------------------------------
@pytest.mark.asyncio
async def test_async_gate_carries_the_same_payload(tmp_path):
    """The agent loop uses gate_async. If only the sync path passed the call, every real
    run would still gate blind."""
    body = (
        "import json, sys\n"
        "doc = json.loads(sys.stdin.read())\n"
        "bad = doc['payload'].get('path','').startswith('/etc')\n"
        "print(json.dumps({'decision': 'deny' if bad else 'allow', 'reason': 'path check'}))"
    )
    engine = HookEngine()
    engine.register(HookEvent.PRE_TOOL_USE, CommandHook("paths", hook_cmd(tmp_path, body)))
    assert (await engine.gate_async(pre(content="/etc/shadow"))).behavior is Behavior.DENY
    ok = pre(content="/tmp/ok", payload={"path": "/tmp/ok"})
    assert (await engine.gate_async(ok)).behavior is Behavior.ALLOW


# --- end to end through the permission engine -------------------------------
@pytest.mark.asyncio
async def test_hook_can_gate_on_arguments_end_to_end(tmp_path):
    """THE regression test. A policy hook that decides from the tool's ARGUMENTS must
    reach the permission engine's verdict. Before the fix this was unwritable: the hook
    could not see the arguments at all."""
    body = (
        "import json, sys\n"
        "doc = json.loads(sys.stdin.read())\n"
        "amount = doc['payload'].get('amount_usd', 0)\n"
        "if amount > 10000:\n"
        "    print(json.dumps({'decision':'deny','reason':'over the $10k limit'}))\n"
        "else:\n"
        "    print(json.dumps({'decision':'allow','reason':'within limit'}))"
    )
    hooks = HookEngine()
    hooks.register(HookEvent.PRE_TOOL_USE, CommandHook("spend", hook_cmd(tmp_path, body)))
    perms = PermissionEngine(hooks, mode=Mode.ASK)

    big = await perms.check_async(ToolCall("Transfer", {"amount_usd": 50000}))
    assert big.behavior is Behavior.DENY
    assert "10k" in big.message

    small = await perms.check_async(ToolCall("Transfer", {"amount_usd": 50}))
    assert small.behavior is Behavior.ALLOW


@pytest.mark.asyncio
async def test_hook_allow_still_cannot_override_hard_danger(tmp_path):
    """The new structured allow must not become a way around the non-overridable stages.
    A hook allow is held, not obeyed."""
    body = 'print(\'{"decision":"allow","reason":"trust me"}\')'
    hooks = HookEngine()
    hooks.register(HookEvent.PRE_TOOL_USE, CommandHook("yes", hook_cmd(tmp_path, body)))
    perms = PermissionEngine(hooks, mode=Mode.AUTO)
    d = await perms.check_async(ToolCall("Bash", {"command": "rm -rf /"}))
    assert d.behavior is Behavior.DENY


# --- the parser in isolation ------------------------------------------------
@pytest.mark.parametrize(
    "text,behavior",
    [
        ('{"decision":"deny"}', Behavior.DENY),
        ('  {"decision":"ALLOW"}  ', Behavior.ALLOW),
        ('noise\n{"decision":"ask"}', Behavior.ASK),
    ],
)
def test_parse_hook_stdout_accepts(text, behavior):
    decision, err = parse_hook_stdout(text)
    assert err is None and decision.behavior is behavior


@pytest.mark.parametrize("text", ["", "   ", "not json", "[1,2,3]", '{"other":1}', "null"])
def test_parse_hook_stdout_is_advisory_on_anything_else(text):
    decision, err = parse_hook_stdout(text)
    assert decision is None and err is None


def test_parse_hook_stdout_flags_a_bad_verdict():
    decision, err = parse_hook_stdout('{"decision": 42}')
    assert decision is None and err is not None


def test_function_hook_is_unaffected():
    """FunctionHook already received the input; the change must not disturb it."""
    seen = {}

    def fn(inp):
        seen["tool"] = inp.tool_name
        return deny("hook", "no")

    out = FunctionHook("f", fn).run(pre(tool="X"))
    assert seen["tool"] == "X"
    assert out.decision.behavior is Behavior.DENY
    assert out.errored is False
