"""Which destinations the network policy can actually see.

`_PAYLOAD_KEYS` exempted 22 common parameter names from destination detection. The
rationale is sound for `HttpRequest` — a URL inside a request body is data being sent, not
a host being dialled — but it was applied to EVERY tool, so any third-party or MCP tool
whose destination parameter happened to be named `query`, `data`, `input`, `json`, `form`,
`content`, `payload`, `text`, `message`, `prompt`, `sql`, `note`, `description` or
`comment` had no target extracted at all, and the allowlist never saw it.

That is exactly the "the policy depends on the tool author's choice of noun" failure the
key-matching rules were written to remove, running in the opposite direction.

The exemption is now CONDITIONAL: it applies only when the call names a destination
somewhere else. Two further gaps closed here: a bare hostname under a destination-ish key
that contained none of the nine magic fragments, and a nesting cutoff that list elements
consumed faster than dicts.
"""

from __future__ import annotations

import pytest

from engine.permissions import Behavior, HostAllowlist, Mode, PermissionEngine, ToolCall
from engine.permissions.engine import network_targets
from engine.hooks import HookEngine


def targets(inp: dict, name: str = "T") -> list[str]:
    return network_targets(ToolCall(name, inp))


# --- the payload exemption, when nothing else names a destination -------------
@pytest.mark.parametrize(
    "key",
    ["body", "data", "content", "payload", "text", "message", "prompt", "input",
     "value", "query", "sql", "json", "form", "note", "description", "comment"],
)
def test_a_url_under_a_payload_key_is_a_target_when_it_is_the_only_candidate(key):
    """No other destination in the call means this URL IS the destination."""
    assert targets({key: "https://evil.com/exfil"}) == ["https://evil.com/exfil"]


def test_a_payload_stays_a_payload_when_a_destination_is_named():
    """The exemption's original purpose still holds: a link inside a request body must not
    become a host the policy denies."""
    got = targets({"url": "https://api.allowed.com/v1", "body": "see https://docs.example.com"})
    assert got == ["https://api.allowed.com/v1"]


def test_the_exemption_holds_for_a_nested_destination_too():
    got = targets({
        "request": {"endpoint": "https://api.allowed.com/v1"},
        "payload": {"message": "read https://blog.example.com"},
    })
    assert "https://blog.example.com" not in got
    assert "https://api.allowed.com/v1" in got


# --- destination keys that named nothing --------------------------------------
@pytest.mark.parametrize(
    "key",
    ["destination", "dest", "remote", "peer", "node", "broker", "bootstrap",
     "cluster", "sink", "forward", "relay", "collector", "receiver", "nameserver",
     "resolver", "registry", "mirror", "gateway", "backend"],
)
def test_a_bare_hostname_under_a_destination_key_is_a_target(key):
    """A host without a scheme, under a key containing none of the nine fragments, produced
    no target at all."""
    assert targets({key: "evil.com"}) == ["evil.com"]


def test_a_host_with_a_port_is_a_target():
    assert targets({"relay": "evil.com:8080"}) == ["evil.com:8080"]


# --- nesting -------------------------------------------------------------------
def test_a_deeply_nested_url_is_still_found():
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"url": "http://evil.com"}}}}}}}}}
    assert targets(deep) == ["http://evil.com"]


def test_list_nesting_does_not_exhaust_the_depth_budget_early():
    """List elements consume depth, so a URL inside a list-of-dicts escaped at a shallower
    nesting than the same URL inside plain dicts."""
    nested = {"x": [{"y": [{"z": [{"w": [{"v": [{"u": {"url": "http://evil.com"}}]}]}]}]}]}
    assert targets(nested) == ["http://evil.com"]


# --- precision -----------------------------------------------------------------
def test_a_call_with_no_network_content_yields_nothing():
    assert targets({"path": "/tmp/x", "count": 3, "flag": True}) == []


def test_a_filename_is_not_mistaken_for_a_host():
    """`report.v2.txt` is host-SHAPED. Treating any dotted string as a destination would
    deny ordinary file operations."""
    assert targets({"filename": "report.v2.txt"}) == []
    assert targets({"path": "src/module.py"}) == []


def test_duplicate_targets_are_reported_once():
    got = targets({"url": "https://x.com/a", "endpoint": "https://x.com/a"})
    assert got == ["https://x.com/a"]


# --- end to end through the policy ---------------------------------------------
def engine_with_allowlist() -> PermissionEngine:
    return PermissionEngine(
        HookEngine(), mode=Mode.AUTO,
        network_policy=HostAllowlist(allow=("*.allowed.com",)),
    )


@pytest.mark.parametrize("key", ["query", "data", "input", "json", "destination"])
def test_the_allowlist_now_denies_a_smuggled_destination(key):
    """THE regression test: before the fix each of these was ALLOWED, because the policy
    was never handed a target to check."""
    decision = engine_with_allowlist().check(ToolCall("Search", {key: "https://evil.com/x"}))
    assert decision.behavior is Behavior.DENY, f"{key}: {decision.message}"


def test_an_allow_listed_host_still_passes():
    decision = engine_with_allowlist().check(
        ToolCall("HttpRequest", {"url": "https://api.allowed.com/v1"})
    )
    assert decision.behavior is Behavior.ALLOW


def test_a_link_in_a_body_to_an_allow_listed_host_still_passes():
    decision = engine_with_allowlist().check(
        ToolCall("HttpRequest", {
            "url": "https://api.allowed.com/v1",
            "body": "please read https://some-other-site.example/page",
        })
    )
    assert decision.behavior is Behavior.ALLOW, decision.message


def test_an_internal_endpoint_smuggled_in_a_payload_key_is_denied():
    decision = engine_with_allowlist().check(
        ToolCall("Search", {"query": "http://169.254.169.254/latest/meta-data/"})
    )
    assert decision.behavior is Behavior.DENY
