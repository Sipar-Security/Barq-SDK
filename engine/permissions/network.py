"""Optional network-target policy for the permission engine.

The permission engine extracts every host/URL a tool call would contact (see
`network_targets` in engine.py). If a `NetworkPolicy` is configured it checks each one;
with no policy configured, network targets are unrestricted and fall through to rules /
mode like any other call.

`HostAllowlist` is a small, self-contained policy: a target is allowed when its host
matches an allow pattern (exact host, label-anchored ``*.suffix`` wildcard, or IP/CIDR)
and does not match a deny pattern. By default it also denies raw reserved/internal IP
literals (loopback, RFC1918, link-local, including cloud metadata 169.254.169.254) as a
basic SSRF guard, unless the operator explicitly opted that address into the allow list.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable
from urllib.parse import urlsplit


def extract_host(target: str) -> str:
    """Reduce a URL / host:port / bare host to a lowercase hostname.

    On anything it cannot parse it returns the stripped input, so a fail-closed caller
    still runs it through matching (which then denies the unrecognised value).
    """
    t = target.strip()
    if not t:
        return ""
    if "://" not in t:
        t = "//" + t  # give urlsplit a scheme so .hostname populates for host:port
    host = urlsplit(t).hostname or ""
    return host.lower().rstrip(".")


@dataclass(frozen=True)
class NetworkVerdict:
    allowed: bool
    reason: str
    matched: Optional[str] = None  # the pattern that decided it


@runtime_checkable
class NetworkPolicy(Protocol):
    """Minimal surface the permission engine needs: decide one target."""

    def check(self, target: str) -> NetworkVerdict: ...


# Reserved / internal ranges an SSRF guard denies by default.
_EXTRA_RESERVED = (
    ipaddress.ip_network("100.64.0.0/10"),   # CGNAT (RFC 6598)
    ipaddress.ip_network("192.0.0.0/24"),    # IETF protocol assignments
    ipaddress.ip_network("198.18.0.0/15"),   # benchmarking
)


def is_reserved_ip(ip: str) -> bool:
    """True if `ip` is a reserved/internal address literal (or unparseable → fail closed)."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if (a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
            or a.is_multicast or a.is_unspecified):
        return True
    return any(a in net for net in _EXTRA_RESERVED)


def _host_matches(host: str, pattern: str) -> bool:
    """Match a host against a pattern: exact, label-anchored ``*.suffix`` wildcard, or
    IP / CIDR. Wildcards match only whole additional labels (``*.acme.com`` matches
    ``api.acme.com`` and ``acme.com`` but never ``evil-acme.com``)."""
    host, pattern = host.lower().rstrip("."), pattern.lower().rstrip(".")
    if not host or not pattern:
        return False
    # IP / CIDR pattern.
    try:
        addr = ipaddress.ip_address(host)
        if "/" in pattern:
            return addr in ipaddress.ip_network(pattern, strict=False)
        return addr == ipaddress.ip_address(pattern)
    except ValueError:
        pass
    if pattern.startswith("*."):
        suffix = pattern[2:]
        return host == suffix or host.endswith("." + suffix)
    return host == pattern


class HostAllowlist:
    """A simple allow/deny `NetworkPolicy`.

    allow: patterns a host must match to be permitted (empty allow = allow anything not
    denied). deny: patterns that always block (win over allow). deny_reserved_ips: when
    True (default), a raw reserved/internal IP literal is blocked unless an allow pattern
    is an IP/CIDR that explicitly contains it.
    """

    def __init__(
        self,
        allow: tuple[str, ...] = (),
        deny: tuple[str, ...] = (),
        *,
        deny_reserved_ips: bool = True,
    ) -> None:
        self._allow = tuple(p.lower() for p in allow)
        self._deny = tuple(p.lower() for p in deny)
        self._deny_reserved_ips = deny_reserved_ips

    def check(self, target: str) -> NetworkVerdict:
        host = extract_host(target)
        if not host:
            return NetworkVerdict(False, "empty/unparseable target")
        for pat in self._deny:
            if _host_matches(host, pat):
                return NetworkVerdict(False, f"host {host} matches deny pattern", pat)
        explicit_ip_allow = any(
            _host_matches(host, pat) and _looks_like_ip_pattern(pat) for pat in self._allow
        )
        # The reserved/internal guard applies only to a raw IP literal - a hostname is
        # resolved by the network layer, not treated as reserved here.
        if (self._deny_reserved_ips and _is_ip_literal(host)
                and is_reserved_ip(host) and not explicit_ip_allow):
            return NetworkVerdict(False, f"host {host} is a reserved/internal address")
        if not self._allow:
            return NetworkVerdict(True, f"host {host} allowed (no allowlist configured)")
        for pat in self._allow:
            if _host_matches(host, pat):
                return NetworkVerdict(True, f"host {host} matches allow pattern", pat)
        return NetworkVerdict(False, f"host {host} matches no allow pattern")


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _looks_like_ip_pattern(pattern: str) -> bool:
    try:
        if "/" in pattern:
            ipaddress.ip_network(pattern, strict=False)
        else:
            ipaddress.ip_address(pattern)
        return True
    except ValueError:
        return False
