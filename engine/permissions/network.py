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


# Hostnames that resolve to an internal endpoint on essentially every platform or cloud.
# `ipaddress` never sees these because they are names, not literals — but blocking
# 169.254.169.254 while allowing `metadata.google.internal` is not a guard.
_INTERNAL_HOSTNAMES = frozenset({
    "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback",
    "metadata", "metadata.google.internal", "metadata.goog",
    "instance-data", "instance-data.ec2.internal",
})
_INTERNAL_SUFFIXES = (
    ".localhost", ".local", ".internal", ".localdomain", ".home.arpa",
)


def normalize_ip_literal(host: str) -> str | None:
    """Return the canonical IP for `host` if it is an IP literal in ANY notation, else None.

    `ipaddress.ip_address` accepts only canonical forms, so `2130706433` (decimal),
    `0x7f000001` (hex), `0177.0.0.1` (octal) and `127.1` (short form) all parsed as
    "not an IP" and sailed straight past the reserved-address guard — while every OS
    resolver happily dials them as 127.0.0.1. They are normalised here first.
    """
    h = (host or "").strip().strip("[]")
    if not h:
        return None
    try:
        return str(ipaddress.ip_address(h))
    except ValueError:
        pass
    # inet_aton-style forms: 1, 2, 3 or 4 parts, each decimal/octal/hex.
    parts = h.split(".")
    if not 1 <= len(parts) <= 4 or any(p == "" for p in parts):
        return None
    values: list[int] = []
    for p in parts:
        try:
            if p.lower().startswith(("0x", "0X")):
                v = int(p, 16)
            elif p.startswith("0") and len(p) > 1:
                v = int(p, 8)
            else:
                v = int(p, 10)
        except ValueError:
            return None
        if v < 0:
            return None
        values.append(v)
    # The LAST part absorbs the remaining bytes (127.1 -> 127.0.0.1; 2130706433 -> 127.0.0.1).
    fill = 4 - len(values)
    if any(v > 255 for v in values[:-1]) or values[-1] >= (1 << (8 * (fill + 1))):
        return None
    packed = 0
    for v in values[:-1]:
        packed = (packed << 8) | v
    packed = (packed << (8 * (fill + 1))) | values[-1]
    try:
        return str(ipaddress.ip_address(packed))
    except ValueError:
        return None


def is_internal_hostname(host: str) -> bool:
    """True if `host` is a name that denotes an internal/loopback endpoint."""
    h = (host or "").strip().lower().rstrip(".")
    if not h:
        return False
    return h in _INTERNAL_HOSTNAMES or h.endswith(_INTERNAL_SUFFIXES)


def is_internal_target(host: str) -> bool:
    """True if `host` DENOTES an internal endpoint: an internal hostname, or an IP literal
    (in any notation) in a reserved range.

    This is the check a caller wants for a host that may be either a name or an address.
    `is_reserved_ip` fails closed on anything it cannot parse as an address, which is the
    right behaviour when the input is known to be an address and the wrong behaviour for
    an ordinary hostname — using it directly on `api.example.com` blocks the whole public
    internet.
    """
    if is_internal_hostname(host):
        return True
    canonical = normalize_ip_literal(host)
    return canonical is not None and is_reserved_ip(canonical)


def is_reserved_ip(ip: str) -> bool:
    """True if `ip` is a reserved/internal ADDRESS (or unparseable → fail closed).

    Accepts any IP notation a resolver would (see `normalize_ip_literal`). Because it fails
    closed, only pass it something you already know to be an address — for a value that
    might be a hostname, use `is_internal_target`.
    """
    if is_internal_hostname(ip):
        return True
    canonical = normalize_ip_literal(ip)
    if canonical is None:
        return True  # not an address we can reason about: fail closed
    a = ipaddress.ip_address(canonical)
    # An IPv4-mapped/compatible IPv6 address (::ffff:127.0.0.1) must be judged on the IPv4
    # address it carries, not on the v6 wrapper.
    mapped = getattr(a, "ipv4_mapped", None) or getattr(a, "sixtofour", None)
    if mapped is not None:
        a = mapped
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
        # The reserved/internal guard covers an IP literal in ANY notation (decimal, hex,
        # octal, short form, v4-mapped v6) plus the hostnames that name an internal
        # endpoint. Checking only canonical dotted-quad left `2130706433`, `0x7f000001`,
        # `127.1` and `localhost` wide open — every one of which a resolver dials as
        # loopback.
        if self._deny_reserved_ips and not explicit_ip_allow:
            if is_internal_hostname(host):
                return NetworkVerdict(False, f"host {host} is an internal hostname")
            canonical = normalize_ip_literal(host)
            if canonical is not None and is_reserved_ip(canonical):
                note = "" if canonical == host else f" ({canonical})"
                return NetworkVerdict(
                    False, f"host {host}{note} is a reserved/internal address"
                )
            # a public hostname falls through to the allow/deny patterns below
        if not self._allow:
            return NetworkVerdict(True, f"host {host} allowed (no allowlist configured)")
        for pat in self._allow:
            if _host_matches(host, pat):
                return NetworkVerdict(True, f"host {host} matches allow pattern", pat)
        return NetworkVerdict(False, f"host {host} matches no allow pattern")


def _looks_like_ip_pattern(pattern: str) -> bool:
    try:
        if "/" in pattern:
            ipaddress.ip_network(pattern, strict=False)
        else:
            ipaddress.ip_address(pattern)
        return True
    except ValueError:
        return False
