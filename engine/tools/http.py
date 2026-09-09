"""A simple HTTP request tool for agents.

Performs an HTTP request and returns status + (truncated) body. If a `rate_limiter` is
given, every request waits on it first.

Redirects and the network policy
--------------------------------
The permission engine authorises the URL the MODEL declared, before this tool runs. That
check is worthless if the HTTP client then silently follows a 302 somewhere else: an
allow-listed host that redirects to `169.254.169.254` hands the agent cloud credentials,
and the allowlist never sees the hop. That was the behaviour of `follow_redirects=True`.

So redirects are followed MANUALLY here, and EVERY hop -- including the first -- is checked
against the same `NetworkPolicy` the permission engine uses. A hop the policy rejects is not
fetched; the tool returns a BLOCKED result naming the host.

Checking only the redirects was not enough. The permission engine consults a NetworkPolicy
only when one is configured, and the `Agent` default is none, so the initial URL reached the
socket unconditionally: a stock agent could fetch `http://169.254.169.254/` and read cloud
credentials. The tool now enforces its own floor regardless of how it was wired, so with no
policy configured hops are still capped (`max_redirects`), non-HTTP schemes are refused, and
reserved/internal addresses are refused in every notation a resolver accepts (decimal, hex,
octal, short-form, v4-mapped v6) plus the hostnames that name one (`localhost`,
`metadata.google.internal`).

Credentials do not follow a cross-origin redirect. `Authorization`, `Cookie` and friends are
scoped to the origin they were issued for; replaying them to whatever host the previous one
nominated hands the caller's bearer token away. httpx and requests both strip them on a
cross-origin hop and so does the manual walker here.

What this still does NOT do: resolve DNS before deciding. The check is on the name, the
connection is made later, so DNS rebinding is unmitigated. Enforce egress at the transport
(an outbound proxy, a network namespace) if that is in your threat model.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from engine.permissions.network import (
    extract_host, is_internal_target, normalize_ip_literal,
)

HTTP_REQUEST_SPEC = {
    "name": "HttpRequest",
    "description": (
        "Send an HTTP request to a URL and return the status code and response body. "
        "Supports an HTTP method (default GET), a query string already in the URL, and "
        "optional headers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL including query string"},
            "method": {"type": "string", "description": "GET|POST|... (default GET)"},
            "headers": {"type": "object", "description": "optional request headers"},
            "body": {"type": "string", "description": "optional request body"},
        },
        "required": ["url"],
    },
}

_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})

# Headers that authenticate the CALLER to a specific origin. They must not follow a
# redirect to a different one.
_CREDENTIAL_HEADERS = frozenset({
    "authorization", "proxy-authorization", "cookie", "www-authenticate",
    "x-api-key", "api-key", "x-auth-token", "x-access-token", "x-csrf-token",
    "x-amz-security-token", "x-goog-api-key", "private-token",
})


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url if "://" in url else "//" + url)
    port = parts.port
    if port is None:
        port = {"http": 80, "https": 443}.get(parts.scheme.lower())
    return (parts.scheme.lower(), (parts.hostname or "").lower(), port)


def _crosses_origin(current: str, nxt: str) -> bool:
    """True if a hop leaves the scheme/host/port the credentials were sent to."""
    return _origin(current) != _origin(nxt)


def _strip_credentials(headers):
    """Drop caller credentials from a header mapping, keeping everything else."""
    if not headers:
        return headers
    return {k: v for k, v in headers.items() if str(k).lower() not in _CREDENTIAL_HEADERS}


def _hop_allowed(url: str, policy, block_reserved: bool, what: str = "redirect to") -> tuple[bool, str]:
    """Decide whether one hop may be dialled. Returns (allowed, reason).

    `what` names the hop in the reason string, so a blocked first request does not report
    itself as a blocked redirect.
    """
    host = extract_host(url)
    if not host:
        return False, f"unparseable target {url!r}"
    scheme = (urlsplit(url).scheme or "").lower()
    if scheme not in ("http", "https"):
        # file://, gopher://, data:// and friends are not HTTP and are classic SSRF
        # escalation targets; this tool speaks HTTP only.
        return False, f"{what} unsupported scheme {scheme or '(none)'!r} in {url!r}"
    if block_reserved:
        # Every IP notation a resolver accepts, plus the hostnames that name an internal
        # endpoint. Judging only canonical dotted-quad literals left decimal/hex/octal/
        # short-form addresses and `localhost` reachable.
        if is_internal_target(host):
            canonical = normalize_ip_literal(host)
            note = f" ({canonical})" if canonical and canonical != host else ""
            return False, f"{what} reserved/internal address {host}{note}"
    if policy is not None:
        verdict = policy.check(url)
        if not verdict.allowed:
            return False, f"{what} {host} rejected by network policy: {verdict.reason}"
    return True, ""


# Methods the tool will issue unless the caller widens the set. GET/HEAD/OPTIONS read and
# POST is the ordinary write, so the useful surface is intact — but PUT, PATCH and DELETE
# are opt-in, because once a host is allowlisted for reading there was nothing else
# stopping the model from issuing `DELETE` against it. The permission rule syntax matches
# on the URL only, so it cannot express a per-method rule either.
DEFAULT_ALLOWED_METHODS: tuple[str, ...] = ("GET", "HEAD", "OPTIONS", "POST")
# Never issuable: they are proxy/debug verbs with no agent use and real abuse potential.
_FORBIDDEN_METHODS = frozenset({"CONNECT", "TRACE", "TRACK"})

# Hard ceiling on how many bytes of a response are read off the wire. The whole body used
# to be buffered into memory and only THEN truncated to 2,000 characters for the model, so
# a single allow-listed URL returning a multi-gigabyte body was a memory-exhaustion vector.
# `max_body` bounds only what reaches the audit log; this bounds the read itself.
DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
# How much of the body the MODEL is shown. Distinct from the read cap above: one bounds
# memory, the other bounds context.
_MODEL_BODY_CHARS = 2000


@dataclass
class _Hop:
    """What one request returned, captured before the streaming response is closed."""

    status: int
    headers: dict
    request_url: str
    text: str
    truncated: bool = False


async def _fetch(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers,
    body,
    max_bytes: int,
    *,
    read_body: bool,
) -> _Hop:
    """Issue one request, reading at most `max_bytes` of the body.

    Streamed rather than buffered, and stopped at the cap instead of after it: an
    oversized response is abandoned mid-read rather than being fully downloaded and then
    discarded.
    """
    async with client.stream(method, url, headers=headers, content=body) as resp:
        request_url = str(resp.request.url)
        hop_headers = dict(resp.headers)
        if not read_body:
            # A redirect: only the Location header matters, so do not spend the transfer.
            return _Hop(resp.status_code, hop_headers, request_url, "", False)

        declared = hop_headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > max_bytes:
                    return _Hop(
                        resp.status_code, hop_headers, request_url,
                        f"[response declared {int(declared)} bytes, over the "
                        f"{max_bytes}-byte limit; not read]",
                        True,
                    )
            except ValueError:
                pass

        chunks: list[bytes] = []
        total = 0
        truncated = False
        async for chunk in resp.aiter_bytes():
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                truncated = True
                break
        raw = b"".join(chunks)[:max_bytes]
        text = raw.decode(resp.encoding or "utf-8", errors="replace")
        # The marker is NOT appended here: the caller slices the body to what the model
        # sees, and a note appended to the far end of a 5 MB string would be cut off by
        # that slice — telling the model nothing.
        return _Hop(resp.status_code, hop_headers, request_url, text, truncated)


def make_http_request(
    audit=None,
    timeout: float = 15.0,
    rate_limiter=None,
    max_body: int = 4000,
    *,
    network_policy=None,
    max_redirects: int = 5,
    block_reserved_redirects: bool = True,
    client: httpx.AsyncClient | None = None,
    redact=None,
    allowed_methods: tuple[str, ...] = DEFAULT_ALLOWED_METHODS,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
):
    """Return an async HTTP tool fn.

    `network_policy`: re-checked on every redirect hop (see the module docstring). Pass the
    same policy object given to the PermissionEngine so pre-flight and in-flight agree.
    `client`: an injected AsyncClient to reuse; without one a pooled client is created once
    and shared by every call, instead of a fresh TLS handshake per request.
    `redact`: (request_headers, response_body) -> (headers, body) applied before auditing.
    `allowed_methods`: the verbs the tool will issue. Widen it to enable PUT/PATCH/DELETE.
    `max_response_bytes`: hard cap on bytes read off the wire per hop.
    """
    methods = frozenset(m.upper() for m in allowed_methods) - _FORBIDDEN_METHODS
    max_response_bytes = max(1024, int(max_response_bytes))
    owned_client: dict = {"c": client}

    def _client() -> httpx.AsyncClient:
        if owned_client["c"] is None:
            # follow_redirects stays OFF: hops are walked manually and policy-checked.
            owned_client["c"] = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        return owned_client["c"]

    async def http_request(inp: dict) -> str:
        url = inp["url"]
        method = (inp.get("method") or "GET").upper()
        headers = inp.get("headers") or None
        body = inp.get("body")

        if method not in methods:
            why = (
                f"method {method} is not permitted (allowed: {', '.join(sorted(methods))})"
            )
            if audit is not None:
                audit.log_blocked("HttpRequest", extract_host(url), why)
            return f"BLOCKED: {why}"

        # The FIRST hop is checked exactly like every later one. It used to be dialled
        # unconditionally on the theory that the permission engine had already vetted it —
        # but the permission engine only checks a network target when a NetworkPolicy is
        # configured, and the Agent default is none. So a stock agent could fetch
        # http://169.254.169.254/ and read cloud credentials. The tool now enforces its own
        # floor regardless of how it was wired.
        ok, why = _hop_allowed(
            url, network_policy, block_reserved_redirects, what="request to",
        )
        if not ok:
            if audit is not None:
                audit.log_blocked("HttpRequest", extract_host(url), why)
            return f"BLOCKED: {why}"

        if rate_limiter is not None:
            await rate_limiter.acquire()

        cl = _client()
        chain = [url]
        # Each hop is streamed under the byte cap. A redirect's body is never shown, so
        # `read_body=False` skips the transfer entirely: only Location matters, and pulling
        # a body down to discard it is the waste the cap exists to prevent.
        nmethod, nbody, hop_url = method, body, url
        hop_headers = headers
        hops = 0
        while True:
            resp = await _fetch(
                cl, nmethod, hop_url, hop_headers, nbody, max_response_bytes,
                read_body=True,
            )
            if resp.status not in _REDIRECT_CODES or hops >= max_redirects:
                break
            location = resp.headers.get("location")
            if not location:
                break
            nxt = urljoin(resp.request_url, location)
            ok, why = _hop_allowed(nxt, network_policy, block_reserved_redirects)
            if not ok:
                if audit is not None:
                    audit.log_blocked("HttpRequest", extract_host(nxt), why)
                return (
                    f"BLOCKED after {hops + 1} redirect(s): {why}\n"
                    f"chain: {' -> '.join(chain)} -> {nxt}"
                )
            # Credentials are scoped to the host they were issued for. Replaying them to a
            # redirect target hands the caller's bearer token or session cookie to whatever
            # host the previous one nominated — httpx and requests both strip these on a
            # cross-origin hop, and the manual walker here has to do the same.
            if _crosses_origin(resp.request_url, nxt):
                hop_headers = _strip_credentials(hop_headers)
            chain.append(nxt)
            hops += 1
            # 303, and 301/302 in practice, degrade to GET without a body.
            if resp.status in (301, 302, 303) and nmethod not in ("GET", "HEAD"):
                nmethod, nbody = "GET", None
            hop_url = nxt

        if resp.status in _REDIRECT_CODES and hops >= max_redirects:
            return f"BLOCKED: exceeded max_redirects={max_redirects}\nchain: {' -> '.join(chain)}"

        text = resp.text
        if audit is not None:
            req_headers, resp_body = headers or {}, text[:max_body]
            if redact is not None:
                req_headers, resp_body = redact(req_headers, resp_body)
            audit.log_exchange(
                method, url,
                request={"method": method, "headers": req_headers, "chain": chain},
                response={"status": resp.status, "body": resp_body},
            )
        note = "" if len(chain) == 1 else f" (via {len(chain) - 1} redirect(s))"
        shown = text[:_MODEL_BODY_CHARS]
        # Say so when the body was cut, at either boundary. Silently handing the model a
        # fragment lets it reason about content that was never in its context and conclude
        # a key is absent when the response simply stopped - the file tool has always
        # reported its truncation, and this one never did.
        if resp.truncated:
            shown += (
                f"\n\n[TRUNCATED at the {max_response_bytes}-byte response limit; "
                f"showing the first {len(shown)} characters. Narrow the request.]"
            )
        elif len(text) > _MODEL_BODY_CHARS:
            shown += (
                f"\n\n[TRUNCATED: showing {_MODEL_BODY_CHARS} of {len(text)} characters.]"
            )
        return f"HTTP {resp.status}{note}\n{shown}"

    async def aclose() -> None:
        if owned_client["c"] is not None and client is None:
            await owned_client["c"].aclose()
            owned_client["c"] = None

    http_request.aclose = aclose  # type: ignore[attr-defined]
    return http_request


# Module-level default (no audit) for simple callers/tests.
http_request = make_http_request()
