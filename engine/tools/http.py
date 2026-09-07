"""A simple HTTP request tool for agents.

Performs an HTTP request and returns status + (truncated) body. If a `rate_limiter` is
given, every request waits on it first.

Redirects and the network policy
--------------------------------
The permission engine authorises the URL the MODEL declared, before this tool runs. That
check is worthless if the HTTP client then silently follows a 302 somewhere else: an
allow-listed host that redirects to `169.254.169.254` hands the agent cloud credentials,
and the allowlist never sees the hop. That was the behaviour of `follow_redirects=True`.

So redirects are followed MANUALLY here, and every hop is re-checked against the same
`NetworkPolicy` the permission engine uses. A hop the policy rejects is not fetched; the
tool returns a BLOCKED result naming the host. With no policy configured, hops are still
capped (`max_redirects`) and reserved/internal addresses are refused by default, so the
tool is not an open SSRF primitive even in its most permissive configuration.
"""

from __future__ import annotations

from urllib.parse import urljoin

import httpx

from engine.permissions.network import extract_host, is_reserved_ip

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


def _hop_allowed(url: str, policy, block_reserved: bool) -> tuple[bool, str]:
    """Decide whether one hop may be dialled. Returns (allowed, reason)."""
    host = extract_host(url)
    if not host:
        return False, f"unparseable redirect target {url!r}"
    if block_reserved:
        # Only a raw IP literal is judged here; a hostname is the policy's business.
        try:
            import ipaddress

            ipaddress.ip_address(host)
            if is_reserved_ip(host):
                return False, f"redirect to reserved/internal address {host}"
        except ValueError:
            pass
    if policy is not None:
        verdict = policy.check(url)
        if not verdict.allowed:
            return False, f"redirect to {host} rejected by network policy: {verdict.reason}"
    return True, ""


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
):
    """Return an async HTTP tool fn.

    `network_policy`: re-checked on every redirect hop (see the module docstring). Pass the
    same policy object given to the PermissionEngine so pre-flight and in-flight agree.
    `client`: an injected AsyncClient to reuse; without one a pooled client is created once
    and shared by every call, instead of a fresh TLS handshake per request.
    `redact`: (request_headers, response_body) -> (headers, body) applied before auditing.
    """
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
        if rate_limiter is not None:
            await rate_limiter.acquire()

        cl = _client()
        chain = [url]
        resp = await cl.request(method, url, headers=headers, content=body)

        hops = 0
        while resp.status_code in _REDIRECT_CODES and hops < max_redirects:
            location = resp.headers.get("location")
            if not location:
                break
            nxt = urljoin(str(resp.request.url), location)
            ok, why = _hop_allowed(nxt, network_policy, block_reserved_redirects)
            if not ok:
                if audit is not None:
                    audit.log_blocked("HttpRequest", extract_host(nxt), why)
                return (
                    f"BLOCKED after {hops + 1} redirect(s): {why}\n"
                    f"chain: {' -> '.join(chain)} -> {nxt}"
                )
            chain.append(nxt)
            hops += 1
            # 303, and 301/302 in practice, degrade to GET without a body.
            nmethod, nbody = (method, body)
            if resp.status_code in (301, 302, 303) and method not in ("GET", "HEAD"):
                nmethod, nbody = "GET", None
            resp = await cl.request(nmethod, nxt, headers=headers, content=nbody)

        if resp.status_code in _REDIRECT_CODES and hops >= max_redirects:
            return f"BLOCKED: exceeded max_redirects={max_redirects}\nchain: {' -> '.join(chain)}"

        text = resp.text
        if audit is not None:
            req_headers, resp_body = headers or {}, text[:max_body]
            if redact is not None:
                req_headers, resp_body = redact(req_headers, resp_body)
            audit.log_exchange(
                method, url,
                request={"method": method, "headers": req_headers, "chain": chain},
                response={"status": resp.status_code, "body": resp_body},
            )
        note = "" if len(chain) == 1 else f" (via {len(chain) - 1} redirect(s))"
        return f"HTTP {resp.status_code}{note}\n{text[:2000]}"

    async def aclose() -> None:
        if owned_client["c"] is not None and client is None:
            await owned_client["c"].aclose()
            owned_client["c"] = None

    http_request.aclose = aclose  # type: ignore[attr-defined]
    return http_request


# Module-level default (no audit) for simple callers/tests.
http_request = make_http_request()
