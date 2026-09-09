"""`HttpRequest` resource and verb limits.

Two reproduced defects:

  * The ENTIRE response body was buffered into memory and only then truncated to 2,000
    characters for the model. No streaming, no `Content-Length` pre-check, no byte cap.
    `max_body` bounds what reaches the AUDIT LOG, not the read — so one allow-listed URL
    returning a multi-gigabyte body was a memory-exhaustion vector.
  * Any method was issuable. Once a host was allow-listed for reading, nothing stopped the
    model issuing `DELETE` against it, and the permission rule syntax matches on the URL
    only so it cannot express a per-method rule either.
"""

from __future__ import annotations

import httpx
import pytest

from engine.tools.http import (
    DEFAULT_ALLOWED_METHODS,
    DEFAULT_MAX_RESPONSE_BYTES,
    make_http_request,
)


def transport(handler):
    return httpx.MockTransport(handler)


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport(handler), follow_redirects=False)


def ok(body: bytes = b"hello", headers=None, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body, headers=headers or {})

    return handler


class RecordingAudit:
    def __init__(self):
        self.blocked: list[tuple] = []
        self.exchanges: list[dict] = []

    def log_blocked(self, tool, target, reason, **kw):
        self.blocked.append((tool, target, reason))
        return ""

    def log_exchange(self, method, url, request, response):
        self.exchanges.append({"method": method, "url": url, "response": response})
        return ""


# --- response size -----------------------------------------------------------
def chunked(total_chunks: int, chunk: bytes, counter: dict):
    """A response with NO Content-Length, so the streaming cap is what stops the read."""

    async def stream():
        for _ in range(total_chunks):
            counter["chunks"] = counter.get("chunks", 0) + 1
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream())

    return handler


@pytest.mark.asyncio
async def test_streamed_response_is_capped_at_the_byte_limit():
    """No Content-Length to check, so the cap has to stop the read itself."""
    counter: dict = {}
    tool = make_http_request(
        client=client_for(chunked(200, b"x" * 1024, counter)), max_response_bytes=4096
    )
    out = await tool({"url": "https://example.com/big"})
    assert "TRUNCATED at the 4096-byte response limit" in out
    # The transfer was ABANDONED, not completed and then discarded: ~4 of 200 chunks read.
    assert counter["chunks"] <= 8, counter


@pytest.mark.asyncio
async def test_a_body_within_the_cap_streams_completely():
    counter: dict = {}
    tool = make_http_request(
        client=client_for(chunked(3, b"abc", counter)), max_response_bytes=4096
    )
    out = await tool({"url": "https://example.com/small"})
    assert "abcabcabc" in out
    assert "TRUNCATED" not in out


@pytest.mark.asyncio
async def test_a_declared_oversized_body_is_not_read_at_all():
    """A Content-Length over the cap means the transfer is refused up front rather than
    streamed and discarded."""
    read = {"bytes": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = b"y" * 50_000
        read["bytes"] += len(body)
        return httpx.Response(200, content=body, headers={"content-length": "50000"})

    tool = make_http_request(client=client_for(handler), max_response_bytes=1024)
    out = await tool({"url": "https://example.com/big"})
    assert "declared 50000 bytes" in out
    assert "not read" in out


@pytest.mark.asyncio
async def test_a_small_response_is_unaffected():
    tool = make_http_request(client=client_for(ok(b"hello world")))
    out = await tool({"url": "https://example.com/x"})
    assert out.startswith("HTTP 200")
    assert "hello world" in out
    assert "TRUNCATED" not in out


@pytest.mark.asyncio
async def test_the_cap_has_a_floor():
    """A nonsensically small cap must not make the tool unusable or crash the read."""
    tool = make_http_request(client=client_for(ok(b"hi")), max_response_bytes=0)
    out = await tool({"url": "https://example.com/x"})
    assert out.startswith("HTTP 200")


@pytest.mark.asyncio
async def test_default_cap_is_finite():
    assert 0 < DEFAULT_MAX_RESPONSE_BYTES < 100 * 1024 * 1024


# --- methods -----------------------------------------------------------------
@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST"])
@pytest.mark.asyncio
async def test_default_methods_are_permitted(method):
    tool = make_http_request(client=client_for(ok()))
    out = await tool({"url": "https://example.com/x", "method": method})
    assert out.startswith("HTTP 200"), out


@pytest.mark.parametrize("method", ["DELETE", "PUT", "PATCH"])
@pytest.mark.asyncio
async def test_destructive_methods_are_refused_by_default(method):
    """An allow-listed host must not be DELETE-able just because it is readable."""
    audit = RecordingAudit()
    tool = make_http_request(client=client_for(ok()), audit=audit)
    out = await tool({"url": "https://example.com/x", "method": method})
    assert out.startswith("BLOCKED")
    assert method in out
    assert audit.blocked and "not permitted" in audit.blocked[0][2]


@pytest.mark.parametrize("method", ["CONNECT", "TRACE", "TRACK"])
@pytest.mark.asyncio
async def test_proxy_and_debug_verbs_can_never_be_enabled(method):
    """Even an operator who widens the set does not get these."""
    tool = make_http_request(
        client=client_for(ok()),
        allowed_methods=("GET", "CONNECT", "TRACE", "TRACK"),
    )
    out = await tool({"url": "https://example.com/x", "method": method})
    assert out.startswith("BLOCKED")


@pytest.mark.asyncio
async def test_an_operator_can_widen_the_method_set():
    tool = make_http_request(
        client=client_for(ok()), allowed_methods=("GET", "DELETE"),
    )
    assert (await tool({"url": "https://example.com/x", "method": "DELETE"})).startswith(
        "HTTP 200"
    )
    # …and widening it does not implicitly allow everything else.
    assert (await tool({"url": "https://example.com/x", "method": "PUT"})).startswith(
        "BLOCKED"
    )


@pytest.mark.asyncio
async def test_method_is_case_insensitive():
    tool = make_http_request(client=client_for(ok()))
    assert (await tool({"url": "https://example.com/x", "method": "get"})).startswith(
        "HTTP 200"
    )


@pytest.mark.asyncio
async def test_method_check_runs_before_the_request_is_made():
    """A refused verb must not reach the socket, or the check is only cosmetic."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=b"ok")

    tool = make_http_request(client=client_for(handler))
    await tool({"url": "https://example.com/x", "method": "DELETE"})
    assert calls["n"] == 0


def test_defaults_are_read_plus_post():
    assert set(DEFAULT_ALLOWED_METHODS) == {"GET", "HEAD", "OPTIONS", "POST"}


# --- the existing guarantees must survive the rewrite ------------------------
@pytest.mark.asyncio
async def test_redirects_are_still_walked_and_policy_checked():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/creds"})
        return httpx.Response(200, content=b"secret")

    tool = make_http_request(client=client_for(handler))
    out = await tool({"url": "https://example.com/start"})
    assert out.startswith("BLOCKED after 1 redirect")
    assert "169.254.169.254" in out


@pytest.mark.asyncio
async def test_credentials_are_still_stripped_across_a_cross_origin_hop():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        if request.url.host == "a.example.com":
            return httpx.Response(302, headers={"location": "https://b.example.com/x"})
        return httpx.Response(200, content=b"done")

    tool = make_http_request(client=client_for(handler))
    out = await tool({
        "url": "https://a.example.com/x",
        "headers": {"Authorization": "Bearer secret", "X-Trace": "keep"},
    })
    assert out.startswith("HTTP 200")
    assert "authorization" in seen[0]
    assert "authorization" not in seen[1]
    assert seen[1].get("x-trace") == "keep"


@pytest.mark.asyncio
async def test_reserved_addresses_are_still_refused_with_no_policy():
    tool = make_http_request(client=client_for(ok()))
    for url in ("http://169.254.169.254/", "http://127.1/", "http://0x7f000001/",
                "http://localhost/", "http://metadata.google.internal/"):
        assert (await tool({"url": url})).startswith("BLOCKED"), url


@pytest.mark.asyncio
async def test_max_redirects_is_still_enforced():
    def handler(request: httpx.Request) -> httpx.Response:
        n = int(request.url.params.get("n", 0))
        return httpx.Response(302, headers={"location": f"https://example.com/?n={n + 1}"})

    tool = make_http_request(client=client_for(handler), max_redirects=3)
    out = await tool({"url": "https://example.com/?n=0"})
    assert "exceeded max_redirects=3" in out


@pytest.mark.asyncio
async def test_the_audit_record_still_carries_the_chain_and_status():
    audit = RecordingAudit()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "https://example.com/b"})
        return httpx.Response(201, content=b"created")

    tool = make_http_request(client=client_for(handler), audit=audit)
    await tool({"url": "https://example.com/a"})
    assert audit.exchanges[0]["response"]["status"] == 201
