"""MCP handler — port of CC's src/services/mcp (client.ts, MCPConnectionManager).

Responsibilities:
  - hold N server connections (stdio/sse/http — CC supports all; stdio is our default);
  - list_tools() per server, namespace names as `{server_id}__{tool_name}` to avoid
    collisions across servers (CC does the same);
  - get_tool_specs() -> Anthropic `tools=` format (merged, deduped);
  - call_tool(namespaced_name, args) -> routes to the owning server.

The connection layer is abstracted behind ServerConnection so the pure routing/
namespacing logic is testable offline. A real stdio connection wraps the `mcp` Python
SDK's ClientSession; see connect_stdio() (requires a live server to exercise).

Any server is registered the same way; its tools become `{server_id}__{tool}`.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_NS_SEP = "__"

# Provider tool-name rule (OpenAI and Anthropic both): ^[a-zA-Z0-9_-]{1,64}$. An MCP server
# is free to name a tool "do it now!" or to use a server id with a dot; passing that straight
# through produced a spec the provider rejects with a 400 that kills the whole run — for a
# reason the operator cannot see. Names are sanitised here instead, and the original is kept
# for routing so the server still receives the name it published.
_NAME_BAD_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_NAME = 64


def sanitize_tool_name(name: str) -> str:
    """Coerce a name into the provider-accepted character set, without collapsing to empty."""
    cleaned = _NAME_BAD_CHARS.sub("_", str(name)).strip("_-")
    return cleaned or "tool"


def _truncate_name(name: str) -> str:
    """Keep a name inside the 64-char limit, preserving the tail (which carries the tool's
    own name) rather than the server prefix."""
    if len(name) <= _MAX_NAME:
        return name
    # A short stable digest keeps two long names from colliding after truncation.
    import hashlib

    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:6]
    return name[: _MAX_NAME - 7] + "_" + digest


@dataclass(frozen=True)
class ToolSpec:
    name: str  # namespaced
    description: str
    input_schema: dict

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@runtime_checkable
class ServerConnection(Protocol):
    """Minimal surface a server must expose. Real impl wraps mcp.ClientSession."""

    server_id: str

    async def list_tools(self) -> list[dict]: ...  # [{name, description, inputSchema}]

    async def call_tool(self, tool_name: str, arguments: dict) -> Any: ...


class MCPHandler:
    def __init__(
        self,
        *,
        call_timeout: float | None = 60.0,
        list_timeout: float | None = 30.0,
        evict_failed: bool = True,
    ) -> None:
        """`call_timeout` bounds a single tool call and `list_timeout` a refresh; without
        them one hung server blocks the agent for the life of the process. `evict_failed`
        withdraws a server's tools from the advertised spec list once it starts failing, so
        the model stops being offered tools that cannot work."""
        self._servers: dict[str, ServerConnection] = {}
        self._specs: dict[str, ToolSpec] = {}  # namespaced name -> spec
        self._owner: dict[str, str] = {}  # namespaced name -> server_id
        self._bare: dict[str, str] = {}  # namespaced name -> the server's own tool name
        # server_id -> error string for servers that connected but failed list_tools on the
        # last refresh(). A single bad server no longer blanks all tools (it is isolated
        # here); the app surfaces these to the user. Empty when every server is healthy.
        self.failures: dict[str, str] = {}
        self._call_timeout = call_timeout
        self._list_timeout = list_timeout
        self._evict_failed = evict_failed
        self._consecutive_failures: dict[str, int] = {}

    def add_server(self, conn: ServerConnection) -> None:
        if conn.server_id in self._servers:
            raise ValueError(f"duplicate server id {conn.server_id}")
        self._servers[conn.server_id] = conn

    @staticmethod
    def namespaced(server_id: str, tool_name: str) -> str:
        raw = f"{sanitize_tool_name(server_id)}{_NS_SEP}{sanitize_tool_name(tool_name)}"
        return _truncate_name(raw)

    def healthy(self) -> dict[str, bool]:
        """Per-server health as of the last refresh/call. False once a server has failed."""
        return {sid: sid not in self.failures for sid in self._servers}

    async def refresh(self) -> None:
        """Pull tool lists from every server and (re)build the spec table.

        Per-server fault isolation: if one server errors on list_tools it is recorded in
        self.failures and skipped — the other servers' tools are still loaded, and the call
        never raises. (Regression guard: a single broken MCP server must not blank every
        tool or crash the session.)"""
        self._specs.clear()
        self._owner.clear()
        self._bare.clear()
        self.failures.clear()
        for sid, conn in self._servers.items():
            try:
                if self._list_timeout is None:
                    tools = await conn.list_tools()
                else:
                    async with asyncio.timeout(self._list_timeout):
                        tools = await conn.list_tools()
            except asyncio.TimeoutError:
                self.failures[sid] = f"list_tools timed out after {self._list_timeout:g}s"
                continue
            except Exception as e:
                self.failures[sid] = f"{type(e).__name__}: {e}"
                continue
            self._consecutive_failures.pop(sid, None)
            for tool in tools:
                bare = tool["name"]
                ns = self.namespaced(sid, bare)
                if ns in self._specs:  # sanitising can map two names onto one
                    n = 2
                    while _truncate_name(f"{ns}_{n}") in self._specs:
                        n += 1
                    ns = _truncate_name(f"{ns}_{n}")
                self._specs[ns] = ToolSpec(
                    name=ns,
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema") or tool.get("input_schema") or {},
                )
                self._owner[ns] = sid
                self._bare[ns] = bare  # route with the server's OWN name, not the sanitised one

    def get_tool_specs(self) -> list[dict]:
        """Anthropic `tools=` payload for all connected MCP tools."""
        return [s.to_anthropic() for s in self._specs.values()]

    def tool_names(self) -> list[str]:
        return list(self._specs)

    async def call_tool(self, namespaced_name: str, arguments: dict) -> Any:
        """Route one call to its owning server, under a wall-clock cap.

        A server that fails or times out is recorded in `failures`; with `evict_failed` its
        tools are withdrawn from the advertised spec list so the model is not repeatedly
        offered tools that cannot work for the rest of the session.
        """
        sid = self._owner.get(namespaced_name)
        if sid is None:
            raise KeyError(f"unknown MCP tool {namespaced_name!r}")
        bare = self._bare.get(namespaced_name)
        if bare is None:
            _, _, bare = namespaced_name.partition(_NS_SEP)
        try:
            if self._call_timeout is None:
                result = await self._servers[sid].call_tool(bare, arguments)
            else:
                async with asyncio.timeout(self._call_timeout):
                    result = await self._servers[sid].call_tool(bare, arguments)
        except asyncio.TimeoutError:
            self._mark_failed(sid, f"call to {bare!r} timed out after {self._call_timeout:g}s")
            raise TimeoutError(
                f"MCP server {sid!r} did not answer {bare!r} within {self._call_timeout:g}s"
            ) from None
        except Exception as e:
            self._mark_failed(sid, f"{type(e).__name__}: {e}")
            raise
        self._consecutive_failures.pop(sid, None)
        return result

    def _mark_failed(self, sid: str, reason: str) -> None:
        self.failures[sid] = reason
        self._consecutive_failures[sid] = self._consecutive_failures.get(sid, 0) + 1
        if self._evict_failed:
            for ns in [n for n, owner in self._owner.items() if owner == sid]:
                self._specs.pop(ns, None)
                self._owner.pop(ns, None)
                self._bare.pop(ns, None)

    async def reconnect(self, server_id: str) -> bool:
        """Re-list one server's tools and, on success, restore them to the spec table.

        A server that dies mid-session used to stay advertised and fail every call for the
        rest of the run, with no way back short of rebuilding the whole handler.
        """
        conn = self._servers.get(server_id)
        if conn is None:
            return False
        reconnect = getattr(conn, "reconnect", None)
        if callable(reconnect):
            try:
                res = reconnect()
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                self.failures[server_id] = f"reconnect failed: {type(e).__name__}: {e}"
                return False
        try:
            if self._list_timeout is None:
                tools = await conn.list_tools()
            else:
                async with asyncio.timeout(self._list_timeout):
                    tools = await conn.list_tools()
        except Exception as e:
            self.failures[server_id] = f"{type(e).__name__}: {e}"
            return False
        for tool in tools:
            bare = tool["name"]
            ns = self.namespaced(server_id, bare)
            self._specs[ns] = ToolSpec(
                name=ns,
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema") or tool.get("input_schema") or {},
            )
            self._owner[ns] = server_id
            self._bare[ns] = bare
        self.failures.pop(server_id, None)
        self._consecutive_failures.pop(server_id, None)
        return True


# --- real stdio connection (live server; integration-tested, not in offline unit suite) ---
class StdioMCPConnection:
    """A live ServerConnection backed by an MCP stdio subprocess (mcp Python SDK).

    Used as an async context manager so the subprocess + session stay open for the
    connection's lifetime:

        async with StdioMCPConnection("codebasememory", CMD, []) as conn:
            handler.add_server(conn); await handler.refresh()

    call_tool() flattens the MCP result's content blocks to text so the model gets a
    readable string.
    """

    def __init__(
        self,
        server_id: str,
        command: str,
        args: list[str] | None = None,
        env: dict | None = None,
        cwd: str | None = None,
    ) -> None:
        self.server_id = server_id
        self._command = command
        self._args = args or []
        self._env = env
        self._cwd = cwd
        self._session = None
        self._stack = None

    async def __aenter__(self) -> "StdioMCPConnection":
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self._command, args=self._args, env=self._env, cwd=self._cwd
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def list_tools(self) -> list[dict]:
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {"type": "object"},
            }
            for t in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = await self._session.call_tool(tool_name, arguments)
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        out = "\n".join(parts)
        if getattr(result, "isError", False):
            return f"MCP_ERROR: {out}"
        return out
