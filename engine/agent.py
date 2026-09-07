"""Agent: the high-level facade that assembles the runtime into one object.

The engine's pieces (coordinator loop, permission engine, tools, MCP handler, memory,
sessions, audit) are usable directly, but most callers want a single object they hand a
model, some tools, and a task. `Agent` wires those together with sensible defaults:

    from engine import Agent
    from engine.providers import OpenAICompatClient, ModelSpec

    model = OpenAICompatClient(ModelSpec(provider="deepseek", model="deepseek-chat"))
    async with Agent(model=model, workdir="./run") as agent:
        answer = await agent.run("List the files here and summarise the README.")

By default the agent gets file tools (read/write/edit/list/find, confined to `workdir`),
memory tools, a permission engine in AUTO mode, an audit log in the workdir, and
crash-resumable sessions. Everything is overridable; pass `tools=` for your own native
tools and `mcp_servers=` for any number of MCP servers (their tool names are namespaced).

Use it as an async context manager (or call `aclose()`) so the audit file handle, the
pooled HTTP clients and the model client are released (a long-lived host that constructs
Agents without closing them leaks a file descriptor and a connection pool each time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from engine.audit import AuditLog
from engine.coordinator import Coordinator
from engine.hooks import HookEngine
from engine.mcp import MCPHandler
from engine.memory import MemoryStore
from engine.permissions import Mode, NetworkPolicy, PermissionEngine
from engine.permissions.readonly import is_read_only_command
from engine.sandbox import FilesystemGuard, FilesystemPolicy
from engine.session import SessionStore, ToolJournal
from engine.tools import (
    EDIT_FILE_SPEC,
    FIND_FILES_SPEC,
    HTTP_REQUEST_SPEC,
    LIST_DIR_SPEC,
    READ_FILE_SPEC,
    RECALL_MEMORY_SPEC,
    SAVE_MEMORY_SPEC,
    WRITE_FILE_SPEC,
    make_edit_file,
    make_find_files,
    make_http_request,
    make_list_dir,
    make_read_file,
    make_recall_memory,
    make_save_memory,
    make_write_file,
)


class _NullAudit:
    """Stand-in when auditing is disabled: the coordinator only needs these three."""ee."""

    def log_decision(self, *a, **k): return ""
    def log_blocked(self, *a, **k): return ""
    def log_exchange(self, *a, **k): return ""
    def close(self): return None


@dataclass
class Agent:
    model: Any                                   # ModelClient (engine.coordinator.ModelClient)
    workdir: str | Path = "."
    # Extra native tools the model can call: [(spec_dict, async_or_sync_handler), ...].
    tools: list = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)  # [ServerConnection, ...]
    mode: Mode = Mode.AUTO
    network_policy: Optional[NetworkPolicy] = None
    allow_rules: tuple[str, ...] = ()
    deny_rules: tuple[str, ...] = ()
    ask_rules: tuple[str, ...] = ()
    max_turns: int = 30
    resume: bool = False
    elicit: Any = None                            # ASK-decision approval callback
    # Built-in toolsets (on by default).
    enable_file_tools: bool = True
    enable_memory_tools: bool = True
    enable_http_tool: bool = False                # opt-in: arbitrary outbound HTTP
    # Storage.
    audit: bool = True
    audit_hmac_key: str | bytes | None = None     # keyed, unforgeable audit chain
    memory_dir: str | Path | None = None
    memory_namespace: Optional[str] = None        # tenant/user scope for memory
    # Prompt-injection posture + optional add-ons.
    trusted_tools: tuple[str, ...] = ()
    loop_guard: Any = None
    context_compactor: Any = None
    system_prompt: Optional[str] = None           # sent as a real system message
    system_preamble: Optional[str] = None         # deprecated alias for system_prompt
    # Safety + budget rails.
    tool_timeout: Optional[float] = 120.0
    max_parallel_tools: int = 8
    token_budget: Optional[int] = None
    validate_tool_input: bool = True
    http_rate_limiter: Any = None                 # awaited before each HttpRequest

    _coord: Optional[Coordinator] = field(default=None, init=False, repr=False)
    _audit: Any = field(default=None, init=False, repr=False)
    _http_tool: Any = field(default=None, init=False, repr=False)
    memory: Optional[MemoryStore] = field(default=None, init=False, repr=False)

    def _effective_system(self) -> Optional[str]:
        return self.system_prompt or self.system_preamble

    async def _build(self) -> Coordinator:
        if self._coord is not None:
            return self._coord
        workdir = Path(self.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        audit = (
            AuditLog(workdir / "audit.jsonl", hmac_key=self.audit_hmac_key)
            if self.audit else _NullAudit()
        )
        self._audit = audit
        hooks = HookEngine()
        perms = PermissionEngine(
            hooks,
            mode=self.mode,
            network_policy=self.network_policy,
            allow_rules=self.allow_rules,
            deny_rules=self.deny_rules,
            ask_rules=self.ask_rules,
            read_only_classifier=lambda call: (
                call.name.lower() in ("bash", "shell", "powershell")
                and is_read_only_command(str(call.input.get("command", "")))
            ),
        )

        native: dict = {}
        specs: list = []
        builtin_trusted: list[str] = []

        if self.enable_file_tools:
            fs = FilesystemGuard(FilesystemPolicy.build(workdir))
            native[READ_FILE_SPEC["name"]] = make_read_file(fs)
            native[WRITE_FILE_SPEC["name"]] = make_write_file(fs)
            native[EDIT_FILE_SPEC["name"]] = make_edit_file(fs)
            native[LIST_DIR_SPEC["name"]] = make_list_dir(fs)
            native[FIND_FILES_SPEC["name"]] = make_find_files(fs, workdir)
            specs += [READ_FILE_SPEC, WRITE_FILE_SPEC, EDIT_FILE_SPEC,
                      LIST_DIR_SPEC, FIND_FILES_SPEC]
            # Engine-controlled output, not untrusted external data: no need to fence it.
            builtin_trusted += ["WriteFile", "EditFile"]

        if self.enable_memory_tools:
            self.memory = MemoryStore(
                self.memory_dir or (workdir / "memory"), namespace=self.memory_namespace
            )
            native[SAVE_MEMORY_SPEC["name"]] = make_save_memory(self.memory)
            native[RECALL_MEMORY_SPEC["name"]] = make_recall_memory(self.memory)
            specs += [SAVE_MEMORY_SPEC, RECALL_MEMORY_SPEC]
            builtin_trusted += ["SaveMemory", "RecallMemory"]

        if self.enable_http_tool:
            # The same policy the permission engine uses, so the pre-flight check and the
            # per-redirect-hop check cannot disagree.
            self._http_tool = make_http_request(
                audit=audit,
                rate_limiter=self.http_rate_limiter,
                network_policy=self.network_policy,
            )
            native[HTTP_REQUEST_SPEC["name"]] = self._http_tool
            specs.append(HTTP_REQUEST_SPEC)

        for spec, handler in self.tools:
            name = spec["name"]
            if name in native:
                raise ValueError(f"tool name {name!r} collides with a built-in tool")
            native[name] = handler
            specs.append(spec)

        handler = None
        if self.mcp_servers:
            handler = MCPHandler()
            for srv in self.mcp_servers:
                handler.add_server(srv)
            await handler.refresh()
            specs += handler.get_tool_specs()

        # Prefer a real system message when the model client supports one.
        system = self._effective_system()
        if system and hasattr(self.model, "system_prompt"):
            if not getattr(self.model, "system_prompt", None):
                self.model.system_prompt = system

        self._coord = Coordinator(
            model=self.model, permissions=perms, audit=audit,
            native_tools=native, tool_specs=specs, mcp_handler=handler,
            max_turns=self.max_turns,
            trusted_tools=frozenset(self.trusted_tools) | frozenset(builtin_trusted),
            elicit=self.elicit,
            session=SessionStore(workdir / "session.json"), resume=self.resume,
            journal=ToolJournal(workdir / "tooljournal.jsonl"),
            loop_guard=self.loop_guard,
            context_compactor=self.context_compactor,
            tool_timeout=self.tool_timeout,
            max_parallel_tools=self.max_parallel_tools,
            token_budget=self.token_budget,
            validate_tool_input=self.validate_tool_input,
        )
        return self._coord

    def _prime(self, message: str) -> str:
        """Fall back to prepending the preamble ONLY when the model client cannot carry a
        real system message (so a custom client still gets the instructions)."""
        system = self._effective_system()
        if system and not hasattr(self.model, "system_prompt"):
            return f"{system}\n\n{message}"
        return message

    async def run(self, message: str) -> str:
        """Run one cold-start task to a stopping point; return the final assistant text."""
        coord = await self._build()
        messages = await coord.run(self._prime(message))
        return _last_text(messages)

    async def send(self, message: str) -> str:
        """Multi-turn: append `message` to the ongoing transcript, run to a stopping
        point, and return the final assistant text. Call repeatedly for a conversation."""
        coord = await self._build()
        first = coord.live_messages is None
        messages = await coord.send(self._prime(message) if first else message)
        return _last_text(messages)

    @property
    def completed(self) -> bool:
        """True if the last run ended on the model's own turn (not truncated at max_turns)."""
        return bool(getattr(self._coord, "completed", False))

    @property
    def usage(self) -> dict:
        """Cumulative token usage for this agent's runs (empty before the first run)."""
        return dict(getattr(self._coord, "usage", {}) or {})

    # --- lifecycle ----------------------------------------------------------
    async def aclose(self) -> None:
        """Release everything the facade opened: the audit file handle, the HTTP tool's
        pooled client, and the model client if it owns one."""
        for closer in (
            getattr(self._http_tool, "aclose", None),
            getattr(self.model, "aclose", None),
        ):
            if callable(closer):
                try:
                    await closer()
                except Exception:
                    pass
        if self._audit is not None and hasattr(self._audit, "close"):
            try:
                self._audit.close()
            except Exception:
                pass

    async def __aenter__(self) -> "Agent":
        await self._build()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


def _last_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, list):
            text = "\n".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            ).strip()
            if text:
                return text
        elif isinstance(content, str) and content.strip():
            return content.strip()
    return ""
