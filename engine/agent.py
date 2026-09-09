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
from engine.coordinator import Coordinator, last_assistant_text
from engine.compact import compact_transcript, estimate_tokens
from engine.events import AgentEvent, CancelToken, EventType
from engine.hooks import HookEngine
from engine.loopguard import LoopGuard
from engine.mcp import MCPHandler
from engine.memory import MemoryStore
from engine.permissions import Mode, NetworkPolicy, PermissionEngine
from engine.permissions.danger import command_text, is_shell_call
from engine.permissions.readonly import is_read_only_command
from engine.sandbox import FilesystemGuard, FilesystemPolicy
from engine.session import SessionStore, ToolJournal
from engine.structured import SCHEMA_INSTRUCTION, schema_for
from engine.subagents import make_spawn_tool
from engine.tools.decorator import as_tool, tool
from engine.tools.http import (
    DEFAULT_ALLOWED_METHODS as HTTP_DEFAULT_METHODS,
    DEFAULT_MAX_RESPONSE_BYTES as HTTP_DEFAULT_MAX_BYTES,
)
from engine.tools import (
    EDIT_FILE_SPEC,
    FIND_FILES_SPEC,
    HTTP_REQUEST_SPEC,
    LIST_DIR_SPEC,
    READ_FILE_SPEC,
    FORGET_MEMORY_SPEC,
    RECALL_MEMORY_SPEC,
    SAVE_MEMORY_SPEC,
    WRITE_FILE_SPEC,
    make_edit_file,
    make_find_files,
    make_forget_memory,
    make_http_request,
    make_list_dir,
    make_read_file,
    make_recall_memory,
    make_save_memory,
    make_write_file,
)


class OutputValidationError(ValueError):
    """Raised by `Agent.run()` when `output_type` was declared and no valid output was
    produced after the repair retries. Returning None instead would surface the schema
    failure in the caller's code, several frames later, as an AttributeError."""


class _NullAudit:
    """Stand-in when auditing is disabled: the coordinator only needs these three."""

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
    # Confine READS to the workdir the way writes are confined. Off by default because an
    # agent working on a codebase legitimately reads outside its workspace; turn it on for
    # any deployment that also grants network egress, where an unconfined read plus an
    # outbound request is a complete exfiltration path.
    confine_reads: bool = False
    # Extra roots the agent may read even under `confine_reads` (a shared library, a
    # sibling repo). Never writable unless also passed to allow_write.
    allow_read: tuple[str, ...] = ()
    # Extra roots the agent may WRITE, outside the workdir. Also the opt-in for the
    # auto-executing paths below: naming one here re-permits it.
    allow_write: tuple[str, ...] = ()
    # Refuse writes to files that execute code later without anyone invoking them: git
    # hooks, CI definitions, `conftest.py`, `sitecustomize.py`, shell rc files, and the
    # agent's own `.claude/` configuration. Workdir confinement bounds WHERE the agent
    # writes; it is not a code-execution boundary, because the human whose repository this
    # is will run `git commit`, `pytest` or `python` in it afterwards and the agent's file
    # then runs outside every policy layer with their privileges. Set False, or name the
    # specific path in `allow_write`, for an agent that legitimately maintains CI.
    protect_auto_executing: bool = True
    # Storage.
    #
    # `state_dir` holds the engine's CONTROL PLANE: the audit log, the session snapshot,
    # the tool journal and memory. It defaults to `workdir/.agent-state` and is made
    # unreadable AND unwritable to the file tools, because the agent must not be able to
    # edit the record of what it did — a prompt-injected agent used to be able to overwrite
    # its own tamper-evident audit log with `WriteFile("audit.jsonl", "[]")`. Point it at a
    # separate volume (or an append-only mount) for a real deployment.
    state_dir: str | Path | None = None
    audit: bool = True
    audit_hmac_key: str | bytes | None = None     # keyed, unforgeable audit chain
    memory_dir: str | Path | None = None
    memory_namespace: Optional[str] = None        # tenant/user scope for memory
    # Delegation. `subagents` maps a type name to a zero-arg factory returning a Subagent;
    # giving it any entry adds the SpawnSubagent tool. `max_spawns` caps how many one run
    # may start, since each spawn carries its own turn budget the parent's cannot see.
    subagents: dict = field(default_factory=dict)
    max_spawns: Optional[int] = 8
    # Prompt-injection posture + optional add-ons.
    trusted_tools: tuple[str, ...] = ()
    # Rails that used to be opt-in and therefore off for anyone following the quick start.
    # `None` means "use the engine default" (on); pass False to disable, or an instance to
    # configure. A degenerate-repetition breaker and a context ceiling are not advanced
    # features — without them a long run burns its budget on a repeated turn, or grows the
    # transcript until the provider rejects it on length.
    loop_guard: Any = None
    context_compactor: Any = None
    # Compact the transcript once it passes this estimated token count. The agent's own
    # model does the summarising. Set 0/None to never compact.
    compact_at_tokens: Optional[int] = 100_000
    # Notes carried VERBATIM across every compaction (verified facts, decisions).
    pinned_notes: list = field(default_factory=list)
    system_prompt: Optional[str] = None           # sent as a real system message
    system_preamble: Optional[str] = None         # deprecated alias for system_prompt
    # Structured output. Declare the shape you need and `run()` returns THAT, validated,
    # instead of prose the caller has to re-parse:
    #
    #     @dataclass
    #     class Triage:
    #         severity: str
    #         summary: str
    #
    #     agent = Agent(model=model, workdir=".", output_type=Triage)
    #     triage = await agent.run("Classify this incident report.")   # -> Triage
    #
    # Accepts a JSON Schema dict, a Pydantic model (duck-typed, not a dependency), a
    # dataclass, a TypedDict, or a primitive. The schema goes into the request where the
    # provider supports it AND into the system prompt, the final message is validated
    # against it, and a mismatch is handed back to the model with the specific errors —
    # `output_retries` times — before the run gives up. `agent.output_error` says why when
    # there is no valid output.
    output_type: Any = None
    output_retries: int = 2
    # Safety + budget rails.
    tool_timeout: Optional[float] = 120.0
    max_parallel_tools: int = 8
    token_budget: Optional[int] = None
    validate_tool_input: bool = True
    http_rate_limiter: Any = None                 # awaited before each HttpRequest
    # Verbs `HttpRequest` may issue. The default is read + POST: once a host is allowlisted
    # for reading, nothing else stopped the model issuing DELETE against it, and the
    # permission rule syntax matches on the URL only so it cannot express a per-method rule.
    http_allowed_methods: tuple[str, ...] = HTTP_DEFAULT_METHODS
    # Bytes read off the wire per response. The whole body used to be buffered and only then
    # truncated for the model, so one allow-listed URL was a memory-exhaustion vector.
    http_max_response_bytes: int = HTTP_DEFAULT_MAX_BYTES
    # Observability + control (see engine/events.py).
    on_event: Any = None                          # called with every AgentEvent
    cancel_token: Any = None                      # cooperative stop signal

    _coord: Optional[Coordinator] = field(default=None, init=False, repr=False)
    _schema: Optional[dict] = field(default=None, init=False, repr=False)
    _audit: Any = field(default=None, init=False, repr=False)
    _http_tool: Any = field(default=None, init=False, repr=False)
    memory: Optional[MemoryStore] = field(default=None, init=False, repr=False)
    _state_dir: Optional[Path] = field(default=None, init=False, repr=False)
    _mcp_handler: Any = field(default=None, init=False, repr=False)
    _owns_mcp: bool = field(default=False, init=False, repr=False)

    def _output_schema(self) -> Optional[dict]:
        """The JSON Schema for `output_type`, derived once."""
        if self.output_type is None:
            return None
        if self._schema is None:
            self._schema = schema_for(self.output_type)
        return self._schema

    def _effective_system(self) -> Optional[str]:
        """The system prompt, with the output-schema instruction appended when one applies.

        Sent in the prompt as well as in `response_format` because most OpenAI-compatible
        endpoints support neither `json_schema` nor `json_object`, and on those the prompt
        is the ONLY thing carrying the requirement.
        """
        base = self.system_prompt or self.system_preamble
        schema = self._output_schema()
        if schema is None:
            return base
        import json as _json

        instruction = SCHEMA_INSTRUCTION.format(
            schema=_json.dumps(schema, indent=2)[:4000]
        )
        return f"{base}\n\n{instruction}" if base else instruction

    def _resolve_loop_guard(self) -> Any:
        """`None` -> a default LoopGuard; `False` -> disabled; anything else -> as given."""
        if self.loop_guard is None:
            return LoopGuard()
        return self.loop_guard or None

    def _resolve_compactor(self) -> Any:
        """Build the default threshold compactor unless the caller supplied or disabled one.

        Compaction was opt-in, so a caller following the quick start had a transcript that
        grew monotonically until the provider rejected it on context length — measured at
        501 -> 17,018 estimated tokens over twelve `send()` calls with nothing trimming it.
        The pieces to prevent that already existed; only the wiring was missing.
        """
        if self.context_compactor is not None:
            return self.context_compactor or None
        if not self.compact_at_tokens:
            return None

        threshold = int(self.compact_at_tokens)
        pinned = self.pinned_notes
        model = self.model

        async def compact_when_large(messages: list[dict]) -> list[dict]:
            if estimate_tokens(messages) < threshold:
                return messages
            try:
                new_messages, _summary = await compact_transcript(
                    model, messages, pinned=list(pinned) or None
                )
            except Exception:
                # A failed summarisation must not end the run: carrying on with the long
                # transcript is strictly better than raising out of the loop.
                return messages
            return new_messages

        return compact_when_large

    async def _build(self) -> Coordinator:
        if self._coord is not None:
            return self._coord
        workdir = Path(self.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        state = Path(self.state_dir).resolve() if self.state_dir else workdir / ".agent-state"
        state.mkdir(parents=True, exist_ok=True)
        self._state_dir = state

        audit = (
            AuditLog(state / "audit.jsonl", hmac_key=self.audit_hmac_key)
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
            # Any command-execution tool, not just one literally named `bash` — the
            # auto-allow must recognise the same calls the danger checks do, or a safe
            # `git status` through a differently-named tool needlessly prompts.
            read_only_classifier=lambda call: (
                is_shell_call(call) and is_read_only_command(command_text(call))
            ),
        )

        native: dict = {}
        specs: list = []
        builtin_trusted: list[str] = []

        if self.enable_file_tools:
            # The control plane is inside the workdir by default, so it must be carved back
            # out of the writable (and readable) region explicitly. deny_write stops the
            # agent editing its own audit trail, session and memory; deny_read stops it
            # reading back an audit log that may hold request/response payloads.
            fs = FilesystemGuard(FilesystemPolicy.build(
                workdir, deny_write=(str(state),), deny_read=(str(state),),
                allow_read=tuple(self.allow_read), allow_write=tuple(self.allow_write),
                confine_reads=self.confine_reads,
                protect_auto_executing=self.protect_auto_executing,
            ))
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
                self.memory_dir or (self._state_dir / "memory"),
                namespace=self.memory_namespace,
                audit=audit,
            )
            native[SAVE_MEMORY_SPEC["name"]] = make_save_memory(self.memory)
            native[RECALL_MEMORY_SPEC["name"]] = make_recall_memory(self.memory)
            native[FORGET_MEMORY_SPEC["name"]] = make_forget_memory(self.memory)
            specs += [SAVE_MEMORY_SPEC, RECALL_MEMORY_SPEC, FORGET_MEMORY_SPEC]
            # SaveMemory/ForgetMemory echo back only what the engine did, so they are
            # trusted. RecallMemory is NOT: its content was authored by the model, and the
            # model's input is untrusted tool output — so a fact poisoned on one run came
            # back to a later run wearing the engine's own authority. It is fenced.
            builtin_trusted += ["SaveMemory", "ForgetMemory"]

        if self.enable_http_tool:
            # The same policy the permission engine uses, so the pre-flight check and the
            # per-redirect-hop check cannot disagree.
            self._http_tool = make_http_request(
                audit=audit,
                rate_limiter=self.http_rate_limiter,
                network_policy=self.network_policy,
                allowed_methods=tuple(self.http_allowed_methods),
                max_response_bytes=int(self.http_max_response_bytes),
            )
            native[HTTP_REQUEST_SPEC["name"]] = self._http_tool
            specs.append(HTTP_REQUEST_SPEC)

        if self.subagents:
            spawn, spawn_spec = make_spawn_tool(
                dict(self.subagents), max_spawns=self.max_spawns
            )
            native[spawn_spec["name"]] = spawn
            specs.append(spawn_spec)
            # The subagent's own summary, written by the engine's wrapper — but its BODY is
            # model-authored text derived from that subagent's tool output, so it is fenced
            # like any other tool result.

        # `tools=` accepts a @tool-decorated function, a plain annotated function, or the
        # original (spec, handler) tuple. One funnel (`as_tool`) rather than a branch per
        # shape, so the low-level pair keeps working exactly as before.
        for entry in self.tools:
            spec, handler = as_tool(entry)
            name = spec["name"]
            if name in native:
                raise ValueError(f"tool name {name!r} collides with a built-in tool")
            native[name] = handler
            specs.append(spec)

        handler = None
        if self.mcp_servers:
            handler = MCPHandler()
            self._mcp_handler = handler
            self._owns_mcp = True
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
            session=SessionStore(state / "session.jsonl"), resume=self.resume,
            journal=ToolJournal(state / "tooljournal.jsonl"),
            loop_guard=self._resolve_loop_guard(),
            context_compactor=self._resolve_compactor(),
            tool_timeout=self.tool_timeout,
            max_parallel_tools=self.max_parallel_tools,
            token_budget=self.token_budget,
            validate_tool_input=self.validate_tool_input,
            on_event=self.on_event,
            cancel_token=self.cancel_token,
            output_schema=self._output_schema(),
            output_type=self.output_type,
            output_retries=self.output_retries,
        )
        return self._coord

    def _prime(self, message: str) -> str:
        """Fall back to prepending the preamble ONLY when the model client cannot carry a
        real system message (so a custom client still gets the instructions)."""
        system = self._effective_system()
        if system and not hasattr(self.model, "system_prompt"):
            return f"{system}\n\n{message}"
        return message

    async def run(self, message: str) -> Any:
        """Run one cold-start task to a stopping point.

        Returns the final assistant text — or, when `output_type` is set, the VALIDATED
        object of that type. A caller who declared a shape asked for that shape; handing
        back prose they must re-parse is what the declaration exists to avoid.

        With `output_type` set and no valid output produced (after `output_retries`
        repairs), this raises `OutputValidationError`. Silently returning `None` would push
        a schema failure into the caller's code as an AttributeError several frames later.
        """
        coord = await self._build()
        messages = await coord.run(self._prime(message))
        if self.output_type is None:
            return _last_text(messages)
        return self._require_output()

    def _require_output(self) -> Any:
        if self._coord is not None and self._coord.output_error:
            raise OutputValidationError(
                f"the model did not produce output matching {self.output_type!r} after "
                f"{self.output_retries} repair attempt(s): {self._coord.output_error}"
            )
        return getattr(self._coord, "output", None)

    async def send(self, message: str) -> Any:
        """Multi-turn: append `message` to the ongoing transcript, run to a stopping
        point, and return the final assistant text — or the validated `output_type` object
        when one is declared. Call repeatedly for a conversation."""
        coord = await self._build()
        first = coord.live_messages is None
        messages = await coord.send(self._prime(message) if first else message)
        if self.output_type is None:
            return _last_text(messages)
        return self._require_output()

    async def stream(self, message: str, *, live: bool = False):
        """Run and yield `AgentEvent`s as they happen, ending with RUN_END.

            async for event in agent.stream("summarise the repo"):
                if event.type is EventType.TEXT_DELTA:
                    print(event.text, end="", flush=True)

        With `live=True` the message continues the ongoing conversation (`send`) instead of
        starting a cold run. Stop consuming to cancel: the run halts at the next safe
        boundary, leaving a resumable session rather than a half-executed tool call.
        """
        coord = await self._build()
        if coord.cancel_token is None:
            coord.cancel_token = CancelToken()
        async for event in coord.astream(self._prime(message), live=live):
            yield event

    def cancel(self, reason: str = "cancelled by the caller") -> None:
        """Ask the current run to stop at the next safe boundary (between turns, or before
        the next tool dispatch). Safe to call from another task."""
        if self._coord is not None:
            if self._coord.cancel_token is None:
                self._coord.cancel_token = CancelToken()
            self._coord.cancel_token.cancel(reason)

    @property
    def cancelled(self) -> bool:
        """True if the last run stopped because it was cancelled."""
        return bool(getattr(self._coord, "cancelled", False))

    @property
    def completed(self) -> bool:
        """True if the last run ended on the model's own turn (not truncated at max_turns)."""
        return bool(getattr(self._coord, "completed", False))

    @property
    def usage(self) -> dict:
        """Cumulative token usage for this agent's runs (empty before the first run)."""
        return dict(getattr(self._coord, "usage", {}) or {})

    @property
    def output(self) -> Any:
        """The validated structured result of the last run, or None.

        `run()` returns the same value; this is for callers that want it without a second
        call, and for the `output_type is None` case where it is always None."""
        return getattr(self._coord, "output", None)

    @property
    def output_error(self) -> str:
        """Why the last run produced no schema-valid output, or "" if it did."""
        return str(getattr(self._coord, "output_error", "") or "")

    @property
    def failures(self) -> list:
        """Tool calls in the last run that produced no result: denied, unapproved,
        rejected by schema validation, or raised.

        A failing tool used to be invisible to the caller. The error text was formatted
        into the transcript for the MODEL and nowhere else, so `run()` returned the model's
        apology, `completed` was True, and nothing distinguished a clean run from one in
        which every tool failed. Check this before trusting a result.
        """
        return list(getattr(self._coord, "failures", []) or [])

    @property
    def mcp_failures(self) -> dict:
        """MCP servers that failed to connect or list their tools, `{server_id: reason}`.

        `MCPHandler.refresh()` isolates a broken server so one bad mount does not blank
        every tool — but the facade never read the result, so mounting three servers and
        getting zero working tools produced a normal return and a plausible answer. The
        only access was `agent._mcp_handler.failures`: a private attribute of a private
        attribute.
        """
        return dict(getattr(self._mcp_handler, "failures", {}) or {})

    @property
    def run_id(self) -> str:
        """The identifier stamped on every audit record for this agent's run.

        Needed to join the evidence chain to anything outside it: the caller's own logs,
        its request id, its APM trace. Empty before the first run.
        """
        return str(getattr(self._coord, "run_id", "") or "")

    @property
    def ok(self) -> bool:
        """True if the last run finished on the model's own turn AND no tool failed AND no
        MCP server is broken. The single check a caller needs before trusting a result."""
        return (
            self.completed
            and not self.failures
            and not self.mcp_failures
            and not self.output_error
        )

    # --- lifecycle ----------------------------------------------------------
    async def aclose(self) -> None:
        """Release everything the facade opened: the audit file handle, the HTTP tool's
        pooled client, and the model client if it owns one."""
        for closer in (
            getattr(self._http_tool, "aclose", None),
            # MCP servers are subprocesses; nothing else closes them, so a long-lived host
            # that built Agents with mcp_servers= leaked one per agent. Only close the
            # handler THIS facade built — a caller managing its own connections in an
            # `async with` still owns them, and aclose() is idempotent either way.
            getattr(self._mcp_handler, "aclose", None) if self._owns_mcp else None,
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


# One definition of "the model's last words" lives in engine.coordinator; this used to
# be a third private copy of it.
_last_text = last_assistant_text
