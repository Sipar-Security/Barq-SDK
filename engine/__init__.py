"""A small, embeddable SDK for building tool-using AI agents in Python.

Modeled on Claude Code's agent architecture patterns, this is a **library**, not an
application: no CLI, no product surface. You bring a model (any OpenAI-compatible
endpoint) and some tools; the engine gives you a permissioned, auditable, crash-resumable
agent loop with native + MCP tools, file-based memory, context compaction, and a
degenerate-loop circuit breaker.

Quick start
-----------
    from engine import Agent
    from engine.providers import OpenAICompatClient, ModelSpec

    model = OpenAICompatClient(ModelSpec(provider="deepseek", model="deepseek-chat"))
    agent = Agent(model=model, workdir="./run")
    answer = await agent.run("Read README.md and summarise it.")

Lower level, wire the pieces yourself:
    from engine import Coordinator, PermissionEngine, HookEngine, Mode
    coord = Coordinator(model=..., permissions=PermissionEngine(HookEngine(), Mode.AUTO),
                        audit=..., native_tools={...}, tool_specs=[...])
    transcript = await coord.run("...")
"""

from __future__ import annotations

from engine.agent import Agent, OutputValidationError
from engine.coordinator import (
    Coordinator,
    ModelClient,
    ModelResponse,
    ToolFailure,
    last_assistant_text,
    run_sync,
)
from engine.hooks import (
    CommandHook,
    FunctionHook,
    HookEngine,
    HookEvent,
    HookInput,
    HookOutcome,
)
from engine.events import AgentEvent, CancelledRun, CancelToken, EventType
from engine.loopguard import LoopGuard
from engine.compact import compact_transcript, estimate_tokens
from engine.mcp import MCPHandler, ServerConnection, StdioMCPConnection
from engine.structured import extract_json, schema_for
from engine.telemetry import OTelTelemetry, otel_available
from engine.tools.decorator import as_tool, is_tool, tool, tool_spec_from
from engine.validation import validate_tool_input
from engine.memory import Memory, MemoryStore, MemoryType
from engine.permissions import (
    Behavior,
    Decision,
    HostAllowlist,
    Mode,
    NetworkPolicy,
    PermissionEngine,
    ToolCall,
    allow,
    ask,
    deny,
)
from engine.ratelimit import NullRateLimiter, RateLimiter
from engine.session import SessionStore, ToolJournal
from engine.subagents import Subagent, SubagentResult, make_spawn_tool

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "OutputValidationError",
    "Coordinator",
    "ToolFailure",
    "last_assistant_text",
    "schema_for",
    "extract_json",
    "tool",
    "as_tool",
    "is_tool",
    "tool_spec_from",
    "OTelTelemetry",
    "otel_available",
    "ModelClient",
    "ModelResponse",
    "run_sync",
    "HookEngine",
    "HookEvent",
    "HookInput",
    "HookOutcome",
    "FunctionHook",
    "CommandHook",
    "AgentEvent",
    "EventType",
    "CancelToken",
    "CancelledRun",
    "LoopGuard",
    "compact_transcript",
    "estimate_tokens",
    "MCPHandler",
    "validate_tool_input",
    "ServerConnection",
    "StdioMCPConnection",
    "Memory",
    "MemoryStore",
    "MemoryType",
    "PermissionEngine",
    "Mode",
    "Behavior",
    "Decision",
    "ToolCall",
    "NetworkPolicy",
    "HostAllowlist",
    "allow",
    "ask",
    "deny",
    "RateLimiter",
    "NullRateLimiter",
    "SessionStore",
    "ToolJournal",
    "Subagent",
    "SubagentResult",
    "make_spawn_tool",
    "__version__",
]
