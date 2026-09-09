# Barq-SDK: a small SDK for building tool-using AI agents

A compact, embeddable Python library for building agents that call tools, use MCP servers,
remember facts across runs, and stay inside a permission policy (with a tamper-evident
audit trail and crash-resume built in). It is a **library, not an application**: you bring a
model and some tools; the engine gives you the loop and the guardrails.

Model access is over any **OpenAI-compatible** `/chat/completions` endpoint (DeepSeek,
Moonshot/Kimi, Zhipu/GLM, or an aggregator like ZenMux/OpenRouter). No vendor SDK.

```bash
pip install -e .          # runtime deps: httpx, mcp
cp .env.example .env      # add a provider key for live runs
python scripts/example_agent.py   # offline demo (no key needed)
```

## Quick start

```python
import asyncio
from barq_sdk import Agent
from barq_sdk.providers import OpenAICompatClient, ModelSpec

model = OpenAICompatClient(ModelSpec(provider="deepseek", model="deepseek-chat"))

async def main():
    async with Agent(model=model, workdir="./run") as agent:
        print(await agent.run("Read README.md and summarise it in three bullets."))
        print(agent.usage)   # {'prompt_tokens': ..., 'total_tokens': ..., 'model_calls': ...}

asyncio.run(main())
```

By default an `Agent` has file tools (read/write/edit/list/find, writes confined to
`workdir`), memory tools (save/recall/forget), a permission engine in `AUTO` mode, a
redacting audit log, and crash-resumable sessions. Everything is overridable.

The engine's control plane -- audit log, session snapshot, tool journal, memory -- lives in
`workdir/.agent-state`, which the file tools can neither read nor write. An agent must not
be able to edit the record of what it did; point `state_dir=` at a separate volume for a
real deployment.

Use it as an async context manager (or call `await agent.aclose()`) so the audit file
handle and pooled HTTP clients are released.

### Rails that are on by default

| Rail | Default | Why |
| :--- | :--- | :--- |
| `tool_timeout` | 120s per call | one unresponsive tool or MCP server cannot hang the agent |
| `max_parallel_tools` | 8 | same-turn tool calls run concurrently; set `1` to force sequencing |
| `validate_tool_input` | on | arguments are checked against the tool's `input_schema` before dispatch, and a violation comes back to the model as a fixable error |
| audit redaction | on | credential headers and secret-shaped values are masked *before* hashing, so the chain still verifies |
| `loop_guard` | on | a degenerate turn (one line repeated, or the same turn three times over) is nudged once, then the run stops rather than burning the budget |
| `compact_at_tokens` | 100k | the transcript is summarised once it passes the threshold, so a long conversation cannot grow until the provider rejects it on length |
| SSRF floor | on | `HttpRequest` refuses reserved/internal addresses in every notation, and drops credentials across a cross-origin redirect, whether or not a `network_policy` is set |
| `token_budget` | off | set it to stop a run at a spend ceiling (`agent.usage` reports the running total) |

Pass `loop_guard=False` or `compact_at_tokens=None` to opt out.

### Watch a run, or stop it

```python
from barq_sdk import EventType

async for event in agent.stream("summarise the repo"):
    if event.type is EventType.TEXT_DELTA:
        print(event.text, end="", flush=True)
    elif event.type is EventType.TOOL_START:
        print(f"
-> {event.tool_name}({event.tool_input})")
    elif event.type is EventType.TOOL_END:
        print(f"   {event.elapsed_ms:.0f}ms")
```

Events also reach an `on_event=` callback (for metrics and spans) without streaming.
`agent.cancel()` stops a run at the next safe boundary -- between turns, or before the next
tool dispatch -- so the session stays resumable instead of being torn down mid-side-effect.

### Delegate to subagents

```python
agent = Agent(model=model, workdir="./run", subagents={"searcher": make_searcher})
```

Any entry adds the `SpawnSubagent` tool. A subagent runs its own loop with its own context;
only its final summary crosses back, so bulk search and crawl output never enters the
parent's window. A subagent can never hold the spawn tool, so delegation cannot recurse.

### Add your own tools

```python
SPEC = {"name": "GetWeather", "description": "current weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}

async def get_weather(inp): return f"{inp['city']}: 21°C clear"

agent = Agent(model=model, workdir="./run", tools=[(SPEC, get_weather)])
```

### Mount MCP servers

```python
from barq_sdk.mcp import StdioMCPConnection

async with StdioMCPConnection("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as fs:
    agent = Agent(model=model, workdir="./run", mcp_servers=[fs])
    await agent.run("List the largest files in this repo.")
```

Tool names are namespaced per server (`fs__list_directory`), sanitised to the character set
providers accept, and truncated to 64 chars, so a server cannot emit a name that gets the
whole run rejected. A server that fails or times out is recorded in `handler.failures`, its
tools are withdrawn from the advertised list, and `await handler.reconnect(server_id)`
brings it back.

### Restrict network access

```python
from barq_sdk import HostAllowlist, Mode

agent = Agent(
    model=model, workdir="./run",
    network_policy=HostAllowlist(allow=("*.example.com",)),  # every other host is denied
    mode=Mode.ASK,                                           # ask a human before each new tool
    enable_http_tool=True,
)
```

## What you get

| Capability | Where |
| :--- | :--- |
| Run events, streaming + cooperative cancellation | `engine/events.py` |
| Tool-argument validation against `input_schema` (incl. `$ref`, `oneOf`, `pattern`, `additionalProperties`) | `engine/validation.py` |
| Secret + PII/PCI redaction before hashing | `engine/audit/redact.py` |
| Tool-use agent loop (native + MCP), untrusted-output fencing | `engine/coordinator.py` |
| High-level facade | `engine/agent.py` (`Agent`) |
| Permissions: hooks → danger → network policy → rules → classifier → mode | `engine/permissions/` |
| File-based memory (`user`/`feedback`/`project`/`reference`) | `engine/memory/` |
| MCP stdio client, namespaced + fault-isolated + per-server tool filters | `engine/mcp/` |
| Crash-resume sessions + exactly-once tool journal | `engine/session.py` |
| Context compaction (summary + verbatim pinned notes) | `engine/compact.py` |
| Degenerate-loop circuit breaker | `engine/loopguard.py` |
| Tamper-evident audit log (SHA-256 chain, optional HMAC/Ed25519) + SIEM export | `engine/audit/` |
| Subagents (isolated context, no recursion) | `engine/subagents/` |
| Filesystem guard (workdir confinement, credential denylist) | `engine/sandbox/filesystem.py` |
| Async rate limiter | `engine/ratelimit.py` |
| OpenAI-compatible provider adapter + role routing | `engine/providers/` |

## Lower-level use

`Agent` is a convenience wrapper. You can wire the pieces directly:

```python
from barq_sdk import Coordinator, PermissionEngine, HookEngine, Mode
from barq_sdk.audit import AuditLog

perms = PermissionEngine(HookEngine(), mode=Mode.AUTO)
coord = Coordinator(model=model, permissions=perms, audit=AuditLog("audit.jsonl"),
                    native_tools={...}, tool_specs=[...])
transcript = await coord.run("...")
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Tests

```bash
python -m pytest        # offline: no API key, no network
```

`tests/test_documented_claims.py` pins every guarantee this README, `ARCHITECTURE.md` and
the module docstrings assert. An audit found six documented guarantees that did not hold
when executed -- each had a working implementation nearby, and nothing that would notice
when the two drifted apart. If one of those tests fails, either the code regressed or the
sentence needs rewriting; both are worth stopping for.

`tests/test_hardening.py` pins the fixes for defects reproduced against the running SDK
(redirect-based policy bypass, fail-open permission gate, unbounded tool calls, audit chain
forking under concurrency, invalid recovery transcripts, and the rest).
`tests/test_security_classifiers.py` covers the read-only classifier and the filesystem
guard - the two modules that decide things without asking a human.

## What this is not

There is no OS-level sandbox: `FilesystemGuard` and `NetworkPolicy` are in-process policy,
not containment. **Reads are not workdir-confined** -- only credential regions and
secret-shaped names are denied, so write confinement is the real boundary. DNS is not
resolved before a policy check, so rebinding is unmitigated. Memory recall is lexical, not
semantic. MCP is stdio-only (`tools/list` and `tools/call`): no remote HTTP/SSE transport,
no OAuth, no resources, prompts, sampling, roots or elicitation. A hash chain cannot detect
truncation of its own tail without an external anchor -- see `AuditLog.anchor()`. These are
deliberate limits, and `SECURITY.md` states which of them are in the threat model.
