# Architecture

`Bark-SQK` (packaged as `bark-sqk` / `bark_sqk`, with backward-compatible `engine` support) is a small library for building tool-using AI agents. It is organised as a set of
decoupled packages around one agent loop; a caller injects a model and tools and drives
either the high-level `Agent` or the lower-level `Coordinator`.

## Topology

```mermaid
graph TD
    subgraph Facade
        A[Agent]
    end
    subgraph Core loop
        C[Coordinator]
        S[SessionStore + ToolJournal]
        CP[Compaction]
        LG[LoopGuard]
    end
    subgraph Policy
        PE[PermissionEngine]
        H[HookEngine]
        D[Danger check]
        NP[NetworkPolicy]
    end
    subgraph Tools & extensions
        NT[Native tools]
        MCP[MCP handler]
        MEM[MemoryStore]
        FS[FilesystemGuard]
    end
    subgraph Model & record
        M[OpenAI-compatible client]
        AU[Audit log]
    end

    A --> C
    C --> M
    C --> PE
    C --> AU
    C --> S
    C --> CP
    C --> LG
    C --> NT
    C --> MCP
    PE --> H
    PE --> D
    PE --> NP
    NT --> MEM
    NT --> FS
```

## The agent loop (`engine/coordinator.py`)

A standard tool-use loop over small Protocols (`ModelClient`, tool callables), so it is
unit-testable with a scripted fake model - no API key.

Each turn: call the model → for every `tool_use` block, run it through the permission path,
then dispatch it (native first, then MCP) → feed results back. Key properties:

- **Every tool call is permissioned and audited** before it runs, and **validated against
  its declared `input_schema`**: a violation is returned to the model as a fixable error
  instead of reaching the handler.
- **Bounded**: every tool call runs under `tool_timeout`, so one unresponsive tool or MCP
  server cannot hang the agent. Same-turn calls run **concurrently** (`max_parallel_tools`),
  with results kept in call order.
- **Metered**: token usage from every response accumulates on `Coordinator.usage`, and
  `token_budget` stops the run at a spend ceiling.
- **Untrusted-output fencing**: tool output is wrapped in an "UNTRUSTED - treat as data"
  frame by default (a cheap prompt-injection mitigation); a caller can exempt trusted tools.
- **Turn budget** with a wrap-up nudge near the end and a `max_tokens` recovery nudge, so a
  run converts work into output instead of being cut off silently. Every steering message
  answers any pending tool call first, so the transcript stays valid on the wire (an
  assistant turn with an unanswered `tool_calls` is a hard 400 on OpenAI/Azure).
- **Crash-resume**: the transcript is snapshotted at each turn boundary (`SessionStore`), and
  a `ToolJournal` makes tool execution **exactly-once** on resume (a tool that already ran
  is not re-run). `Coordinator.completed` distinguishes "finished" from "hit max_turns".
- **`send()`** keeps a transcript across messages for multi-turn conversations. `run()` and
  `send()` are thin entry points onto one driver (`_drive`), so a loop fix cannot land in
  one and be missed in the other.
- Optional **`LoopGuard`** (degenerate-repetition circuit breaker) and **`context_compactor`**
  (invoked before each model round-trip) plug in here.

## Permissions (`engine/permissions/`)

`PermissionEngine.check(_async)` resolves each `ToolCall` in this order, fail-closed:

0. **Fail-closed wrapper**: if any step below raises, the decision is **ASK**, never
   allow, and never an exception escaping into the loop. A hook that *crashes* has no
   verdict, so it cannot be read as "no objection".
1. **PreToolUse hook gate**: a registered hook may hard allow/deny. In the async path
   hooks run off the event loop (`asyncio.to_thread`), so a hook that shells out to a
   policy service cannot stall every other coroutine in the process.
2. **Hard-danger DENY**: catastrophic shell commands (`rm -rf /`, fork bomb, `mkfs`, raw
   device writes, `DROP TABLE`, …) are denied *before any rule*, so an allow rule can never
   open a path to them. (`engine/permissions/danger.py`)
3. **Network policy** (optional): every host the call would contact (`network_targets`) is
   checked; a rejected host is denied, or asked if `network_ask` is set. Reserved/internal
   IP literals are denied by default. A destination field is recognised by name *fragment*
   (`webhook_url`, `callbackUri`, `api_endpoint`), not by an exact-match list that any
   unanticipated field name slipped past. The HTTP tool re-checks **every redirect hop**
   against the same policy: a pre-flight check alone is void the moment a permitted host
   returns a 302. (`engine/permissions/network.py`)
4. **Explicit rules**: deny beats ask beats allow (`ToolName` / `ToolName(glob)` syntax).
5. **Soft-danger ASK**: dangerous-but-legit commands (`sudo`, force-push, pipe-to-shell).
6. **Injected classifier** (optional): an async LLM check; on error it ASKs, never allows.
7. **Mode default**: `AUTO` allow / `ASK` ask / `LOCKED` deny. A provably read-only shell
   command can be auto-allowed at this step via the read-only classifier.

Hooks (`engine/hooks/`) fire at exactly three lifecycle points, all wired in the loop:
`PreToolUse`, `PostToolUse`, `PermissionDenied` - there are no declared-but-never-fired
events. `FunctionHook` (in-process callable) and `CommandHook` (shell, exit 2 = block) are
supported.

## Tools & extensions

- **Native tools** are `(spec, async_or_sync_handler)` pairs; the built-ins are file
  read/write, memory save/recall, and an opt-in HTTP tool.
- **MCP** (`engine/mcp/`) mounts any number of stdio servers; tool names are namespaced
  `server__tool`, and one server failing `list_tools` is isolated (recorded in `.failures`)
  rather than blanking every tool.
- **Memory** (`engine/memory/`) is a file-per-fact store with a `MEMORY.md` index and four
  types (`user`/`feedback`/`project`/`reference`). Recall scores **name, description and
  body** through an inverted index refreshed only for changed files, so a query costs
  roughly the number of matching memories rather than a re-read of the whole corpus.
  `namespace=` scopes a store to one tenant/user/session. Recall is deterministic keyword
  overlap; swap in a vector store for semantic recall.
- **FilesystemGuard** (`engine/sandbox/filesystem.py`) confines writes to the workdir and
  makes credential paths (`~/.ssh`, cloud creds, `.env`, `*.pem`, …) unreadable. Relative
  paths resolve against the workdir, not the process CWD.

## Model provider (`engine/providers/`)

One adapter (`OpenAICompatClient`) speaks the OpenAI `/chat/completions` wire format for all
providers; translation functions are pure and unit-tested. It retries transient failures
(429/5xx/transport) with backoff + `Retry-After`, surfaces reasoning-token usage, and takes
an injectable transport (mock in tests). `ModelRouter` maps roles (`FAST`, `SMART`) to
specs so a caller can split cheap vs. strong models.

## Audit (`engine/audit/`)

Records are **redacted before hashing** (credential headers, secret-shaped values), so the
chain covers exactly the bytes on disk and evidence verification still works: the log is
durable and non-repudiable, which is precisely why it must not become a secret store.
Appends take an OS lock on a sidecar file and re-read the chain head under it, so
concurrent Agents or processes cannot fork the chain.

An append-only, fsync'd JSONL log where every entry is **hash-chained** to the previous one,
so any insert/delete/reorder/edit is detectable by `verify()`. Optional HMAC keying makes the
chain unforgeable without the key; optional Ed25519 `seal()` produces third-party-verifiable
checkpoints. `log_exchange()` content-addresses the exact request/response so a later claim
can bind to precise evidence. Records export to ECS/CEF for SIEM ingestion.

## What this is not

There is no OS-level sandbox (network namespaces, bubblewrap) - the FilesystemGuard and
network policy are in-process policy layers, not containment; run the agent in a container
if you need isolation. **Reads are not workdir-confined**: only credential regions and
secret-looking names are denied, so the write confinement is the real boundary.

Memory recall is keyword-based, not semantic. There is **no streaming** (`ModelClient` has a
single `create`) and **no tracing/OpenTelemetry**: usage is counted, but there are no spans.
MCP support is **stdio-only** and covers `list_tools`/`call_tool`; there is no resources,
prompts, or remote-HTTP/OAuth support. These are deliberate simplifications; the seams to
replace them are in the modules above.
