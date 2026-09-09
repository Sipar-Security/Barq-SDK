# Changelog

## Unreleased

A remediation pass over an engineering audit that reproduced every finding against the
running SDK. Grouped by what breaks if you upgrade blind.

### Breaking

- **The package is imported as `barq_sdk`.** The directory was `barq-sdk/` (a hyphen is not
  a legal module name), `pyproject.toml` declared `bark_sqk*`, and the README documented a
  third spelling — so the package could not be imported under any of them, and
  `tests/test_bark_sqk.py` failed at *collection*, which aborted the whole suite. `engine`
  still works and resolves to the same objects.
- **The control plane moved to `workdir/.agent-state/`** (`audit.jsonl`, `session.jsonl`,
  `tooljournal.jsonl`, `memory/`), and the file tools can neither read nor write it. The
  agent could previously destroy its own audit log with `WriteFile("audit.jsonl", "[]")`.
  Pass `state_dir=` to relocate it.
- **`SaveMemory` no longer overwrites by default.** A colliding name lands beside the
  existing memory unless the call sets `replace: true`. A single injected instruction could
  otherwise silently replace a true fact with a false one, leaving no trace.
- **A `PreToolUse` hook returning `allow()` no longer bypasses hard-danger and the network
  policy.** It still suppresses rules, soft-danger, the classifier and the mode default.
- **The memory namespace directory now includes a digest of the exact namespace string**, so
  `TENANT-A`, `tenant_a` and `tenant a` are distinct stores. Existing namespaced memories
  live under the old slug-only directory; move them or re-save them.
- **`to_openai_messages` raises on an image/document/audio block** instead of stringifying
  the dict into a user message, which sent a vision model the Python repr of a dict and
  raised nothing.
- **`run_sync()` raises a named error inside a running event loop** rather than surfacing
  asyncio's message, which explained neither the cause nor the fix.

### Security

- `HttpRequest` checks the **first** request, not only redirect hops. With no
  `network_policy` configured — the `Agent` default — a stock agent could fetch
  `http://169.254.169.254/` and read cloud credentials.
- Credentials (`Authorization`, `Cookie`, …) are stripped across a **cross-origin
  redirect**; same-origin hops keep them. Non-credential headers always survive.
- Reserved/internal addresses are recognised in every notation a resolver accepts —
  decimal (`2130706433`), hex (`0x7f000001`), octal (`0177.0.0.1`), short-form (`127.1`),
  v4-mapped v6 — plus the hostnames that name one (`localhost`,
  `metadata.google.internal`, `*.internal`). Non-HTTP schemes are refused.
- **Memory frontmatter cannot be forged.** Model-supplied fields are flattened to one line,
  so a newline can no longer close the block, set `type`, and replace the body — which
  persisted an attacker-authored "fact" into a later run.
- **`RecallMemory` output is fenced as untrusted.** It was exempted as "engine-controlled
  output", but memory content is model-authored and the model's input is untrusted tool
  output, so a poisoned fact came back to a clean run wearing the engine's own authority.
- Dangerous commands are caught **under any tool name** (`RunCommand`, `exec`, `Terminal`,
  or anything with a command-shaped argument). The detector keyed on the literal names
  `bash`/`shell`/`powershell`, and this SDK ships no shell tool, so the whole layer
  disengaged unless the integrator happened to pick one of them.
- New danger coverage: `find / -delete`, interpreter one-liners deleting a system path,
  `> /etc/passwd`, `mv /etc`, `shred`, credential-file exfiltration, `kubectl delete`,
  `terraform destroy`, cloud resource deletion, history clearing, package installs.
- **Network destinations are found at any depth and under any field name** — the policy
  previously depended on the tool author naming their field with one of nine magic
  substrings, so `destination`, `to`, `where`, `cmd` and anything nested were invisible.
  Bash `/dev/tcp/host/port` sockets are recognised. Payload fields are still excluded.
- A `PreToolUse` hook that **raises** escalates to ASK even when another hook allowed.
- Audit redaction covers connection-string passwords, `password=`/`token=`-style fields,
  payment cards (Luhn-validated), emails, SSNs, phone numbers and IBANs. Opt out with
  `default_redactor(..., mask_pii=False)`.

### Fixed

- **`verify_audit_file` no longer raises** on a line that is valid JSON but not an object —
  exactly what an agent overwriting the log with `[]` leaves behind. Tampering now always
  comes back as a report.
- **Tail truncation is addressable.** `AuditLog.anchor()` returns `{chain_id, head, count}`
  for out-of-band storage; `verify_audit_file(expected_head=…, expected_count=…)` uses it.
  `VerifyReport.unsealed_tail` reports the exposure when no anchor is available. The claim
  that the chain detected truncation was false and has been corrected.
- **Audit records carry the tool arguments**, a `tool_use_id` and a `run_id` (bounded, with
  a sha256 for truncated values). The log previously proved it was unaltered while saying
  nothing about what happened.
- **`MCPHandler.reconnect()` applies the same collision handling as `refresh()`.** Two tool
  names that sanitise alike collapsed onto one entry after a reconnect, silently
  dispatching a name the model already knew to a *different* tool. Stale tools a server no
  longer publishes are withdrawn.
- Schema validation honours `$ref`, `oneOf`/`anyOf`/`allOf`/`not`, `const`, `pattern`,
  `format`, `minLength`/`maxLength`, `minItems`/`maxItems`, `uniqueItems` and
  `additionalProperties`. MCP servers routinely ship schemas using these, and every one of
  those constraints previously vanished between the spec shown to the model and the check
  performed before dispatch.
- Permission rules accept a glob in the **tool-name** half (`mcp__github__*`, `*`), which
  previously raised `ValueError`.

### Performance

- **File tools and memory reads run off the event loop.** A `FindFiles` over 1,800 files
  froze it for 1,227 ms — stalling every other agent in the process and making
  `max_parallel_tools` meaningless for the built-ins. Worst measured stall is now 65 ms.
- **Memory recall: 667 ms → 5 ms** on a 1,600-memory store. One `scandir` pass instead of a
  glob plus a stat per file, and the postings index is rebuilt only when something changed.
- **Audit appends: ~48 → ~886 entries/sec** with fsync unchanged (~2,000 with
  `durability="flush"`). The chain head is re-read from disk only when the file grew since
  this writer's own last append.

### Added

- **Run events, streaming and cancellation** (`engine/events.py`). `Agent.stream()` /
  `Coordinator.astream()` yield typed `AgentEvent`s; `on_event=` receives the same stream
  for metrics and spans; `CancelToken` stops a run at the next safe boundary rather than
  tearing it down mid side-effect. `OpenAICompatClient.stream()` parses provider SSE into
  text deltas and reassembles the same `ModelResponse` the blocking path returns.
- **Subagents are wired into the facade**: `Agent(subagents={"searcher": factory})` adds the
  `SpawnSubagent` tool, capped by `max_spawns`.
- **The loop guard and context compaction are on by default** (`loop_guard=False` /
  `compact_at_tokens=None` to opt out). Both existed but were opt-in, so a caller following
  the quick start had a transcript that grew until the provider rejected it on length.
- `ForgetMemory` tool, so a fact that turns out to be wrong can be retracted.
- `MCPHandler.set_tool_filter()` for per-server allow/deny lists, and `aclose()`;
  `Agent.aclose()` now shuts down the MCP servers it started.
- `AuditLog(seal_every=…)` writes periodic checkpoints; `durability=` trades throughput
  against crash exposure.
- `tests/test_documented_claims.py` pins every guarantee the README, `ARCHITECTURE.md` and
  the module docstrings assert. Six were false when the audit ran.
- Packaging: `LICENSE` (MIT), `py.typed`, `SECURITY.md`, GitHub Actions CI across
  Linux/Windows and Python 3.12/3.13, and upper bounds on dependencies.
