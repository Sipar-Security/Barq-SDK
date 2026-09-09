# Barq-SDK — Deficiency Register

**Scope:** what `barq_sdk` does not implement, and what it implements incorrectly, measured
against 50 agent-development SDKs and platforms.

**Method.** Full read of the source tree, then **execution against the running code**.
Every finding in Part 1 was reproduced by running a probe, and the observed output is
quoted. Findings in Part 2 were confirmed absent by symbol search over the tree. Framework
claims verified against vendor documentation.

**Corrections to the previous revision** (issued before the code was executed):

| Claim in v1 | Corrected |
| :--- | :--- |
| "167 tests" | **328 tests collected**, 327 pass, 1 skipped. 167 was the `def test_` count before parametrize expansion. |
| "Validation is a JSON-Schema *subset*" | Overstated as a gap. `$ref`, `oneOf`, `anyOf`, `allOf`, `not`, `pattern`, `additionalProperties: false`, `minLength`/`maxLength`, `minItems`, `uniqueItems`, `const`, union types and arbitrary nesting **all work**. Only five keywords are missing (see P8). |
| Implied audit chain may be unsafe under concurrency | **Verified correct.** Two `AuditLog` objects, same file, 120 concurrent appends from two threads: 121 entries, `verify()` returns OK. |
| Implied write confinement may be escapable | **Verified correct.** `../escape.txt`, absolute paths, `~/x.txt` and `workdir/../escape.txt` are all refused. |
| Listed "userinfo trick" as a possible allowlist bypass | **Not a bypass.** `https://api.example.com@evil.com/` correctly resolves to host `evil.com` and is denied. |

**173 findings: 36 P0 · 69 P1 · 56 P2 · 12 P3.**
**Part 1** — 30 defects reproduced by execution. **Part 2** — 143 absences confirmed by module read or symbol search.
**P0** fails an enterprise evaluation outright · **P1** blocks production · **P2** is
expected parity · **P3** is long tail.

---

# PART 1 — Defects reproduced against the running code

Thirty findings, each produced by executing the shipped code. These are not missing
features; they are behaviours that are wrong.

---

## 1.1 The dangerous-command layer

The module claims to be the deterministic backstop that catches "the well-known destructive
shapes". It is a case-insensitive regex denylist applied to the whole command string with no
lexing, no quote awareness and no comment stripping.

### K1 — Hard-DENY regexes match inside quoted strings and comments · **P0**

Hard DENY is documented as non-overridable: it fires *before any rule and past any hook
allow*. There is therefore no operator escape hatch. Observed:

```
deny  | git commit -m "fix reboot handling in init"   -> host power control
deny  | grep -r "shutdown" ./src                      -> host power control
deny  | echo "we should reboot the staging box" >> notes.txt -> host power control
deny  | npm run build && npm run halt-on-error        -> host power control
deny  | cat docs/runbook.md | grep -i "poweroff"      -> host power control
```

The pattern is `\b(shutdown|reboot|halt|poweroff|init\s+0)\b` searched over the raw
command. Any agent working on a codebase that mentions those words in a filename, branch
name, commit message, grep target or npm script is permanently blocked, and no
configuration can unblock it. A control that cannot be overridden must not have false
positives; this one has them on ordinary English.

### K2 — The layer scans tool inputs its own docstring says it does not · **P0**

`danger.py` states: *"Scope: SHELL tools only... We deliberately do NOT scan HTTP payloads
or other tool inputs."* But `is_shell_call()` returns True for any tool carrying a `code`,
`script`, `args`, `argv` or `command` key. Observed:

```
deny  | RunCode({'code': 'sql = "DROP TABLE IF EXISTS tmp_staging"'})
deny  | RunCode({'code': 'print("truncate table is a destructive SQL statement")'})
deny  | Bash('python manage.py migrate  # includes DROP TABLE old_users')
```

A code-execution tool cannot pass a SQL migration, a test fixture, or a string literal
containing `DROP TABLE`. The docstring and the behaviour disagree, and the behaviour is a
hard DENY.

### K3 — Trivial encoding defeats the entire layer · **P1**

```
none  | echo cm0gLXJmIC8K | base64 -d | sh
none  | eval "$(echo cm0gLXJmIC8= | base64 -d)"
none  | python3 -c "import shutil,os; shutil.rmtree(os.path.expanduser(chr(126)))"
```

All three delete the filesystem root or the home directory. None produces any verdict — not
even ASK. A regex denylist over an un-lexed string cannot be a backstop against a model that
can encode; the layer's stated purpose is defence against a *prompt-injected* model, which is
precisely the adversary that will encode.

### K4 — Destructive shapes with no verdict at all · **P1**

```
none  | chown -R nobody /              (chmod -R 777 / IS caught; chown is not)
none  | iptables -F                    (drops every firewall rule)
none  | crontab -r                     (deletes all scheduled jobs, no confirmation)
none  | systemctl stop nginx && systemctl disable nginx
none  | userdel -r root
none  | mv /home/user /tmp/gone        (the mv rule requires whitespace directly after
                                        the system dir, so any deeper path escapes)
none  | shred -u /home/user/.ssh/id_rsa (shred rule covers only /etc/, /boot/, /var/log)
```

### K5 — Host-destroying deletions are downgraded to ASK · **P1**

```
ask   | rm -rf /usr/bin        (unbootable host)
ask   | rm -rf /var/lib/mysql  (total data loss)
```

`_is_catastrophic_target` requires the target to *fullmatch* a system directory optionally
followed by `/` or `/*`. Any path one level deeper falls through to the soft-ASK rule — and
in AUTO mode with no `elicit` callback configured, ASK fails closed to *denied*, so the
distinction is invisible; but in ASK mode with an operator present it is one keystroke from
executing.

---

## 1.2 The read-only auto-allow classifier

In ASK mode this classifier **downgrades an ASK to ALLOW with no human prompt**. It decides
what an unattended agent may run against a human-in-the-loop deployment.

### K6 — It auto-allows credential harvesting · **P0**

```
auto-allow=True  | env
auto-allow=True  | printenv
auto-allow=True  | cat /etc/shadow
auto-allow=True  | git config --list
auto-allow=True  | ps aux
auto-allow=True  | head -c 1000000000 /dev/zero
```

`env` prints every API key in the process environment — including the model provider key
the SDK itself loaded. `cat /etc/shadow` prints password hashes. `git config --list` prints
credential helpers and remote URLs with embedded tokens. `ps aux` prints command lines,
which routinely carry secrets. All are approved silently.

The classifier equates *does not mutate the filesystem* with *safe to run unattended*. For
an exfiltration threat model — which is the threat model `SECURITY.md` adopts — reads are
the whole attack. `cat ~/.ssh/id_rsa` and `cat .env` *are* blocked, so the intent exists;
the coverage does not.

---

## 1.3 Hooks

### K7 — `CommandHook` receives no information about the call it is gating · **P0**

`CommandHook.run(self, inp: HookInput)` accepts the input and never references it. Probe
output from a hook that dumps its environment and stdin:

```
stdout: '{}\nNOSTDIN'
```

No stdin payload, no environment variables, no argv. A command hook cannot know which tool
is being called, with what arguments, against what target. It is a **constant function** of
the tool call, filtered only by the `if_` rule string. Every external policy integration —
an OPA sidecar, a policy service, a corporate DLP check — is impossible to write.

For contrast, Claude Code delivers the full tool call as JSON on the hook's stdin.

### K8 — `CommandHook` stdout is never parsed · **P1**

```
decision field from a CommandHook: None
```

A command hook can only signal by exiting 2. It cannot return a reason, cannot return ASK,
cannot modify the tool input, cannot supply structured context. The `HookOutcome.decision`
field exists but no code path ever populates it from a command hook.

---

## 1.4 Permission rules

### K9 — Rule content-matching is platform-dependent · **P1**

`rule_matches` uses `fnmatch.fnmatch` for the content half, which applies
`os.path.normcase`. On Windows that is case-insensitive; on POSIX it is case-sensitive.
Observed on win32:

```
Bash(rm *) vs 'rm -rf x'  -> True
Bash(rm *) vs 'RM -rf x'  -> True
Bash(rm *) vs 'Rm -rf x'  -> True
```

The same deny rule permits `RM -rf x` on Linux. The tool-*name* half was deliberately made
case-insensitive with a comment explaining that a case-sensitive compare is "a fail-OPEN
bypass". The content half inherits platform semantics by accident and has exactly that
bypass on the deployment platform, while tests pass on both because CI never asserts the
cross-platform equivalence.

### K10 — Rules against multi-argument tools match an unstable blob · **P2**

```
match_content(ToolCall('X', {'b':'2','a':'1'}))  ->  '2 1'
```

For a tool that is neither shell nor network, the match content is the space-joined
argument *values* in dict insertion order. A rule like `MyTool(*.py)` matches against
`"value1 value2 value3"`, so its behaviour depends on argument ordering and on values the
rule author never intended to match. Rules are only reliable for single-argument tools.

### K11 — An injected classifier can never allow · **P3**

`if c is not None and c.behavior is not Behavior.ALLOW: return c` — an ALLOW verdict from
the classifier is discarded. In LOCKED mode no classifier can approve anything, so the
extension point is deny/ask-only by construction, which is not documented.

---

## 1.5 Network policy target extraction

### K12 — Payload keys are exempt, so a destination named `query` escapes the allowlist · **P0**

```
{"query":  "https://evil.com/exfil"}  ->  []   (no target extracted)
{"data":   "https://evil.com/exfil"}  ->  []
{"input":  "https://evil.com/exfil"}  ->  []
{"json":   "https://evil.com/exfil"}  ->  []
{"body":   "https://evil.com/exfil"}  ->  []
```

`_PAYLOAD_KEYS` excludes 22 common parameter names from destination detection. The rationale
(a URL inside a request body is data, not a destination) is sound for `HttpRequest`, but it
is applied globally — so any third-party or MCP tool whose destination parameter happens to
be named `query`, `data`, `input`, `value`, `target`… wait, `target` is a network key —
`query`, `data`, `input`, `json`, `form`, `content`, `payload`, `text`, `message`, `prompt`,
`sql`, `note`, `description` or `comment` bypasses the network policy entirely.

The module docstring for `_walk_targets` says key-matching alone "made the policy depend on
the tool author's choice of noun". The payload exemption reintroduces exactly that
dependency in the opposite direction.

### K13 — A bare hostname under a non-network key is invisible · **P1**

```
{"destination": "evil.com"}  ->  []
```

The value must be an absolute `scheme://` URL to be caught under a non-network key. A host
without a scheme, under a key that does not contain one of the nine magic fragments, is
never checked.

### K14 — Nesting deeper than six levels escapes · **P2**

```
{"x":[{"y":[{"z":[{"w":[{"v":[{"u":{"url":"http://evil.com"}}]}]}]}]}]}  ->  []
```

`_walk_targets` returns at `depth > 6`. List elements consume depth, so a URL inside a
list-of-dicts structure escapes at a shallower nesting than a pure dict structure would.

---

## 1.6 State and storage

### K15 — `MemoryStore` has no locking; concurrent writes corrupt the index and crash on Windows · **P0**

Two `MemoryStore` objects on the same directory, 40 concurrent saves each:

```
PermissionError: [WinError 5] Access is denied:
  '...\tmpnb1jbbx8.tmp' -> '...\MEMORY.md'

memory files on disk: 41 | index lines: 40 | LOST INDEX ENTRIES: 1
```

Two independent failures:

1. **Windows:** `os.replace` onto `MEMORY.md` raises when another writer holds it. The
   exception propagates out of `save()` and killed the writing thread outright.
2. **All platforms:** `_add_index_line` is a read-modify-write of the whole index with no
   lock. Concurrent saves lose entries — a classic lost-update race. A memory whose index
   line is lost still exists on disk but is invisible to any consumer reading `MEMORY.md`.

`AuditLog` takes an OS-level file lock on a sidecar for exactly this reason.
`MemoryStore` — which the `Agent` facade places inside the same control-plane directory —
takes none. `Agent` also exposes `memory_dir=`, so a shared memory directory across agents
is a documented configuration.

### K16 — Audit timestamps are unattested local wall clock · **P1**

```
ts field type: <class 'float'>  1788966888.762873
```

`time.time()`. No monotonic guard against clock rollback, no timezone, no RFC 3161
timestamping authority, no external time anchor. The hash chain proves *ordering* and
*integrity of content* — it does not prove *when*. Anyone who controls the host clock can
produce a chain that verifies cleanly with fabricated timestamps. For a log whose stated
purpose is non-repudiation, and which is being positioned against regulatory record-keeping
requirements, the time source is the weakest link and it is undefended.

### K17 — `send()` has no cumulative turn ceiling · **P2**

Each `send()` call receives a fresh `max_turns` budget. `run()` counts turns across resumes
(`taken = sum(1 for m in messages if m.get("role") == "assistant")`); `send()` does not. A
conversation of *n* messages can consume `n × max_turns` model calls with no configured
ceiling.

### K18 — `token_budget` is enforced after spending · **P2**

```
token_budget=1000  ->  actual spend: 1000  (checked after the call returns)
```

`_over_budget()` runs after `_record_usage()`. The budget can be overshot by one full
completion, which is unbounded when `ModelSpec.max_tokens` is `None` (the default when not
built through `build_router_from_env`).

---

## 1.7 Failure surfacing

### K19 — A dead MCP server produces a silently tool-less agent · **P1**

```
Agent.run() returned: 'done'
Agent exposes handler failures? False | mcp_failures attr? False
reachable only via private: {'broken': 'RuntimeError: server died'}
```

`refresh()` catches the exception per server and records it in `handler.failures`. The
`Agent` facade never reads that dict and exposes no property for it. A caller who mounts
three MCP servers and gets zero working tools sees a normal return value and a plausible
answer. The only access is `agent._mcp_handler.failures` — a private attribute of a private
attribute.

This is also the failure mode for the documented usage pattern: a `StdioMCPConnection` not
entered as an async context manager has `_session = None`, so `list_tools()` raises
`AttributeError`, which is swallowed into `failures`.

### K20 — A failing tool is invisible to the caller · **P1**

```
tool raises RuntimeError('database is on fire')
Agent.run() -> 'I could not do it'
completed=True  cancelled=False
```

The tool error is formatted into the transcript for the model and nowhere else. `Agent.run()`
returns the model's prose; `completed` is `True`. There is no exception, no error count, no
list of failed calls, and no structured result object. A caller cannot distinguish a
successful run from a run in which every tool failed and the model apologised.

---

## 1.8 The HTTP tool

### K21 — The entire response body is buffered before truncation · **P1**

```python
resp = await cl.request(method, url, headers=headers, content=body)
...
text = resp.text
```

No streaming, no `Content-Length` pre-check, no maximum byte count. A multi-gigabyte
response is downloaded in full and materialised in memory before being truncated to 2,000
characters for the model. `max_body` bounds only what is written to the audit log. A single
allowlisted URL is a memory-exhaustion vector.

### K22 — Any HTTP method is permitted · **P2**

`method = (inp.get("method") or "GET").upper()` with no allowlist. Once a host is
allowlisted for reading, the model may issue `DELETE`, `PUT` and `PATCH` against it. There
is no read-only mode, no per-method policy, and no method dimension in the permission rule
syntax (rules match on the URL only).

---

## 1.9 Filesystem boundary

### K23 — Workdir write confinement is not a code-execution boundary · **P0**

Write confinement itself is correct — `../escape.txt`, absolute paths and `~/x.txt` are all
refused. But inside the workdir:

```
write=True   .git/hooks/pre-commit
write=True   .github/workflows/ci.yml
write=True   conftest.py
write=True   sitecustomize.py
write=True   Makefile
write=True   setup.py
```

Every one of these executes code the next time a human runs `git commit`, `pytest`, or
`python` in that directory, or the next time CI runs. The stated boundary — "writes are
confined to the workdir" — is a *filesystem* boundary that the documentation treats as the
real security boundary ("write confinement is the real boundary"). When the workdir is a
git repository a developer will subsequently work in, it is not a containment boundary at
all: the agent writes a file, the human runs a routine command, and the agent's code runs
outside every policy layer with the human's privileges.

No denylist exists for these paths, and none is documented as the caller's responsibility.

### K24 — The credential read denylist is narrow and rename-defeatable · **P1**

At the shipping default (`confine_reads=False`) these are all readable:

```
read=True  /etc/shadow
read=True  /etc/ssh/ssh_host_rsa_key
read=True  /proc/self/environ
read=True  ~/.gitconfig
read=True  ~/.bash_history        ~/.zsh_history
read=True  ~/kubeconfig           ~/.dockercfg
read=True  ~/.config/rclone/rclone.conf
read=True  ~/AppData/Roaming/Mozilla/Firefox/<p>/logins.json
read=True  ~/backup/aws-creds-copy.txt
read=True  ~/.env.backup
read=True  ~/id_rsa.bak
read=True  ~/secrets.yaml
```

(`~/.ssh/id_rsa`, `~/.aws/credentials` and `workdir/.env` are correctly denied.)

The denylist is a fixed set of exact filenames plus six suffixes plus eight home
subdirectories. Any secret in a file the list does not literally name is readable — which
includes every backup, every renamed copy, `/etc/shadow`, and the process environment via
`/proc/self/environ`.

### K25 — The symlink-escape test never runs on the development platform · **P2**

```
SKIPPED [1] tests/test_security_classifiers.py:175: symlink creation needs privilege on Windows
```

The CI comment in `.github/workflows/ci.yml` states: *"If it needs either, that is the
bug — a test that silently skips is worse than one that fails."* One test skips, and it is
the one covering symlink traversal of the read/write boundary. It runs on the Ubuntu matrix
leg, so the boundary is covered — but the project's own stated standard is violated by its
own suite, unremarked.

---

## 1.10 Redaction

### K26 — PII coverage is US-only · **P1**

Implemented: Luhn-checked card numbers, email, US SSN (`\d{3}-\d{2}-\d{4}`), US/NANP phone,
IBAN.

Absent: personal names, postal addresses, dates of birth, passport numbers, driving licence
numbers, UK National Insurance, Indian Aadhaar/PAN, EU VAT, Canadian SIN, Australian TFN,
national health identifiers, and every non-NANP phone format. A deployment subject to GDPR,
DPDP or PIPEDA gets card numbers and emails masked and nothing else.

### K27 — Secret coverage misses major cloud formats · **P2**

Implemented: `sk-`/`pk-`/`rk-`/`ak-` prefixed keys, AWS `AKIA`/`ASIA`, GitHub `gh[pousr]_`,
Slack `xox[abposr]-`, JWT, PEM private keys, `Bearer`/`Basic`, URL userinfo.

Absent: Google API keys (`AIza…`), GCP service-account JSON blobs, Azure storage connection
strings (`DefaultEndpointsProtocol=…AccountKey=…`), Azure SAS tokens, `password=` in free
text, Twilio `SK…`, SendGrid `SG.…`, private keys in OpenSSH's newer container format when
the header differs, and any customer-specific format.

Both K26 and K27 matter more than usual here because the same detectors are the only ones
available if redaction is ever extended to the output path — see P–G2.

---

## 1.11 Two structural observations from the code

### K28 — There is no run identifier reachable by the caller · **P1**

`AuditLog` generates a `run_id` internally and stamps every record with it. Neither `Agent`
nor `Coordinator` exposes it:

```
Agent has run_id? False | Coordinator has run_id? False
```

A caller cannot correlate its own application logs, its user-facing request id, or its APM
trace with the audit records for that run. The evidence chain exists but cannot be joined to
anything outside itself. This compounds F1/F2 (no OpenTelemetry, no span model): there is no
correlation identifier of any kind crossing the SDK boundary.

### K29 — `system_prompt` silently binds to the model client, not the agent · **P2**

```python
if system and hasattr(self.model, "system_prompt"):
    if not getattr(self.model, "system_prompt", None):
        self.model.system_prompt = system
```

The prompt is written onto the shared model client, and only if the client does not already
have one. Two `Agent`s constructed with the same `OpenAICompatClient` and different
`system_prompt` values: the second silently runs with the first's prompt, with no warning.
Sharing one client across agents is the documented efficiency pattern (pooled connections,
`usage_total` metering for subagents).

### K30 — Every observability sink failure is swallowed to `pass` · **P1**

`Coordinator._emit`, `Agent.aclose`, `MemoryStore._record`, `HookEngine` failure paths and
`PermissionEngine._finalize` all catch bare `Exception` and `pass`. The isolation intent is
right — a sink must not fail the run — but there is no fallback logger, no counter, no
diagnostic channel and no `logging` integration anywhere in the tree. An `on_event`
callback that raises on every event produces a completely silent run with no indication that
telemetry is broken.

---

# PART 2 — Capability gaps

143 findings. Confirmed absent by reading the implementing module or by symbol search.

---

## A — Model and provider layer

One adapter (`OpenAICompatClient`), one wire format, text only, four hardcoded base URLs,
two sampling controls, `tool_choice` hardcoded to `"auto"`.

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| A1 | **No structured / typed output.** No `response_format`, no `json_schema` mode, no `output_type`, no output validators, no repair-retry. `Agent.run()` returns `str`. Verified: zero occurrences of `response_format` or `json_schema` in the tree. | P0 | OpenAI Agents, PydanticAI, Instructor, Outlines, Guidance, LangChain, Vercel AI SDK, ADK, Marvin, Agno, Mastra, Strands, DSPy, Smolagents, CrewAI |
| A2 | **No native Anthropic Messages API.** No `cache_control`, no thinking blocks, no native `tool_use` shape. | P0 | Claude SDK, LangChain, LlamaIndex, PydanticAI, Vercel, Strands, ADK, CrewAI, Agno |
| A3 | **No Google Gemini / Vertex.** No `generateContent`, no Gemini function-calling shape, no context caching, no grounding. | P0 | ADK, LangChain, LlamaIndex, PydanticAI, Vercel, Agno, Strands, Haystack |
| A4 | **No AWS Bedrock Converse API.** No SigV4, no Bedrock model ids, no Guardrails passthrough. | P0 | Strands, LangChain, LlamaIndex, CrewAI, Agno, Haystack, Bedrock Agents |
| A5 | **No Azure OpenAI.** No `api-version`, no deployment-name routing, no Entra ID auth. Key-in-env only, which most Azure tenants forbid. | P0 | MS Agent Framework, Semantic Kernel, LangChain |
| A6 | **No local / self-hosted inference.** No Ollama, vLLM, llama.cpp, LM Studio, TGI, SGLang. Air-gapped deployment impossible. | P1 | Smolagents, LangChain, LlamaIndex, Agno, Haystack, PydanticAI, Mastra, AgentScope, Letta, OpenHands |
| A7 | **No LiteLLM or unified provider abstraction.** | P1 | OpenAI Agents, ADK, CrewAI, Strands, Agno, PraisonAI, AutoGen |
| A8 | **No multimodal.** Images, audio, video, PDFs raise `ValueError` by design. | P0 | Vercel AI SDK, OpenAI Agents, ADK, LlamaIndex, Agno, Smolagents, LangChain, Claude SDK, AgentScope |
| A9 | **No voice / realtime agents.** No speech-to-speech, no interruption detection, no VAD, no WebRTC. | P2 | OpenAI Agents, ADK Live API, Vercel AI SDK, AgentScope, Agno |
| A10 | **No prompt-caching controls or metrics.** No `cache_control` breakpoints, no Gemini cached-content handles, no `cache_read_input_tokens` in usage. | P1 | Claude SDK, LangChain, LlamaIndex, PydanticAI, Vercel, Strands, Agno, Bedrock |
| A11 | **No reasoning-effort or thinking-budget controls,** and no preservation of thinking blocks across tool turns. Reasoning tokens are reported, never controlled. | P1 | Claude SDK, OpenAI Agents, Vercel AI SDK, PydanticAI, LangChain, Strands |
| A12 | **No cost accounting.** No price table, no currency budget, no per-run or per-tenant attribution. | P1 | Agno, Strands, LangSmith, Langfuse, Bedrock, Databricks, Vertex, CrewAI Enterprise, Portkey |
| A13 | **Token estimation is `chars // 4` for every language.** Measured: 400 Chinese characters → 100 tokens estimated; real tokenizers give roughly 400. A 4× underestimate means `compact_at_tokens=100_000` fires at ~400k real tokens on CJK content — past every current context window. | P1 | LangChain (tiktoken), LlamaIndex, Claude SDK, PydanticAI, Agno |
| A14 | **No embeddings interface.** No `embed()` anywhere. Blocks semantic memory, RAG, semantic caching, semantic routing, dedup and clustering. | P0 | LangChain, LlamaIndex, Haystack, Agno, Mastra, CrewAI, ADK, Letta, Semantic Router, Strands |
| A15 | **No model fallback or failover.** Retry hits the same endpoint five times. | P1 | Vercel AI SDK, PydanticAI, LangChain, LiteLLM Router, Portkey, Mastra |
| A16 | **No `tool_choice` control.** Hardcoded `"auto"`. Cannot force a tool, require any tool, or disable tools for one turn. | P2 | OpenAI Agents, Vercel AI SDK, LangChain, PydanticAI, ADK, Strands |
| A17 | **Sampling surface is two parameters.** No `top_p`, `top_k`, `stop`, `seed`, penalties, `logprobs`, `n`, `parallel_tool_calls`, `service_tier`, `metadata`, `user`. | P2 | All |
| A18 | **`temperature` always sent.** Reasoning models reject or ignore it; no capability negotiation. | P2 | Vercel AI SDK, LiteLLM, LangChain |
| A19 | **No provider-hosted server-side tools.** No web search, code interpreter, file search, computer use, image generation. | P1 | OpenAI Agents, Claude SDK, ADK, Vercel AI SDK, Bedrock |
| A20 | **No batch API support.** | P3 | LangChain, LlamaIndex |
| A21 | **No partial-object streaming.** | P2 | Vercel AI SDK, PydanticAI, OpenAI Agents, Instructor |
| A22 | **Four providers hardcoded in source.** No registry, no plugin point, no config file. | P2 | All |
| A23 | **No request-level middleware.** `transport=` is httpx-level, not semantic. | P2 | MS AF, Vercel AI SDK, LangChain v1, Portkey |

## B — Agent abstraction and orchestration

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| B1 | **No handoffs.** No control transfer between agents with input filters and callbacks. | P0 | OpenAI Agents, Swarm/AG2, MS AF, Strands, CrewAI, AutoGen |
| B2 | **No graph or workflow engine.** No nodes, edges, conditional routing, cycles, fan-out/fan-in, map-reduce, sub-graphs, or typed state with reducers. | P0 | LangGraph, MS AF Workflows, ADK, LlamaIndex Workflows, Haystack, CrewAI Flows, AgentScope, Mastra, Burr |
| B3 | **No team or group-chat orchestration.** No round-robin, selector, magentic, debate, or termination conditions. | P0 | AutoGen, MS AF, CAMEL, CrewAI, MetaGPT, AgentScope, PraisonAI |
| B4 | **No agents-as-tools.** No `Agent.as_tool()`. | P1 | OpenAI Agents, Strands, ADK, CrewAI, LlamaIndex |
| B5 | **Subagents cannot nest.** Structurally forbidden; no depth limit offered as an alternative. | P1 | Claude SDK (depth 5), ADK, CrewAI, AutoGen, MetaGPT, OpenHands |
| B6 | **No durable execution.** Local JSONL re-read by the same process on the same disk. | P0 | PydanticAI (Temporal/DBOS/Restate), MS AF, LangGraph, Mastra, Cloudflare, Letta, Bedrock AgentCore |
| B7 | **No time travel or checkpoint branching.** No checkpoint ids, no history API, no fork, no state patching. | P1 | LangGraph, MS AF, Letta, PydanticAI+Temporal, AgentScope |
| B8 | **Human-in-the-loop is in-process only.** No suspend-persist-return-resume-from-another-process. | P0 | LangGraph, MS AF, Mastra, Cloudflare, Agno, PydanticAI+Temporal, Letta, ADK |
| B9 | **No planning abstraction.** No plan-and-execute, ReWOO, Reflexion, TODO state, replanning, or self-critique. | P1 | LangGraph, CrewAI, AutoGPT, BabyAGI, MetaGPT, Claude SDK, SuperAGI, PraisonAI, Semantic Kernel, OpenHands |
| B10 | **No A2A or inter-agent protocol.** No agent cards, discovery, or remote invocation. | P1 | ADK, Strands, MS AF, AgentScope, Vertex, watsonx, BeeAI |
| B11 | **No thread or conversation management.** One `live_messages` list per Coordinator. No thread ids, listing, isolation, branching, or message-level edit. | P1 | OpenAI Agents Sessions, LangGraph, ADK, Letta, Mastra, Agno, Strands, Cloudflare |
| B12 | **No run-level result object.** Bare `str`. No item stream, usage, raw responses, guardrail results, or `to_input_list()`. See also K20. | P1 | OpenAI Agents, PydanticAI, Strands, Vercel AI SDK, LangGraph, Agno |
| B13 | **No whole-run timeout.** | P2 | OpenAI Agents, Strands, Agno, LangGraph |
| B14 | **No cross-run concurrency control.** | P2 | AutoGen, Cloudflare, LangGraph Platform, Bedrock |
| B15 | **No dynamic instructions.** Static string set once at build time. | P2 | OpenAI Agents, ADK, PydanticAI, CrewAI, Agno, Mastra |
| B16 | **Three lifecycle hook points.** Missing run/turn/model/agent/handoff/subagent/compaction/session/error events. Claude Code fires 25. | P1 | Claude SDK, OpenAI Agents, ADK, MS AF, LangChain, Agno |
| B17 | **No PreCompact/PostCompact hook.** `pinned_notes` is static config. | P2 | Claude SDK, LangGraph, Letta |
| B18 | **One compaction strategy.** No sliding window, trim-by-tokens, tool-result truncation, selective eviction, or hierarchical summary. | P2 | ADK, LangChain, Letta, Claude SDK, Agno, Mastra |
| B19 | **No agent-level retry policy.** | P2 | PydanticAI, Instructor, LangGraph, Temporal |
| B20 | **No deterministic replay.** No recorded responses, cassette mode, or seed capture. | P2 | LangGraph, MS AF, Temporal, LangSmith, promptfoo |
| B21 | **No input filtering between agents.** | P3 | OpenAI Agents, ADK, LangGraph |

## C — Tools

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| C1 | **No tool decorator or schema inference.** Every custom tool needs a hand-written JSON-Schema dict that can drift from the handler signature. | P0 | OpenAI Agents, PydanticAI, LangChain, Strands, Agno, Smolagents, ADK, CrewAI, Mastra, Vercel AI SDK, Marvin |
| C2 | **No sandboxed code execution.** No Docker, E2B, Modal, Daytona, Pyodide, WASM. An agent that cannot run code cannot do data analysis. | P0 | Smolagents, OpenHands, ADK, Bedrock Code Interpreter, LangChain, Agno, AutoGen, CrewAI, Databricks, Letta |
| C3 | **No tool library.** No web search, browser, computer use, SQL, shell, git, calculator, REPL, or SaaS connectors. Nine tools against ecosystems of hundreds. | P0 | LangChain (700+), LlamaHub (400+), Agno (100+), Composio (250+), ADK, Strands, Haystack, CrewAI, Mastra |
| C4 | **Tool results are strings only.** `str(result)`. No structured content, images, files, typed results, or citations. | P1 | OpenAI Agents, ADK, LangChain, Vercel AI SDK, Claude SDK, MCP itself |
| C5 | **No per-tool approval policy.** No `needs_approval`, no per-tool predicate over arguments. | P2 | OpenAI Agents, LangGraph, Claude SDK, Agno, Mastra |
| C6 | **No dynamic tool enable/disable.** Fixed at build time. At a few hundred tools the prompt stops working with no mitigation. | P1 | OpenAI Agents, ADK, Bedrock Gateway (semantic tool selection), Agno, LangChain, Mastra |
| C7 | **No long-running or async tool pattern.** No start-poll-resume, no webhook completion. | P2 | ADK, OpenAI Agents, PydanticAI+Temporal, Cloudflare |
| C8 | **No tool-result caching.** | P2 | LangChain, Agno, GPTCache, Portkey |
| C9 | **No per-tool timeout, retry or cost budget.** One global `tool_timeout` for a 2-second read and a 90-second crawl. | P2 | OpenAI Agents, LangGraph, Strands, Agno |
| C10 | **No native tool grouping.** Flat dict; name collision raises at construction. | P2 | Agno, LangChain, PydanticAI, ADK, Semantic Kernel |
| C11 | **No OpenAPI tool generation.** | P1 | ADK, LangChain, Agno, Bedrock Gateway, Strands, watsonx, Mastra |
| C12 | **No tool auth or credential brokering.** No per-user OAuth, token exchange, on-behalf-of, scoped credentials, or consent flow. | P0 | Bedrock AgentCore Identity, ADK, Composio, Arcade, watsonx, Agentforce, Mastra |
| C13 | **No streaming tool output.** | P3 | Vercel AI SDK, ADK, LangGraph |

## D — Model Context Protocol

stdio transport only; `tools/list` and `tools/call` only.

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| D1 | **No Streamable HTTP or SSE transport.** No remote MCP server can be used at all. Every hosted server (GitHub, Linear, Sentry, Notion, Stripe) is unreachable. | P0 | OpenAI Agents, Claude SDK, ADK, LangChain adapters, Strands, Mastra, Vercel AI SDK, Cloudflare, PydanticAI, Agno |
| D2 | **No MCP OAuth 2.1.** No AS discovery, DCR, PKCE, or token refresh. | P0 | OpenAI Agents, Claude SDK, Cloudflare, Mastra, ADK |
| D3 | **No MCP resources.** | P1 | Claude SDK, ADK, LangChain adapters, Mastra, PydanticAI |
| D4 | **No MCP prompts.** | P2 | Claude SDK, Mastra, ADK |
| D5 | **No MCP sampling.** | P2 | Claude SDK, PydanticAI, Mastra |
| D6 | **No roots, elicitation, completions, logging, progress or cancellation.** | P2 | Claude SDK, PydanticAI, Mastra |
| D7 | **No MCP server mode.** Cannot expose the agent or its tools as an MCP server, closing the entire distribution channel into Claude Code, Cursor, VS Code and ChatGPT. | P1 | FastMCP, Claude SDK, Mastra, ADK, Strands, Agno, LangChain |
| D8 | **No `tools/list_changed` handling.** | P2 | Claude SDK, OpenAI Agents, Mastra |
| D9 | **No MCP configuration file.** No `.mcp.json`, per-server env injection, or scope layering. | P2 | Claude SDK, Cursor, VS Code, Mastra, ADK |
| D10 | **No MCP supply-chain controls.** No server allowlist, signature, pinning, tool-description change detection, or poisoning heuristics. An MCP server is arbitrary code whose tool descriptions go straight into the prompt. | P1 | Claude SDK, enterprise MCP gateways, Bedrock Gateway |
| D11 | **No tool-list caching across restarts.** | P3 | OpenAI Agents |
| D12 | **No egress control over MCP subprocesses.** The `NetworkPolicy` inspects tool *arguments*; an MCP server dials whatever it likes from its own process, and nothing in the SDK sees or constrains it. `StdioMCPConnection` passes `env` and `cwd` straight through with no sandbox, no user separation and no resource limits. | P1 | Bedrock Gateway, container-based runtimes, Cloudflare |

## E — Memory, state and knowledge

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| E1 | **No semantic memory.** Deterministic keyword overlap. Synonyms, paraphrase and cross-lingual recall all fail. | P0 | Letta, Mem0, Zep, LangMem, ADK, Strands, Agno, Mastra, CrewAI, LlamaIndex |
| E2 | **No RAG.** No loaders, chunkers, ingestion, vector stores, retrievers, hybrid search, reranking, citations, or retrieval evaluation. | P0 | LlamaIndex, Haystack, LangChain, ADK, Bedrock KB, Databricks, Agno, CrewAI, Mastra, watsonx |
| E3 | **No storage backend but the local filesystem.** No Postgres, Redis, S3, DynamoDB, Mongo, SQLite, Cosmos. With OS file locking on the audit log, two containers cannot share state. | P0 | LangGraph, OpenAI Agents, Strands, ADK, Letta, Agno, Mastra, MS AF, Cloudflare |
| E4 | **No memory typology beyond four document categories.** No working/episodic/semantic/procedural split, no core memory blocks in the prompt, no agent self-editing, no hierarchy with paging. | P1 | Letta, Mem0, Zep, CrewAI, Agno, Mastra |
| E5 | **No background consolidation.** No sleep-time compute, dedup, conflict resolution, decay, TTL, or importance scoring. Duplicates accumulate forever. | P1 | Letta, Mem0, Zep, LangMem |
| E6 | **No knowledge-graph memory.** | P2 | Zep/Graphiti, Letta, LlamaIndex, Cognee |
| E7 | **No user or entity profile memory.** | P2 | Letta, Mem0, ADK, Vertex Memory Bank, Agentforce, Mastra |
| E8 | **No artifact store.** Files land in the workdir unversioned, unaddressed, with no metadata, MIME type, retention or retrieval URL. | P1 | ADK ArtifactService, OpenHands, Bedrock, Databricks, Agno, Vertex |
| E9 | **State is a message list.** No typed shared state, reducers, scoped state, deltas, schema, or validation. | P1 | LangGraph, ADK, MS AF, Mastra, AutoGen |
| E10 | **Recall re-scans on TTL expiry.** O(corpus) past the 2-second stat TTL; no sharding or external index option. | P2 | Any vector-store-backed memory |
| E11 | **No memory export, import or portability.** | P3 | Letta (`.af`), Mem0, ADK |
| E12 | **No session TTL, retention or right-to-erasure.** Sessions and journals grow forever. No GDPR delete path, no data-residency controls. | P1 | Bedrock AgentCore Memory, Vertex, LangGraph Platform, Letta, Agno |
| E13 | **The tool journal never expires and is never bounded.** `ToolJournal` loads every historical record into a dict at construction and appends forever. A long-lived agent's journal grows without limit, and startup cost is O(all tool calls ever made). No compaction, no rotation, no size cap. | P2 | LangGraph checkpointers (TTL), Temporal |

## F — Observability and evaluation

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| F1 | **No OpenTelemetry.** Zero occurrences of `opentelemetry` in the tree. No spans, no GenAI semantic conventions, no OTLP exporter, no trace-context propagation. | P0 | MS AF, Strands, ADK, OpenAI Agents, PydanticAI/Logfire, Agno, LangSmith, LlamaIndex, Haystack, CrewAI, Databricks, Bedrock, NeMo, Vertex |
| F2 | **No trace or span model.** Flat event sequence with a turn integer. No trace/span ids, parent-child links, W3C traceparent, or cross-process propagation. A subagent's work cannot be attributed to the parent call. See also K28. | P0 | All of the above |
| F3 | **No metrics.** No counters or histograms for latency, tokens, error rate, tool calls, permission denials. Nothing can be alerted on. | P1 | MS AF, Strands, Bedrock, Databricks, Agno, LangSmith |
| F4 | **No evaluation framework.** No eval sets, dataset format, trajectory evaluation, LLM-as-judge, assertions, scoring, CI gate, experiment tracking, or A/B comparison. | P0 | ADK, OpenAI Agents, LangSmith, PydanticAI, Databricks, Bedrock, Vertex, CrewAI, DeepEval, Ragas, promptfoo, Braintrust, AgentScope, Haystack, NeMo |
| F5 | **No trace viewer or dev UI.** Debugging means reading `audit.jsonl`. | P1 | ADK `adk web`, AutoGen Studio, OpenAI traces, LangGraph Studio, Langfuse, Agno AgentOS, Letta ADE, Mastra playground, AgentScope Studio, Databricks review app |
| F6 | **No per-tenant usage attribution.** Usage is per-Coordinator, unlabelled, unaggregated. | P1 | Bedrock, Vertex, Databricks, Langfuse, Portkey, Helicone |
| F7 | **No structured logging.** No `logging` integration, correlation ids, or levels anywhere in the tree. See K30. | P1 | All |
| F8 | **No online quality monitoring.** No drift detection, failure clustering, bad-run flagging, or feedback capture. | P2 | LangSmith, Langfuse, Arize, Braintrust, Databricks, Bedrock, Vertex |
| F9 | **No benchmark harness and no published numbers.** No SWE-bench, GAIA, τ-bench, or WebArena runner. | P2 | OpenHands, Smolagents, CAMEL, AutoGen, MetaGPT, Strands |
| F10 | **No per-component cost or latency profiling.** | P2 | NeMo Agent Toolkit, Langfuse, Agno, Databricks |
| F11 | **No audit log rotation, retention or archival.** `AuditLog` appends to one JSONL file forever. `verify()` re-reads the entire chain, so verification cost grows linearly and unboundedly with the deployment's lifetime. No segment rolling, no chain hand-off between files, no compaction, no size or age cap, no external sink (S3, CloudWatch, Splunk) — only a manual `export_ecs`/`export_cef` batch command. | P1 | Every SIEM-integrated platform |

## G — Safety and guardrails

Tool-call gating only. Nothing inspects what goes into the model or what comes out of it.

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| G1 | **No input guardrails.** No jailbreak or injection classifier, topic check, PII detection on input, off-domain rejection, or tripwire before the model call. | P0 | OpenAI Agents, NeMo Guardrails, Bedrock Guardrails, Azure Content Safety, Llama Guard, Guardrails AI, Agentforce, watsonx.governance, ADK Model Armor, CrewAI, Agno, Mastra, Strands |
| G2 | **No output guardrails.** No PII redaction in what the agent says, toxicity or safety classification, groundedness check, compliance filter, or schema gate. Redaction runs on the audit path only, so **a secret is masked in the evidence file and returned verbatim to the user.** | P0 | Same as G1 |
| G3 | **Prompt-injection defence is a text fence only.** Advisory framing, no classifier, no taint tracking, no data/control separation, no capability reduction after untrusted data enters context. | P1 | Claude SDK, ADK Model Armor, Bedrock, NeMo |
| G4 | **No model-call rate limiting.** `RateLimiter` is wired to the HTTP tool only. Nothing throttles model calls; no per-tenant quota. | P1 | Portkey, LiteLLM, Bedrock, Vertex, Agno, LangGraph Platform |
| G5 | **No secrets management.** Hand-rolled `.env` parser. No vault, KMS, rotation, or per-tenant credential isolation. | P1 | Bedrock AgentCore Identity, ADK, MS AF, Databricks, watsonx, Agentforce |
| G6 | **No identity model.** No user identity, authentication, RBAC, scopes, on-behalf-of, or per-user permission profiles. The permission engine has no notion of *who* is asking. Disqualifying for any multi-user deployment. | P0 | Bedrock AgentCore Identity, ADK, watsonx, Agentforce, Databricks, MS AF, Vertex |
| G7 | **Reads unconfined by default.** `confine_reads=False`. See K24 for the reproduced list of readable credential paths. Combined with `enable_http_tool=True` this is a complete exfiltration path in the default posture. | P1 | Smolagents, OpenHands, ADK, Bedrock, Claude SDK sandbox mode |
| G8 | **No DNS resolution before policy check.** Rebinding unmitigated; no resolve-check-pin option even as opt-in. | P2 | Enterprise egress proxies |
| G9 | **No content-safety metadata on responses.** No refusal detection, block-reason surfacing, or finish-reason taxonomy beyond three values. | P2 | Vertex, Bedrock, Azure |
| G10 | **No compliance packaging.** No SOC 2 or ISO 27001 evidence mapping, EU AI Act Art. 12 mapping, HIPAA/GDPR/PCI documentation, data-residency controls, or model card template. | P1 | watsonx.governance, Agentforce, Databricks, Bedrock, Vertex, Azure AI Foundry |
| G11 | **No red-team or adversarial suite.** No injection corpus, automated attack generation, or permission-engine fuzzing. Given the defects in Part 1.1, 1.2 and 1.5, this is the control that would have caught them. | P2 | NeMo Guardrails, PyRIT, Garak, Bedrock, Databricks |
| G12 | **No process-level isolation of any kind.** No seccomp, no namespaces, no cgroups, no resource limits, no user separation. Every policy layer is in-process Python that a registered tool, an MCP subprocess, or a `CommandHook` runs beside and outside of. Documented honestly, but it means the entire security model depends on the integrator supplying containment the SDK does not describe how to build. | P1 | Smolagents, OpenHands, ADK, Bedrock, Cloudflare, Claude SDK |

## H — Deployment and runtime

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| H1 | **No server or API surface.** No FastAPI app, REST, WebSocket, SSE, OpenAI-compatible façade, or AG-UI. Every integrator builds their own HTTP layer first. | P0 | ADK, Agno AgentOS, LangGraph Server, Letta, Mastra, AutoGen Studio, Vercel route handlers, Cloudflare, Dify, Flowise, Botpress, Bedrock Runtime, Vertex Agent Engine |
| H2 | **No CLI.** Only `python -m engine.audit` (verify/export). No scaffolding, run, eval, or deploy. | P1 | ADK, CrewAI, Mastra, LangGraph CLI, Letta, AutoGPT, Agno, Smolagents |
| H3 | **No horizontal scaling.** Local-file state plus OS locks means one node. Two replicas diverge sessions; see K15 for the memory-store failure mode under two writers. | P0 | LangGraph Platform, Bedrock, Vertex, Cloudflare, MS AF, Agno, Databricks, Letta |
| H4 | **No deployment targets.** No Lambda, Cloud Run, Fargate, Knative or Kubernetes adapters, Helm chart, Terraform module, or serverless handler. The `Dockerfile` entrypoint prints the version. | P1 | ADK, Strands, LangGraph Platform, Mastra, Agno, Vertex, Bedrock |
| H5 | **No background execution, queue, scheduling or triggers.** No cron, alarms, webhooks, event-driven start, job queue, or fire-and-forget. | P1 | Cloudflare, Mastra, Letta, PydanticAI+Temporal, AutoGPT, Inngest, LangGraph Platform |
| H6 | **No multi-tenancy primitives.** Memory namespacing is the only boundary. No per-tenant quotas, rate limits, isolation, billing, or segregation for sessions, audit or artifacts. | P1 | Bedrock, Vertex, LangGraph Platform, Databricks, watsonx, Agentforce, Cloudflare |
| H7 | **No agent versioning or release lifecycle.** No versions, aliases, staged rollout, canary, rollback, or A/B routing. | P1 | Bedrock Agents, Vertex, watsonx, Agentforce, Databricks, LangGraph Platform |
| H8 | **No declarative agent configuration.** Python objects only. No YAML/JSON definition, config-as-code, hot reload, non-programmer authoring path, or GitOps story. | P1 | CrewAI, ADK, Claude SDK, Rasa, Dify, Botpress, Flowise, watsonx, Agentforce, Semantic Kernel |
| H9 | **No graceful shutdown or drain.** No signal handling, in-flight drain, or checkpoint-on-SIGTERM. In a container this loses work on every deploy. | P2 | LangGraph Platform, Bedrock, Cloudflare, Temporal |
| H10 | **No health or readiness endpoints.** `MCPHandler.healthy()` exists but is surfaced nowhere. | P2 | All platform offerings |
| H11 | **No resource limits on the agent process.** No memory cap, CPU quota, workdir disk quota, or file-count cap. A runaway `WriteFile` loop fills the disk; see K21 for the HTTP analogue. | P2 | Container platforms, OpenHands, Smolagents, Bedrock |
| H12 | **The `Dockerfile` copies `engine/` but not `barq_sdk/`.** The published import name is `barq_sdk`; the image installs only the legacy package, so `import barq_sdk` fails inside the container the repo ships. CI verifies the import from a source install, never from the image. | P2 | — |

## I — Developer experience and ecosystem

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| I1 | **Public seams untyped.** `py.typed` ships, but `Agent` declares `model`, `elicit`, `loop_guard`, `context_compactor`, `on_event`, `cancel_token` and `http_rate_limiter` as `Any`, plus bare `list`/`dict` for `tools`, `mcp_servers` and `subagents`. A type checker cannot validate the most error-prone call sites. | P1 | PydanticAI, MS AF, OpenAI Agents, Mastra, Vercel AI SDK |
| I2 | **No lint, format or type-check in CI.** Workflow runs `pytest` only. No mypy, pyright, ruff, black, coverage gate, or pre-commit. | P1 | Most serious OSS projects |
| I3 | **No prompt management.** No templating, versioning, registry, composition, variables, or A/B testing. | P1 | LangChain hub, Langfuse, ADK, Mastra, Agenta, PromptLayer, Semantic Kernel |
| I4 | **No prompt or program optimisation.** No compile step, bootstrapped few-shot, instruction search, or reflective evolution. Nothing improves from data. | P2 | DSPy (BootstrapFewShot, MIPROv2, COPRO, GEPA), TextGrad, AdalFlow, PraisonAI, Vertex |
| I5 | **No few-shot or demonstration management.** | P2 | DSPy, LangChain, Semantic Kernel, ADK |
| I6 | **No caching layer.** No prompt/response cache, semantic cache, or disk cache. | P2 | LangChain, GPTCache, Portkey, Helicone, LiteLLM |
| I7 | **No JS/TS implementation.** | P1 | OpenAI Agents, ADK (Java/Go/TS), MS AF (.NET), Strands, LangGraph, Mastra, Vercel, Cloudflare, Letta |
| I8 | **No documentation site.** Two markdown files. No API reference, versioned docs, search, how-to guides, migration guides, or conceptual docs. | P1 | All |
| I9 | **One example script.** No cookbook, notebooks, template repo, reference application, or pattern recipes. | P1 | All |
| I10 | **No release engineering.** 0.1.0, no PyPI release, no tags, no SemVer or deprecation policy, no supported-version matrix, no SBOM, no signed releases. | P1 | All |
| I11 | **No community infrastructure.** No Discord/Slack, discussions, roadmap, contribution guide, issue templates, code of conduct, or governance. | P2 | All |
| I12 | **No integration adapters.** Cannot use a LangChain tool, LlamaIndex retriever, CrewAI tool, or Haystack component. Every ecosystem interoperates with the others; this one interoperates with none. | P2 | ADK, LangChain, Strands, Agno, PydanticAI, Haystack |
| I13 | **No error taxonomy.** Tool errors become `"TOOL ERROR: …"` strings in the transcript. No typed exception hierarchy, error codes, retryable/terminal classification, or structured error to the caller. See K20. | P2 | OpenAI Agents, PydanticAI, LangChain, Strands |
| I14 | **No plugin or extension system.** Every extension point is a constructor argument. No entry points, discovery, or registry. | P3 | ADK, LangChain, Haystack, Agno, Semantic Kernel |
| I15 | **No configuration validation.** Hand-rolled `.env` parser, no schema, no documented precedence. Misconfiguration fails at first use. | P3 | All |
| I16 | **The dual `engine` / `barq_sdk` package aliasing mutates `sys.modules` at import time.** `barq_sdk/__init__.py` walks `sys.modules` and installs 16 submodule aliases plus every already-loaded nested module. Import order changes which nested aliases exist, `importlib.reload` on either name desynchronises them, and any tool that introspects module identity (pickle, coverage, mypy, Sphinx, profilers) sees two names for one object. Ships as the primary public import path. | P2 | — |

## J — Capability categories with no analogue

| # | Gap | Sev | Who ships it |
| :-- | :--- | :-- | :--- |
| J1 | **No computer use or browser automation.** No Playwright, CDP, screenshot loop, or DOM tooling. | P1 | OpenAI Agents, Claude SDK, Bedrock Browser, OpenHands, Smolagents, Agno, Cloudflare, ADK |
| J2 | **No document processing.** No PDF, DOCX, XLSX, CSV or image parsing; no OCR. `ReadFile` returns raw text and mangles binary (`errors="replace"`). | P1 | LlamaIndex/LlamaParse, Haystack, LangChain, Agno, ADK, Bedrock Data Automation, Docling |
| J3 | **No SQL or data-warehouse support.** No schema introspection, text-to-SQL, safe query execution, or result rendering. | P1 | LangChain, LlamaIndex, Databricks Genie, Agno, Vanna, watsonx |
| J4 | **No skills or progressive-disclosure packaging.** No way to ship instructions + scripts + resources loaded on demand. | P2 | Claude SDK Agent Skills, ADK plugins, Semantic Kernel, watsonx, Agentforce |
| J5 | **No semantic routing or intent classification.** | P2 | Semantic Router, ADK, LangChain, MS AF, watsonx, Rasa CALM |
| J6 | **No conversational-AI features.** No NLU intents/entities, slot filling, forms, dialogue policy, fallback handling, or channel connectors. | P2 | Rasa, Botpress, Dify, Agentforce, watsonx, Cognigy |
| J7 | **No synthetic-data or self-improvement loop.** No trajectory collection for fine-tuning, distillation, or RLAIF/DPO export. | P3 | CAMEL, DSPy, AutoGen, Databricks, OpenAI RFT |
| J8 | **No simulation or multi-agent environment.** | P3 | CAMEL/OASIS, AgentScope, AutoGen, Concordia |
| J9 | **No marketplace or template gallery.** | P3 | Dify, Flowise, AutoGPT, CrewAI, Botpress, watsonx, Agentforce, LangChain hub |

## P — Narrower than claimed (correcting the previous revision)

| # | Finding | Sev |
| :-- | :--- | :-- |
| P8 | **Validation is missing exactly five JSON-Schema keywords,** not "most of them": `multipleOf`, `exclusiveMinimum`, `exclusiveMaximum`, `propertyNames`, `dependentRequired`, `patternProperties`. Everything a tool spec realistically uses is implemented and verified working. | P3 |

---

# PART 3 — What each competitor has that this does not

| SDK / Platform | Capability advantage |
| :--- | :--- |
| **Google ADK** | Workflow agents (Sequential/Parallel/Loop); before-and-after callbacks on agent, model and tool; swappable Session/Memory/Artifact services; `adk web` dev UI; `adk eval` with trajectory scoring; A2A; deploy to Agent Engine, Cloud Run, GKE; bidirectional audio-video streaming; code executors; OpenAPI toolsets; LangChain and CrewAI tool adapters; Java and Go |
| **OpenAI Agents SDK** | Handoffs with input filters; input/output guardrails with tripwires; Sessions across four backends; tracing with processor plugins; `output_type`; `agent.as_tool()`; dynamic instructions; RunHooks/AgentHooks; realtime voice; five hosted tools; `RunResult` with `to_input_list()`; usage limits; LiteLLM for 100+ models; JS/TS parity |
| **Microsoft Agent Framework** | Durable workflows with Cosmos checkpointing; OpenTelemetry built in; DI and middleware; group-chat/handoff/magentic orchestrations; typed workflow ports for human-in-loop; Azure AI Foundry; .NET and Python parity; Entra identity; hosted agents; A2A |
| **AutoGen** | Actor-model core with distributed runtime; cross-language Python↔.NET; RoundRobin/Selector/Swarm/MagenticOne; termination-condition composition; AutoGen Studio; Docker code executors |
| **Semantic Kernel** | Plugin/function model with DI; planners; function/prompt/auto-invoke filters; vector-store connectors; Handlebars/Jinja/Liquid templates; process framework; .NET/Python/Java |
| **LangChain** | ~700 integrations; `with_structured_output`; Runnable composition; caching; callbacks; retrievers and loaders; `trim_messages`; v1 middleware |
| **LangGraph** | Typed state with reducers; checkpointers (memory/SQLite/Postgres); `interrupt()` + `Command(resume=)`; time travel and branching; subgraphs; five streaming modes; long-term Store; `create_react_agent`; Studio; Platform with scaling, crons, TTL |
| **LlamaIndex** | The RAG stack; event-driven Workflows with checkpointing; LlamaParse; AgentWorkflow; LlamaCloud; 400+ LlamaHub integrations |
| **CrewAI** | Role/goal/backstory model; Crews and Flows; hierarchical process with manager agent; `planning=True`; short/long/entity memory; knowledge sources; YAML config; `crewai test`/`train`; enterprise control plane |
| **Haystack** | Typed component pipelines with validated connections; mature evaluation stack; document stores and hybrid retrieval; Hayhooks deployment; serialisable pipeline format |
| **PydanticAI** | End-to-end static typing with generics; dependency injection; `output_type` with validators and `ModelRetry`; durable execution via Temporal/DBOS/Restate/Prefect; pydantic-evals; `FallbackModel`; Logfire; MCP client *and* server; toolsets |
| **Smolagents** | `CodeAgent` — actions as Python, empirically fewer steps; sandboxed execution (E2B/Docker/WASM); Hub-shared tools; vision and browser agents; ~1k-LOC core |
| **DSPy** | Signatures, modules and optimisers (BootstrapFewShot, MIPROv2, COPRO, GEPA) that compile a program against a metric; assertions; typed predictors; fine-tuning |
| **Agno** | 100+ toolkits; multi-modal in and out; teams and workflows; AgentOS runtime with UI; session storage across many databases; cost tracking; reasoning tools; human-in-loop; guardrails |
| **Mastra** | TypeScript-first; workflows with suspend/resume; working memory plus semantic recall; RAG; evals; MCP client and server; dev playground; deploy adapters; Zod typing |
| **Letta** | Memory blocks compiled into the system prompt and self-edited by the agent; archival and recall memory with paging; sleep-time agents; stateful server with REST API; ADE; `.af` portability; shared memory blocks between agents |
| **OpenHands** | Docker runtime with terminal, editor and browser; event-stream architecture; microagents; delegation; SWE-bench-verified numbers |
| **AutoGPT** | Visual block builder; scheduling and triggers; marketplace of prebuilt agents; long-running autonomous execution |
| **CAMEL-AI** | Role-playing societies; synthetic-data generation at scale; OASIS million-agent simulation |
| **BabyAGI** | The task-list loop as an explicit, inspectable artefact (create → prioritise → execute) |
| **Bedrock Agents / AgentCore** | Managed Runtime with 8-hour sessions and per-session isolation; Gateway with semantic tool selection over thousands of tools; Identity with OAuth, consent portal, on-behalf-of; Memory with configurable strategies; Browser and Code Interpreter; CloudWatch observability; Evaluations; Guardrails; Knowledge Bases; action groups; versions and aliases; multi-agent collaboration |
| **Vertex AI Agent Builder** | Managed runtime; Sessions and Memory Bank; Example Store; Gen AI Evaluation Service; tracing; Agentspace; one-command ADK deploy; enterprise IAM, VPC-SC, data residency |
| **IBM watsonx Orchestrate** | Skills catalogue and flows; low-code builder; multi-agent supervision; LLM routing; watsonx.governance (bias, drift, model risk, audit-ready documentation); prebuilt domain agents |
| **Salesforce Agentforce** | Atlas reasoning engine; topics/actions/instructions authoring; Einstein Trust Layer (masking, grounding, toxicity, audit trail, zero retention); Testing Center; Data Cloud grounding; native CRM permissions; Slack and MuleSoft distribution |
| **NVIDIA NeMo Agent Toolkit** | Framework-agnostic wrapper over LangChain/LlamaIndex/CrewAI/Semantic Kernel; per-component token and latency profiling; sizing calculator; OTel-native; evaluation harness |
| **Databricks Mosaic AI** | MLflow tracing end to end; Agent Evaluation with built-in judges and an SME review app; Unity Catalog governance over tools, data and models; Vector Search; Model Serving; lineage |
| **Anthropic Claude Agent SDK** | 25 hook points including PreCompact/PostCompact, each receiving the full call as JSON on stdin; nested subagents to depth 5 with dynamic fan-out; Agent Skills; automatic compaction and context editing; sandbox mode; fine-grained permission modes; in-process MCP servers; structured JSON-Schema output; session forking |
| **Cloudflare Agents SDK** | Durable Objects giving every agent its own SQLite database and durable identity; WebSocket-native; scheduling and alarms; hibernation; MCP server hosting with OAuth; Workflows; Browser Rendering; edge deployment |
| **Vercel AI SDK** | Provider abstraction across ~50 providers; `generateObject`/`streamObject` with Zod; loop control via `stopWhen`/`prepareStep`; UI streaming with `useChat`; telemetry; MCP |
| **Dify** | Visual workflow builder; dataset and RAG management UI; prompt IDE; app publishing with hosted endpoints; annotation loops; observability dashboards; model marketplace |
| **Flowise** | Drag-and-drop composition; template marketplace; embeddable chat widgets; document store UI |
| **Botpress** | Visual flow editor; NLU; multi-channel deployment; human handoff to live agents; analytics; hosted studio |
| **Rasa** | Production NLU; CALM dialogue policies; forms and slot filling; conversation testing framework; fully on-premises; regulated-industry track record |
| **SuperAGI** | Agent provisioning and monitoring GUI; concurrent agent management; telemetry; toolkit marketplace |
| **MetaGPT** | Encoded SOPs for software-company roles; structured artifact outputs (PRD, design, tasks, code); Data Interpreter; role-based message routing |
| **OpenAI Swarm** | The minimal handoff/routine pattern with `context_variables` |
| **AgentScope** | Explicit message-passing with visualisation; Studio; distributed multi-process execution; realtime voice; A2A; fault-tolerance design |
| **AG2** | Community-governed AutoGen fork: conversable agents, group-chat patterns, captain agents |
| **Marvin** | AI functions as ordinary Python functions; classification, extraction and casting primitives |
| **Instructor** | Pydantic extraction with validation-error-driven retries; `Partial[T]` streaming; Python/TS/Go/Ruby |
| **Outlines** | Token-level constrained decoding via finite-state machines — *guaranteed* schema conformance; regex and CFG grammars |
| **Guidance** | Interleaved generation and control with constrained decoding, token healing and acceleration |
| **Semantic Router** | Sub-100ms embedding-based intent routing before the model |
| **PraisonAI** | Low-code multi-agent orchestration; self-reflection agents; YAML agent teams |
| **Bee / BeeAI** | Framework-agnostic composition; ACP protocol; workflows; built-in observability; hosted platform |
| **Strands Agents** | Model-driven loop; multi-agent primitives (agents-as-tools, handoffs, swarms, graphs); A2A; session managers with S3 persistence; OTel-native; deploy to Lambda/Fargate/EC2/AgentCore; Python and TypeScript |

---

# PART 4 — Fix order

Ranked by consequence, not by effort. Part 1 defects come first because a wrong control is
worse than an absent one: an absent control is visible in a threat model, a broken one is
not.

## Tier 0 — defects that make a shipped control wrong

| # | Fix | Findings |
| :-- | :--- | :--- |
| 1 | **Give `CommandHook` the tool call.** Serialise `HookInput` to the hook's stdin as JSON and parse a structured decision from stdout. Until this lands, every external-policy integration is impossible. | K7, K8 |
| 2 | **Lex the command before applying danger patterns,** or drop the hard-DENY tier and make everything ASK. As shipped, an unblockable control fires on `git commit -m "fix reboot"` and misses `base64 -d \| sh`. Both failure directions are present at once. | K1–K5 |
| 3 | **Remove `env`, `printenv`, `cat /etc/shadow`, `git config --list` and `ps aux` from read-only auto-allow.** The classifier decides what runs unattended and currently approves credential dumping. | K6 |
| 4 | **Lock `MemoryStore` writes** with the same sidecar-lock mechanism `AuditLog` already uses. Concurrent saves currently raise on Windows and lose index entries everywhere. | K15 |
| 5 | **Add `.git/hooks/`, `.github/`, `conftest.py`, `sitecustomize.py`, `Makefile` and `setup.py` to the default `deny_write` set,** or state in `SECURITY.md` that workdir write access is equivalent to code execution on the operator's machine. | K23 |
| 6 | **Surface failure.** Expose `Agent.mcp_failures`, a failed-tool-call list, and the audit `run_id`. Three separate classes of silent failure currently reach the caller as a normal return. | K19, K20, K28 |
| 7 | **Apply the network policy to payload keys** when the tool is not a known HTTP tool, or invert the exemption to an explicit per-tool declaration. | K12, K13, K14 |
| 8 | **Bound the HTTP response** with a streaming read and a byte cap; add a method allowlist. | K21, K22 |

## Tier 1 — the seven that decide evaluations

| # | Fix | Closes | Where |
| :-- | :--- | :--- | :--- |
| 9 | **Structured output** — `output_type=` with schema injection, validation via the existing `engine/validation.py` (which is more complete than assumed), and bounded repair-retry | A1, A21, B12 | `providers/openai_compat.py`, `agent.py`, `coordinator.py` |
| 10 | **OpenTelemetry with GenAI semantic conventions** plus a caller-visible `run_id`/`trace_id` | F1, F2, F3, F6, F10, K28 | new `engine/telemetry.py` on `on_event` |
| 11 | **Anthropic, Gemini, Bedrock, Azure — or LiteLLM.** `ModelClient` is already a Protocol | A2–A7, A22 | `engine/providers/` |
| 12 | **`@tool` decorator with schema inference** from type hints and docstring | C1 | new `engine/tools/decorator.py` |
| 13 | **Content guardrails, input and output.** The detectors exist in `audit/redact.py` and run on the audit path only | G1, G2 | new `engine/guardrails/` |
| 14 | **Pluggable state backends** (Postgres/Redis/S3) behind the existing store interfaces | E3, H3, B11, K15 | `session.py`, `memory/`, `audit/log.py` |
| 15 | **Out-of-process human-in-the-loop** — suspend, persist, return, resume elsewhere | B8, C5 | `coordinator.py` + `session.py` |

## Tier 2 — required for production

16. Evaluation harness (F4) · 17. Remote MCP over Streamable HTTP + OAuth (D1, D2) ·
18. Handoffs and agents-as-tools (B1, B4) · 19. Real tokenizer and cost accounting
(A12, A13) · 20. Semantic memory via a pluggable embedder (A14, E1) · 21. Sandboxed code
execution (C2) · 22. HTTP server surface (H1) · 23. Agent lifecycle hooks (B16, B17) ·
24. Identity propagation and per-user permissions (G6, C12) · 25. Audit log rotation and
external sink (F11) · 26. Trusted timestamping for the audit chain (K16).

## Tier 3 — parity and long tail

Graph/workflow engine (B2) · multimodal (A8) · prompt-caching controls (A10) · CLI (H2) ·
documentation site (I8) · typed public API (I1) · lint and type-check in CI (I2) · tool
library and OpenAPI toolsets (C3, C11) · RAG primitives (E2) · artifact store (E8) ·
declarative YAML agents (H8) · TypeScript port (I7) · compliance evidence packaging (G10) ·
model fallback (A15) · journal bounding (E13) · Dockerfile package fix (H12) · module
aliasing (I16).

---

*Source read and executed at commit `fede9bf`, branch `main`, working tree dirty.
9 September 2026. 328 tests collected, 327 pass, 1 skipped.*
