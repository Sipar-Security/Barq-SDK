# Security policy

## Reporting a vulnerability

Report privately through GitHub's **Report a vulnerability** (Security → Advisories) rather
than opening a public issue. Include a reproduction, the version, and what an attacker
gains. Expect an acknowledgement within three working days.

## What is in the threat model

The engine assumes **the model is untrusted**. Tool output is attacker-influenceable (a
fetched page, an MCP result, a file the agent read), the model may be steered by it, and
the guardrails exist to bound what a steered model can do. Reports in that shape are in
scope:

- a tool call that reaches a resource the permission engine should have denied;
- a way to make the audit log unverifiable, or to change it without `verify()` failing;
- a path out of `workdir` for a write, or into a credential region for a read;
- a network destination the configured `NetworkPolicy` should have caught;
- content that persists into memory and comes back to a later run unfenced.

## What is NOT in the threat model

These are documented limits, not vulnerabilities. Design around them:

- **There is no OS-level sandbox.** `FilesystemGuard` and `NetworkPolicy` are in-process
  policy, not containment. A tool the caller registers can do anything the process can.
  Run the agent in a container or a jail if you need isolation.
- **Reads are not confined to the workdir.** Only credential regions and secret-shaped
  names are denied. Write confinement is the real boundary.
- **DNS is not resolved before a policy check**, so DNS rebinding is unmitigated. Enforce
  egress at the transport (an outbound proxy, a network namespace) if that matters.
- **A hash chain cannot detect tail truncation from inside the file.** Capture
  `AuditLog.anchor()` somewhere the agent cannot reach and pass it back to
  `verify_audit_file(expected_head=..., expected_count=...)`.
- **Redaction is pattern-based**, so a novel secret format can pass through. Treat the
  audit log as sensitive at rest regardless.
- Anything requiring the operator's own credentials, host access, or a malicious tool
  supplied by the caller.
