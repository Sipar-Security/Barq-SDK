"""Context compaction: Claude Code's autoCompact, ported.

Recovered helper: when a conversation grows too long, summarize the whole transcript into
one dense, structured summary and continue from it, so the context window never overflows.

Contract:
  * estimate_tokens(messages) -> int              rough token estimate of a message list
  * async compact_transcript(client, messages)    -> (new_messages, summary_text)
        calls the model with a SYSTEM summarize prompt + the rendered transcript, and
        returns a replacement transcript that begins with the summary as a user message.
The summarizer uses its OWN system prompt (passed here), not the agent's, so the summary
is neutral. Only turn-level content is summarized; the audit log remains the source of truth.
"""

from __future__ import annotations

from typing import Any

# Must contain the word "summariz…" - callers/tests detect the summarizer turn by it.
SUMMARY_SYSTEM = (
    "You are a compaction assistant. Summarize the conversation below into a dense, "
    "structured summary that preserves everything needed to CONTINUE the work without the "
    "original transcript. Include, as applicable: the user's goal and constraints; key "
    "decisions and facts established; files/URLs/endpoints touched and their state; "
    "commands run and their results; and the exact next step in progress. Be "
    "specific (names, numbers, paths) and omit chit-chat. Output only the summary."
)


def estimate_tokens(messages: list[dict] | None) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token) over a message list."""
    if not messages:
        return 0
    chars = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            chars += sum(len(str(b.get("text", b))) for b in c)
        else:
            chars += len(str(c))
    return chars // 4


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for b in content:
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    parts.append(f"[tool_use {b.get('name','')}({b.get('input','')})]")
                elif b.get("type") == "tool_result":
                    parts.append(f"[tool_result {str(b.get('content',''))[:400]}]")
            content = "\n".join(parts)
        lines.append(f"{role.upper()}: {content}")
    return "\n\n".join(lines)


def _pinned_block(pinned: list[str] | None) -> str | None:
    """Render pinned notes as a verbatim, clearly-labelled block (deterministic: never
    routed through the summarizer, so verified facts/decisions cannot be summarized away)."""
    items = [str(p).strip() for p in (pinned or []) if str(p).strip()]
    if not items:
        return None
    body = "\n".join(f"- {it}" for it in items)
    return (
        "PINNED NOTES CARRIED ACROSS COMPACTION (verified: do not drop or contradict; "
        "these survive verbatim):\n" + body
    )


async def compact_transcript(
    client: Any, messages: list[dict], pinned: list[str] | None = None,
) -> tuple[list[dict], str]:
    """Summarize `messages` via `client` and return (new_messages, summary_text).

    `client` is a bare ModelClient with `async create(messages, tools)` returning a
    ModelResponse whose `.content` is a list of blocks. The returned transcript starts
    fresh with the summary carried as a single user message.

    `pinned`, if given, is a list of durable notes (verified facts, key decisions) that are
    re-injected VERBATIM as a separate leading message (deterministically, not via the model)
    so they survive compaction even if the LLM summary omits or garbles them. This is what
    stops a run from losing already-verified work when the window fills.
    """
    transcript = _render(messages or [])
    resp = await client.create(
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        tools=[],
    )
    summary = "".join(
        b.get("text", "") for b in getattr(resp, "content", []) if b.get("type") == "text"
    ).strip()
    new_messages: list[dict] = []
    block = _pinned_block(pinned)
    if block is not None:
        new_messages.append({"role": "user", "content": block})
    new_messages.append({"role": "user", "content": summary})
    return new_messages, summary
