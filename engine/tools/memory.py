"""Memory tools backed by MemoryStore.

Wire the file-based memory into a run: the agent can persist a durable fact (SaveMemory)
and recall relevant ones later (RecallMemory).
"""

from __future__ import annotations

from engine.memory import Memory, MemoryStore, MemoryType

SAVE_MEMORY_SPEC = {
    "name": "SaveMemory",
    "description": (
        "Persist ONE durable fact (user/feedback/project/reference). Do NOT save "
        "re-derivable noise — only context worth carrying to a future run."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
            "body": {"type": "string"},
        },
        "required": ["name", "body"],
    },
}

RECALL_MEMORY_SPEC = {
    "name": "RecallMemory",
    "description": "Recall memories relevant to a query (keyword overlap over name+description).",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    },
}


def _render(mems: list[Memory]) -> str:
    return "\n".join(
        f"- [{m.type.value}] {m.name}: {m.description}\n  {m.body}" for m in mems
    )


def make_save_memory(store: MemoryStore):
    async def save_memory(inp: dict) -> str:
        try:
            mem = Memory(
                name=str(inp["name"]),
                description=str(inp.get("description", "")),
                type=MemoryType(str(inp.get("type", "project"))),
                body=str(inp["body"]),
            )
        except Exception as e:
            return f"REJECTED: {e}"
        store.save(mem)
        return f"saved memory '{mem.name}'"

    return save_memory


def make_recall_memory(store: MemoryStore):
    async def recall_memory(inp: dict) -> str:
        limit = int(inp.get("limit", 5) or 5)
        mems = store.find_relevant(str(inp.get("query", "")), limit=limit)
        return _render(mems) if mems else "no relevant memories"

    return recall_memory
