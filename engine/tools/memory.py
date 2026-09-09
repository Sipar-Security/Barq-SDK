"""Memory tools backed by MemoryStore.

Wire the file-based memory into a run: the agent can persist a durable fact (SaveMemory),
recall relevant ones later (RecallMemory), and retract one that turns out to be wrong
(ForgetMemory).

Trust boundary
--------------
Memory content is authored by the MODEL, and the model's own input is untrusted tool
output. So a memory is not engine-controlled data: recall output is fenced as untrusted
like any other tool result (see `Agent._build`), and saving never silently destroys an
existing fact — `SaveMemory` refuses to clobber unless the caller explicitly asks. Without
that, a single injected instruction could overwrite a true fact with a false one and leave
no trace.
"""

from __future__ import annotations

from engine.memory import Memory, MemoryStore, MemoryType

SAVE_MEMORY_SPEC = {
    "name": "SaveMemory",
    "description": (
        "Persist ONE durable fact (user/feedback/project/reference). Do NOT save "
        "re-derivable noise — only context worth carrying to a future run. Saving under a "
        "name that already exists is refused unless you set replace=true, so read the "
        "existing memory first with RecallMemory before replacing it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
            "body": {"type": "string"},
            "replace": {
                "type": "boolean",
                "description": "overwrite an existing memory of the same name (default false)",
            },
        },
        "required": ["name", "body"],
    },
}

RECALL_MEMORY_SPEC = {
    "name": "RecallMemory",
    "description": (
        "Recall memories relevant to a query (keyword overlap over name, description and "
        "body)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    },
}

FORGET_MEMORY_SPEC = {
    "name": "ForgetMemory",
    "description": (
        "Delete a stored memory by name. Use this when a fact turns out to be wrong — "
        "retracting it is better than leaving a false fact to be recalled again."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
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
        # Non-destructive by default: a colliding name lands beside the existing memory
        # rather than replacing it, and the model is told which file it actually got.
        path = await store.asave(mem, overwrite=bool(inp.get("replace")))
        stored = path.stem
        if inp.get("replace"):
            return f"saved memory '{stored}' (replaced any previous memory of that name)"
        return f"saved memory '{stored}'"

    return save_memory


def make_recall_memory(store: MemoryStore):
    async def recall_memory(inp: dict) -> str:
        limit = int(inp.get("limit", 5) or 5)
        mems = await store.afind_relevant(str(inp.get("query", "")), limit=limit)
        return _render(mems) if mems else "no relevant memories"

    return recall_memory


def make_forget_memory(store: MemoryStore):
    async def forget_memory(inp: dict) -> str:
        name = str(inp.get("name", "")).strip()
        if not name:
            return "REJECTED: 'name' is required"
        existed = await store.adelete(name)
        return f"deleted memory '{name}'" if existed else f"no memory named '{name}'"

    return forget_memory
