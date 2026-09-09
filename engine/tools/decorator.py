"""`@tool`: turn a Python function into a tool the model can call.

A custom tool was a hand-written pair — a JSON-Schema dict plus a handler taking a raw
`dict` — with nothing tying the two together:

    SPEC = {"name": "GetWeather", "description": "current weather",
            "input_schema": {"type": "object",
                             "properties": {"city": {"type": "string"}},
                             "required": ["city"]}}

    async def get_weather(inp): return f"{inp['city']}: 21C"

    agent = Agent(model=model, tools=[(SPEC, get_weather)])

Every argument is named twice, its type is declared in a place the type checker cannot see,
and the two halves drift silently: rename the parameter and the schema still advertises the
old name, so the model sends `city`, the handler reads `inp['city']`, and the mismatch
surfaces as a KeyError at run time. Every competing SDK generates the schema from the
signature; this does the same:

    @tool
    async def get_weather(city: str, units: str = "celsius") -> str:
        \"\"\"Current weather for a city.

        Args:
            city: the city to look up
            units: celsius or fahrenheit
        \"\"\"
        return f"{city}: 21 {units}"

    agent = Agent(model=model, tools=[get_weather])

The description comes from the docstring's summary, per-argument descriptions from its
`Args:` (or `:param:`) section, the JSON Schema from the annotations, and `required` from
which parameters have no default. Sync and async functions both work.

Nothing here is magic at call time: the decorator attaches a `__tool_spec__` and a
dict-taking `__tool_handler__`, and `as_tool()` returns the same `(spec, handler)` tuple a
caller would have written by hand — so the low-level API is unchanged and a decorated
function can be passed to `Coordinator(native_tools=…)` directly.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import typing
from typing import Any, Callable, Optional

from engine.structured import _annotation_schema

__all__ = ["as_tool", "is_tool", "tool", "tool_spec_from"]

# Google-style `Args:` / `Arguments:` / `Parameters:` block.
_ARGS_HEADER = re.compile(r"^\s*(?:Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_SECTION_HEADER = re.compile(
    r"^\s*(?:Returns|Yields|Raises|Examples?|Note|Notes|Attributes)\s*:\s*$", re.IGNORECASE
)
_ARG_LINE = re.compile(r"^\s{1,}(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
# Sphinx-style `:param name: description`.
_SPHINX_PARAM = re.compile(r"^\s*:param\s+(?:\S+\s+)?(\w+)\s*:\s*(.*)$")

# A tool's arguments arrive as JSON, so these never appear in a schema.
_SKIP_PARAM_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def parse_docstring(doc: Optional[str]) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {param: description}).

    Google and Sphinx forms are both recognised because both are common and picking one
    would silently drop the other's argument descriptions — which are the only per-argument
    guidance the model ever sees.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()

    summary: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    current: Optional[str] = None

    for line in lines:
        sphinx = _SPHINX_PARAM.match(line)
        if sphinx:
            in_args, current = False, None
            params[sphinx.group(1)] = sphinx.group(2).strip()
            continue
        if _ARGS_HEADER.match(line):
            in_args, current = True, None
            continue
        if _SECTION_HEADER.match(line):
            in_args, current = False, None
            continue
        if in_args:
            match = _ARG_LINE.match(line)
            if match:
                current = match.group(1).lstrip("*")
                params[current] = match.group(2).strip()
            elif current and line.strip():
                params[current] = f"{params[current]} {line.strip()}".strip()
            continue
        if not params:
            summary.append(line)

    # The summary is everything before the first section, trimmed of trailing blank lines.
    while summary and not summary[-1].strip():
        summary.pop()
    return "\n".join(summary).strip(), params


def _camel(name: str) -> str:
    """`get_weather` -> `GetWeather`. Tool names are conventionally CamelCase in prompts,
    and the built-ins (`ReadFile`, `SaveMemory`) already are."""
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


def tool_spec_from(
    fn: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Build a tool spec from a function's signature and docstring."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - an unresolvable forward ref must not break the tool
        hints = getattr(fn, "__annotations__", {}) or {}
    summary, arg_docs = parse_docstring(inspect.getdoc(fn))
    signature = inspect.signature(fn)

    properties: dict[str, dict] = {}
    required: list[str] = []
    for param_name, param in signature.parameters.items():
        if param.kind in _SKIP_PARAM_KINDS or param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, Any)
        schema = _annotation_schema(annotation) if annotation is not Any else {}
        doc = arg_docs.get(param_name)
        if doc:
            schema = {**schema, "description": doc}
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    resolved_description = description or summary
    if not resolved_description:
        raise ValueError(
            f"tool {fn.__name__!r} has no description: give it a docstring, or pass "
            "description= to @tool. The description is the only thing telling the model "
            "when to call it."
        )
    return {
        "name": name or _camel(fn.__name__),
        "description": resolved_description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _make_handler(fn: Callable) -> Callable:
    """Wrap `fn` so it accepts the engine's `dict` input and is always awaitable."""
    signature = inspect.signature(fn)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    known = {
        n for n, p in signature.parameters.items() if p.kind not in _SKIP_PARAM_KINDS
    }
    is_async = asyncio.iscoroutinefunction(fn)

    async def handler(payload: dict) -> Any:
        arguments = dict(payload or {})
        if not accepts_kwargs:
            # The model can emit a field the signature does not have. Dropping it beats a
            # TypeError the model cannot act on — argument validation has already run
            # against the declared schema, so anything left here is genuinely extra.
            arguments = {k: v for k, v in arguments.items() if k in known}
        result = fn(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    handler.__name__ = f"{fn.__name__}_tool"
    handler.__doc__ = fn.__doc__
    handler.__wrapped_tool__ = fn  # type: ignore[attr-defined]
    handler.__is_async_tool__ = is_async  # type: ignore[attr-defined]
    return handler


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """Turn a function into a tool. Usable bare or with arguments.

        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        @tool(name="Sum", description="Add two numbers")
        def add(a: int, b: int) -> int: ...

    The function stays callable exactly as written; the spec and the dict-taking handler
    are attached as attributes, so nothing about its normal use changes.
    """

    def decorate(target: Callable) -> Callable:
        spec = tool_spec_from(target, name=name, description=description)
        target.__tool_spec__ = spec  # type: ignore[attr-defined]
        target.__tool_handler__ = _make_handler(target)  # type: ignore[attr-defined]
        return target

    if fn is not None:
        return decorate(fn)
    return decorate


def is_tool(obj: Any) -> bool:
    """True if `obj` was decorated with `@tool`."""
    return callable(obj) and hasattr(obj, "__tool_spec__")


def as_tool(obj: Any) -> tuple[dict, Callable]:
    """Normalise anything tool-shaped to the `(spec, handler)` pair the engine wants.

    Accepts a `@tool`-decorated function, an explicit `(spec, handler)` tuple, or a bare
    annotated function (which is decorated on the spot). One funnel, so the facade does not
    grow a branch per accepted shape and the low-level `(spec, handler)` API keeps working
    unchanged.
    """
    if is_tool(obj):
        return obj.__tool_spec__, obj.__tool_handler__
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        spec, handler = obj
        if not isinstance(spec, dict) or "name" not in spec:
            raise TypeError(
                "a (spec, handler) tool needs a spec dict with a 'name'; got "
                f"{type(spec).__name__}"
            )
        return spec, handler
    if callable(obj):
        decorated = tool(obj)
        return decorated.__tool_spec__, decorated.__tool_handler__
    raise TypeError(
        f"cannot use {obj!r} as a tool. Pass a @tool-decorated function, a plain "
        "annotated function, or a (spec, handler) tuple."
    )
