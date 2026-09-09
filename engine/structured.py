"""Structured output: make a run return a validated object, not prose.

`Agent.run()` returned a bare string. Every integration that needs a field out of an agent
then re-parses free text, and there is no point at which the engine can say the model got
the shape wrong — so the failure surfaces downstream, in the caller's code, as a KeyError
on data that was never checked.

This module supplies the three pieces the loop needs:

  * `schema_for(output_type)` — a JSON Schema for whatever the caller declared. A plain
    schema dict, a Pydantic model, a dataclass or a TypedDict all work, with **no new
    dependency**: Pydantic is duck-typed (`model_json_schema`), never imported.
  * `extract_json(text)` — the model's answer, recovered from prose. Models wrap JSON in
    ```json fences and preface it with a sentence; refusing to parse that would make the
    feature fail on its most common success case.
  * `build(output_type, data)` — the validated data as the caller's own type, so a
    dataclass or Pydantic model comes back as an instance rather than a dict.

Validation itself reuses `engine.validation.validate_tool_input`, which already implements
the JSON-Schema subset this SDK cares about and is exercised by its own suite. Nothing here
re-implements it.
"""

from __future__ import annotations

import dataclasses
import json
import re
import typing
from typing import Any, Optional

__all__ = [
    "SCHEMA_INSTRUCTION",
    "build",
    "extract_json",
    "repair_prompt",
    "schema_for",
    "type_name",
]

# Appended to the system prompt when an output type is declared. Kept short: the schema is
# also sent in `response_format` where the provider supports it, and a long restatement of
# the same constraint costs tokens on every turn of the run.
SCHEMA_INSTRUCTION = (
    "When you have finished all tool use, your FINAL message must be a single JSON value "
    "matching this schema, and nothing else — no prose before or after, no code fence:\n"
    "{schema}"
)

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}


# Annotations that survive as text. Deliberately only the unambiguous ones: reading
# `list[Foo]` out of a string would mean re-implementing a type parser, and a wrong schema
# rejects valid calls, which is worse than an unchecked field.
_STRING_ANNOTATIONS: dict[str, dict] = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "dict": {"type": "object"},
    "list": {"type": "array"},
    "list[str]": {"type": "array", "items": {"type": "string"}},
    "list[int]": {"type": "array", "items": {"type": "integer"}},
    "list[float]": {"type": "array", "items": {"type": "number"}},
    "optional[str]": {"type": ["string", "null"]},
    "optional[int]": {"type": ["integer", "null"]},
    "str | none": {"type": ["string", "null"]},
    "int | none": {"type": ["integer", "null"]},
}


def type_name(output_type: Any) -> str:
    """A short name for the declared type, used in `response_format` and error messages."""
    for attr in ("__name__", "_name"):
        name = getattr(output_type, attr, None)
        if isinstance(name, str) and name:
            return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:60]
    if isinstance(output_type, dict):
        title = output_type.get("title")
        if isinstance(title, str) and title:
            return re.sub(r"[^A-Za-z0-9_-]", "_", title)[:60]
    return "output"


def _annotation_schema(annotation: Any) -> dict:
    """A JSON Schema fragment for one Python annotation.

    Deliberately shallow. Anything it cannot model becomes an unconstrained `{}`, which
    validates everything — the alternative, guessing a constraint, would reject valid
    output, and a wrong "invalid output" is worse than an unchecked field.
    """
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    # Bare containers: `dict`, `list`, `set` with no parameter. `get_origin` returns None
    # for these, so without this they fell through to the unconstrained `{}` and a `dict`
    # argument was advertised to the model with no type at all.
    if annotation is dict:
        return {"type": "object"}
    if annotation in (list, set, tuple, frozenset):
        return {"type": "array"}
    # A string annotation, which is every annotation in a module using
    # `from __future__ import annotations` when the name cannot be resolved (a locally
    # defined class, a TYPE_CHECKING-only import). Map the ones we can read off the text
    # and leave the rest unconstrained rather than guessing.
    if isinstance(annotation, str):
        return _STRING_ANNOTATIONS.get(annotation.strip().lower(), {})
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (list, set, tuple, frozenset):
        item = _annotation_schema(args[0]) if args else {}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Union or str(origin) == "types.UnionType":
        # Optional[X] is Union[X, None]: model it as the nullable form of X.
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            inner = _annotation_schema(non_none[0])
            if "type" in inner:
                return {**inner, "type": [inner["type"], "null"]}
            return inner
        return {"anyOf": [_annotation_schema(a) for a in args]}
    if dataclasses.is_dataclass(annotation) or _is_typed_dict(annotation):
        return schema_for(annotation)
    return {}


def _is_typed_dict(t: Any) -> bool:
    return (
        isinstance(t, type)
        and issubclass(t, dict)
        and hasattr(t, "__annotations__")
        and hasattr(t, "__total__")
    )


def _schema_from_annotations(annotations: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {k: _annotation_schema(v) for k, v in annotations.items()},
        "required": required,
        "additionalProperties": False,
    }


def schema_for(output_type: Any) -> dict:
    """A JSON Schema for whatever the caller declared.

    Accepts, in order of preference:
      * a JSON Schema `dict`, used verbatim;
      * a Pydantic model — duck-typed via `model_json_schema()` / `schema()`, so Pydantic
        is supported without being imported or depended on;
      * a dataclass;
      * a TypedDict;
      * a bare primitive (`str`, `int`, `list[str]`, …).
    """
    if output_type is None:
        raise ValueError("output_type is None")
    if isinstance(output_type, dict):
        return output_type

    for method in ("model_json_schema", "schema"):
        fn = getattr(output_type, method, None)
        if callable(fn):
            try:
                produced = fn()
            except Exception:  # noqa: BLE001 - a broken model must not kill the run setup
                continue
            if isinstance(produced, dict):
                return produced

    if dataclasses.is_dataclass(output_type):
        hints = typing.get_type_hints(output_type)
        required = [
            f.name
            for f in dataclasses.fields(output_type)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ]
        return _schema_from_annotations(
            {f.name: hints.get(f.name, Any) for f in dataclasses.fields(output_type)},
            required,
        )

    if _is_typed_dict(output_type):
        hints = typing.get_type_hints(output_type)
        required = list(getattr(output_type, "__required_keys__", hints.keys()))
        return _schema_from_annotations(hints, required)

    fragment = _annotation_schema(output_type)
    if fragment:
        return fragment
    raise TypeError(
        f"cannot derive a JSON Schema from {output_type!r}. Pass a JSON Schema dict, a "
        "Pydantic model, a dataclass, a TypedDict, or a primitive type."
    )


def _balanced_span(text: str) -> Optional[str]:
    """The first balanced JSON object or array in `text`, quote- and escape-aware."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def extract_json(text: str) -> tuple[Any, Optional[str]]:
    """Recover the JSON value from a model's final message.

    Returns `(value, error)`. Models routinely wrap the answer in a ```json fence or put a
    sentence in front of it; treating that as a failure would make structured output fail
    on its most common success case, so three strategies are tried in order of strictness:
    the whole message, the contents of a fence, then the first balanced object or array.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "the model returned no final message to parse"

    candidates = [raw]
    for match in _FENCE.finditer(raw):
        body = match.group(1).strip()
        if body:
            candidates.append(body)
    span = _balanced_span(raw)
    if span:
        candidates.append(span)

    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except (json.JSONDecodeError, ValueError):
            continue
    return None, f"the final message was not valid JSON (starts: {raw[:120]!r})"


def build(output_type: Any, data: Any) -> Any:
    """Return `data` as the caller's own type where that is meaningful.

    A caller who declared a dataclass wants an instance, not a dict that happens to have
    the right keys — otherwise every call site does the construction itself and the
    declaration bought nothing. A plain schema dict has no type to build, so the validated
    data is returned unchanged.
    """
    if output_type is None or isinstance(output_type, dict):
        return data

    validate = getattr(output_type, "model_validate", None)  # Pydantic v2
    if callable(validate):
        try:
            return validate(data)
        except Exception:  # noqa: BLE001 - fall back to the plain data
            return data
    parse_obj = getattr(output_type, "parse_obj", None)  # Pydantic v1
    if callable(parse_obj):
        try:
            return parse_obj(data)
        except Exception:  # noqa: BLE001
            return data

    if dataclasses.is_dataclass(output_type) and isinstance(data, dict):
        names = {f.name for f in dataclasses.fields(output_type)}
        try:
            return output_type(**{k: v for k, v in data.items() if k in names})
        except TypeError:
            return data
    return data


def repair_prompt(errors: list[str], schema: dict) -> str:
    """The message sent back when the final answer did not match the schema.

    It names what was wrong rather than restating the schema alone: an error list the model
    can act on converts most failures on the first retry, where "try again" does not.
    """
    listed = "\n".join(f"  - {e}" for e in errors[:20])
    return (
        "SYSTEM: Your final message did not match the required output schema.\n"
        f"{listed}\n\n"
        "Reply with ONLY the corrected JSON value — no prose, no code fence. It must "
        "satisfy this schema:\n"
        f"{json.dumps(schema, indent=2)[:4000]}"
    )
