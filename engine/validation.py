"""Tool-argument validation against a tool spec's declared `input_schema`.

A tool spec advertises an `input_schema` to the model; without a check on the way back in,
whatever the model emits reaches the handler unchanged: a missing required field, a string
where an integer was declared, an unknown key. Every tool author then has to re-implement
the same validation, and the ones who forget get a stack trace instead of a usable error.

This validates the JSON-Schema subset that tool specs actually use (type, required,
properties, enum, items, minimum/maximum, and nested objects/arrays), with no dependency.
It is deliberately NOT a full JSON-Schema implementation: anything it does not understand
is passed through rather than rejected, so an exotic schema can never block a legitimate
call. Failures come back as a list of human-readable messages that the coordinator returns
to the model as a tool error, which the model can act on and retry.
"""

from __future__ import annotations

from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
    "null": (type(None),),
}


def _type_ok(value: Any, declared: str) -> bool:
    expected = _TYPES.get(declared)
    if expected is None:
        return True  # unknown type keyword: don't invent a failure
    # bool is a subclass of int in Python; an integer field must not accept True.
    if declared in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _describe(value: Any) -> str:
    for name, types in _TYPES.items():
        if name in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, types):
            return name
    return type(value).__name__


def _check(value: Any, schema: dict, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        return

    declared = schema.get("type")
    if isinstance(declared, str) and not _type_ok(value, declared):
        errors.append(f"{path}: expected {declared}, got {_describe(value)}")
        return  # a wrong type makes every deeper check meaningless
    if isinstance(declared, list):  # union type, e.g. ["string", "null"]
        if not any(_type_ok(value, d) for d in declared if isinstance(d, str)):
            errors.append(f"{path}: expected one of {declared}, got {_describe(value)}")
            return

    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        errors.append(f"{path}: {value!r} is not one of {enum}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if isinstance(lo, (int, float)) and value < lo:
            errors.append(f"{path}: {value} is below the minimum of {lo}")
        if isinstance(hi, (int, float)) and value > hi:
            errors.append(f"{path}: {value} is above the maximum of {hi}")

    if isinstance(value, dict):
        props = schema.get("properties")
        required = schema.get("required")
        if isinstance(required, list):
            for field in required:
                if isinstance(field, str) and field not in value:
                    errors.append(f"{path}: missing required field {field!r}")
        if isinstance(props, dict):
            for key, sub in props.items():
                if key in value and isinstance(sub, dict):
                    _check(value[key], sub, f"{path}.{key}" if path != "input" else key, errors)

    if isinstance(value, (list, tuple)):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                _check(item, items, f"{path}[{i}]", errors)


def validate_tool_input(value: Any, schema: dict | None) -> list[str]:
    """Return a list of validation errors ([] when the input is acceptable).

    An absent or non-dict schema validates everything; a tool that declares no schema
    has, by definition, no contract to break.
    """
    if not isinstance(schema, dict) or not schema:
        return []
    errors: list[str] = []
    _check(value, schema, "input", errors)
    return errors
