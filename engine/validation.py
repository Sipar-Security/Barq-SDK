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

import json
import re
from typing import Any


def _safe_match(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text) is not None
    except re.error:
        return False


# `format` checks for the keywords tool schemas actually use. Deliberately shallow: the
# point is to catch a model handing a handler something structurally wrong, not to be a
# conformant validator.
_FORMATS: dict[str, Any] = {
    "uri": lambda v: "://" in v and " " not in v.strip(),
    "url": lambda v: "://" in v and " " not in v.strip(),
    "email": lambda v: _safe_match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v),
    "uuid": lambda v: _safe_match(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", v),
    "date": lambda v: _safe_match(r"^\d{4}-\d{2}-\d{2}$", v),
    "date-time": lambda v: _safe_match(r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}", v),
    "ipv4": lambda v: _safe_match(r"^(?:\d{1,3}\.){3}\d{1,3}$", v),
}

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


def _resolve(schema: dict, root: dict, seen: frozenset[str] = frozenset()) -> dict:
    """Follow a local `$ref` (`#/$defs/Name`, `#/definitions/Name`) to the schema it names.

    MCP servers routinely ship schemas built from `$ref` + `$defs`. An unresolved `$ref`
    validated as an empty schema, so every constraint behind it silently vanished between
    what the model was shown and what was checked. Remote refs and recursion are left
    unresolved (returned as-is) rather than guessed at.
    """
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return schema  # dangling ref: validate nothing rather than invent a failure
        node = node[part]
    if not isinstance(node, dict):
        return schema
    merged = {k: v for k, v in schema.items() if k != "$ref"}
    return {**node, **merged} if merged else _resolve(node, root, seen | {ref})


def _branch_errors(value: Any, branch: dict, root: dict) -> list[str]:
    sub: list[str] = []
    _check(value, branch, "input", sub, root)
    return sub


def _check_combinators(value: Any, schema: dict, path: str, errors: list[str],
                       root: dict) -> None:
    """oneOf / anyOf / allOf / not. Common in MCP schemas and previously ignored entirely."""
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if all(_branch_errors(value, b, root) for b in any_of if isinstance(b, dict)):
            errors.append(f"{path}: matches none of the {len(any_of)} allowed shapes")
    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = sum(
            1 for b in one_of if isinstance(b, dict) and not _branch_errors(value, b, root)
        )
        if matches != 1:
            errors.append(
                f"{path}: must match exactly one of the {len(one_of)} allowed shapes "
                f"(matched {matches})"
            )
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for branch in all_of:
            if isinstance(branch, dict):
                _check(value, branch, path, errors, root)
    negated = schema.get("not")
    if isinstance(negated, dict) and not _branch_errors(value, negated, root):
        errors.append(f"{path}: matches a forbidden shape")


def _check(value: Any, schema: dict, path: str, errors: list[str],
           root: dict | None = None) -> None:
    if not isinstance(schema, dict):
        return
    if root is None:
        root = schema
    schema = _resolve(schema, root)

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must be {schema['const']!r}")

    _check_combinators(value, schema, path, errors, root)

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

    if isinstance(value, str):
        lo, hi = schema.get("minLength"), schema.get("maxLength")
        if isinstance(lo, int) and len(value) < lo:
            errors.append(f"{path}: is {len(value)} characters, shorter than the minimum {lo}")
        if isinstance(hi, int) and len(value) > hi:
            errors.append(f"{path}: is {len(value)} characters, longer than the maximum {hi}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    errors.append(f"{path}: does not match the required pattern {pattern!r}")
            except re.error:
                pass  # an unusable pattern is the schema author's bug, not the model's
        fmt = schema.get("format")
        checker = _FORMATS.get(fmt) if isinstance(fmt, str) else None
        if checker is not None and not checker(value):
            errors.append(f"{path}: is not a valid {fmt}")

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
                    _check(value[key], sub, f"{path}.{key}" if path != "input" else key,
                           errors, root)
        # `additionalProperties: false` was ignored, so a model inventing an argument got
        # no correction — it just silently reached the handler.
        extra_schema = schema.get("additionalProperties")
        pattern_props = schema.get("patternProperties")
        if extra_schema is False or isinstance(extra_schema, dict):
            known = set(props) if isinstance(props, dict) else set()
            for key in value:
                if key in known:
                    continue
                if isinstance(pattern_props, dict) and any(
                    _safe_match(p, str(key)) for p in pattern_props
                ):
                    continue
                if extra_schema is False:
                    errors.append(f"{path}: unexpected field {key!r} is not allowed")
                else:
                    _check(value[key], extra_schema,
                           f"{path}.{key}" if path != "input" else str(key), errors, root)

    if isinstance(value, (list, tuple)):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                _check(item, items, f"{path}[{i}]", errors, root)
        lo, hi = schema.get("minItems"), schema.get("maxItems")
        if isinstance(lo, int) and len(value) < lo:
            errors.append(f"{path}: has {len(value)} items, fewer than the minimum {lo}")
        if isinstance(hi, int) and len(value) > hi:
            errors.append(f"{path}: has {len(value)} items, more than the maximum {hi}")
        if schema.get("uniqueItems") is True:
            try:
                if len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
                    errors.append(f"{path}: items must be unique")
            except TypeError:
                pass


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
