"""`@tool`: a function becomes a tool, schema and all.

A custom tool was a hand-written JSON-Schema dict plus a handler taking a raw `dict`, with
nothing tying the two together. Every argument was named twice, its type was declared where
no type checker could see it, and the halves drifted silently: rename a parameter and the
spec still advertised the old name, so the model sent one key and the handler read another
— surfacing as a KeyError at run time, inside the tool, on the model's turn.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Optional

import pytest

from engine import Agent, ModelResponse, as_tool, is_tool, tool, tool_spec_from
from engine.tools.decorator import parse_docstring


@dataclasses.dataclass
class Point:
    """A module-level dataclass, so its name resolves from the module namespace."""

    x: int
    y: int


class Script:
    def __init__(self, responses):
        self.responses = list(responses)

    async def create(self, messages, tools):
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse([{"type": "text", "text": "done"}], "end_turn", {})


def call(name="GetWeather", inp=None, call_id="1"):
    return ModelResponse(
        [{"type": "tool_use", "id": call_id, "name": name, "input": inp or {}}],
        "tool_use", {},
    )


# --- schema inference ---------------------------------------------------------
def test_schema_is_derived_from_the_signature():
    @tool
    def get_weather(city: str, days: int = 1) -> str:
        """Current weather for a city."""
        return ""

    spec = get_weather.__tool_spec__
    assert spec["name"] == "GetWeather"
    assert spec["description"] == "Current weather for a city."
    assert spec["input_schema"]["properties"]["city"]["type"] == "string"
    assert spec["input_schema"]["properties"]["days"]["type"] == "integer"
    # a parameter with a default is optional
    assert spec["input_schema"]["required"] == ["city"]


def test_argument_descriptions_come_from_a_google_docstring():
    @tool
    def search(query: str, limit: int = 10):
        """Search the index.

        Args:
            query: what to look for
            limit: how many results to return

        Returns:
            A list of hits.
        """
        return ""

    props = search.__tool_spec__["input_schema"]["properties"]
    assert props["query"]["description"] == "what to look for"
    assert props["limit"]["description"] == "how many results to return"
    assert search.__tool_spec__["description"] == "Search the index."


def test_argument_descriptions_come_from_a_sphinx_docstring():
    @tool
    def search(query: str):
        """Search the index.

        :param query: what to look for
        :returns: hits
        """
        return ""

    assert (
        search.__tool_spec__["input_schema"]["properties"]["query"]["description"]
        == "what to look for"
    )


def test_a_wrapped_argument_description_is_joined():
    @tool
    def f(a: str):
        """Do a thing.

        Args:
            a: a description that runs on
                to a second line
        """
        return ""

    assert "second line" in f.__tool_spec__["input_schema"]["properties"]["a"]["description"]


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (str, "string"), (int, "integer"), (float, "number"), (bool, "boolean"),
    ],
)
def test_primitive_annotations_map_to_json_types(annotation, expected):
    def f(x):
        """d"""

    f.__annotations__ = {"x": annotation}
    assert tool_spec_from(f)["input_schema"]["properties"]["x"]["type"] == expected


def test_container_and_optional_annotations():
    @tool
    def f(tags: list[str], note: Optional[str] = None, meta: dict = {}):
        """d"""
        return ""

    props = f.__tool_spec__["input_schema"]["properties"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["note"]["type"] == ["string", "null"]
    assert props["meta"]["type"] == "object"


def test_a_nested_dataclass_argument_is_expanded():
    @tool
    def move(to: Point):
        """Move somewhere."""
        return ""

    schema = move.__tool_spec__["input_schema"]["properties"]["to"]
    assert schema["properties"]["x"] == {"type": "integer"}


def test_an_unresolvable_annotation_is_unconstrained_not_wrong():
    """Under `from __future__ import annotations` every annotation is a string, and a name
    defined inside a function cannot be resolved from the module namespace. Leaving the
    field unconstrained is right: a guessed schema would reject valid calls."""
    @dataclasses.dataclass
    class LocalOnly:
        x: int

    @tool
    def f(v: LocalOnly):
        """d"""
        return ""

    assert f.__tool_spec__["input_schema"]["properties"]["v"] == {}
    # …but the primitives still resolve from their text.
    @tool
    def g(a: str, b: int, c: list[str]):
        """d"""
        return ""

    props = g.__tool_spec__["input_schema"]["properties"]
    assert props["a"]["type"] == "string"
    assert props["b"]["type"] == "integer"
    assert props["c"] == {"type": "array", "items": {"type": "string"}}


def test_an_unannotated_parameter_is_unconstrained_not_rejected():
    """Guessing a constraint would reject valid calls; a wrong 'invalid arguments' is
    worse than an unchecked field."""
    @tool
    def f(anything):
        """d"""
        return ""

    assert f.__tool_spec__["input_schema"]["properties"]["anything"] == {}


def test_varargs_and_kwargs_are_not_in_the_schema():
    @tool
    def f(a: str, *args, **kwargs):
        """d"""
        return ""

    assert list(f.__tool_spec__["input_schema"]["properties"]) == ["a"]


def test_name_and_description_can_be_overridden():
    @tool(name="Sum", description="Add two numbers")
    def add(a: int, b: int):
        return a + b

    assert add.__tool_spec__["name"] == "Sum"
    assert add.__tool_spec__["description"] == "Add two numbers"


def test_a_tool_with_no_description_is_refused_loudly():
    """The description is the only thing telling the model when to call it, so a silent
    empty string would produce a tool the model never uses and nobody can debug."""
    with pytest.raises(ValueError, match="no description"):
        @tool
        def f(a: int):
            return a


def test_snake_case_becomes_camel_case():
    @tool
    def read_the_file(path: str):
        """d"""
        return ""

    assert read_the_file.__tool_spec__["name"] == "ReadTheFile"


# --- the handler ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_handler_maps_a_dict_onto_keyword_arguments():
    @tool
    def add(a: int, b: int):
        """Add."""
        return a + b

    assert await add.__tool_handler__({"a": 2, "b": 3}) == 5


@pytest.mark.asyncio
async def test_an_async_function_works_unchanged():
    @tool
    async def slow(x: int):
        """Wait then return."""
        await asyncio.sleep(0)
        return x * 2

    assert await slow.__tool_handler__({"x": 21}) == 42


@pytest.mark.asyncio
async def test_defaults_apply_when_the_model_omits_an_argument():
    @tool
    def greet(name: str, greeting: str = "hello"):
        """Greet."""
        return f"{greeting} {name}"

    assert await greet.__tool_handler__({"name": "x"}) == "hello x"


@pytest.mark.asyncio
async def test_an_unexpected_field_is_dropped_not_a_type_error():
    """Argument validation has already run against the declared schema, so anything left
    is genuinely extra — and a TypeError is not something the model can act on."""
    @tool
    def f(a: int):
        """d"""
        return a

    assert await f.__tool_handler__({"a": 1, "hallucinated": True}) == 1


@pytest.mark.asyncio
async def test_a_function_taking_kwargs_receives_everything():
    @tool
    def f(a: int, **rest):
        """d"""
        return (a, rest)

    assert await f.__tool_handler__({"a": 1, "extra": 2}) == (1, {"extra": 2})


def test_the_function_stays_callable_as_written():
    @tool
    def add(a: int, b: int):
        """Add."""
        return a + b

    assert add(2, 3) == 5  # decoration must not change ordinary use


# --- normalisation --------------------------------------------------------------
def test_as_tool_accepts_a_decorated_function():
    @tool
    def f(a: int):
        """d"""
        return a

    spec, handler = as_tool(f)
    assert spec["name"] == "F" and callable(handler)


def test_as_tool_accepts_the_original_spec_handler_pair():
    """The low-level API must keep working unchanged."""
    spec = {"name": "X", "description": "d", "input_schema": {"type": "object"}}
    got_spec, handler = as_tool((spec, lambda i: "ok"))
    assert got_spec is spec


def test_as_tool_decorates_a_bare_annotated_function():
    def f(a: int):
        """A bare function."""
        return a

    spec, _ = as_tool(f)
    assert spec["input_schema"]["properties"]["a"]["type"] == "integer"


def test_as_tool_refuses_something_that_is_not_a_tool():
    with pytest.raises(TypeError, match="cannot use"):
        as_tool(42)


def test_as_tool_refuses_a_malformed_pair():
    with pytest.raises(TypeError, match="spec dict"):
        as_tool(("not-a-dict", lambda i: None))


def test_is_tool():
    @tool
    def f(a: int):
        """d"""
        return a

    assert is_tool(f)
    assert not is_tool(lambda: None)


# --- docstring parsing in isolation ----------------------------------------------
def test_parse_docstring_summary_only():
    summary, params = parse_docstring("Just a summary.")
    assert summary == "Just a summary." and params == {}


def test_parse_docstring_ignores_other_sections():
    summary, params = parse_docstring(
        "Sum.\n\nArgs:\n    a: first\n\nRaises:\n    ValueError: never\n"
    )
    assert params == {"a": "first"}
    assert "Raises" not in summary


def test_parse_docstring_handles_a_typed_arg_line():
    _, params = parse_docstring("d\n\nArgs:\n    a (int): the number\n")
    assert params == {"a": "the number"}


def test_parse_docstring_of_none():
    assert parse_docstring(None) == ("", {})


# --- end to end through the Agent --------------------------------------------------
@pytest.mark.asyncio
async def test_a_decorated_tool_runs_in_an_agent(tmp_path):
    seen: dict = {}

    @tool
    async def get_weather(city: str, units: str = "celsius") -> str:
        """Current weather for a city.

        Args:
            city: the city to look up
        """
        seen["city"] = city
        seen["units"] = units
        return f"{city}: 21 {units}"

    agent = Agent(
        model=Script([call("GetWeather", {"city": "Lahore"})]),
        workdir=tmp_path, tools=[get_weather],
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("weather in Lahore?")
    assert seen == {"city": "Lahore", "units": "celsius"}
    assert agent.ok is True
    await agent.aclose()


@pytest.mark.asyncio
async def test_the_generated_schema_is_enforced_by_the_loop(tmp_path):
    """The point of deriving the schema is that argument validation then applies to it."""
    @tool
    def add(a: int, b: int):
        """Add two numbers."""
        return a + b

    agent = Agent(
        model=Script([call("Add", {"a": "not-a-number", "b": 1})]),
        workdir=tmp_path, tools=[add],
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("add them")
    assert [f.kind for f in agent.failures] == ["invalid"]
    assert "expected integer" in agent.failures[0].error
    await agent.aclose()


@pytest.mark.asyncio
async def test_a_missing_required_argument_is_caught(tmp_path):
    @tool
    def add(a: int, b: int):
        """Add two numbers."""
        return a + b

    agent = Agent(
        model=Script([call("Add", {"a": 1})]),
        workdir=tmp_path, tools=[add],
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("add them")
    assert agent.failures and agent.failures[0].kind == "invalid"
    await agent.aclose()


@pytest.mark.asyncio
async def test_decorated_and_tuple_tools_coexist(tmp_path):
    @tool
    def alpha(x: int):
        """First."""
        return x

    spec = {"name": "Beta", "description": "Second", "input_schema": {"type": "object"}}

    agent = Agent(
        model=Script([]), workdir=tmp_path,
        tools=[alpha, (spec, lambda i: "ok")],
        enable_file_tools=False, enable_memory_tools=False,
    )
    await agent.run("hi")
    names = {s["name"] for s in agent._coord.tool_specs}
    assert {"Alpha", "Beta"} <= names
    await agent.aclose()


@pytest.mark.asyncio
async def test_a_name_collision_with_a_builtin_is_refused(tmp_path):
    @tool(name="ReadFile")
    def shadow(path: str):
        """Shadow the built-in."""
        return ""

    agent = Agent(model=Script([]), workdir=tmp_path, tools=[shadow])
    with pytest.raises(ValueError, match="collides"):
        await agent.run("hi")
