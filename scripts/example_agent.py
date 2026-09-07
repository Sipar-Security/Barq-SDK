"""Runnable example: build an agent, give it a tool, run a task.

Two modes:

  * Offline (default): drives the agent with a scripted fake model, so it runs with no API
    key and no network. Good for seeing the mechanics and for CI.

  * Live: set a provider key (see .env.example) and pass --live to use a real
    OpenAI-compatible model.

    python scripts/example_agent.py            # offline, deterministic
    python scripts/example_agent.py --live     # real model from .env
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import Agent, ModelResponse


# A tiny custom tool the model can call.
WEATHER_SPEC = {
    "name": "GetWeather",
    "description": "Return the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


async def get_weather(inp: dict) -> str:
    return f"{inp.get('city', '?')}: 21°C, clear"


class ScriptedModel:
    """Stand-in model for the offline demo: calls the tool, then answers."""

    def __init__(self):
        self._turns = [
            ModelResponse(
                content=[{"type": "tool_use", "id": "1", "name": "GetWeather",
                          "input": {"city": "Lisbon"}}],
                stop_reason="tool_use",
            ),
            ModelResponse(
                content=[{"type": "text", "text": "It's 21°C and clear in Lisbon."}],
                stop_reason="end_turn",
            ),
        ]

    async def create(self, messages, tools):
        return self._turns.pop(0)


async def main(live: bool) -> None:
    if live:
        from engine.providers import OpenAICompatClient
        from engine.providers.config import build_router_from_env
        from engine.providers import ModelRole
        router = build_router_from_env()               # reads .env
        model = OpenAICompatClient(router.spec(ModelRole.SMART))
    else:
        model = ScriptedModel()

    agent = Agent(
        model=model,
        workdir=tempfile.mkdtemp(prefix="agent-"),
        tools=[(WEATHER_SPEC, get_weather)],
    )
    answer = await agent.run("What's the weather in Lisbon?")
    print("Answer:", answer)
    print("Completed cleanly:", agent.completed)


if __name__ == "__main__":
    asyncio.run(main(live="--live" in sys.argv))
