"""Model providers: OpenAI-compatible endpoints (DeepSeek / Moonshot / Zhipu / any
aggregator like ZenMux or OpenRouter).

Every provider here exposes an OpenAI-style /chat/completions endpoint with function/tool
calling, so one adapter (openai_compat.py) serves them all; only base_url, api-key env
var, and model ids differ. (There is no vendor SDK dependency on purpose: this avoids
SDK-vs-provider drift.)

Base URLs below are sensible defaults but change over time; confirm against each
provider's docs. `api_key()` reads from the named env var and raises if missing, so a
misconfiguration fails loudly at construction rather than silently degrading.

Role routing: map a `ModelRole` to a `(provider, model)` so a caller can use a cheap/fast
model for hot-path work and a strong model for the reasoning-heavy work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    base_url: str  # OpenAI-compatible root; /chat/completions is appended
    api_key_env: str


# Defaults: confirm against provider docs. Overridable via ModelSpec.base_url.
PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    # Moonshot / Kimi: use .cn for mainland China, .ai for international.
    "moonshot": ProviderConfig("moonshot", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "zhipu": ProviderConfig("zhipu", "https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
    # ZenMux: an OpenAI-compatible model aggregator (like OpenRouter) behind one key.
    "zenmux": ProviderConfig("zenmux", "https://zenmux.ai/api/v1", "ZENMUX_API_KEY"),
}


class ModelRole(str, Enum):
    # Cheap / low-latency: routing, classification, short hot-path calls.
    FAST = "fast"
    # Strong reasoning: the main agent work.
    SMART = "smart"


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # key into PROVIDERS
    model: str  # e.g. "deepseek-chat", "kimi-k2", "glm-4.6" (confirm ids)
    base_url: str | None = None  # override PROVIDERS[provider].base_url
    temperature: float = 0.0
    # Per-completion cap. None = provider default (unbounded). A finite cap bounds any
    # single response so a degenerate "repeat until the token limit" turn can't balloon
    # the context: belt-and-suspenders with the loop guard.
    max_tokens: int | None = None

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return PROVIDERS[self.provider].base_url

    def api_key(self) -> str:
        env = PROVIDERS[self.provider].api_key_env
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(
                f"missing API key: set ${env} for provider {self.provider!r}"
            )
        return key


class ModelRouter:
    """Maps each role to a ModelSpec. Callers ask the router for the model appropriate to
    the work rather than hardcoding one provider."""

    def __init__(self, routes: dict[ModelRole, ModelSpec]) -> None:
        missing = set(ModelRole) - set(routes)
        if missing:
            raise ValueError(f"router missing roles: {sorted(m.value for m in missing)}")
        self._routes = dict(routes)

    def spec(self, role: ModelRole) -> ModelSpec:
        return self._routes[role]
