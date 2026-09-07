"""Provider config from environment / .env.

Reads DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL (and the moonshot/zhipu
equivalents if present) and builds a ModelRouter. With only DeepSeek configured, all
roles route to DeepSeek; add MOONSHOT_* / ZHIPU_* to split roles across providers.

Tiny .env loader here so there is no python-dotenv dependency. It only sets keys that
are not already in the environment (real env wins over the file).
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import ModelRole, ModelRouter, ModelSpec


def _parse_value(raw: str) -> str:
    """Parse the right-hand side of a .env line.

    Quoted values take the quoted span verbatim and ignore anything after the closing
    quote (so `KEY="sk-x" # note` -> `sk-x`). Unquoted values drop a trailing ` #`
    comment (so `KEY=sk-x # note` -> `sk-x`). The old code blindly stripped quotes and
    whitespace, which folded inline comments into the value and corrupted API keys.
    """
    v = raw.strip()
    if v[:1] in ("'", '"'):
        quote = v[0]
        end = v.find(quote, 1)
        return v[1:end] if end != -1 else v[1:]  # unterminated quote: take the rest
    hash_idx = v.find(" #")
    if hash_idx != -1:
        v = v[:hash_idx]
    return v.strip()


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), _parse_value(v)
        os.environ.setdefault(k, v)


def _max_tokens_for(prefix: str) -> int | None:
    """Per-completion cap: {prefix}_MAX_TOKENS overrides AGENT_MAX_TOKENS (default 8192).
    Set to 0 to opt out (provider default / unbounded)."""
    raw = os.environ.get(f"{prefix}_MAX_TOKENS") or os.environ.get("AGENT_MAX_TOKENS") or "8192"
    try:
        n = int(raw)
    except ValueError:
        return 8192
    return n if n > 0 else None


def _spec_for(prefix: str, provider: str, default_model: str) -> ModelSpec | None:
    if not os.environ.get(f"{prefix}_API_KEY"):
        return None
    return ModelSpec(
        provider=provider,
        model=os.environ.get(f"{prefix}_MODEL", default_model),
        base_url=os.environ.get(f"{prefix}_BASE_URL") or None,
        max_tokens=_max_tokens_for(prefix),
    )


def build_router_from_env(dotenv: str | Path | None = ".env") -> ModelRouter:
    if dotenv is not None:
        load_dotenv(dotenv)

    deepseek = _spec_for("DEEPSEEK", "deepseek", "deepseek-chat")
    moonshot = _spec_for("MOONSHOT", "moonshot", "kimi-k2")
    zhipu = _spec_for("ZHIPU", "zhipu", "glm-4.6")
    zenmux = _spec_for("ZENMUX", "zenmux", "x-ai/grok-4.5-free")

    # Prefer a cheap/fast model for hot-path work and a strong one for reasoning when more
    # than one provider is configured; otherwise everything routes to whatever is present.
    by_provider = {s.provider: s for s in (deepseek, moonshot, zhipu, zenmux) if s is not None}
    available = list(by_provider.values())
    if not available:
        raise RuntimeError(
            "no provider configured: set DEEPSEEK_API_KEY (and/or MOONSHOT_/ZHIPU_/ZENMUX_)"
        )

    # Explicit role→provider overrides:
    #   AGENT_SMART_PROVIDER: the reasoning-heavy work (default: a strong model if present)
    #   AGENT_FAST_PROVIDER : fast hot-path calls (default: a cheap model)
    smart = (by_provider.get(os.environ.get("AGENT_SMART_PROVIDER", ""))
             or zenmux or zhipu or moonshot or deepseek or available[0])
    fast = (by_provider.get(os.environ.get("AGENT_FAST_PROVIDER", ""))
            or deepseek or moonshot or available[0])

    return ModelRouter(
        {
            ModelRole.FAST: fast,
            ModelRole.SMART: smart,
        }
    )
