"""Barq-SDK: a small SDK for building tool-using AI agents in Python.

This is the public import name. The implementation lives in the `engine` package, which
stays importable so existing code keeps working:

    from barq_sdk import Agent                 # preferred
    from barq_sdk.providers import ModelSpec   # subpackages work too
    from engine import Agent                   # legacy, still supported

Every name is the *same object* under both spellings — this module re-exports rather than
re-implements, so `barq_sdk.Agent is engine.Agent`.
"""

from __future__ import annotations

import sys

import engine
from engine import *  # noqa: F401,F403 — re-export the curated __all__
from engine import __version__

from engine import (  # noqa: F401 — bound as attributes AND aliased in sys.modules below
    agent,
    audit,
    compact,
    coordinator,
    hooks,
    loopguard,
    mcp,
    memory,
    permissions,
    providers,
    ratelimit,
    sandbox,
    session,
    subagents,
    tools,
    validation,
)

# Register subpackage aliases in sys.modules so `from barq_sdk.audit import AuditLog`
# resolves without a separate physical package. Derived from the imports above so the two
# lists cannot drift apart.
_SUBMODULES = (
    "agent", "audit", "compact", "coordinator", "hooks", "loopguard", "mcp", "memory",
    "permissions", "providers", "ratelimit", "sandbox", "session", "subagents", "tools",
    "validation",
)

for _name in _SUBMODULES:
    _mod = getattr(engine, _name)
    sys.modules[f"{__name__}.{_name}"] = _mod
    # Nested packages (engine.audit.integrity, engine.permissions.network, ...) are aliased
    # on demand by their parent's import; alias the ones already loaded so a deep import
    # like `barq_sdk.audit.integrity` works without importing `engine.audit.integrity` first.
    for _full, _loaded in list(sys.modules.items()):
        if _full.startswith(f"engine.{_name}."):
            sys.modules[f"{__name__}.{_full[len('engine.'):]}"] = _loaded

del _name, _mod, _full, _loaded

__all__ = list(engine.__all__) + list(_SUBMODULES)
