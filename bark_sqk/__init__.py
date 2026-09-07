"""Bark-SQK: a small SDK for building tool-using AI agents in Python.

This package provides top-level access to the Bark-SQK agent framework and exports.
It maintains full backwards compatibility with the legacy `engine` package.
"""

from __future__ import annotations

import sys
import engine
from engine import *
from engine import __version__

import engine.agent as agent
import engine.audit as audit
import engine.compact as compact
import engine.coordinator as coordinator
import engine.hooks as hooks
import engine.loopguard as loopguard
import engine.mcp as mcp
import engine.memory as memory
import engine.permissions as permissions
import engine.providers as providers
import engine.ratelimit as ratelimit
import engine.sandbox as sandbox
import engine.session as session
import engine.subagents as subagents
import engine.tools as tools
import engine.validation as validation

# Register subpackage aliases in sys.modules so `from bark_sqk.subpackage import ...` works seamlessly
_submodules = {
    "agent": agent,
    "audit": audit,
    "compact": compact,
    "coordinator": coordinator,
    "hooks": hooks,
    "loopguard": loopguard,
    "mcp": mcp,
    "memory": memory,
    "permissions": permissions,
    "providers": providers,
    "ratelimit": ratelimit,
    "sandbox": sandbox,
    "session": session,
    "subagents": subagents,
    "tools": tools,
    "validation": validation,
}

for name, mod in _submodules.items():
    sys.modules[f"bark_sqk.{name}"] = mod

__all__ = engine.__all__
