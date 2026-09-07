"""Async rate limiter.

Not advisory: a caller `await`s `acquire()` before each throttled action (e.g. every
outbound HTTP request), and `acquire()` sleeps to keep actions spaced at least `1/rps`
apart. Async-safe via a lock so concurrent tasks serialize through the same spacing.
"""

from __future__ import annotations

import asyncio
import threading


class RateLimiter:
    # Process-wide registry so every limiter for the same key shares ONE bucket. Without
    # this, each worker/subagent builds its own limiter and the aggregate rate is N × rps.
    _registry: dict[str, "RateLimiter"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, rps: float) -> None:
        if rps <= 0:
            raise ValueError("rps must be > 0")
        self.rps = rps
        self._min_interval = 1.0 / rps
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0
        self.acquired = 0  # count, for observability/tests

    @classmethod
    def shared(cls, key: str, rps: float) -> "RateLimiter":
        """Return the one limiter for `key`, creating it once. All callers throttling the
        same resource must use the same key so their actions serialize through one bucket."""
        with cls._registry_lock:
            inst = cls._registry.get(key)
            if inst is None:
                inst = cls(rps)
                cls._registry[key] = inst
            return inst

    @classmethod
    def reset_shared(cls) -> None:
        """Drop the shared registry (test isolation / a new session)."""
        with cls._registry_lock:
            cls._registry.clear()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
            self.acquired += 1


class NullRateLimiter:
    """No-op limiter for tests / unthrottled contexts."""

    async def acquire(self) -> None:
        return None
