import asyncio
import time

import pytest

from engine.ratelimit import NullRateLimiter, RateLimiter


def test_rps_must_be_positive():
    with pytest.raises(ValueError):
        RateLimiter(0)
    with pytest.raises(ValueError):
        RateLimiter(-1)


def test_limiter_spaces_requests():
    async def scenario():
        rl = RateLimiter(rps=50)  # 20ms apart
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for _ in range(6):
            await rl.acquire()
        elapsed = loop.time() - t0
        # 6 acquisitions => at least 5 intervals of 20ms = 100ms
        assert elapsed >= 0.09, elapsed
        assert rl.acquired == 6
    asyncio.run(scenario())


def test_first_acquire_is_immediate():
    async def scenario():
        rl = RateLimiter(rps=1)  # 1s apart, but first must not wait
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await rl.acquire()
        assert loop.time() - t0 < 0.2
    asyncio.run(scenario())


def test_null_limiter_is_noop():
    asyncio.run(NullRateLimiter().acquire())


def test_concurrent_acquires_serialize():
    async def scenario():
        rl = RateLimiter(rps=100)  # 10ms
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.gather(*(rl.acquire() for _ in range(5)))
        assert loop.time() - t0 >= 0.035  # 4 intervals of 10ms
        assert rl.acquired == 5
    asyncio.run(scenario())
