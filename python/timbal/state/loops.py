"""Handing work to another event loop and waiting for it while that loop runs.

A session, task or future belongs to the loop it was made on, and only that loop can finish it. These helpers hand
the work over (``run_coroutine_threadsafe``) and wait for it by polling, with no callback into the waiting loop,
which may be gone by the time the work ends.
"""

import asyncio
import concurrent.futures
from typing import Any

# How often a wait for work handed to another loop checks that the loop still runs (no event says it stopped).
POLL_S = 0.02


def hand_to(owner: asyncio.AbstractEventLoop, coroutine: Any) -> "concurrent.futures.Future[Any] | None":
    """``run_coroutine_threadsafe(coroutine, owner)``, or None when ``owner`` closed after the caller saw it run (the
    call raises then): the coroutine is closed, never left unawaited."""
    try:
        return asyncio.run_coroutine_threadsafe(coroutine, owner)
    except RuntimeError:
        coroutine.close()
        return None


async def wait_on_owner(
    handed: "concurrent.futures.Future[Any]", owner: asyncio.AbstractEventLoop, timeout: float | None = None
) -> bool:
    """Wait for work handed to ``owner`` while ``owner`` runs, for at most ``timeout`` seconds, kept on the waiting
    loop: a loop that runs but is busy (a callback that does not yield) may not even start the work. False when it
    stops before the work is done, or the time is up (the work goes on there): a wait for a paused loop would never
    end. True means only that it is done: whether it succeeded, was cancelled there or raised is the future's."""
    end = None if timeout is None else asyncio.get_running_loop().time() + timeout
    while not handed.done():
        if owner.is_closed() or not owner.is_running():
            return handed.done()
        if end is not None and asyncio.get_running_loop().time() >= end:
            return False
        await asyncio.sleep(POLL_S)
    return True


def succeeded(handed: "concurrent.futures.Future[Any]") -> bool:
    """A done future that neither was cancelled nor raised."""
    return handed.done() and not handed.cancelled() and handed.exception() is None
