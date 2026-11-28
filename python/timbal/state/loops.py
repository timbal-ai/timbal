"""Handing work to another event loop and waiting for it while that loop runs.

A task or future belongs to the loop it was made on, and only that loop can finish it. These helpers hand the work
over (``run_coroutine_threadsafe``) and wait for it by polling, with no callback into the waiting loop, which may be
gone by the time the work ends. Used by ``RunContext``'s end hooks (``context._awaited``): a hook may return a pending
future of another loop than the one ending the run.
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


async def wait_on_owner(handed: "concurrent.futures.Future[Any]", owner: asyncio.AbstractEventLoop) -> bool:
    """Wait for work handed to ``owner`` while ``owner`` runs, kept on the waiting loop. False when it stops before
    the work is done (the work goes on there once it runs again): a wait for a paused loop would never end. True
    means only that it is done: whether it succeeded, was cancelled there or raised is the future's."""
    while not handed.done():
        if owner.is_closed() or not owner.is_running():
            return handed.done()
        await asyncio.sleep(POLL_S)
    return True
