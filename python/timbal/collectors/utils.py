import asyncio
import contextvars
from collections.abc import AsyncGenerator, Generator
from typing import Any


async def sync_to_async_gen(
    gen: Generator[Any, None, None], 
    loop: asyncio.AbstractEventLoop,
    ctx: contextvars.Context,
    # Hack to pass the runnable path and call id to the executor and enable usage tracking with inspect
    _path: str = None,
    _call_id: str = None,
) -> AsyncGenerator[Any, None]:
    """Auxiliary function to convert a sync generator to an async generator.
    This function also shares the context of the caller to the executor.
    """
    while True:
        # StopIteration is special in Python. It's used to implement generator protocol and can't
        # be pickled/transferred across threads properly. By catching it explicitly in the executor 
        # function and converting it to a sentinel value, we avoid problematic exception propagation.
        def _next(_path=_path, _call_id=_call_id):
            try:
                return next(gen)
            except StopIteration: 
                return None
        value = await loop.run_in_executor(None, lambda: ctx.run(_next))
        if value is None:
            break
        yield value
