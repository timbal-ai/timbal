import asyncio
import contextvars
import math
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..types.file import File
from ..types.message import Message
from .context import get_run_context


async def sync_to_async_gen(
    gen: Generator[Any, None, None], 
    loop: asyncio.AbstractEventLoop,
    ctx: contextvars.Context,
) -> AsyncGenerator[Any, None]:
    """Auxiliary function to convert a sync generator to an async generator.
    This function also shares the context of the caller to the executor.
    """
    while True:
        # StopIteration is special in Python. It's used to implement generator protocol and can't
        # be pickled/transferred across threads properly. By catching it explicitly in the executor 
        # function and converting it to a sentinel value, we avoid problematic exception propagation.
        def _next():
            try:
                return next(gen)
            except StopIteration: 
                return None
        value = await loop.run_in_executor(None, lambda: ctx.run(_next))
        if value is None:
            break
        yield value


def safe_is_nan(value: Any) -> bool:
    """Utility function to check if a value is NaN. Even for pd.NA values (without pandas dependency)."""
    if value is None:
        return True
    
    # Catch pd.NA values.
    if type(value).__name__ == "NAType":
        return True

    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def dump(value: Any) -> Any:
    """Dumps all models that live within a nested structure of arbitrary depth."""
    # Handle float("nan"), np.nan, pd.NA, etc. (might need to handle more scenarios here)
    if safe_is_nan(value):
        return None
    elif isinstance(value, float):
        # Handle non-finite floats and limit precision to avoid JSON serialization issues
        if not isinstance(value, float) or not value.is_integer():
            return round(value, 10)  # 10 decimal places should be enough for most use cases
    elif isinstance(value, Path):
        return value.as_posix()
    elif isinstance(value, Message): # Message is no longer a BaseModel.
        return {
            "role": value.role,
            "content": [dump(c) for c in value.content],
        }
    elif isinstance(value, BaseModel): # Use @model_serializer of the class
        return value.model_dump()
    elif isinstance(value, dict):
        return {k: dump(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)): # noqa: UP038
        return [dump(v) for v in value]
    elif isinstance(value, File):
        run_context = get_run_context()
        return File.serialize(value, run_context)
    elif isinstance(value, Exception):
        return {
            "error_type": type(value).__name__,
            "message": str(value),
            # "traceback": traceback.format_exc()
        }
    return value
