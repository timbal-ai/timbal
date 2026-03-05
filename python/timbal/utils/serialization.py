import asyncio
import contextvars
import json
import math
from ast import literal_eval
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


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


async def dump(value: Any) -> Any:
    """Dumps all models that live within a nested structure of arbitrary depth."""
    from ..types.file import File
    from ..types.message import Message

    # Handle float("nan"), np.nan, pd.NA, etc. (might need to handle more scenarios here)
    if safe_is_nan(value):
        return None
    elif isinstance(value, float):
        # Handle non-finite floats and limit precision to avoid JSON serialization issues
        if not isinstance(value, float) or not value.is_integer():
            return round(value, 10)  # 10 decimal places should be enough for most use cases
    elif isinstance(value, Path):
        return value.as_posix()
    elif isinstance(value, Message):
        result = {
            "role": value.role,
            "content": await asyncio.gather(*[dump(c) for c in value.content]),
        }
        if value.stop_reason is not None:
            result["stop_reason"] = value.stop_reason
        return result
    # Perform the check via mro to avoid circular imports between Runnable and dump
    elif any(cls.__name__ == "Runnable" for cls in value.__class__.__mro__):
        return value.model_dump()
    # Handle the rest of BaseModel instances as we handle dictionaries
    elif isinstance(value, BaseModel):
        items = await asyncio.gather(*[dump(v) for v in value.__dict__.values()])
        return dict(zip(value.__dict__.keys(), items, strict=False))
    elif isinstance(value, dict):
        keys, values = zip(*value.items(), strict=False) if value else ([], [])
        dumped_values = await asyncio.gather(*[dump(v) for v in values])
        return dict(zip(keys, dumped_values, strict=False))
    elif isinstance(value, (list, tuple)):  # noqa: UP038
        dumped_items = await asyncio.gather(*[dump(v) for v in value])
        return dumped_items if isinstance(value, list) else tuple(dumped_items)
    elif isinstance(value, File):
        return await value.persist()
    elif isinstance(value, Exception):
        return {
            "error_type": type(value).__name__,
            "message": str(value),
            # "traceback": traceback.format_exc()
        }
    # Try to serialize the value as JSON, if it fails, convert it to a string
    try:
        json.dumps(value)
    except:
        value = str(value)
    return value


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


def coerce_to_dict(v: Any) -> dict[str, Any]:
    """Utility function to convert LLM outputs into python objects."""
    if isinstance(v, dict):
        return v
    elif isinstance(v, str):
        if v.strip() == "":
            return {}
        try:
            v = json.loads(v)
            return v
        except Exception:
            try:
                v = literal_eval(v)
                return v
            except Exception as e:
                raise ValueError(f"Cannot coerce value to dict: {v}") from e
    else:
        raise ValueError(f"Cannot coerce value to dict: {v}")
