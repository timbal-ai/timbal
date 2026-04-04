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


# ---------------------------------------------------------------------------
# Cached type references (lazy-loaded once to avoid per-call import overhead)
# ---------------------------------------------------------------------------
_File = None
_Message = None


def _ensure_types():
    global _File, _Message
    if _File is None:
        from ..types.file import File
        from ..types.message import Message

        _File = File
        _Message = Message


# Pre-allocated singleton to avoid creating exception objects on every File hit
class _NeedsAsync(Exception):
    """Sentinel: value tree contains a File that requires async I/O."""


_NEEDS_ASYNC = _NeedsAsync()


# ---------------------------------------------------------------------------
# Sync fast path — zero asyncio overhead for the 99% case (no File objects)
# ---------------------------------------------------------------------------

# Primitive types that are always JSON-safe and need zero processing
_PASSTHROUGH_TYPES = (int, str, bool, type(None))


def _dump_sync(value: Any) -> Any:
    # Fast exit for the most common types — no function call overhead
    if isinstance(value, _PASSTHROUGH_TYPES):
        return value

    if isinstance(value, float):
        # safe_is_nan inlined for hot path
        if value != value:  # NaN check: NaN != NaN
            return None
        if not value.is_integer():
            return round(value, 10)
        return value

    if isinstance(value, dict):
        return {k: _dump_sync(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_dump_sync(v) for v in value]

    _ensure_types()

    if isinstance(value, _File):
        raise _NEEDS_ASYNC

    if isinstance(value, _Message):
        result = {
            "role": value.role,
            "content": [_dump_sync(c) for c in value.content],
        }
        if value.stop_reason is not None:
            result["stop_reason"] = value.stop_reason
        return result

    # Marker attribute check — O(1) vs O(n) MRO scan
    if getattr(value, "_is_timbal_runnable", False):
        return value.model_dump()

    if isinstance(value, BaseModel):
        return {k: _dump_sync(v) for k, v in value.__dict__.items()}

    if isinstance(value, tuple):
        return tuple(_dump_sync(v) for v in value)

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, Exception):
        return {"error_type": type(value).__name__, "message": str(value)}

    # NaN/NA check for non-float types (pd.NA, np.nan boxed in object, etc.)
    if safe_is_nan(value):
        return None

    return str(value)


# ---------------------------------------------------------------------------
# Async path — only entered when a File object exists in the value tree
# ---------------------------------------------------------------------------

async def _dump_async(value: Any) -> Any:
    _ensure_types()

    if isinstance(value, _PASSTHROUGH_TYPES):
        return value

    if isinstance(value, float):
        if value != value:
            return None
        if not value.is_integer():
            return round(value, 10)
        return value

    if isinstance(value, dict):
        keys, values = zip(*value.items(), strict=False) if value else ([], [])
        dumped_values = await asyncio.gather(*[_dump_async(v) for v in values])
        return dict(zip(keys, dumped_values, strict=False))

    if isinstance(value, (list, tuple)):  # noqa: UP038
        dumped_items = await asyncio.gather(*[_dump_async(v) for v in value])
        return dumped_items if isinstance(value, list) else tuple(dumped_items)

    if isinstance(value, _File):
        return await value.persist()

    if isinstance(value, _Message):
        result = {
            "role": value.role,
            "content": await asyncio.gather(*[_dump_async(c) for c in value.content]),
        }
        if value.stop_reason is not None:
            result["stop_reason"] = value.stop_reason
        return result

    if getattr(value, "_is_timbal_runnable", False):
        return value.model_dump()

    if isinstance(value, BaseModel):
        items = await asyncio.gather(*[_dump_async(v) for v in value.__dict__.values()])
        return dict(zip(value.__dict__.keys(), items, strict=False))

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, Exception):
        return {"error_type": type(value).__name__, "message": str(value)}

    if safe_is_nan(value):
        return None

    return str(value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def dump(value: Any) -> Any:
    """Dumps all models that live within a nested structure of arbitrary depth.

    Uses a sync fast path for value trees without File objects (zero asyncio overhead).
    Falls back to the async path only when a File requiring I/O is present.
    """
    try:
        return _dump_sync(value)
    except _NeedsAsync:
        return await _dump_async(value)


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
