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
    """Utility function to check if a value is NaN or NA-like."""
    if value is None:
        return True
    # Catch NA sentinel types (e.g. NAType from numpy or similar libraries).
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
_FileContent = None
_Runnable = None


def _ensure_types():
    global _File, _Message, _FileContent, _Runnable
    if _File is None:
        from ..core.runnable import Runnable
        from ..types.content.file import FileContent
        from ..types.file import File
        from ..types.message import Message

        _File = File
        _Message = Message
        _FileContent = FileContent
        _Runnable = Runnable


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

    if isinstance(value, _FileContent):
        raise _NEEDS_ASYNC

    if isinstance(value, _Message):
        # Per-message dump cache: long conversations re-dump the same Message
        # objects several times per turn. Messages are immutable after
        # construction except for in-place content appends, so validate the
        # cache against len(content). The returned dict is a shared read-only
        # snapshot — dump consumers never mutate the inner dicts.
        content = value.content
        if value._cached_dump is not None and value._cached_dump_len == len(content):
            return value._cached_dump
        result = {
            "role": value.role,
            "content": [_dump_sync(c) for c in content],
        }
        if value.stop_reason is not None:
            result["stop_reason"] = value.stop_reason
        object.__setattr__(value, "_cached_dump", result)
        object.__setattr__(value, "_cached_dump_len", len(content))
        return result

    # BaseModel branch comes before the marker probes: failed getattr on a
    # pydantic model raises AttributeError inside pydantic's __getattr__, which
    # is expensive when probing thousands of content items per dump.
    if isinstance(value, BaseModel):
        if isinstance(value, _Runnable):
            return value.model_dump()
        return {k: _dump_sync(v) for k, v in value.__dict__.items()}

    # Slotted timbal models (events, Span, RunStatus) — plain classes with a
    # class-attribute marker (cheap getattr); no __dict__; use model_dump()
    if getattr(value, "__timbal_serializable__", False):
        return _dump_sync(value.model_dump())

    if isinstance(value, tuple):
        return tuple(_dump_sync(v) for v in value)

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, Exception):
        return {"error_type": type(value).__name__, "message": str(value)}

    # NaN/NA check for non-float types (NA sentinels, np.nan boxed in object, etc.)
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

    if isinstance(value, _FileContent):
        result: dict[str, Any] = {
            "type": value.type,
            "file": await _dump_async(value.file),
        }
        if value.name is not None:
            result["name"] = value.name
        return result

    if isinstance(value, _Message):
        # See _dump_sync: cached, len-validated snapshot. Also caches messages
        # with File content after their (potentially expensive) persist step.
        content = value.content
        if value._cached_dump is not None and value._cached_dump_len == len(content):
            return value._cached_dump
        result = {
            "role": value.role,
            "content": await asyncio.gather(*[_dump_async(c) for c in content]),
        }
        if value.stop_reason is not None:
            result["stop_reason"] = value.stop_reason
        object.__setattr__(value, "_cached_dump", result)
        object.__setattr__(value, "_cached_dump_len", len(content))
        return result

    # BaseModel before marker probes — see _dump_sync.
    if isinstance(value, BaseModel):
        if isinstance(value, _Runnable):
            return value.model_dump()
        items = await asyncio.gather(*[_dump_async(v) for v in value.__dict__.values()])
        return dict(zip(value.__dict__.keys(), items, strict=False))

    # Slotted timbal models (events, Span, RunStatus) — no __dict__; use model_dump()
    if getattr(value, "__timbal_serializable__", False):
        return await _dump_async(value.model_dump())

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

def invalidate_message_dump_caches(value: Any) -> None:
    """Reset the cached dump on any Message reachable in *value* (shallow containers).

    Hooks are allowed to mutate message content in place (e.g. a post_hook
    rewriting the output text); the framework re-dumps afterwards and must not
    serve the pre-mutation cache. Called after post_hook execution.
    """
    _ensure_types()
    if isinstance(value, _Message):
        object.__setattr__(value, "_cached_dump", None)
        object.__setattr__(value, "_cached_dump_len", -1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            invalidate_message_dump_caches(item)
    elif isinstance(value, dict):
        for item in value.values():
            invalidate_message_dump_caches(item)


async def dump(value: Any) -> Any:
    """Dumps all models that live within a nested structure of arbitrary depth.

    Uses a sync fast path for value trees without File objects (zero asyncio overhead).
    Falls back to the async path only when a File requiring I/O is present.
    """
    try:
        return _dump_sync(value)
    except _NeedsAsync:
        return await _dump_async(value)


_GEN_DONE = object()
"""Unique end-of-generator sentinel for sync_to_async_gen. Using None would
silently truncate sync generators that legitimately yield None."""


async def sync_to_async_gen(
    gen: Generator[Any, None, None],
    loop: asyncio.AbstractEventLoop,
    ctx: contextvars.Context,
) -> AsyncGenerator[Any, None]:
    """Auxiliary function to convert a sync generator to an async generator.
    This function also shares the context of the caller to the executor.
    """

    # StopIteration is special in Python. It's used to implement generator protocol and can't
    # be pickled/transferred across threads properly. By catching it explicitly in the executor
    # function and converting it to a sentinel value, we avoid problematic exception propagation.
    def _next():
        try:
            return next(gen)
        except StopIteration:
            return _GEN_DONE

    def _next_with_ctx():
        return ctx.run(_next)

    while True:
        value = await loop.run_in_executor(None, _next_with_ctx)
        if value is _GEN_DONE:
            break
        yield value


def coerce_to_dict(v: Any) -> dict[str, Any]:
    """Utility function to convert LLM outputs into python objects.

    Providers often emit ``null`` / ``"null"`` / ``None`` for tools with no
    parameters (Groq + OpenAI-compatible chat completions). Treat those as ``{}``.
    """
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        stripped = v.strip()
        if not stripped or stripped.lower() in ("null", "none"):
            return {}
        try:
            parsed = json.loads(stripped)
        except Exception:
            try:
                parsed = literal_eval(stripped)
            except Exception as e:
                raise ValueError(f"Cannot coerce value to dict: {v}") from e
        if parsed is None:
            return {}
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Cannot coerce value to dict: {v}")
    raise ValueError(f"Cannot coerce value to dict: {v}")
