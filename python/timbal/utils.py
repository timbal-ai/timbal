import asyncio
import contextvars
import importlib.util
import math
import socket
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any, NamedTuple

import structlog
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

logger = structlog.get_logger("timbal.utils")


class ImportSpec(BaseModel):
    """Specification for importing an object from a Python module."""
    path: Path
    target: str | None = None

    def load(self) -> Any:
        """Load and return the target object from the module."""
        spec = importlib.util.spec_from_file_location(self.path.stem, self.path.as_posix())
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if self.target:
                if hasattr(module, self.target):
                    obj = getattr(module, self.target)
                    return obj
                else:
                    raise ValueError(f"Module {self.path} has no target {self.target}")
            else:
                raise NotImplementedError("Does not support loading entire module")
        else:
            raise ValueError(f"Failed to load module {self.path}")


def is_port_in_use(port: int) -> bool:
    """Check if a TCP port is currently in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0


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
    from .types.file import File
    from .types.message import Message
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
        return {
            "role": value.role,
            "content": await asyncio.gather(*[dump(c) for c in value.content]),
        }
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
    elif isinstance(value, (list, tuple)): # noqa: UP038
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
    return value


def create_model_from_argspec(name: str, argspec: NamedTuple) -> BaseModel:
    """Create a dynamic pydantic model from the argspec of a function."""
    fields = {}
    defaults = {}
    if argspec.defaults:
        defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults, strict=True))
    for field_name in argspec.args:
        field_info = defaults.get(field_name, ...) # Pydantic will use ... to mark this field as required.
        if not isinstance(field_info, FieldInfo):
            field_info = Field(field_info)
        field_type = argspec.annotations.get(field_name, Any)
        fields[field_name] = (field_type, field_info)
    return create_model(name, __config__=ConfigDict(extra="ignore"), **fields)


def issubclass_safe(candidate: Any, base_class: type[Any]) -> bool:
    """Helper function to avoid checking on annotations types always before calling issubclass."""
    if isinstance(candidate, type):
        return issubclass(candidate, base_class)
    else:
        return False


# TODO We might implement a decorator to wrap all handlers to handle this automatically
def resolve_default(key: str, value: Any) -> Any:
    """Resolve the default value of a field.
    Use this function to resolve default kwargs when calling a function that uses Field defaults.
    """
    if isinstance(value, FieldInfo):
        if value.default == PydanticUndefined:
            raise ValueError(f"{key} is required")
        return value.default
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
