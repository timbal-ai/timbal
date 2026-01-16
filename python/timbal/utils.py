import asyncio
import contextvars
import importlib.util
import inspect
import json
import math
import socket
from ast import literal_eval
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path
from typing import Any, Literal, cast

import pydantic
import structlog
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

logger = structlog.get_logger("timbal.utils")

# Type definitions from anthropic SDK for schema transformation
SupportedTypes = Literal["object", "array", "string", "number", "integer", "boolean", "null"]
SupportedStringFormats = Literal["date", "date-time", "email", "hostname", "ipv4", "ipv6", "uri", "uuid"]


def is_list(obj: object) -> bool:
    """Helper to check if an object is a list."""
    return isinstance(obj, list)


def assert_never(value: object) -> None:
    """Helper for exhaustiveness checking."""
    raise AssertionError(f"Unexpected value: {value}")


def transform_schema(
    json_schema: type[pydantic.BaseModel] | dict[str, Any],
) -> dict[str, Any]:
    """
    Transforms a JSON schema to ensure compatibility with OpenAI strict mode.

    This is a modified version of anthropic.transform_schema that enforces
    OpenAI's structured output requirements:
    - All object properties are marked as required (strict mode requirement)
    - additionalProperties is set to false
    - Converts oneOf to anyOf (OpenAI doesn't support allOf but supports anyOf)
    - Rejects unsupported composition keywords (allOf, not, etc.)

    Modified from: anthropic.transform_schema
    Original: https://github.com/anthropics/anthropic-sdk-python

    OpenAI Structured Outputs Requirements:
    - All fields must be in the 'required' array when strict=true
    - additionalProperties must be false
    - Root object cannot use anyOf
    - allOf, not, dependentRequired, dependentSchemas, if/then/else not supported

    References:
    - https://platform.openai.com/docs/guides/structured-outputs
    - https://docs.anthropic.com/en/docs/build-with-claude/structured-output

    Args:
        json_schema: A Pydantic BaseModel class or a JSON schema dictionary.

    Returns:
        The transformed JSON schema compatible with both OpenAI and Anthropic.

    Examples:
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     age: Optional[int] = None
        >>> transform_schema(MyModel)
        {'type': 'object', 'properties': {...}, 'required': ['name', 'age'], ...}
    """
    if inspect.isclass(json_schema) and issubclass(json_schema, pydantic.BaseModel):
        json_schema = json_schema.model_json_schema()

    strict_schema: dict[str, Any] = {}
    json_schema = {**json_schema}

    ref = json_schema.pop("$ref", None)
    if ref is not None:
        strict_schema["$ref"] = ref
        return strict_schema

    defs = json_schema.pop("$defs", None)
    if defs is not None:
        strict_defs: dict[str, Any] = {}
        strict_schema["$defs"] = strict_defs

        for name, schema in defs.items():
            strict_defs[name] = transform_schema(schema)

    type_: SupportedTypes | None = json_schema.pop("type", None)
    any_of = json_schema.pop("anyOf", None)
    one_of = json_schema.pop("oneOf", None)
    all_of = json_schema.pop("allOf", None)

    # Reject unsupported composition keywords per OpenAI strict mode
    unsupported_keywords = ["not", "dependentRequired", "dependentSchemas", "if", "then", "else"]
    for keyword in unsupported_keywords:
        if keyword in json_schema:
            raise ValueError(
                f"Unsupported JSON Schema keyword '{keyword}' for OpenAI strict mode. "
                f"See: https://platform.openai.com/docs/guides/structured-outputs"
            )

    if is_list(any_of):
        strict_schema["anyOf"] = [transform_schema(cast("dict[str, Any]", variant)) for variant in any_of]
    elif is_list(one_of):
        # Convert oneOf to anyOf for compatibility (OpenAI supports anyOf but not oneOf in strict mode)
        strict_schema["anyOf"] = [transform_schema(cast("dict[str, Any]", variant)) for variant in one_of]
    elif is_list(all_of):
        # allOf is not supported in OpenAI strict mode
        raise ValueError(
            "allOf is not supported in OpenAI strict mode. "
            "Consider flattening the schema or using anyOf instead. "
            "See: https://platform.openai.com/docs/guides/structured-outputs"
        )
    else:
        if type_ is None:
            raise ValueError("Schema must have a 'type', 'anyOf', 'oneOf', or 'allOf' field.")

        strict_schema["type"] = type_

    description = json_schema.pop("description", None)
    if description is not None:
        strict_schema["description"] = description

    title = json_schema.pop("title", None)
    if title is not None:
        strict_schema["title"] = title

    if type_ == "object":
        strict_schema["properties"] = {
            key: transform_schema(prop_schema) for key, prop_schema in json_schema.pop("properties", {}).items()
        }
        json_schema.pop("additionalProperties", None)
        strict_schema["additionalProperties"] = False

        # MODIFICATION: Force all properties to be required
        # Original code: required = json_schema.pop("required", None)
        # This modification extracts all property keys and marks them as required
        properties = strict_schema.get("properties", {})
        if properties:
            strict_schema["required"] = list(properties.keys())

    elif type_ == "string":
        format = json_schema.pop("format", None)
        if format and format in SupportedStringFormats:
            strict_schema["format"] = format
        elif format:
            # add it back so its treated as an extra property and appended to the description
            json_schema["format"] = format
    elif type_ == "array":
        items = json_schema.pop("items", None)
        if items is not None:
            strict_schema["items"] = transform_schema(items)

        min_items = json_schema.pop("minItems", None)
        if min_items is not None and (min_items == 0 or min_items == 1):
            strict_schema["minItems"] = min_items
        elif min_items is not None:
            # add it back so its treated as an extra property and appended to the description
            json_schema["minItems"] = min_items

    elif type_ == "boolean" or type_ == "integer" or type_ == "number" or type_ == "null" or type_ is None:
        pass
    else:
        assert_never(type_)

    # if there are any props leftover then they aren't supported, so we add them to the description
    # so that the model *might* follow them.
    if json_schema:
        description = strict_schema.get("description")
        strict_schema["description"] = (
            (description + "\n\n" if description is not None else "")
            + "{"
            + ", ".join(f"{key}: {value}" for key, value in json_schema.items())
            + "}"
        )

    return strict_schema


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


def create_model_from_handler(name: str, handler: Callable[..., Any]) -> BaseModel:
    """Create a dynamic pydantic model from the argspec of a function."""
    argspec = inspect.getfullargspec(handler)

    # If the handler is a bound method, we need to skip the first argument (normally self or cls)
    is_bound = inspect.ismethod(handler)
    skip_args_idx = 1 if is_bound else 0

    fields = {}
    defaults = {}
    if argspec.defaults:
        defaults = dict(zip(argspec.args[-len(argspec.defaults) :], argspec.defaults, strict=True))
    for field_name in argspec.args[skip_args_idx:]:
        field_info = defaults.get(field_name, ...)  # Pydantic will use ... to mark this field as required.
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
