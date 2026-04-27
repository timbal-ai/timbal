import inspect
import typing
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


def create_model_from_handler(name: str, handler: Any) -> BaseModel:
    """Create a dynamic pydantic model from the argspec of a function."""
    argspec = inspect.getfullargspec(handler)
    try:
        globalns = {**vars(typing), **getattr(handler, "__globals__", {})}
        annotations = typing.get_type_hints(handler, globalns=globalns, include_extras=True)
    except Exception:
        annotations = argspec.annotations

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
        field_type = annotations.get(field_name, Any)
        fields[field_name] = (field_type, field_info)
    extra_mode = "allow" if argspec.varkw else "ignore"
    return create_model(name, __config__=ConfigDict(extra=extra_mode), **fields)


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
