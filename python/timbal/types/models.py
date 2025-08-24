import asyncio
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import (
    Any,
    Literal,
    NamedTuple,
    TypeVar,
    get_args,
    get_origin,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
)
from pydantic.fields import FieldInfo

from .file import File
from .message import Message


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


def get_base_type(annotation: type[Any]) -> type[Any]:
    """Function that returns the base type of a type with an arbitrary number of layers
    (e.g. Annotation[List[List[T]]] -> T)."""
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = get_args(annotation)
    if args:
        return get_base_type(args[0])
    return origin


def create_generics_validator(fields_names: list[str]) -> Callable:
    """Factory function to create a Pydantic model validator that ensures all 
    model generics (base types) are of the same type."""
    def validate_generics(fields_values: dict[str, Any], _info) -> dict[str, Any]:
        field_generic_type = None

        for field_name in fields_names:
            if field_name not in fields_values:
                continue

            field_value = fields_values[field_name]
            field_value_type = type(field_value)

            if isinstance(field_value, list):
                # Skip validation for empty lists
                if len(field_value) == 0:
                    continue
                field_value_type = type(field_value[0])
                if not all(isinstance(field_value_i, field_value_type) for field_value_i in field_value):
                    raise ValueError(f"All elements in the list '{field_name}' must be of the same base type!")

            if field_generic_type is None:
                field_generic_type = field_value_type
            elif field_generic_type != field_value_type:
                raise ValueError(f"Fields {fields_names} must have the same base types!")

        return fields_values

    return validate_generics


def create_model_from_argspec(name: str, argspec: NamedTuple) -> BaseModel:
    """Create a dynamic pydantic model from the argspec of a function."""
    fields = {}
    validators = {}
    generics = defaultdict(list)

    defaults = {}
    if argspec.defaults:
        defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults, strict=True))

    for field_name in argspec.args:
        field_default = defaults.get(field_name, ...) # Pydantic will use ... to mark this field as required.
        if not isinstance(field_default, FieldInfo):
            field_default = Field(field_default)

        json_schema_extra = getattr(field_default, "json_schema_extra", None) or {}
        if json_schema_extra.get("private", False):
            continue

        field_type = argspec.annotations.get(field_name, Any)

        # If base type is a generic, we need to make sure that all values passed as this generic are of the same type.
        field_base_type = get_base_type(field_type)
        if isinstance(field_base_type, TypeVar):
            generics[field_base_type].append(field_name)

        if "choices" in json_schema_extra:
            choices = json_schema_extra.pop("choices")
            field_type = Literal.__getitem__(tuple(choices))

        fields[field_name] = (field_type, field_default)

    for generic, generic_fields_names in generics.items():
        generic_validator = create_generics_validator(generic_fields_names)
        validators[f"{generic}_validator"] = model_validator(mode="before")(generic_validator)
    
    config = ConfigDict(extra="ignore")
    model = create_model(name, __config__=config, __validators__=validators, **fields)
    return model


def issubclass_safe(candidate: Any, base_class: type[Any]) -> bool:
    """Helper function to avoid checking on annotations types always before calling issubclass."""
    if isinstance(candidate, type):
        return issubclass(candidate, base_class)
    else:
        return False
