"""
This script provides functions for creating and validating dynamic models using Pydantic.
It supports advanced use cases, such as handling generic types, validating field choices,
and dynamically generating models based on function argument specifications.

Usage:

1. Creating a dynamic model from a function argument specification:
    >>> import inspect
    >>> def my_function(a: int, b: str, c: List[int]):
    ...     pass
    >>> argspec = inspect.getfullargspec(my_function)
    >>> model = create_model_from_argspec('MyModel', argspec)
    >>> print(model.model_dump())

2. Generating OpenAPI schemas:
    >>> from typing import List, Annotated
    >>> schema, metadata = get_schema_from_annotation(List[int])
    >>> print(schema)  # {'type': 'array', 'items': {'type': 'integer'}}
"""
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

from annotated_types import Ge, Le, MaxLen, MinLen
from pydantic import (
    BaseModel,
    create_model,
    model_validator,
)
from pydantic.fields import FieldInfo

from ..state.context import RunContext
from . import Field, File, Message


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


def dump(value: Any, context: RunContext | None = None) -> Any:
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
            "content": [dump(c, context) for c in value.content],
        }
    elif isinstance(value, BaseModel): # Handle BaseModel instances as we handle dictionaries.
        return {k: dump(v, context) for k, v in value.__dict__.items()}
    elif isinstance(value, dict):
        return {k: dump(v, context) for k, v in value.items()}
    elif isinstance(value, (list, tuple)): # noqa: UP038
        return [dump(v, context) for v in value]
    elif isinstance(value, File):
        return File.serialize(value, context)
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


def create_choices_validator(choices: list[str | int | float]) -> Callable:
    """Factory function to create a Pydantic validator for a list of choices."""
    def validate_choice(value: str | int | float, _info) -> str | int | float:
        if value not in choices:
            raise ValueError(f"Choice {value} is not one of: {choices}.")
        return value
    return validate_choice


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
            field_default = Field(default=field_default)

        field_type = argspec.annotations.get(field_name, Any)

        # If base type is a generic, we need to make sure that all values passed as this generic are of the same type.
        field_base_type = get_base_type(field_type)
        if isinstance(field_base_type, TypeVar):
            generics[field_base_type].append(field_name)

        json_schema_extra = getattr(field_default, "json_schema_extra", None)
        if isinstance(json_schema_extra, dict) and "choices" in json_schema_extra:
            choices = json_schema_extra.pop("choices")
            field_type = Literal.__getitem__(tuple(choices))
            # choices_validator = create_choices_validator(choices)
            # validators[f"{field_name}_validator"] = field_validator(field_name)(choices_validator)

        fields[field_name] = (field_type, field_default)

    for generic, generic_fields_names in generics.items():
        generic_validator = create_generics_validator(generic_fields_names)
        validators[f"{generic}_validator"] = model_validator(mode="before")(generic_validator)
    
    model = create_model(name, __validators__=validators, **fields)
    return model


def issubclass_safe(candidate: Any, base_class: type[Any]) -> bool:
    """Helper function to avoid checking on annotations types always before calling issubclass."""
    if isinstance(candidate, type):
        return issubclass(candidate, base_class)
    else:
        return False


def create_model_from_annotation(name: str, annotation: type[Any] | None) -> BaseModel:
    """Generate an openapi compliant json schema from a function return annotation."""
    # If the type is already a BaseModel, simply return the schema.
    if issubclass_safe(annotation, BaseModel):
        return annotation

    if annotation is None:
        model = create_model(name, **{"return": (Literal[None], ...)})
    elif annotation is Ellipsis:
        model = create_model(name, **{"return": (Any, ...)})
    else:
        model = create_model(name, **{"return": (annotation, ...)})

    # If the annotation is Annotated, pydantic will store the metadata in the FieldInfos.
    
    return model


def create_model_from_fields(name: str, model_fields: dict[str, FieldInfo]) -> BaseModel:
    """Create a pydantic model from a dictionary of fields."""
    fields = {}
    validators = {}
    generics = defaultdict(list)

    for field_name, field_info in model_fields.items():
        field_type = field_info.annotation
        fields[field_name] = (field_type, field_info)

        # If base type is a generic, we need to make sure that all values passed as this generic are of the same type.
        field_base_type = get_base_type(field_type)
        if isinstance(field_base_type, TypeVar):
            generics[field_base_type].append(field_name)

        # No need to check for json schema extra choices here. Since we'll be getting fields from steps models.
    
    for generic, generic_fields_names in generics.items():
        generic_validator = create_generics_validator(generic_fields_names)
        validators[f"{generic}_validator"] = model_validator(mode="before")(generic_validator)
    
    model = create_model(name, __validators__=validators, **fields)
    return model


def merge_model_fields(*args: FieldInfo) -> FieldInfo:
    """Merge an arbitrary number of pydantic fields into a single one if possible.
    It enforces the most restrictive validators."""
    field_annotation = None
    field_metadata = []
    field_descriptions = []
    field_kwargs = {"default": None}

    for field_info in args:
        if field_annotation is None:
            field_annotation = field_info.annotation
        else:
            if field_annotation != field_info.annotation:
                raise ValueError("Cannot merge fields with different annotations.")
        # If any of the merged fields is required, the merged field is required.
        if field_info.is_required():
            field_kwargs.pop("default", None)
        # Collect all descriptions for later formatting.
        field_descriptions.append(field_info.description)
        # The remaining validators live inside metadata as common annotated types.
        for metadata in field_info.metadata:
            if hasattr(metadata, "pattern"):
                raise NotImplementedError("Regex merging is not implemented.")
            elif isinstance(metadata, Ge):
                current_ge = field_kwargs.get("ge", float("-inf"))
                current_le = field_kwargs.get("le", float("inf"))
                new_ge = metadata.ge
                if new_ge > current_le:
                    raise ValueError(f"Cannot merge fields with incompatible constraints: ge {new_ge} > le {current_le}.")
                field_kwargs["ge"] = max(current_ge, new_ge)
            elif isinstance(metadata, Le):
                current_le = field_kwargs.get("le", float("inf"))
                current_ge = field_kwargs.get("ge", float("-inf"))
                new_le = metadata.le
                if new_le < current_ge:
                    raise ValueError(f"Cannot merge fields with incompatible constraints: le {new_le} < ge {current_ge}.")
                field_kwargs["le"] = min(current_le, new_le)
            elif isinstance(metadata, MaxLen):
                current_max_length = field_kwargs.get("max_length", float("inf"))
                current_min_length = field_kwargs.get("min_length", 0)
                new_max_length = metadata.max_length
                if new_max_length < current_min_length:
                    raise ValueError(f"Cannot merge fields with incompatible constraints: max_length {new_max_length} < min_length {current_min_length}.")
                field_kwargs["max_length"] = min(current_max_length, new_max_length)
            elif isinstance(metadata, MinLen):
                current_min_length = field_kwargs.get("min_length", 0)
                current_max_length = field_kwargs.get("max_length", float("inf"))
                new_min_length = metadata.min_length
                if new_min_length > current_max_length:
                    raise ValueError(f"Cannot merge fields with incompatible constraints: min_length {new_min_length} > max_length {current_max_length}.")
                field_kwargs["min_length"] = max(current_min_length, new_min_length)
            else:
                field_metadata.append(metadata)

        # Merge choices with the intersection of the choices of all fields.
        if isinstance(field_info.json_schema_extra, dict):
            choices = field_info.json_schema_extra.get("choices", None)
            if choices is not None:
                if "choices" not in field_kwargs:
                    field_kwargs["choices"] = choices
                else:
                    current_choices = field_kwargs["choices"]
                    field_kwargs["choices"] = list(set(current_choices) & set(choices))

    # Check if any of these choices are in conflict with the other validators.
    if "choices" in field_kwargs:
        valid_choices = []
        ge = field_kwargs.get("ge", float("-inf"))
        le = field_kwargs.get("le", float("inf"))
        min_length = field_kwargs.get("min_length", 0)
        max_length = field_kwargs.get("max_length", float("inf"))
        for choice in field_kwargs["choices"]:
            if isinstance(choice, int | float):
                if ge <= choice <= le:
                    valid_choices.append(choice)
            elif isinstance(choice, str):
                if min_length <= len(choice) <= max_length:
                    valid_choices.append(choice)
            else:
                valid_choices.append(choice)
        field_kwargs["choices"] = valid_choices
        # After validating the choices, we can remove the remaining validators since they are less restrictive in nature.
        field_kwargs.pop("ge", None)
        field_kwargs.pop("le", None)
        field_kwargs.pop("min_length", None)
        field_kwargs.pop("max_length", None)

    # Merge descriptions.
    if len(field_descriptions):
        field_kwargs["description"] = "Merged field descriptions:"
        for field_description in field_descriptions:
            field_kwargs["description"] += f"\n- {field_description}"

    # Init field info, and manually add annotation and extra metadata.
    field_info = Field(**field_kwargs)
    field_info.annotation = field_annotation
    field_info.metadata.extend(field_metadata)
    return field_info
