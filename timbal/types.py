from pydantic import Field
from typing import Any, List, Union


def Input(
    default: Any = ...,
    description: str = None,
    ge: float = None,
    le: float = None,
    min_length: int = None,
    max_length: int = None,
    regex: str = None,
    choices: List[Union[str, int]] = None,
) -> Any:
    """
    Input is similar to pydantic.Field, but doesn't require a default value to be the first argument.
    Original source: https://github.com/replicate/cog/blob/main/python/cog/types.py
    Parameters are kept the same to maintain compatibility with cog.

    If default is not provided, the field is required.
    If default is explicitly set to None, then the field is optional.
    """
    field_info = {
        "description": description,
        "ge": ge,
        "le": le,
        "min_length": min_length,
        "max_length": max_length,
        "pattern": regex,
    }
    # Choices is not implemented in pydantic Field. This will be added in json_schema_extra.
    if choices is not None:
        field_info["choices"] = choices
    return Field(default, **field_info)
