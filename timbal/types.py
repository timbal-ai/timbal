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

    If default is not provided, the field is required.
    If default is explicitly set to None, then the field is optional.
    """
    return Field(
        default,
        description = description,
        ge = ge,
        le = le,
        min_length = min_length,
        max_length = max_length,
        regex = regex,
        choices = choices,
    )
