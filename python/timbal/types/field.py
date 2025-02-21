"""Field wrapper around Pydantic's Field, adapted for Replicate's Cog field specification.
Enables model parameter definitions with validation rules including required/optional fields,
numeric bounds, string constraints, regex patterns, and enumerated choices.

See Also:
    - Pydantic Field: https://docs.pydantic.dev/latest/api/fields/
    - Cog Specification: https://github.com/replicate/cog/blob/main/python/cog/types.py
"""

from typing import Any

from pydantic import Field as PydanticField


def Field(
    default: Any = ...,
    description: str = None,
    ge: float = None,
    le: float = None,
    min_length: int = None,
    max_length: int = None,
    regex: str = None,
    choices: list[str | int | float] = None,
) -> Any:
    """Create a field definition with validation rules for Timbal models.

    Args:
        default: Default value for the field. Leave blank for required fields.
        description: Human-readable description of the field.
        ge: Greater than or equal to (minimum value).
        le: Less than or equal to (maximum value).
        min_length: Minimum length for strings/sequences.
        max_length: Maximum length for strings/sequences.
        regex: Regular expression pattern for string validation.
        choices: List of valid values for this field.

    Returns:
        A Pydantic Field instance with the specified validation rules.

    Note:
        - Required fields: Omit default.
        - Optional fields: Set default=None explicitly.
        - The choices parameter extends Pydantic's functionality to match Cog's spec.
    """
    field_info = {
        "description": description,
        "ge": ge,
        "le": le,
        "min_length": min_length,
        "max_length": max_length,
        "pattern": regex,
        "json_schema_extra": {},
    }

    # The `choices` parameter is deprecated in Pydantic v2.
    # Instead, the user should use `Literal[...]` to specify the valid values.
    if choices is not None:
        field_info["json_schema_extra"]["choices"] = choices

    return PydanticField(default, **field_info)
