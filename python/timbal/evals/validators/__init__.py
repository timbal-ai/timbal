from typing import Annotated

from pydantic import Discriminator, TypeAdapter

from .base import BaseValidator
from .contains import ContainsValidator
from .eq import EqValidator
from .pattern import PatternValidator

# Discriminated union of all validators
Validator = Annotated[
    EqValidator | ContainsValidator | PatternValidator,
    Discriminator("name"),
]

# TypeAdapter for parsing validators from dicts
ValidatorAdapter = TypeAdapter(Validator)


def parse_validator(data: dict) -> BaseValidator:
    """Parse a dict into a validator instance.

    Args:
        data: Dict with 'name', 'target', and validator-specific fields.

    Returns:
        Parsed validator instance.

    Raises:
        ValueError: If validator name is unknown or data is invalid.
    """
    return ValidatorAdapter.validate_python(data)
