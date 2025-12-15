from typing import Annotated

from pydantic import Discriminator, TypeAdapter

from .base import BaseValidator
from .contains import ContainsValidator
from .ends_with import EndsWithValidator
from .eq import EqValidator
from .gt import GtValidator
from .gte import GteValidator
from .lt import LtValidator
from .lte import LteValidator
from .max_length import MaxLengthValidator
from .min_length import MinLengthValidator
from .not_null import NotNullValidator
from .pattern import PatternValidator
from .starts_with import StartsWithValidator
from .type import TypeValidator

# Discriminated union of all validators
Validator = Annotated[
    EqValidator
    | ContainsValidator
    | PatternValidator
    | TypeValidator
    | StartsWithValidator
    | EndsWithValidator
    | MinLengthValidator
    | MaxLengthValidator
    | NotNullValidator
    | LtValidator
    | LteValidator
    | GtValidator
    | GteValidator,
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
