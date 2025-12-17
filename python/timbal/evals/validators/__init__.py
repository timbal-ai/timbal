from typing import Annotated

from pydantic import Discriminator, TypeAdapter

from .base import BaseValidator
from .comparison import ComparisonValidator
from .contains import ContainsValidator
from .email import EmailValidator
from .ends_with import EndsWithValidator
from .eq import EqValidator
from .gt import GtValidator
from .gte import GteValidator
from .json import JsonValidator
from .language import LanguageValidator
from .length import LengthValidator
from .lt import LtValidator
from .lte import LteValidator
from .max_length import MaxLengthValidator
from .min_length import MinLengthValidator
from .not_contains import NotContainsValidator
from .not_null import NotNullValidator
from .parallel import ParallelValidator
from .pattern import PatternValidator
from .semantic import SemanticValidator
from .seq import SeqValidator
from .starts_with import StartsWithValidator
from .type import TypeValidator

# Discriminated union of all validators
Validator = Annotated[
    EqValidator
    | ComparisonValidator
    | ContainsValidator
    | NotContainsValidator
    | PatternValidator
    | TypeValidator
    | StartsWithValidator
    | EndsWithValidator
    | LengthValidator
    | MinLengthValidator
    | MaxLengthValidator
    | NotNullValidator
    | LtValidator
    | LteValidator
    | GtValidator
    | GteValidator
    | SemanticValidator
    | LanguageValidator
    | JsonValidator
    | EmailValidator
    | SeqValidator
    | ParallelValidator,
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
