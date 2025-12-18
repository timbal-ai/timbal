from typing import Annotated

from pydantic import Discriminator, TypeAdapter

from .base import BaseValidator
from .contains import ContainsValidator
from .contains_all import ContainsAllValidator
from .contains_any import ContainsAnyValidator
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
    | ContainsValidator
    | ContainsAllValidator
    | ContainsAnyValidator
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

# Aliases that map to existing validators with negate=True
# Format: alias -> (real_validator_name, negate)
VALIDATOR_ALIASES: dict[str, tuple[str, bool]] = {
    # ne! as alias for eq! with negate
    "ne!": ("eq!", True),
    # not_* aliases
    "not_contains!": ("contains!", True),
    "not_contains_all!": ("contains_all!", True),
    "not_contains_any!": ("contains_any!", True),
    "not_starts_with!": ("starts_with!", True),
    "not_ends_with!": ("ends_with!", True),
    "not_pattern!": ("pattern!", True),
    "not_type!": ("type!", True),
    "not_semantic!": ("semantic!", True),
    "not_language!": ("language!", True),
}


def parse_validator(data: dict) -> BaseValidator:
    """Parse a dict into a validator instance.

    Args:
        data: Dict with 'name', 'target', and validator-specific fields.

    Returns:
        Parsed validator instance.

    Raises:
        ValueError: If validator name is unknown or data is invalid.
    """
    name = data.get("name")

    # Check if this is an alias
    if name in VALIDATOR_ALIASES:
        real_name, negate = VALIDATOR_ALIASES[name]
        data = {**data, "name": real_name, "negate": negate}

    return ValidatorAdapter.validate_python(data)
