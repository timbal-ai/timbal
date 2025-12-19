from typing import Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class LengthValidator(BaseValidator):
    """Length validator - checks if value has exact length.

    Can be subclassed to create min_length and max_length variants.
    """

    name: Literal["length!"] = "length!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self):
        if not isinstance(self.value, int):
            raise ValueError(f"expected integer length, got {type(self.value).__name__}")
        if self.value < 0:
            raise ValueError(f"expected non-negative length, got {self.value}")
        return self

    def check_length(self, actual_length: int, expected_length: int) -> bool:
        """Check if length satisfies the constraint.

        Override in subclasses for different comparisons.
        Default: exact match.
        """
        return actual_length == expected_length

    def get_error_message(self, actual_length: int, expected_length: int) -> str:
        """Generate error message for failed validation.

        Override in subclasses for different messages.
        """
        return f"expected length {expected_length}, got length {actual_length}"

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value satisfies the length constraint.

        Works with strings, lists, dicts, and any object implementing len().

        Raises:
            AssertionError: If length check fails.
            TypeError: If value doesn't support len().
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target, self.path_key)

        # Handle Message objects
        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        try:
            actual_length = len(actual_value)
        except TypeError as e:
            raise AssertionError(f"value of type {type(actual_value).__name__!r} doesn't support len()") from e

        if not self.check_length(actual_length, self.value):
            raise AssertionError(self.get_error_message(actual_length, self.value))
