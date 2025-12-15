from typing import Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class MinLengthValidator(BaseValidator):
    """Min length validator - checks if value has minimum length."""

    name: Literal["min_length!"] = "min_length!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "MinLengthValidator":
        if not isinstance(self.value, int):
            raise ValueError(f"expected integer length, got {type(self.value).__name__}")
        if self.value < 0:
            raise ValueError(f"expected non-negative length, got {self.value}")
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value has minimum length.

        Works with strings, lists, dicts, and any object implementing len().

        Raises:
            AssertionError: If length is less than expected minimum.
            TypeError: If value doesn't support len().
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        # Handle Message objects
        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        try:
            actual_length = len(actual_value)
        except TypeError as e:
            raise AssertionError(f"value of type {type(actual_value).__name__!r} doesn't support len()") from e

        if actual_length < self.value:
            raise AssertionError(f"expected length >= {self.value}, got length {actual_length}")
