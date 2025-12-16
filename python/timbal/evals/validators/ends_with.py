from typing import Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class EndsWithValidator(BaseValidator):
    """EndsWith validator - checks if value ends with a suffix."""

    name: Literal["ends_with!"] = "ends_with!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "EndsWithValidator":
        if not isinstance(self.value, str):
            raise ValueError(f"expected string suffix, got {type(self.value).__name__}")
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value ends with expected suffix.

        Raises:
            AssertionError: If value doesn't end with the suffix.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        if not isinstance(actual_value, str):
            raise AssertionError(f"expected string value, got {type(actual_value).__name__}")

        if not actual_value.endswith(self.value):
            raise AssertionError(f"expected {actual_value!r} to end with {self.value!r}")
