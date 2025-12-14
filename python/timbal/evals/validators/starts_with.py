from typing import Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class StartsWithValidator(BaseValidator):
    """StartsWith validator - checks if value starts with a prefix."""

    name: Literal["starts_with!"] = "starts_with!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "StartsWithValidator":
        if not isinstance(self.value, str):
            raise ValueError(f"expected string prefix, got {type(self.value).__name__}")
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value starts with expected prefix.

        Raises:
            AssertionError: If value doesn't start with the prefix.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        if not isinstance(actual_value, str):
            raise AssertionError(f"expected string value, got {type(actual_value).__name__}")

        if not actual_value.startswith(self.value):
            raise AssertionError(f"expected {actual_value!r} to start with {self.value!r}")
