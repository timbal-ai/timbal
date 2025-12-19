from typing import Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class NotNullValidator(BaseValidator):
    """Not null validator - checks if value matches expected null/non-null state."""

    name: Literal["not_null!"] = "not_null!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "NotNullValidator":
        # Normalize value to boolean (default True if not provided)
        if self.value is None:
            self.value = True
        else:
            self.value = bool(self.value)
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value matches expected null/non-null state.

        If value is True: checks that value is not None
        If value is False: checks that value is None

        Raises:
            AssertionError: If value doesn't match expected null state.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target, self.path_key)

        if self.value:
            # Should be not null
            if actual_value is None:
                raise AssertionError(f"expected non-null value, got None")
        else:
            # Should be null
            if actual_value is not None:
                raise AssertionError(f"expected null value, got {actual_value!r}")
