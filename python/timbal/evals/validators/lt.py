from datetime import datetime
from typing import Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class LtValidator(BaseValidator):
    """Less than validator - checks if value < expected."""

    name: Literal["lt!"] = "lt!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "LtValidator":
        # Ensure value is comparable (int, float, or date string)
        if not isinstance(self.value, (int, float, str)):
            raise ValueError(f"expected int, float, or date string, got {type(self.value).__name__}")
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value is less than expected.

        Supports numeric comparisons and date string comparisons.

        Raises:
            AssertionError: If value is greater than or equal to expected.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        # Try to parse as dates if both are strings
        if isinstance(actual_value, str) and isinstance(self.value, str):
            try:
                actual_dt = datetime.fromisoformat(actual_value.replace("Z", "+00:00"))
                expected_dt = datetime.fromisoformat(self.value.replace("Z", "+00:00"))
                if not (actual_dt < expected_dt):
                    raise AssertionError(f"expected {actual_value!r} < {self.value!r}")
                return
            except (ValueError, AttributeError):
                # Not valid date strings, fall through to string comparison
                pass

        # Numeric or string comparison
        try:
            if not (actual_value < self.value):
                raise AssertionError(f"expected {actual_value!r} < {self.value!r}")
        except TypeError as e:
            raise AssertionError(
                f"cannot compare {type(actual_value).__name__} with {type(self.value).__name__}"
            ) from e
