from typing import Literal

from dateutil import parser as date_parser
from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class GtValidator(BaseValidator):
    """Greater than validator - checks if value > expected."""

    name: Literal["gt!"] = "gt!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "GtValidator":
        # Ensure value is comparable (int, float, or date string)
        if not isinstance(self.value, (int, float, str)):
            raise ValueError(f"expected int, float, or date string, got {type(self.value).__name__}")
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value is greater than expected.

        Supports numeric comparisons and date string comparisons.

        Raises:
            AssertionError: If value is less than or equal to expected.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        # Try to parse as dates if both are strings
        if isinstance(actual_value, str) and isinstance(self.value, str):
            try:
                actual_dt = date_parser.parse(actual_value)
                expected_dt = date_parser.parse(self.value)
                if not (actual_dt > expected_dt):
                    raise AssertionError(f"expected {actual_value!r} > {self.value!r}")
                return
            except (ValueError, TypeError):
                # Not valid date strings, fall through to string comparison
                pass

        # Numeric or string comparison
        try:
            if not (actual_value > self.value):
                raise AssertionError(f"expected {actual_value!r} > {self.value!r}")
        except TypeError as e:
            raise AssertionError(
                f"cannot compare {type(actual_value).__name__} with {type(self.value).__name__}"
            ) from e
