from abc import abstractmethod
from datetime import datetime
from typing import Any, Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class ComparisonValidator(BaseValidator):
    """Base class for comparison validators (gt, gte, lt, lte).

    Subclasses should implement:
    - compare(actual, expected): Returns True if comparison passes
    - operator_symbol: The symbol for error messages (e.g., ">", ">=")
    """

    name: Literal["comparison!"] = "comparison!"  # type: ignore

    @property
    @abstractmethod
    def operator_symbol(self) -> str:
        """Return the operator symbol for error messages (e.g., '>', '>=')."""
        ...

    @abstractmethod
    def compare(self, actual: Any, expected: Any) -> bool:
        """Perform the comparison.

        Args:
            actual: The actual value from the trace
            expected: The expected value from the validator

        Returns:
            True if comparison passes, False otherwise.
        """
        ...

    @model_validator(mode="after")
    def validate_value(self) -> "ComparisonValidator":
        # Ensure value is comparable (int, float, or date string)
        if not isinstance(self.value, (int, float, str)):
            raise ValueError(f"expected int, float, or date string, got {type(self.value).__name__}")
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value satisfies the comparison.

        Supports numeric comparisons and date string comparisons.

        Raises:
            AssertionError: If comparison fails.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target, self.path_key)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        # Try to parse as dates if both are strings
        if isinstance(actual_value, str) and isinstance(self.value, str):
            try:
                actual_dt = datetime.fromisoformat(actual_value.replace("Z", "+00:00"))
                expected_dt = datetime.fromisoformat(self.value.replace("Z", "+00:00"))
                if not self.compare(actual_dt, expected_dt):
                    raise AssertionError(f"expected {actual_value!r} {self.operator_symbol} {self.value!r}")
                return
            except (ValueError, AttributeError):
                # Not valid date strings, fall through to numeric/string comparison
                pass

        # Numeric or string comparison
        try:
            if not self.compare(actual_value, self.value):
                raise AssertionError(f"expected {actual_value!r} {self.operator_symbol} {self.value!r}")
        except TypeError as e:
            raise AssertionError(
                f"cannot compare {type(actual_value).__name__} with {type(self.value).__name__}"
            ) from e
