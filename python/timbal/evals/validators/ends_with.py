from typing import Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class EndsWithValidator(BaseValidator):
    """EndsWith validator - checks if value ends with a suffix.

    With negate=True, checks that value does NOT end with the suffix.
    """

    name: Literal["ends_with!"] = "ends_with!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "EndsWithValidator":
        if not isinstance(self.value, str):
            raise ValueError(f"expected string suffix, got {type(self.value).__name__}")
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value ends with expected suffix.

        Raises:
            AssertionError: If value doesn't end with the suffix (or ends with when negated).
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target, self.path_key)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        if not isinstance(actual_value, str):
            raise AssertionError(f"expected string value, got {type(actual_value).__name__}")

        actual_value = self.apply_transform(actual_value)
        ref_value = self.apply_transform(self.value)

        ends_with = actual_value.endswith(ref_value)

        if self.negate:
            if ends_with:
                raise AssertionError(f"expected {actual_value!r} to not end with {self.value!r}")
        else:
            if not ends_with:
                raise AssertionError(f"expected {actual_value!r} to end with {self.value!r}")
