from typing import Literal

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class NotContainsValidator(BaseValidator):
    """Not contains validator - checks if value does not contain substring/item."""

    name: Literal["not_contains!"] = "not_contains!"  # type: ignore

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value does not contain expected.

        Raises:
            AssertionError: If value contains expected.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        ref_value = self.value

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()
        if isinstance(actual_value, str):
            ref_value = str(self.value)

        if ref_value in actual_value:
            raise AssertionError(f"expected {actual_value!r} to not contain {self.value!r}")
