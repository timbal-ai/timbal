from typing import Literal

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class ContainsValidator(BaseValidator):
    """Contains validator - checks if value contains substring/item."""

    name: Literal["contains!"] = "contains!"  # type: ignore

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value contains expected.

        Raises:
            AssertionError: If value doesn't contain expected.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        ref_value = self.value

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()
        if isinstance(actual_value, str):
            ref_value = str(self.value)

        if ref_value not in actual_value:
            raise AssertionError(f"expected {actual_value!r} to contain {self.value!r}")
