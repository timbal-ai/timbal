from typing import Literal

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class EqValidator(BaseValidator):
    """Equality validator - checks if value equals expected."""

    name: Literal["eq!"] = "eq!"  # type: ignore

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value equals expected.

        Raises:
            AssertionError: If values don't match.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        ref_value = self.value

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()
        if isinstance(actual_value, str):
            ref_value = str(self.value)

        if ref_value != actual_value:
            raise AssertionError(f"expected {ref_value!r}, got {actual_value!r}")
