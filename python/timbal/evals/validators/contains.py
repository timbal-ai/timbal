from typing import Any, Literal

from .base import BaseValidator


class ContainsValidator(BaseValidator):
    """Contains validator - checks if value contains substring/item."""

    name: Literal["contains!"] = "contains!"

    async def __call__(self, ctx: Any) -> None:
        """Check if resolved value contains expected.

        Raises:
            AssertionError: If value doesn't contain expected.
        """
        # TODO: resolve target from ctx
        actual = ctx.value

        if self.value not in actual:
            raise AssertionError(f"expected {actual!r} to contain {self.value!r}")
