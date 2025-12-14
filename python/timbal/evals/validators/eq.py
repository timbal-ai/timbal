from typing import Any, Literal

from .base import BaseValidator


class EqValidator(BaseValidator):
    """Equality validator - checks if value equals expected."""

    name: Literal["eq!"] = "eq!"

    async def __call__(self, ctx: Any) -> None:
        """Check if resolved value equals expected.

        Raises:
            AssertionError: If values don't match.
        """
        # TODO: resolve target from ctx
        actual = ctx.value

        if actual != self.value:
            raise AssertionError(f"expected {self.value!r}, got {actual!r}")
