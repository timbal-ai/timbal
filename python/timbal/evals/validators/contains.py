from typing import Literal

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class ContainsValidator(BaseValidator):
    """Contains validator - checks if value contains substring/item.

    With negate=True, checks that value does NOT contain the substring/item.
    """

    name: Literal["contains!"] = "contains!"  # type: ignore

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value contains expected.

        Raises:
            AssertionError: If value doesn't contain expected (or contains when negated).
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        ref_value = self.value

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()
        if isinstance(actual_value, str):
            actual_value = self.apply_transform(actual_value)
            ref_value = self.apply_transform(str(self.value))

        contains = ref_value in actual_value

        if self.negate:
            if contains:
                raise AssertionError(f"expected {actual_value!r} to not contain {self.value!r}")
        else:
            if not contains:
                raise AssertionError(f"expected {actual_value!r} to contain {self.value!r}")
