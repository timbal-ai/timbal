from typing import Literal

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class ContainsAnyValidator(BaseValidator):
    """Contains any validator - checks if value contains at least one of the specified items.

    With negate=True, checks that value contains NONE of the specified items.
    """

    name: Literal["contains_any!"] = "contains_any!"  # type: ignore

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value contains at least one of the expected items.

        Raises:
            AssertionError: If value doesn't contain any of the expected items (or contains any when negated).
            ValueError: If value is not a list.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        if not isinstance(self.value, list):
            raise ValueError(f"contains_any! expects a list, got {type(self.value).__name__}")

        is_string = isinstance(actual_value, str)
        if is_string:
            actual_value = self.apply_transform(actual_value)

        found = []
        for item in self.value:
            ref_value = self.apply_transform(str(item)) if is_string else item
            if ref_value in actual_value:
                found.append(item)

        if self.negate:
            if found:
                raise AssertionError(
                    f"expected {actual_value!r} to contain none of {self.value!r}, but found {found!r}"
                )
        else:
            if not found:
                raise AssertionError(f"expected {actual_value!r} to contain at least one of {self.value!r}")
