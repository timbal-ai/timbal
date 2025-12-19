from typing import Literal

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class ContainsAllValidator(BaseValidator):
    """Contains all validator - checks if value contains all of the specified items.

    With negate=True, checks that value does NOT contain all of the specified items
    (i.e., at least one item is missing).
    """

    name: Literal["contains_all!"] = "contains_all!"  # type: ignore

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value contains all of the expected items.

        Raises:
            AssertionError: If value doesn't contain all expected items (or contains all when negated).
            ValueError: If value is not a list.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target, self.path_key)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        if not isinstance(self.value, list):
            raise ValueError(f"contains_all! expects a list, got {type(self.value).__name__}")

        is_string = isinstance(actual_value, str)
        if is_string:
            actual_value = self.apply_transform(actual_value)

        missing = []
        for item in self.value:
            ref_value = self.apply_transform(str(item)) if is_string else item
            if ref_value not in actual_value:
                missing.append(item)

        if self.negate:
            if not missing:
                raise AssertionError(
                    f"expected {actual_value!r} to not contain all of {self.value!r}, but all were found"
                )
        else:
            if missing:
                raise AssertionError(f"expected {actual_value!r} to contain all of {self.value!r}, missing {missing!r}")
