from typing import Literal

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class EqValidator(BaseValidator):
    """Equality validator - checks if value equals expected.

    With negate=True, checks that value does NOT equal expected.
    """

    name: Literal["eq!"] = "eq!"  # type: ignore

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value equals expected.

        Raises:
            AssertionError: If values don't match (or match when negated).
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target, self.path_key)

        ref_value = self.value

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()
        if isinstance(actual_value, str):
            actual_value = self.apply_transform(actual_value)
            ref_value = self.apply_transform(str(self.value))

        equals = ref_value == actual_value

        if self.negate:
            if equals:
                raise AssertionError(f"expected value to not equal {ref_value!r}, got {actual_value!r}")
        else:
            if not equals:
                raise AssertionError(f"expected {ref_value!r}, got {actual_value!r}")
