from typing import Literal

from pydantic import model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class StartsWithValidator(BaseValidator):
    """StartsWith validator - checks if value starts with a prefix.

    With negate=True, checks that value does NOT start with the prefix.
    """

    name: Literal["starts_with!"] = "starts_with!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "StartsWithValidator":
        if not isinstance(self.value, str):
            raise ValueError(f"expected string prefix, got {type(self.value).__name__}")
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value starts with expected prefix.

        Raises:
            AssertionError: If value doesn't start with the prefix (or starts with when negated).
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target, self.path_key)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        if not isinstance(actual_value, str):
            raise AssertionError(f"expected string value, got {type(actual_value).__name__}")

        actual_value = self.apply_transform(actual_value)
        ref_value = self.apply_transform(self.value)

        starts_with = actual_value.startswith(ref_value)

        if self.negate:
            if starts_with:
                raise AssertionError(f"expected {actual_value!r} to not start with {self.value!r}")
        else:
            if not starts_with:
                raise AssertionError(f"expected {actual_value!r} to start with {self.value!r}")
