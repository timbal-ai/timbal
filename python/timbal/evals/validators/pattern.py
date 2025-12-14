import re
from typing import Literal

from pydantic import PrivateAttr, model_validator

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class PatternValidator(BaseValidator):
    """Pattern validator - checks if value matches a regex pattern."""

    name: Literal["pattern!"] = "pattern!"  # type: ignore

    _compiled: re.Pattern[str] = PrivateAttr()

    @model_validator(mode="after")
    def compile_pattern(self) -> "PatternValidator":
        if not isinstance(self.value, str):
            raise ValueError(f"expected string pattern, got {type(self.value).__name__}")
        self._compiled = re.compile(self.value)
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value matches the regex pattern.

        Raises:
            AssertionError: If value doesn't match the pattern.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        if not isinstance(actual_value, str):
            raise AssertionError(f"expected string value, got {type(actual_value).__name__}")

        if not self._compiled.search(actual_value):
            raise AssertionError(f"expected {actual_value!r} to match pattern {self.value!r}")
