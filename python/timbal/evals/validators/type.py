from typing import Literal

from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext

# Mapping of type names to Python types
TYPE_MAP = {
    "string": str,
    "str": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": (int, float),
    "bool": bool,
    "boolean": bool,
    "list": list,
    "array": list,
    "dict": dict,
    "object": dict,
    "none": type(None),
    "null": type(None),
}


class TypeValidator(BaseValidator):
    """Type validator - checks if value is of the expected type.

    With negate=True, checks that value is NOT of the expected type.
    """

    name: Literal["type!"] = "type!"  # type: ignore

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value is of the expected type.

        Raises:
            AssertionError: If value is not of the expected type (or is when negated).
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        # Treat Message as string
        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        expected_type_name = self.value
        if not isinstance(expected_type_name, str):
            raise AssertionError(f"expected type name as string, got {type(expected_type_name).__name__}")

        expected_type_name = expected_type_name.lower()
        if expected_type_name not in TYPE_MAP:
            raise AssertionError(f"unknown type {self.value!r}, valid types: {', '.join(TYPE_MAP.keys())}")

        expected_type = TYPE_MAP[expected_type_name]
        is_expected_type = isinstance(actual_value, expected_type)
        actual_type_name = type(actual_value).__name__

        if self.negate:
            if is_expected_type:
                raise AssertionError(f"expected type to not be {self.value!r}, got {actual_type_name!r}")
        else:
            if not is_expected_type:
                raise AssertionError(f"expected type {self.value!r}, got {actual_type_name!r}")
