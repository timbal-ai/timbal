from typing import Any, Literal

from .comparison_base import ComparisonValidator


class GtValidator(ComparisonValidator):
    """Greater than validator - checks if value > expected."""

    name: Literal["gt!"] = "gt!"  # type: ignore

    @property
    def operator_symbol(self) -> str:
        return ">"

    def compare(self, actual: Any, expected: Any) -> bool:
        return actual > expected
