from typing import Any, Literal

from .comparison_base import ComparisonValidator


class GteValidator(ComparisonValidator):
    """Greater than or equal validator - checks if value >= expected."""

    name: Literal["gte!"] = "gte!"  # type: ignore

    @property
    def operator_symbol(self) -> str:
        return ">="

    def compare(self, actual: Any, expected: Any) -> bool:
        return actual >= expected
