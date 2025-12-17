from typing import Any, Literal

from .comparison import ComparisonValidator


class LteValidator(ComparisonValidator):
    """Less than or equal validator - checks if value <= expected."""

    name: Literal["lte!"] = "lte!"  # type: ignore

    @property
    def operator_symbol(self) -> str:
        return "<="

    def compare(self, actual: Any, expected: Any) -> bool:
        return actual <= expected
