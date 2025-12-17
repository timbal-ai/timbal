from typing import Any, Literal

from .comparison import ComparisonValidator


class LtValidator(ComparisonValidator):
    """Less than validator - checks if value < expected."""

    name: Literal["lt!"] = "lt!"  # type: ignore

    @property
    def operator_symbol(self) -> str:
        return "<"

    def compare(self, actual: Any, expected: Any) -> bool:
        return actual < expected
