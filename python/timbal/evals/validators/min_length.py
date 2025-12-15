from typing import Literal

from .length import LengthValidator


class MinLengthValidator(LengthValidator):
    """Min length validator - checks if value has minimum length."""

    name: Literal["min_length!"] = "min_length!"  # type: ignore

    def check_length(self, actual_length: int, expected_length: int) -> bool:
        return actual_length >= expected_length

    def get_error_message(self, actual_length: int, expected_length: int) -> str:
        return f"expected length >= {expected_length}, got length {actual_length}"
