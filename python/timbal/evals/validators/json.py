from typing import Literal

from pydantic import model_validator

from .base import BaseValidator
from .context import ValidationContext


class JsonValidator(BaseValidator):
    """JSON validator - checks if value is valid JSON string."""

    name: Literal["json!"] = "json!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "JsonValidator":
        self._normalize_bool_value()
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value is valid JSON.

        Uses the normalized boolean value to determine expectation.

        Raises:
            AssertionError: If JSON validity doesn't match expected state.
        """
        actual_value = self._resolve_expected_str(ctx, "JSON")
        if actual_value is None:
            # Non-string is not JSON, which is what we wanted
            return

        # Strip markdown code blocks (```json ... ``` or ``` ... ```)
        import re

        actual_value = actual_value.strip()
        # Remove markdown code fences with optional language identifier
        actual_value = re.sub(r"^```(?:json)?\s*\n?", "", actual_value)
        actual_value = re.sub(r"\n?```\s*$", "", actual_value)
        actual_value = actual_value.strip()

        # Try to parse as JSON
        import json

        try:
            json.loads(actual_value)
            is_valid_json = True
        except (json.JSONDecodeError, ValueError):
            is_valid_json = False

        if self.value:
            # Should be valid JSON
            if not is_valid_json:
                raise AssertionError(f"expected valid JSON string, got invalid JSON: {actual_value}")
        else:
            # Should NOT be valid JSON
            if is_valid_json:
                raise AssertionError(f"expected invalid JSON string, got valid JSON: {actual_value}")
