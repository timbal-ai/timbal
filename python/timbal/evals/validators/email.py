from typing import Literal

from pydantic import BaseModel, EmailStr, ValidationError, model_validator

from .base import BaseValidator
from .context import ValidationContext


class _EmailModel(BaseModel):
    """Helper model for email validation."""

    email: EmailStr


class EmailValidator(BaseValidator):
    """Email validator - checks if value is a valid email address using Pydantic."""

    name: Literal["email!"] = "email!"  # type: ignore

    @model_validator(mode="after")
    def validate_value(self) -> "EmailValidator":
        self._normalize_bool_value()
        return self

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if resolved value is a valid email address.

        Uses Pydantic's EmailStr for validation.

        Raises:
            AssertionError: If email validity doesn't match expected state.
        """
        actual_value = self._resolve_expected_str(ctx, "email")
        if actual_value is None:
            # Non-string is not an email, which is what we wanted
            return

        actual_value = actual_value.strip()

        # Use Pydantic model to validate email
        try:
            _EmailModel(email=actual_value)
            is_valid_email = True
        except ValidationError:
            is_valid_email = False

        if self.value:
            # Should be valid email
            if not is_valid_email:
                raise AssertionError(f"expected valid email address, got {actual_value!r}")
        else:
            # Should NOT be valid email
            if is_valid_email:
                raise AssertionError(f"expected invalid email address, got valid email {actual_value!r}")
