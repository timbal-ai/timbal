from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseValidator(ABC, BaseModel):
    """Base class for all validators.

    Validators are callable Pydantic models that check conditions against a ValidationContext.

    Attributes:
        target: Dot-separated path specifying what to validate within the context.
                Examples:
                    - "input.query" -> look at span.input.query
                    - "output.items.0" -> look at first item in output
                    - None -> validate the current context value directly
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target: str | None = None

    @abstractmethod
    async def __call__(self, ctx: Any) -> Any:
        """Execute the validation.

        Args:
            ctx: ValidationContext containing the trace and current state.

        Returns:
            ValidationResult indicating pass/fail with optional message.
        """
        ...
