from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from .context import ValidationContext


class BaseValidator(ABC, BaseModel):
    """Base class for all validators.

    Validators are callable Pydantic models that check conditions against a ValidationContext.
    Uses Pydantic discriminated unions with `name` as the discriminator.

    Attributes:
        name: The validator keyword (e.g., "eq!", "contains!"). Used as discriminator.
        target: Dot-separated path specifying what to validate within the context.
                Examples:
                    - "input.query" -> look at span.input.query
                    - "output.items.0" -> look at first item in output
                    - None -> validate the current context value directly
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str  # Discriminator field - subclasses use Literal["eq!"], etc.
    target: str
    value: Any = None  # The value passed to the validator (e.g., "foo" for eq!: "foo")

    @abstractmethod
    async def __call__(self, ctx: ValidationContext) -> None:
        """Execute the validation.

        Args:
            ctx: ValidationContext containing the trace and current state.

        Raises:
            AssertionError: If validation fails.
        """
        ...
