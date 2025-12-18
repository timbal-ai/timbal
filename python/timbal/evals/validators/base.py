import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from .context import ValidationContext

# Supported transforms
TRANSFORMS = frozenset(
    [
        "lowercase",
        "uppercase",
        "trim",
        "collapse_whitespace",
    ]
)


def apply_transforms(value: str, transforms: list[str]) -> str:
    """Apply a list of transforms to a string value.

    Args:
        value: The string to transform.
        transforms: List of transform names to apply in order.

    Returns:
        The transformed string.
    """
    for t in transforms:
        match t:
            case "lowercase":
                value = value.lower()
            case "uppercase":
                value = value.upper()
            case "trim":
                value = value.strip()
            case "collapse_whitespace":
                value = re.sub(r"\s+", " ", value)
            case _:
                raise ValueError(f"unknown transform: {t}")
    return value


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
        value: The value to compare against.
        transform: Optional transform(s) to apply to the actual value before comparison.
                   Can be a single string or a list of strings.
                   Supported: lowercase, uppercase, trim, collapse_whitespace
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str  # Discriminator field - subclasses use Literal["eq!"], etc.
    target: str
    value: Any = None  # The value passed to the validator (e.g., "foo" for eq!: "foo")
    transform: list[str] = []  # Transforms to apply before validation
    negate: bool = False  # If True, the validation logic is negated

    @model_validator(mode="before")
    @classmethod
    def extract_options_from_value(cls, data: Any) -> Any:
        """Extract transform and negate from value if value is a dict with 'value' key.

        This allows syntax like:
            contains!:
              value: "hello"
              transform: lowercase
              negate: true
        """
        if not isinstance(data, dict):
            return data

        value = data.get("value")

        # Check if value is a dict with 'value' and optionally 'transform'/'negate'
        if isinstance(value, dict) and "value" in value:
            actual_value = value.get("value")
            transform = value.get("transform")
            # Preserve existing negate from data (e.g., from alias resolution) unless explicitly set in value
            negate = value.get("negate") if "negate" in value else data.get("negate", False)

            # Normalize transform to list
            if transform is None:
                transform = []
            elif isinstance(transform, str):
                transform = [transform]

            # Validate transforms
            for t in transform:
                if t not in TRANSFORMS:
                    raise ValueError(f"unknown transform: {t}. Supported: {sorted(TRANSFORMS)}")

            return {
                **data,
                "value": actual_value,
                "transform": transform,
                "negate": negate,
            }

        return data

    def apply_transform(self, value: str) -> str:
        """Apply configured transforms to a value.

        Args:
            value: The string to transform.

        Returns:
            The transformed string.
        """
        if not self.transform:
            return value
        return apply_transforms(value, self.transform)

    @abstractmethod
    async def __call__(self, ctx: ValidationContext) -> None:
        """Execute the validation.

        Args:
            ctx: ValidationContext containing the trace and current state.

        Raises:
            AssertionError: If validation fails.
        """
        ...
