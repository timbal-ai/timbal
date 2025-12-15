from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...types.message import Message
from .base import BaseValidator
from .context import ValidationContext


class LLMValidationResult(BaseModel):
    """Result of an LLM validation."""

    passes: bool = Field(description="Whether the validation passed")
    reason: str = Field(description="Explanation of why the validation passed or failed")


class LLMValidator(BaseValidator):
    """Base class for validators that use LLM evaluation.

    Attributes:
        system_prompt: System prompt for the LLM. Can be customized in YAML.
                      Subclasses should provide a default value.
        model: LLM model to use for validation. Defaults to "gpt-4o-mini".
               Can be customized in YAML.

    Subclasses should implement:
    - get_user_prompt(actual_value): Returns the user prompt with the value to evaluate
    - Set a default system_prompt as a class field
    """

    system_prompt: str  # Must be set in subclass or YAML
    model: str = "openai/gpt-4.1-nano"  # Can be overridden in YAML

    @abstractmethod
    def get_user_prompt(self, actual_value: Any) -> str:
        """Return the user prompt with the actual value to evaluate.

        Args:
            actual_value: The value extracted from the trace to evaluate

        Returns:
            Formatted prompt string with the value to evaluate
        """
        ...

    async def _evaluate_with_llm(self, value: Any) -> LLMValidationResult:
        """Call LLM to evaluate the value using a Timbal Agent.

        Args:
            value: The value to evaluate

        Returns:
            LLMValidationResult with passes and reason

        Raises:
            Exception: If LLM call fails
        """
        user_prompt = self.get_user_prompt(value)

        agent = Agent(
            name=f"{self.name}_evaluator",
            system_prompt=self.system_prompt,
            model=self.model,
            output_model=LLMValidationResult,  # Force structured output
            model_params={
                "temperature": 0.0,
                "max_tokens": 2048,
            },
        )

        output_event = await agent(prompt=user_prompt).collect()  # type: ignore

        if output_event.error is not None:
            raise Exception(f"LLM call failed: {output_event.error}")

        return output_event.output

    async def __call__(self, ctx: ValidationContext) -> None:
        """Execute the LLM-based validation.

        Raises:
            AssertionError: If validation fails.
        """
        from ..utils import resolve_target

        _, actual_value = resolve_target(ctx.trace, self.target)

        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()

        result = await self._evaluate_with_llm(actual_value)

        if not result.passes:
            raise AssertionError(f"{self.name} validation failed: {result.reason}")
