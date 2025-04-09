from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..state import RunContext


class BaseStep(BaseModel, ABC):
    """Abstract base class for defining processing steps in a workflow.

    BaseStep combines Pydantic's data validation with abstract methods to create a
    standardized interface for workflow steps. Each step must define its parameter
    and return value schemas, as well as the actual processing logic.
    """
    # Allow storing extra fields in the model.
    model_config = ConfigDict(extra="allow")

    id: str 
    """Unique identifier for the step instance."""
    path: str 
    """Any step will be a part of a flow. With potentially multiple nested sub-flows.
    We will use the path to uniquely identify the step's position in the overall flow.
    """
    metadata: dict[str, Any] = {}
    """Optional metadata associated with the step."""


    @abstractmethod
    def prefix_path(self, prefix: str) -> None:
        """Prefix the step's path with a given path."""
        pass


    @abstractmethod
    def params_model(self) -> BaseModel:
        """Returns the Pydantic model defining the expected parameters for this step."""
        pass


    @abstractmethod
    def params_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the step's parameter model."""
        pass
    

    @abstractmethod
    def return_model(self) -> Any:
        """Returns the expected return type for this step."""
        pass


    @abstractmethod
    def return_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the step's return value model."""
        pass


    def to_openai_tool(self) -> dict[str, Any]:
        """Convert the step to OpenAI's expected tool format."""
        tool_description = ""
        if hasattr(self, "tool_description"):
            tool_description = self.tool_description or ""

        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": tool_description,
                "parameters": self.params_model_schema(),
            }
        }


    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert the step to Anthropic's expected tool format."""
        tool_description = ""
        if hasattr(self, "tool_description"):
            tool_description = self.tool_description or ""

        return {
            "name": self.id,
            "description": tool_description,
            "input_schema": self.params_model_schema(),
        }


    # TODO Better method definition. Then we can use "See base class" in the child classes.
    @abstractmethod
    async def run(
        self, 
        context: RunContext | None = None, # noqa: ARG002
        **kwargs: Any,
    ) -> Any:
        """Executes the step's processing logic."""
        pass
