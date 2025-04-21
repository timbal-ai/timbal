from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..state import RunContext


class BaseStep(BaseModel, ABC):
    """Abstract base class for defining processing steps in a workflow.

    BaseStep combines Pydantic's data validation with abstract methods to create a
    standardized interface for workflow steps and agent tools. Each step must define its parameter
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


    def _get_tool_input_schema(self) -> dict[str, Any]:
        """Aux function to get, filter and format the input schema for a tool."""
        input_schema = self.params_model_schema()

        tool_params_mode = "all"
        if hasattr(self, "tool_params_mode"):
            tool_params_mode = self.tool_params_mode

        selected_params = set()
        if tool_params_mode == "required":
            selected_params = set(input_schema.get("required", []))
        else:
            selected_params = set(input_schema["properties"].keys())
        
        if hasattr(self, "tool_include_params") and self.tool_include_params is not None:
            selected_params.update(self.tool_include_params)

        if hasattr(self, "tool_exclude_params") and self.tool_exclude_params is not None:
            selected_params.difference_update(self.tool_exclude_params)

        input_schema["properties"] = {
            k: v 
            for k, v in input_schema["properties"].items()
            if k in selected_params
        }

        return input_schema


    def to_openai_tool(self) -> dict[str, Any]:
        """Convert the step to OpenAI's expected tool format."""
        tool_description = ""
        if hasattr(self, "tool_description"):
            tool_description = self.tool_description or ""

        input_schema = self._get_tool_input_schema()

        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": tool_description,
                "parameters": input_schema,
            }
        }


    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert the step to Anthropic's expected tool format."""
        tool_description = ""
        if hasattr(self, "tool_description"):
            tool_description = self.tool_description or ""

        input_schema = self._get_tool_input_schema()

        return {
            "name": self.id,
            "description": tool_description,
            "input_schema": input_schema,
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
