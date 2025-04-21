from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, computed_field

from ...state.data import DATA_KEY_INTERPOLATION_PATTERN, BaseData, get_data_key
from ...types import Message, ToolUseContent

logger = structlog.get_logger("timbal.core.flow.link")


class Link(BaseModel):
    """A Link represents a connection between two steps in a workflow graph.
    
    Links can be conditional (using a condition string that evaluates against workflow data),
    or represent tool calls and their results. Each link connects a source step (step_id)
    to a destination step (next_step_id).
    """
    # Allow storing extra fields in the model.
    # Validate dynamic assignments to the model.
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    step_id: str 
    """The ID of the source step."""
    next_step_id: str
    """The ID of the destination step."""
    condition: str | None = None
    """Optional string containing a condition to evaluate."""
    is_tool: bool = False
    """ Whether this link represents a tool call."""
    is_tool_result: bool = False
    """Whether this link represents a tool result."""
    metadata: dict[str, Any] = {}
    """Additional metadata associated with the link."""


    @computed_field
    @property
    def id(self) -> str:
        """The id of the link is the concatenation of the step_id and next_step_id."""
        return f"{self.step_id}-{self.next_step_id}"

    
    def evaluate_condition(self, data: dict[str, Any]) -> bool:
        """Evaluates if this link's condition is met given the current data."""

        if self.is_tool:
            step_output = data.get(f"{self.step_id}.return", None)

            assert isinstance(step_output, BaseData), "Step output should always be an instance of BaseData."
            step_output = step_output.resolve(context_data=data)
            
            # Check if this is an LLM output with tool calls
            if isinstance(step_output, Message):
                for content in step_output.content:
                    if isinstance(content, ToolUseContent) and content.name == self.next_step_id:
                        return True
            return False

        if self.condition:
            # Search for all keys we may have in the context data.
            template_keys = DATA_KEY_INTERPOLATION_PATTERN.findall(self.condition)
            condition = self.condition  # Work with a copy instead of modifying self.condition 
        
            for template_key in template_keys:
                replacement_value = get_data_key(data, template_key)
                # TODO Handle more scenarios here.
                if isinstance(replacement_value, str):
                    replacement_value = f"'{replacement_value}'"
                else:
                    replacement_value = str(replacement_value)
                # In order to evaluate the condition, we need to place the value in the condition as a string.
                condition = condition.replace(f"{{{{{template_key}}}}}", replacement_value)

            if not eval(condition):
                return False
        
        # By default, if the link was not added as a tool, or no additional condition is passed, we always return True.
        return True
