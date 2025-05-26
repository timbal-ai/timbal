
from pydantic import BaseModel, ConfigDict

from .input import Input
from .output import Output
from .steps import Steps


class Turn(BaseModel):
    """Represents a single turn in an evaluation test.

    A turn defines one interaction between a user and an agent within a test case.
    Each turn specifies the input provided to the agent and the output that should result.
    The output can serve as an expected response (with validators for checking correctness)
    or as a fixed output to be used as agent memory/history (when no validators are present).
    """
    model_config = ConfigDict(extra="ignore")

    input: Input
    """The input for this turn, representing what is sent to the agent. This may include text and/or files."""
    output: Output
    """The output for this turn. When validators are present, this represents the expected output to be checked for correctness.
    When no validators are present, this output is used as a fixed response to store in the agent's memory or conversation history.
    """
    steps: Steps | None = None
    """Defines the expected sequence of tool-use or action steps for this turn, if available.
    If this field is omitted or empty, no step validation is performed for the turn.
    """

    usage: list[dict] | None = None
    """Defines the expected resource or token usage constraints for this turn, if available.
    Each constraint is represented as a dictionary, and may include `max` and/or `min` 
    values for a particular usage type (e.g., input or output tokens).
    If this field is omitted or empty, no usage constraints are applied for the turn.
    """