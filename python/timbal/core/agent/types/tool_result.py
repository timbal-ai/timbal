from typing import Any

from pydantic import BaseModel

from ....types.message import Message


class ToolResult(BaseModel):
    """Helper class to wrap general tool results."""

    id: str
    """The id of the tool use that will be matched in the generated ToolResultContent."""
    input: dict[str, Any]
    """The input to the tool. This is stored for tracing and debugging."""
    output: Message 
    """The output of the tool. Always as an LLM ready message so we can pass it to the LLMs."""
    error: dict[str, Any] | None = None
    """Store if any error occurred while running the tool."""
    t0: int
    """The start time of the tool in milliseconds."""
    t1: int
    """The end time of the tool in milliseconds."""
    usage: dict[str, int] = {}
    """The usage of the tool."""
    force_exit: bool
    """Whether the tool should force the agent to exit."""
    # skip_summarization: bool = False
    # """If set to True, instructs the ADK to bypass the LLM call that typically summarizes the tool's output. This is useful if your tool's return value is already a user-ready message."""
    # transfer_to_agent: str | None = None
    # """Set this to the name of another agent. The framework will halt the current agent's execution and transfer control of the conversation to the specified agent. This allows tools to dynamically hand off tasks to more specialized agents."""
    # escalate: bool = False
    # """Setting this to True signals that the current agent cannot handle the request and should pass control up to its parent agent (if in a hierarchy). In a LoopAgent, setting escalate=True in a sub-agent's tool will terminate the loop."""
