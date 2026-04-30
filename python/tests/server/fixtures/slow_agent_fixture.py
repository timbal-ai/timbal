"""A slow agent fixture for testing cancellation."""

import asyncio

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

async def slow_task(message: str) -> str:
    """A slow task that takes time to complete."""
    await asyncio.sleep(5)
    return f"Completed: {message}"


slow_agent = Agent(
    name="slow_agent",
    model=TestModel(responses=[
        Message(
            role="assistant",
            content=[ToolUseContent(id="c1", name="slow_task", input={"message": "slow work"})],
            stop_reason="tool_use",
        ),
        "Done.",
    ]),
    tools=[slow_task],
)
