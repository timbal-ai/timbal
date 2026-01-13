"""A slow agent fixture for testing cancellation."""

import asyncio

from timbal import Agent


async def slow_task(message: str) -> str:
    """A slow task that takes time to complete."""
    await asyncio.sleep(5)
    return f"Completed: {message}"


slow_agent = Agent(
    name="slow_agent",
    model="openai/gpt-4o-mini",
    tools=[slow_task],
)
