import pytest

from timbal.core_v2.agent import Agent
from timbal.state.context import RunContext
from timbal.state import set_run_context


@pytest.mark.asyncio
async def test_memory():
    agent = Agent(
        name="agent",
        model="gpt-4.1-mini",
    )

    async for _ in agent(prompt="Hello, my name is David"):
        pass
            
    async for _ in agent(prompt="What is my name?"):
        pass
        