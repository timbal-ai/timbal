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

    async for event in agent(prompt="Hello, my name is David"):
        if event.type == "OUTPUT" and event.path == "agent":
            run_id = event.run_id
            
    run_context = RunContext(parent_id="...")
    set_run_context(run_context)
    async for event in agent(prompt="What is my name?"):
        pass
        