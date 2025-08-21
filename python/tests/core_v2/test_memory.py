import pytest
from timbal.core_v2.agent import Agent


@pytest.mark.asyncio
async def test_memory():
    agent = Agent(
        name="agent",
        model="openai/gpt-4.1-mini",
    )

    await agent(prompt="Hello, my name is David").collect()
            
    async_gen = agent(prompt="What is my name?")
    res = await async_gen.collect()
    print(res)
        