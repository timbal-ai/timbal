import pytest

from timbal import Agent
from timbal.steps.perplexity import search


@pytest.mark.asyncio
async def test_parallel_search():
    agent = Agent(
        model="gpt-4.1-nano",
        tools=[{
            "runnable": search,
            "description": "Search the weather on the internet.",
            "params_mode": "required",
        }],
        stream=True,
    )

    # prompt = "Hello"
    prompt = "What is the weather in Tokyo and in London?"
    async for event in agent.run(prompt=prompt):
        # print(event)
        pass
