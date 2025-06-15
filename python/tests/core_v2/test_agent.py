from typing import Any

import pytest

from timbal.core_v2.agent import Agent


def test_init_callable_tool():
    def identity(x: Any) -> Any:
        return x
    Agent(
        name="agent",
        model="gpt-4.1",
        tools=[identity],
    )


def test_init_dict_tool():
    tool = {
        "handler": lambda x: x,
        "name": "identity",
    }
    Agent(
        name="agent",
        model="gpt-4.1",
        tools=[tool],
    )


def test_init_agent_tool():
    sales_expert = Agent(
        name="sales_expert",
        model="gpt-4.1",
    )
    Agent(
        name="agent",
        model="gpt-4.1",
        tools=[sales_expert],
    )


def test_init_duplicate_tool():
    def identity(x: Any) -> Any:
        return x
    
    with pytest.raises(ValueError):
        Agent(
            name="agent",
            model="gpt-4.1",
            tools=[identity, identity],
        )
    

@pytest.mark.asyncio
async def test_agent_without_tools():
    agent = Agent(
        name="agent",
        model="gpt-4.1",
    )

    async for event in agent(prompt="Hello"):
        pass
        

@pytest.mark.asyncio
async def test_agent_with_tools():
    from datetime import datetime

    def get_datetime(city: str) -> str:
        return datetime.now().isoformat()
    

    agent = Agent(
        name="agent",
        model="gpt-4.1",
        tools=[get_datetime],
    )

    async for event in agent(prompt="Hello"):
        pass

    async for event in agent(prompt="What time is it in barcelona and in new york?"):
        pass


@pytest.mark.asyncio
async def test_agent_with_agent():
    def get_weather(city: str) -> str:
        return "Sunny"

    weather_agent = Agent(
        name="weather_agent",
        description="This agent answers questions about the weather.",
        model="gpt-4.1",
        tools=[get_weather],
    )

    agent = Agent(
        name="agent",
        model="gpt-4.1",
        tools=[weather_agent],
    )

    async for event in agent(prompt="What is the weather in Barcelona?"):
        pass
