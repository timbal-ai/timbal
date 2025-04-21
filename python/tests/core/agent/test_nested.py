import pytest

from timbal import Agent
from timbal.steps.perplexity import search
from timbal.types.events import OutputEvent, StartEvent


@pytest.mark.asyncio
async def test_nested_paths():
    agent_0 = Agent(id="agent_0", tools=[search])
    agent_1 = Agent(id="agent_1", tools=[agent_0])
    agent_2 = Agent(id="agent_2", tools=[agent_1])

    assert len(agent_2.tools) == 1
    agent_2_tool = agent_2.tools[0]
    assert agent_2_tool.path == "agent_2.agent_1"

    assert len(agent_2_tool.tools) == 1
    agent_1_tool = agent_1.tools[0]
    assert agent_1_tool.path == "agent_2.agent_1.agent_0"

    assert len(agent_1_tool.tools) == 1
    agent_0_tool = agent_0.tools[0]
    assert agent_0_tool.path == "agent_2.agent_1.agent_0.search"
        

@pytest.mark.asyncio
async def test_nested_1():
    weather_expert = Agent(
        id="weather_expert",
        tools=[{
            "runnable": search,
            "description": "Search the internet for any information.",
            "params_mode": "required",
        }],
    )

    agent = Agent(tools=[weather_expert])

    events = []
    async for event in agent.run(prompt="What is the weather like in New York?"):
        # Hack for being able to check the events in the test easily.
        if isinstance(event, StartEvent):
            events.append(f"START:{event.path}")
        elif isinstance(event, OutputEvent):
            events.append(f"OUTPUT:{event.path}")

    assert "START:agent" in events
    assert "START:agent.llm-0" in events
    assert "OUTPUT:agent.llm-0" in events
    assert any(event.startswith("START:agent.weather_expert-") for event in events)
    assert any(event.startswith("OUTPUT:agent.weather_expert-") for event in events)
    assert "START:agent.llm-1" in events
    assert "OUTPUT:agent.llm-1" in events
    assert "OUTPUT:agent" in events

    # Hence nested events are not yielded by the parent agent.
    # TODO Think what to do if we want to yield them.
