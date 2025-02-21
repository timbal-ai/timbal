import pytest
from timbal import Flow
from timbal.types import Field


def get_weather(
    location: str = Field(description="The location to get the weather for"), # noqa: ARG001
) -> str:
    return "15 celsius"

def get_traffic(
    location: str = Field(description="The location to get the weather for"), # noqa: ARG001
) -> str:
    return "pretty heavy"

def convert_to_fahrenheit(
    temperature: str = Field(description="The temperature to convert to fahrenheit"), # noqa: ARG001
) -> str:
    return "59 fahrenheit"


@pytest.mark.asyncio
async def test_add_agent():
    flow = Flow(id="test_add_agent")
    prompt="What's the weather and traffic like in New York?"
    flow.add_agent(model="gpt-4o-mini", tools=[get_weather, get_traffic])
    executed_steps = set()
    async for event in flow.run(prompt=prompt):
        if event.type == "STEP_START":
            executed_steps.add(event.step_id)
    assert any(step.startswith("get_weather") for step in executed_steps)
    assert any(step.startswith("get_traffic") for step in executed_steps)


@pytest.mark.asyncio
async def test_add_tool_description():
    flow = Flow(id="test_add_tool_description")
    prompt="What's the weather and traffic like in New York?"
    flow.add_agent(model="gpt-4o-mini", 
                   tools=[get_weather, 
                          {"tool": get_traffic, "description": "Get the traffic for a given location"}]
                )
    executed_steps = set()
    async for event in flow.run(prompt=prompt):
        if event.type == "STEP_START":
            executed_steps.add(event.step_id)
    assert any(step.startswith("get_weather") for step in executed_steps)
    assert any(step.startswith("get_traffic") for step in executed_steps)


@pytest.mark.asyncio
async def test_max_iter():
    flow = Flow(id="test_max_iter")
    prompt="What's the weather in New York in fahrenheit?"
    flow.add_agent(model="gpt-4o-mini", 
                   tools=[get_weather, get_traffic, convert_to_fahrenheit],
                   max_iter=2
                )
    executed_steps = set()
    async for event in flow.run(prompt=prompt):
        if event.type == "STEP_START":
            executed_steps.add(event.step_id)
    assert any(step.startswith("get_weather") for step in executed_steps)
    assert any(step.startswith("convert_to_fahrenheit") for step in executed_steps)
