from datetime import datetime

import pytest
from timbal import Agent
from timbal.state.savers import InMemorySaver
from timbal.types import Field, Message


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
async def test_agent():
    prompt="What's the weather and traffic like in New York?"

    flow = Agent(tools=[get_weather, get_traffic])

    flow_output_event = await flow.complete(prompt=prompt)

    assert isinstance(flow_output_event.output, Message)

    message = flow_output_event.output
    assert "15" in message.content[0].text
    assert "pretty heavy" in message.content[0].text.lower()
    

@pytest.mark.asyncio
async def test_agent_with_tool_description():
    prompt="What's the weather and traffic like in New York?"

    flow = Agent(
        model="gpt-4o-mini", 
        tools=[
            get_weather, 
            {
                "tool": get_traffic, 
                "description": "Get the traffic for a given location"
            }
        ],
    ).compile(state_saver=InMemorySaver())

    flow_output_event = await flow.complete(prompt=prompt)

    assert isinstance(flow_output_event.output, Message)

    message = flow_output_event.output
    assert "15" in message.content[0].text
    assert "pretty heavy" in message.content[0].text.lower()


@pytest.mark.asyncio
async def test_agent_with_multiple_iterations():
    prompt = "What's the weather in New York in fahrenheit?"

    flow = Agent(
        model="gpt-4o-mini", 
        tools=[get_weather, get_traffic, convert_to_fahrenheit],
        max_iter=2
    ).compile(state_saver=InMemorySaver())

    flow_output_event = await flow.complete(prompt=prompt)

    assert isinstance(flow_output_event.output, Message)

    message = flow_output_event.output
    assert "59" in message.content[0].text.lower()
    assert "pretty heavy" not in message.content[0].text.lower()


@pytest.mark.asyncio
async def test_agent_with_no_tools():
    prompt = "Hello are you an LLM with no tools?"

    flow = Agent()

    flow_output_event = await flow.complete(prompt=prompt)

    assert isinstance(flow_output_event.output, Message)


@pytest.mark.asyncio
async def test_agent_with_custom_outputs():
    flow = Agent(tools=[get_datetime])
    flow.set_output("agent_get_datetime_0.return", "datetime")

    assert not isinstance(flow.return_model(), Message)

    prompt = "What's the current time?"
    flow_output_event = await flow.complete(prompt=prompt)
    assert "datetime" in flow_output_event.output
    assert flow_output_event.output["datetime"] is not None

    prompt = "What are you?"
    flow_output_event = await flow.complete(prompt=prompt)
    assert "datetime" in flow_output_event.output
    assert flow_output_event.output["datetime"] is None
