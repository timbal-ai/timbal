import pytest
from timbal import Flow
from timbal.state.savers import InMemorySaver
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
    prompt="What's the weather and traffic like in New York?"
    flow = (Flow(id="test_add_agent")
            .add_agent(model="gpt-4o-mini", tools=[get_weather, get_traffic])
            .set_data_map("agent.prompt", "prompt")
            .set_output("response", "agent.return.response")
            .compile(state_saver=InMemorySaver())
    )
    result = await flow.complete(prompt=prompt)
    assert "15" in result["response"].content[0].text
    assert "pretty heavy" in result["response"].content[0].text.lower()
    

@pytest.mark.asyncio
async def test_add_tool_description():
    prompt="What's the weather and traffic like in New York?"
    flow = (Flow(id="test_add_tool_description")
            .add_agent(model="gpt-4o-mini", 
                       tools=[get_weather, 
                              {"tool": get_traffic, "description": "Get the traffic for a given location"}]
                    )
            .set_data_map("agent.prompt", "prompt")
            .set_output("response", "agent.return.response")
            .compile(state_saver=InMemorySaver())
    )
    result = await flow.complete(prompt=prompt)
    assert "15" in result["response"].content[0].text
    assert "pretty heavy" in result["response"].content[0].text.lower()


@pytest.mark.asyncio
async def test_max_iter():
    prompt="What's the weather in New York in fahrenheit?"
    flow = (Flow(id="test_max_iter")
            .add_agent(model="gpt-4o-mini", 
                       tools=[get_weather, get_traffic, convert_to_fahrenheit],
                       max_iter=2
                    )
            .set_data_map("agent.prompt", "prompt")
            .set_output("response", "agent.return.response")
            .compile(state_saver=InMemorySaver())
    )
    result = await flow.complete(prompt=prompt)
    assert "59" in result["response"].content[0].text.lower()
    assert "pretty heavy" not in result["response"].content[0].text.lower()
