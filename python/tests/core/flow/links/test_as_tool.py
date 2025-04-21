import pytest
from timbal import Flow
from timbal.types import Field


def get_weather(
    location: str = Field(description="The location to get the weather for"), # noqa: ARG001
    unit: str = Field(choices=["celsius", "fahrenheit"], description="The unit for the temperature") # noqa: ARG001
) -> str:
    return "sunny"


def get_traffic(
    location: str = Field(description="The location to get the weather for"), # noqa: ARG001
) -> str:
    return "pretty heavy"


@pytest.mark.asyncio
async def test_tool_use():
    flow = (
        Flow(id="test_parallel_tools")
        .add_llm("ask_llm")
        .set_data_map("ask_llm.prompt", "prompt")
        .add_step("get_weather", get_weather)
        .add_link("ask_llm", "get_weather", is_tool=True)
        .add_step("get_traffic", get_traffic)
        .add_link("ask_llm", "get_traffic", is_tool=True)
    )

    prompt = "What's the weather and traffic like in New York?"
    executed_steps = set()
    async for event in flow.run(prompt=prompt):
        if event.type == "START":
            executed_steps.add(event.path)
    assert "test_parallel_tools.get_weather" in executed_steps
    assert "test_parallel_tools.get_traffic" in executed_steps
    
    # Use only the tool that is relevant to the prompt
    prompt = "What's the weather like in New York?"
    executed_steps.clear()
    async for event in flow.run(prompt=prompt):
        if event.type == "START":
            executed_steps.add(event.path)
    assert "test_parallel_tools.get_weather" in executed_steps
    assert "test_parallel_tools.get_traffic" not in executed_steps

    # No tools should be used
    prompt = "What's 2 + 2?"
    executed_steps.clear()
    async for event in flow.run(prompt=prompt):
        if event.type == "START":
            executed_steps.add(event.path)
    assert "test_parallel_tools.get_weather" not in executed_steps
    assert "test_parallel_tools.get_traffic" not in executed_steps


@pytest.mark.asyncio
async def test_tool_result():
    flow = (
        Flow(id="test_tool_result")
        .add_llm("ask_llm", memory_id="ask_llm")
        .set_data_map("ask_llm.prompt", "prompt")
        .add_step("get_weather", get_weather)
        .add_link("ask_llm", "get_weather", is_tool=True)
        .add_step("get_traffic", get_traffic)
        .add_link("ask_llm", "get_traffic", is_tool=True)
        .add_llm("ask_llm_2", memory_id="ask_llm")
        .add_link("get_weather", "ask_llm_2", is_tool_result=True)
        .add_link("get_traffic", "ask_llm_2", is_tool_result=True)
    )

    prompt = "What's the weather and traffic like in New York?"
    executed_steps = set()
    async for event in flow.run(prompt=prompt):
        if event.type == "START":
            executed_steps.add(event.path)
    assert "test_tool_result.get_weather" in executed_steps
    assert "test_tool_result.get_traffic" in executed_steps
    assert "test_tool_result.ask_llm_2" in executed_steps
