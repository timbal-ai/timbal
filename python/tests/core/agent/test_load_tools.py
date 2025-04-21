import pytest

from timbal import Agent
from timbal.steps.perplexity import search
from timbal.core.agent.types.tool import Tool
from timbal.core.step import Step


@pytest.mark.asyncio
async def test_init_as_callable():
    agent = Agent(tools=[search])

    openai_tool_input = agent.tools[0].to_openai_tool()
    assert len(openai_tool_input["function"]["parameters"]["properties"]) == 14

    anthropic_tool_input = agent.tools[0].to_anthropic_tool()
    assert len(anthropic_tool_input["input_schema"]["properties"]) == 14


@pytest.mark.asyncio
async def test_init_as_step():
    search_step = Step(id="search", handler_fn=search)
    agent = Agent(tools=[search_step])

    openai_tool_input = agent.tools[0].to_openai_tool()
    assert len(openai_tool_input["function"]["parameters"]["properties"]) == 14

    anthropic_tool_input = agent.tools[0].to_anthropic_tool()
    assert len(anthropic_tool_input["input_schema"]["properties"]) == 14


@pytest.mark.asyncio
async def test_required_params():
    agent = Agent(
        model="gemini-2.0-flash-lite",
        tools=[{
            "runnable": search, 
            "description": "Search information on the internet.",
            "params_mode": "required",
        }],
    )

    openai_tool_input = agent.tools[0].to_openai_tool()
    assert openai_tool_input["function"]["parameters"]["properties"].keys() == {"query"}

    anthropic_tool_input = agent.tools[0].to_anthropic_tool()
    assert anthropic_tool_input["input_schema"]["properties"].keys() == {"query"}


@pytest.mark.asyncio
async def test_required_params_with_include():
    agent = Agent(
        model="gpt-4.1",
        tools=[Tool(
            runnable=search, 
            description="Search information on the internet.",
            params_mode="required",
            include_params=["model"],
        )],
    )

    openai_tool_input = agent.tools[0].to_openai_tool()
    assert openai_tool_input["function"]["parameters"]["properties"].keys() == {"query", "model"}

    anthropic_tool_input = agent.tools[0].to_anthropic_tool()
    assert anthropic_tool_input["input_schema"]["properties"].keys() == {"query", "model"}


@pytest.mark.asyncio
async def test_required_params_with_exclude():
    agent = Agent(
        model="gpt-4.1",
        tools=[Tool(
            runnable=search, 
            description="Search information on the internet.",
            exclude_params=["query"],
        )],
    )

    openai_tool_input = agent.tools[0].to_openai_tool()
    assert "query" not in openai_tool_input["function"]["parameters"]["properties"].keys()

    anthropic_tool_input = agent.tools[0].to_anthropic_tool()
    assert "query" not in anthropic_tool_input["input_schema"]["properties"].keys()


@pytest.mark.asyncio
async def test_init_invalid_dict():
    # Extra properties are ignored.
    agent = Agent(tools=[{
        "runnable": search, 
        "description": "Search information on the internet.",
        "params_mode": "required",
        "invalid_key": "invalid_value",
    }])

    # Invalid types or missing required properties raise an error.
    with pytest.raises(ValueError):
        agent = Agent(tools=[{
            "description": {},
        }])
