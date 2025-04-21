import asyncio
import pytest
from datetime import datetime

from timbal import Agent
from timbal.errors import AgentError
from timbal.state.context import RunContext
from timbal.state.savers import InMemorySaver, JSONLSaver
from timbal.steps.perplexity import search


def get_current_time():
    return datetime.now().isoformat()


async def get_current_time_async():
    await asyncio.sleep(1)
    return datetime.now().isoformat()


def get_company_info(company_name: str):
    yield "Company name: Timbal. "
    yield "Company description: Timbal is a platform for building and running agents."


async def get_company_info_async(company_name: str):
    await asyncio.sleep(1)
    yield "Company name: Timbal. "
    await asyncio.sleep(1)
    yield "Company description: Timbal is a platform for building and running agents."


@pytest.mark.asyncio
async def test_init():
    agent = Agent()
    print(agent)


@pytest.mark.asyncio
async def test_init_with_tools():
    agent = Agent(tools=[get_current_time, search])
    print("Tools: ", agent.tools)
    print("Tools lookup: ", agent.tools_lookup)
    for tool in agent.tools:
        print(tool.to_openai_tool())
        print(tool.to_anthropic_tool())


@pytest.mark.asyncio
async def test_params_model():
    agent = Agent()
    print(agent.params_model())


@pytest.mark.asyncio
async def test_params_model_schema():
    agent = Agent()
    print(agent.params_model_schema())


@pytest.mark.asyncio
async def test_return_model():
    agent = Agent()
    print(agent.return_model())


@pytest.mark.asyncio
async def test_return_model_schema():
    agent = Agent()
    print(agent.return_model_schema())


@pytest.mark.asyncio
async def test_run():
    agent = Agent()
    async for event in agent.run(prompt="Hello, world!"):
        print(event)


@pytest.mark.asyncio
async def test_run_with_sync_tool():
    agent = Agent(tools=[get_current_time])
    await agent.complete(prompt="What time is it?")


@pytest.mark.asyncio
async def test_run_with_async_tool():
    agent = Agent(tools=[get_current_time_async])
    async for event in agent.run(prompt="What time is it?"):
        print(event)


@pytest.mark.asyncio
async def test_run_with_sync_gen_tool():
    agent = Agent(tools=[get_company_info])
    async for event in agent.run(prompt="Give me the info of the company Timbal."):
        print(event)


@pytest.mark.asyncio
async def test_run_with_async_gen_tool():
    agent = Agent(tools=[get_company_info_async])
    async for event in agent.run(prompt="Give me the info of the company Timbal."):
        print(event)


@pytest.mark.asyncio
async def test_run_with_multiple_tools():
    agent = Agent(tools=[get_current_time, get_company_info_async])
    async for event in agent.run(prompt="Give me the time and the info of the company Timbal."):
        print(event)


def get_customer(customer_id):
    if isinstance(customer_id, int):
        raise ValueError("Customer ID must be a string.")
    return {"id": customer_id, "name": "John Doe", "email": "john.doe@example.com"}


@pytest.mark.asyncio
async def test_run_incomplete_type_annotations():
    agent = Agent(tools=[get_customer])
    await agent.complete(prompt="Give me the info of the customer 123")


@pytest.mark.asyncio
async def test_run_with_max_iter():
    agent = Agent(tools=[get_customer], max_iter=1)
    async for event in agent.run(prompt="Give me the info of the customer 123"):
        print(event)


async def get_weather(city: str):
    yield "The weather in " 
    await asyncio.sleep(3)
    yield city
    await asyncio.sleep(3)
    yield " is sunny."


@pytest.mark.asyncio
async def test_run_parallel_tool_calls():
    agent = Agent(tools=[get_weather])
    async for event in agent.run(prompt="What is the weather in Tokyo and in London?"):
        print(event)


@pytest.mark.asyncio
async def test_run_with_memory():
    agent = Agent(
        tools=[get_weather],
        state_saver=InMemorySaver(),
    )

    async for event in agent.run(prompt="What is the weather in Tokyo and in London?"):
        parent_run_id = event.run_id
        print()
        print("Event: ", event)

    async for event in agent.run(
        context=RunContext(parent_id=parent_run_id),
        prompt="What was I talking about?"
    ):
        print()
        print("Event: ", event)


@pytest.mark.asyncio
async def test_run_with_parallel_internet_search():
    agent = Agent(tools=[search])

    async for event in agent.run(prompt="What is the weather in Tokyo and in London?"):
        print()
        print("Event: ", event)


@pytest.mark.asyncio
async def test_run_llm_error():
    agent = Agent(model="this-model-does-not-exist")
    with pytest.raises(AgentError):
        async for event in agent.run(prompt="What is the weather in Tokyo and in London?"):
            print()
            print("Event: ", event)


# TODO Test initial kwargs validation error.
# TODO Test data is being saved properly after any type of error.


@pytest.mark.asyncio
async def test_other_models():
    state_saver = JSONLSaver("test_jsonl.jsonl")
    agent = Agent(
        model="gpt-4o-mini",
        # model="gemini-2.0-flash-lite",
        # model="o3-mini",
        # model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        # model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # model="claude-3-5-sonnet-20241022",
        # model="o1",
        tools=[get_current_time],
        state_saver=state_saver,
        max_tokens=2048,
    )
    res = await agent.complete(prompt="What is the time?")
    print(res)


@pytest.mark.asyncio
async def test_parallel_search():
    # TODO Since gemini does not return an id the paths look like this:  path='agent.search-'
    agent = Agent(
        model="gemini-2.0-flash-lite",
        tools=[{
            "runnable": search,
            "description": "Search the weather on the internet.",
            "params_mode": "required",
        }],
    )
    res = await agent.complete(prompt="What is the weather in Tokyo and in London?")
    print(res)
