import pytest

from timbal import Agent, Flow
from timbal.state.context import RunContext
from timbal.state.savers import InMemorySaver


@pytest.mark.asyncio
async def test_usage_openai():
    flow = Agent(
        model="gpt-4o-mini",
    ).compile(state_saver=InMemorySaver())

    flow_output_event = await flow.complete(prompt="My name is David.")

    llm_usage = flow.state_saver.snapshots[-1].steps["agent.agent_llm_0"]["usage"]
    print(flow_output_event)
    print(llm_usage)

    flow_output_event = await flow.complete(
        context=RunContext(parent_id=flow_output_event.run_id), 
        prompt="What is my name?"
    )

    llm_usage = flow.state_saver.snapshots[-1].steps["agent.agent_llm_0"]["usage"]
    print(flow_output_event)
    print(llm_usage)


@pytest.mark.asyncio
async def test_usage_anthropic():
    flow = Agent(
        model="claude-3-5-sonnet-20241022",
    ).compile(state_saver=InMemorySaver())

    flow_output_event = await flow.complete(prompt="My name is David.")

    llm_usage = flow.state_saver.snapshots[-1].steps["agent.agent_llm_0"]["usage"]
    print(flow_output_event)
    print(llm_usage)

    flow_output_event = await flow.complete(
        context=RunContext(parent_id=flow_output_event.run_id), 
        prompt="What is my name?"
    )

    llm_usage = flow.state_saver.snapshots[-1].steps["agent.agent_llm_0"]["usage"]
    print(flow_output_event)
    print(llm_usage)


@pytest.mark.asyncio
async def test_usage_subflow():
    subflow = Agent(model="gpt-4o-mini").compile(state_saver=InMemorySaver())

    flow = (Flow()
        .add_step("subflow", subflow)
        .set_input("subflow.prompt", "prompt")
        .add_llm("llm0", model="gpt-4o-mini")
        .set_input("llm0.prompt", "prompt")
    ).compile(state_saver=InMemorySaver())

    await flow.complete(prompt="Tell me something interesting")
    print()
    print(flow.state_saver.snapshots)
