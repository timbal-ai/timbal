import pytest
from timbal import Agent
from timbal.state import RunContext
from timbal.state.savers import InMemorySaver


@pytest.mark.asyncio
async def test_in_memory():

    agent_id = "custom_name"

    flow = Agent(
        id=agent_id,
        model="gpt-4o-mini",
    ).compile(state_saver=InMemorySaver())

    flow_output_event = await flow.complete(prompt="My name is David")

    assert len(flow.state_saver.snapshots) == 1
    for i, snapshot in enumerate(flow.state_saver.snapshots):
        agent_memory = snapshot.data[f"{agent_id}.memory"].resolve()
        assert len(agent_memory) == (i + 1) * 2

    flow_output_event = await flow.complete(
        context=RunContext(parent_id=flow_output_event.run_id),
        prompt="What's my name?"
    )

    assert len(flow.state_saver.snapshots) == 2
    for i, snapshot in enumerate(flow.state_saver.snapshots):
        agent_memory = snapshot.data[f"{agent_id}.memory"].resolve()
        assert len(agent_memory) == (i + 1) * 2

    await flow.complete(
        context=RunContext(parent_id=flow_output_event.run_id),
        prompt="And what's yours?"
    )

    assert len(flow.state_saver.snapshots) == 3
    for i, snapshot in enumerate(flow.state_saver.snapshots):
        agent_memory = snapshot.data[f"{agent_id}.memory"].resolve()
        assert len(agent_memory) == (i + 1) * 2
