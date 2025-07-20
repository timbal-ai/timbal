from pathlib import Path

import pytest
from timbal import Agent
from timbal.state import RunContext
from timbal.state.savers import InMemorySaver
from timbal.types.file import File


@pytest.fixture(params=[Path(__file__).parent / "fixtures" / "test.csv"])
def csv(request):
    return File.validate(request.param)


@pytest.mark.asyncio
async def test_in_memory(csv):
    agent = Agent(
        model="gpt-4.1-nano",
        state_saver=InMemorySaver(),
    )

    agent_output_event = await agent.complete(prompt=[csv, "Explain the data."])

    assert len(agent.state_saver.snapshots) == 1
    for i, snapshot in enumerate(agent.state_saver.snapshots):
        agent_memory = snapshot.data["memory"].resolve()
        assert len(agent_memory) == (i + 1) * 2

    run_context = RunContext(parent_id=agent_output_event.run_id)
    agent_output_event = await agent.complete(
        context=run_context,
        prompt="Too verbose",
    )

    assert len(agent.state_saver.snapshots) == 2
    for i, snapshot in enumerate(agent.state_saver.snapshots):
        agent_memory = snapshot.data["memory"].resolve()
        assert len(agent_memory) == (i + 1) * 2

    run_context = RunContext(parent_id=agent_output_event.run_id)
    _ = await agent.complete(
        context=run_context,
        prompt="Thanks",
    )

    assert len(agent.state_saver.snapshots) == 3
    for i, snapshot in enumerate(agent.state_saver.snapshots):
        agent_memory = snapshot.data["memory"].resolve()
        assert len(agent_memory) == (i + 1) * 2
