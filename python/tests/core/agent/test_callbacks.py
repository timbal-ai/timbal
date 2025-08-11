import asyncio
import pytest

from timbal import Agent
from timbal.state.context import RunContext
from timbal.state import set_run_context


@pytest.mark.asyncio
async def test_invalid_callbacks():
    # Test for invalid arguments.
    def invalid_callback(a, b, c): pass

    with pytest.raises(TypeError):
        Agent(before_agent_callback=invalid_callback)

    with pytest.raises(TypeError):
        Agent(after_agent_callback=invalid_callback)

    # Test for generators.
    def invalid_callback(run_context):
        yield

    with pytest.raises(TypeError):
        Agent(before_agent_callback=invalid_callback)

    with pytest.raises(TypeError):
        Agent(after_agent_callback=invalid_callback)


@pytest.mark.asyncio
async def test_sync_callbacks():
    def before_agent_callback(run_context: RunContext):
        run_context.data["before_agent_callback_called"] = True

    def after_agent_callback(run_context: RunContext):
        run_context.data["after_agent_callback_called"] = True

    agent = Agent(
        before_agent_callback=before_agent_callback,
        after_agent_callback=after_agent_callback,
    )

    run_context = RunContext(id="test_sync_callbacks")
    set_run_context(run_context)
    await agent.complete(prompt="Hello")

    assert run_context.data["before_agent_callback_called"]
    assert run_context.data["after_agent_callback_called"]


@pytest.mark.asyncio
async def test_async_callbacks():
    async def before_agent_callback(run_context: RunContext):
        await asyncio.sleep(1)
        run_context.data["before_agent_callback_called"] = True

    async def after_agent_callback(run_context: RunContext):
        await asyncio.sleep(1)
        run_context.data["after_agent_callback_called"] = True

    agent = Agent(
        before_agent_callback=before_agent_callback,
        after_agent_callback=after_agent_callback,
    )

    run_context = RunContext(id="test_async_callbacks")
    set_run_context(run_context)
    await agent.complete(prompt="Hello")

    assert run_context.data["before_agent_callback_called"]
    assert run_context.data["after_agent_callback_called"]
