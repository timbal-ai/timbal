import pytest
from timbal import Agent
from timbal.errors import bail


@pytest.mark.asyncio
async def test_post_hook_bail():
    def post_hook():
        bail("Bailing out!")
    agent = Agent(name="test_agent", model="openai/gpt-4o-mini", post_hook=post_hook)
    res = await agent(prompt="what is the capital of france").collect()
    assert res.status.code == "cancelled"
    assert res.status.reason == "early_exit"
    assert res.output is None


@pytest.mark.asyncio
async def test_pre_hook_bail():
    def pre_hook():
        bail("Bailing out!")
    agent = Agent(name="test_agent", model="openai/gpt-4o-mini", pre_hook=pre_hook)
    res = await agent(prompt="what is the capital of france").collect()
    assert res.status.code == "cancelled"
    assert res.status.reason == "early_exit"
    assert res.output is None
    

@pytest.mark.asyncio
async def test_bail_in_tool():
    def get_datetime():
        bail("Bailing out!")
    agent = Agent(name="test_agent", model="openai/gpt-4o-mini", tools=[get_datetime])
    res = await agent(prompt="what time is it?").collect()
    assert res.status.code == "cancelled"
    assert res.status.reason == "early_exit"
    assert res.output is None
