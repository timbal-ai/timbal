import os
import pytest

from timbal import Agent
from timbal.state.context import RunContext, TimbalPlatformConfig
from timbal.steps.timbal.orgs import get_orgs


@pytest.mark.asyncio
async def test_get_orgs():
    # We can pass the timbal platform config directly as a function parameter.
    timbal_platform_config = TimbalPlatformConfig(**{
        "host": "dev.timbal.ai",
        "auth_config": {
            "type": "bearer",
            "token": os.getenv("TIMBAL_API_TOKEN")
        },
        "scope": {}
    })

    orgs = await get_orgs(timbal_platform_config=timbal_platform_config)
    print(orgs)


@pytest.mark.asyncio
async def test_get_orgs_agent():
    agent = Agent(
        model="gpt-4.1",
        tools=[{
            "runnable": get_orgs,
            "description": "Get all your organizations.",
        }],
        max_iter=1,
    )

    run_context = RunContext(
        timbal_platform_config=TimbalPlatformConfig(**{
            "host": "dev.timbal.ai",
            "auth_config": {
                "type": "bearer",
                "token": os.getenv("TIMBAL_API_TOKEN")
            },
            "app_config": {}
        })
    )

    await agent.complete(context=run_context, prompt="is Timbal one of my organizations?"),
