"""Static typing smoke for Agent / Tool / OutputEvent generics.

Not collected by pytest (outside ``python/tests``). Check with:

    pyright python/typing_tests/agent_generics.py
"""

from typing import assert_type

from pydantic import BaseModel
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.types.events import OutputEvent
from timbal.types.message import Message


class Summary(BaseModel):
    title: str


async def _agent_with_output_model() -> None:
    agent = Agent(
        name="typed",
        model=TestModel(responses=['{"title": "hi"}']),
        output_model=Summary,
        tools=[],
        max_tokens=64,
    )
    assert_type(agent, Agent[Summary])
    result = await agent(prompt="hi").collect()
    assert_type(result, OutputEvent[Summary] | None)
    if result is not None:
        assert_type(result.output, Summary)


def _agent_default_message() -> None:
    agent = Agent(
        name="plain",
        model=TestModel(responses=["hello"]),
        tools=[],
        max_tokens=64,
    )
    assert_type(agent, Agent[Message])


def _tool_payload() -> None:
    tool: Tool[int] = Tool(name="add", handler=lambda x: x + 1)
    assert_type(tool, Tool[int])


def _output_event_payload() -> None:
    from timbal.types.run_status import RunStatus

    ev: OutputEvent[Summary] = OutputEvent(
        run_id="r",
        path="a",
        call_id="c",
        status=RunStatus(code="success"),
        t0=0,
        t1=1,
        output=Summary(title="x"),
    )
    assert_type(ev.output, Summary)
