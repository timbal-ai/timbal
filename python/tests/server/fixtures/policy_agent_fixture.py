"""An agent whose model calls ``write`` (not read-only) and then answers; ``WRITES`` records the calls that ran."""

from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

WRITES: list[str] = []


def _write() -> str:
    WRITES.append("write")
    return "written"


def _handler(messages):
    if messages[-1].role == "tool":
        return "done"
    return Message(
        role="assistant",
        content=[ToolUseContent(id="call-write", name="write", input={})],
        stop_reason="tool_use",
    )


policy_agent = Agent(
    name="policy_agent",
    model=TestModel(handler=_handler),
    tools=[Tool(name="write", handler=_write), Tool(name="peek", handler=lambda: "peeked", read_only=True)],
)
