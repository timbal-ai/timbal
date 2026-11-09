"""An echo agent for chat-session tests: ``start`` runs a short tool, a window to send messages into."""

import asyncio

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


async def work() -> str:
    """Take a moment."""
    await asyncio.sleep(0.4)
    return "worked"


def echo(messages):
    last = messages[-1]
    if last.role == "tool":
        return "done"
    text = next(m.collect_text() for m in reversed(messages) if m.role == "user" and not m.is_runtime())
    if text == "start":
        return Message(
            role="assistant",
            content=[ToolUseContent(id=f"call-{len(messages)}", name="work", input={})],
            stop_reason="tool_use",
        )
    return f"echo: {text}"


chat_agent = Agent(
    name="chat_agent",
    model=TestModel(handler=echo),
    tools=[work],
    tracing_provider=InMemoryTracingProvider,
)
