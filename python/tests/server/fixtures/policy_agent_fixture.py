"""An agent whose model calls one tool, picked by the prompt, then answers; ``WRITES`` records the writes that ran.

- ``"go"``: ``write`` (not read-only).
- ``"deep"``: ``peek_deep`` (read-only), which runs ``inner``, an agent whose model calls ``write``.
- ``"spawn"``: ``spawn`` (read-only, ``background_mode="always"``), which does the same in the background.
- ``"peek"``: ``peek`` (read-only).

``DONE`` records which of ``peek_deep`` / ``spawn`` finished, so a test knows the background one has run.
"""

import asyncio

from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

WRITES: list[str] = []
DONE: list[str] = []
TOOL_FOR_PROMPT = {"go": "write", "deep": "peek_deep", "spawn": "spawn", "peek": "peek"}


def _write() -> str:
    WRITES.append("write")
    return "written"


def _call(name: str) -> Message:
    return Message(
        role="assistant", content=[ToolUseContent(id=f"call-{name}", name=name, input={})], stop_reason="tool_use"
    )


def _inner_handler(messages):
    return "inner done" if messages[-1].role == "tool" else _call("write")


inner = Agent(
    name="inner",
    model=TestModel(handler=_inner_handler),
    tools=[Tool(name="write", handler=_write)],
    tracing_provider=InMemoryTracingProvider,
)


async def _run_inner(marker: str) -> str:
    await inner(prompt="write it").collect()
    DONE.append(marker)
    return "peeked"


async def _peek_deep() -> str:
    return await _run_inner("peek_deep")


async def _spawn() -> str:
    await asyncio.sleep(0.2)  # the run that started it has ended by now: the branch a background child takes
    return await _run_inner("spawn")


def _handler(messages):
    if messages[-1].role == "tool":
        return "done"
    prompt = next(m for m in messages if m.role == "user").collect_text()
    return _call(TOOL_FOR_PROMPT.get(prompt, "write"))


policy_agent = Agent(
    name="policy_agent",
    model=TestModel(handler=_handler),
    tools=[
        Tool(name="write", handler=_write),
        Tool(name="peek", handler=lambda: "peeked", read_only=True),
        Tool(name="peek_deep", handler=_peek_deep, read_only=True),
        Tool(name="spawn", handler=_spawn, read_only=True, background_mode="always"),
    ],
    tracing_provider=InMemoryTracingProvider,
)
