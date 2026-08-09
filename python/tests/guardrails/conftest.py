"""Shared fixtures for guardrail tests.

``StreamingTestModel`` is the missing piece TestModel can't provide: it streams real
``TextDelta`` / ``ThinkingDelta`` / ``ToolUse`` items through the LLM router path, so the
agent's delta handling — in-flight scrubbing, buffer-until-verdict, tail flushing,
per-block scrubbers — gets true end-to-end coverage instead of unit-only coverage.
"""

import json
from typing import Any

from timbal.collectors import _collector_registry
from timbal.collectors.base import BaseCollector
from timbal.types.content import TextContent, ThinkingContent, ToolUseContent
from timbal.types.events.delta import DeltaItem, TextDelta, ThinkingDelta, ToolUse
from timbal.types.message import Message


class _StreamChunk:
    """Marker wrapper so the collector registry can dispatch on our chunks."""

    __test__ = False

    def __init__(self, item: DeltaItem) -> None:
        self.item = item


class StreamingTestCollector(BaseCollector):
    """Accumulates streamed delta items into a final Message.

    ``process()`` returns the DeltaItem itself, which ``Runnable._execute_handler``
    wraps into a real ``DeltaEvent`` — exactly like provider collectors do.
    """

    __test__ = False

    def __init__(self, async_gen: Any, **kwargs: Any) -> None:  # noqa: ARG002 — collectors are constructed with start=
        super().__init__(async_gen)
        self._text: dict[str, str] = {}
        self._thinking: dict[str, str] = {}
        self._tool_uses: dict[str, ToolUse] = {}
        self._order: list[tuple[str, str]] = []  # (kind, block_id) in first-seen order

    @classmethod
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, _StreamChunk)

    def _track(self, kind: str, block_id: str) -> None:
        if (kind, block_id) not in self._order:
            self._order.append((kind, block_id))

    def process(self, event: Any) -> Any:
        item = event.item
        if isinstance(item, TextDelta):
            self._track("text", item.id)
            self._text[item.id] = self._text.get(item.id, "") + item.text_delta
        elif isinstance(item, ThinkingDelta):
            self._track("thinking", item.id)
            self._thinking[item.id] = self._thinking.get(item.id, "") + item.thinking_delta
        elif isinstance(item, ToolUse):
            self._track("tool_use", item.id)
            self._tool_uses[item.id] = item
        return item

    def result(self) -> Message:
        content: list[Any] = []
        for kind, block_id in self._order:
            if kind == "thinking":
                content.append(ThinkingContent(thinking=self._thinking[block_id]))
            elif kind == "text":
                content.append(TextContent(text=self._text[block_id]))
            else:
                tool_use = self._tool_uses[block_id]
                content.append(
                    ToolUseContent(
                        id=tool_use.id,
                        name=tool_use.name,
                        input=json.loads(tool_use.input) if tool_use.input else {},
                    )
                )
        stop_reason = "tool_use" if self._tool_uses else "end_turn"
        return Message(role="assistant", content=content, stop_reason=stop_reason)


def text_stream(text: str, *, block_id: str = "t1", chunk_size: int = 7) -> list[DeltaItem]:
    """Split text into TextDelta chunks (default size chosen to split patterns mid-way)."""
    return [
        TextDelta(id=block_id, text_delta=text[i : i + chunk_size]) for i in range(0, len(text), chunk_size)
    ]


def thinking_stream(text: str, *, block_id: str = "th1", chunk_size: int = 7) -> list[DeltaItem]:
    return [
        ThinkingDelta(id=block_id, thinking_delta=text[i : i + chunk_size])
        for i in range(0, len(text), chunk_size)
    ]


def tool_use_item(name: str, input: dict, *, block_id: str = "call_1") -> DeltaItem:
    return ToolUse(id=block_id, name=name, input=json.dumps(input))


class StreamingTestModel:
    """Drop-in model that streams scripted DeltaItems. No network calls.

    ``scripts`` is a list of turns; each turn is a list of DeltaItems (build with
    ``text_stream`` / ``thinking_stream`` / ``tool_use_item``). Turn selection mirrors
    TestModel: the number of assistant messages already in the conversation picks the
    script, cycling to the last one when exhausted.
    """

    __test__ = False

    provider: str = "test"
    model_name: str = "streaming"

    _collector_registered: bool = False

    def __init__(self, scripts: list[list[DeltaItem]]) -> None:
        if not scripts:
            raise ValueError("StreamingTestModel requires at least one script.")
        self.scripts = scripts
        self.call_count = 0

    async def stream(self, messages: list, **_kwargs: Any) -> Any:
        if not StreamingTestModel._collector_registered:
            _collector_registry.register(StreamingTestCollector)
            StreamingTestModel._collector_registered = True

        self.call_count += 1
        step = sum(1 for m in messages if m.role == "assistant")
        script = self.scripts[min(step, len(self.scripts) - 1)]
        for item in script:
            yield _StreamChunk(item)

    def __str__(self) -> str:
        return "test/streaming"
