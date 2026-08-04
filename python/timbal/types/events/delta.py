"""Delta event system for fine-grained streaming output.

This module provides a structured event system for streaming LLM outputs with
rich semantic information. DeltaEvents carry typed, structured information about
different types of content being streamed (text, tool calls, thinking, etc.),
providing better observability and control over streaming LLM responses.
"""

from typing import Any

from ..._slots import SlotModel
from .base import BaseEvent


class DeltaItem(SlotModel):
    __slots__ = ("id",)

    type: str = ""

    _FIELDS = ("id", "type")

    def __init__(self, *, id: str, **_ignored: Any) -> None:
        self.id = id


class ToolUse(DeltaItem):
    __slots__ = ("name", "input", "is_server_tool_use")

    type = "tool_use"

    _FIELDS = DeltaItem._FIELDS + ("name", "input", "is_server_tool_use")

    def __init__(self, *, id: str, name: str, input: str = "", is_server_tool_use: bool = False, **_ignored: Any) -> None:
        super().__init__(id=id)
        self.name = name
        self.input = input
        self.is_server_tool_use = is_server_tool_use


class ToolUseDelta(DeltaItem):
    __slots__ = ("input_delta",)

    type = "tool_use_delta"

    _FIELDS = DeltaItem._FIELDS + ("input_delta",)

    def __init__(self, *, id: str, input_delta: str, **_ignored: Any) -> None:
        super().__init__(id=id)
        self.input_delta = input_delta


class Text(DeltaItem):
    __slots__ = ("text",)

    type = "text"

    _FIELDS = DeltaItem._FIELDS + ("text",)

    def __init__(self, *, id: str, text: str, **_ignored: Any) -> None:
        super().__init__(id=id)
        self.text = text


class TextDelta(DeltaItem):
    __slots__ = ("text_delta",)

    type = "text_delta"

    _FIELDS = DeltaItem._FIELDS + ("text_delta",)

    def __init__(self, *, id: str, text_delta: str, **_ignored: Any) -> None:
        super().__init__(id=id)
        self.text_delta = text_delta


class Thinking(DeltaItem):
    __slots__ = ("thinking",)

    type = "thinking"

    _FIELDS = DeltaItem._FIELDS + ("thinking",)

    def __init__(self, *, id: str, thinking: str, **_ignored: Any) -> None:
        super().__init__(id=id)
        self.thinking = thinking


class ThinkingDelta(DeltaItem):
    __slots__ = ("thinking_delta",)

    type = "thinking_delta"

    _FIELDS = DeltaItem._FIELDS + ("thinking_delta",)

    def __init__(self, *, id: str, thinking_delta: str, **_ignored: Any) -> None:
        super().__init__(id=id)
        self.thinking_delta = thinking_delta


class Custom(DeltaItem):
    __slots__ = ("data",)

    type = "custom"

    _FIELDS = DeltaItem._FIELDS + ("data",)

    def __init__(self, *, id: str, data: Any = None, **_ignored: Any) -> None:
        super().__init__(id=id)
        self.data = data


class ContentBlockStop(DeltaItem):
    __slots__ = ()

    type = "content_block_stop"


_DELTA_ITEM_TYPES: dict[str, type[DeltaItem]] = {
    cls.type: cls
    for cls in (ToolUse, ToolUseDelta, Text, TextDelta, Thinking, ThinkingDelta, Custom, ContentBlockStop)
}


def validate_delta_item(data: DeltaItem | dict[str, Any]) -> DeltaItem:
    """Build a DeltaItem from a dict using the ``type`` discriminator."""
    if isinstance(data, DeltaItem):
        return data
    item_type = data.get("type")
    cls = _DELTA_ITEM_TYPES.get(item_type)
    if cls is None:
        raise ValueError(f"Unknown delta item type {item_type!r}.")
    return cls(**{k: v for k, v in data.items() if k != "type"})


class DeltaEvent(BaseEvent):
    __slots__ = ("item",)

    type = "DELTA"

    _FIELDS = BaseEvent._FIELDS + ("item",)

    def __init__(
        self,
        *,
        run_id: str,
        path: str,
        call_id: str,
        item: DeltaItem | dict[str, Any],
        parent_run_id: str | None = None,
        parent_call_id: str | None = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(
            run_id=run_id,
            path=path,
            call_id=call_id,
            parent_run_id=parent_run_id,
            parent_call_id=parent_call_id,
        )
        self.item = validate_delta_item(item)
