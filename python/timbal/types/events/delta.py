"""Delta event system for fine-grained streaming output.

This module provides a structured event system for streaming LLM outputs with
rich semantic information. Unlike the simple ChunkEvent which only carries raw
data, DeltaEvents provide typed, structured information about different types
of content being streamed (text, tool calls, thinking, etc.).

The delta event system is enabled via the TIMBAL_DELTA_EVENTS environment variable
and provides better observability and control over streaming LLM responses.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .base import BaseEvent


class DeltaItem(BaseModel):
    id: str
    type: str


class ToolUse(DeltaItem):
    type: Literal["tool_use"] = "tool_use"
    name: str
    input: str = ""
    is_server_tool_use: bool = False


class ToolUseDelta(DeltaItem):
    type: Literal["tool_use_delta"] = "tool_use_delta"
    input_delta: str


class Text(DeltaItem):
    type: Literal["text"] = "text"
    text: str


class TextDelta(DeltaItem):
    type: Literal["text_delta"] = "text_delta"
    text_delta: str


class Thinking(DeltaItem):
    type: Literal["thinking"] = "thinking"
    thinking: str


class ThinkingDelta(DeltaItem):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking_delta: str


class Custom(DeltaItem):
    type: Literal["custom"] = "custom"
    data: Any


class ContentBlockStop(DeltaItem):
    type: Literal["content_block_stop"] = "content_block_stop"


class DeltaEvent(BaseEvent):
    type: Literal["DELTA"] = "DELTA"
    item: Annotated[
        ToolUse | ToolUseDelta | 
        Text | TextDelta | 
        Thinking | ThinkingDelta | 
        Custom | ContentBlockStop,
        Field(discriminator="type")
    ]
