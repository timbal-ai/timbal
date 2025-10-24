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
    """Base class for all delta items in the streaming output.
    
    DeltaItems represent different types of content that can be streamed from
    an LLM or other runnable. Each item type has specific fields relevant to
    that content type.
    
    The discriminator field 'type' is used to determine which concrete subclass
    to instantiate when deserializing events.
    """
    type: Literal["tool", "tool_input", "tool_result", "text", "thinking", "custom"]
    # ? id: str -> should we add an id to every block


class ToolItem(DeltaItem):
    """Represents the start of a tool call in the streaming output.
    
    Emitted when an LLM begins calling a tool. Contains the tool's identifier,
    name, and initial (usually empty) input. Subsequent ToolInputItem events
    will provide the actual input parameters as they stream in.
    
    Attributes:
        type: Discriminator value, always "tool"
        id: Unique identifier for this tool call instance
        name: Name of the tool being called
        input: Initial input string (typically empty at start)
        is_server_tool_use: Whether this is a remote/server-side tool execution
    """
    type: Literal["tool"] = "tool"
    id: str
    name: str
    input: str = ""
    is_server_tool_use: bool = False


class ToolInputItem(DeltaItem):
    """Represents a chunk of streaming tool input parameters.
    
    Emitted as the LLM streams the input parameters for a tool call. Multiple
    ToolInputItem events are emitted for a single tool call, each containing
    a delta (incremental chunk) of the JSON input string.
    
    Attributes:
        type: Discriminator value, always "tool_input"
        id: Identifier of the tool call this input belongs to
        delta: Incremental chunk of the JSON input string
    """
    type: Literal["tool_input"] = "tool_input"
    id: str
    delta: str


class ToolResultItem(DeltaItem):
    """Represents the result of a completed tool execution.
    
    Emitted when a tool call completes and returns its result. This is typically
    used for remote/server-side tool executions where the result is provided by
    the LLM provider rather than executed locally.
    
    Attributes:
        type: Discriminator value, always "tool_result"
        id: Identifier of the tool call this result belongs to
        result: The actual result data from the tool execution
    """
    type: Literal["tool_result"] = "tool_result"
    id: str
    result: Any


class TextItem(DeltaItem):
    """Represents a chunk of streaming text output.
    
    Emitted as the LLM generates text responses. Multiple TextItem events
    are emitted for a single text response, each containing a delta
    (incremental chunk) of the text.
    
    This is the most common delta item type for standard LLM text generation.
    
    Attributes:
        type: Discriminator value, always "text"
        delta: Incremental chunk of text content
    """
    type: Literal["text"] = "text"
    delta: str


class ThinkingItem(DeltaItem):
    """Represents a chunk of streaming 'thinking' or reasoning output.
    
    Emitted when an LLM with extended thinking capabilities (e.g., Claude with
    thinking blocks) streams its internal reasoning process. This allows
    observability into the model's thought process before generating the
    final response.
    
    Attributes:
        type: Discriminator value, always "thinking"
        delta: Incremental chunk of thinking/reasoning content
    """
    type: Literal["thinking"] = "thinking"
    delta: str


class CustomItem(DeltaItem):
    """Represents arbitrary custom content in the streaming output.
    
    This is an escape hatch for streaming content that doesn't fit into the
    standard delta item types. It allows users and custom collectors to emit
    arbitrary typed data while still participating in the delta event system.
    
    Use cases:
    - Custom LLM provider-specific features
    - Multimodal content (images, audio) not yet standardized
    - Application-specific streaming data
    - Experimental features before standardization
    
    Example:
        ```python
        # Streaming image generation progress
        CustomItem(
            data={"type": "image_progress", "percent": 45, "url": "..."}
        )
        
        # Custom provider feature
        CustomItem(
            data={"type": "confidence_score", "score": 0.95}
        )
        ```
    
    Attributes:
        type: Discriminator value, always "custom"
        data: Arbitrary data payload of any type
    """
    type: Literal["custom"] = "custom"
    data: Any


class DeltaEvent(BaseEvent):
    """Event emitted when a structured delta of data is received during streaming.
    
    DeltaEvents provide fine-grained, typed information about streaming LLM outputs.
    Unlike ChunkEvents which carry raw, untyped data, DeltaEvents use discriminated
    unions of DeltaItem subclasses to provide semantic information about what type
    of content is being streamed.
    
    This enables:
    - Better observability into LLM reasoning and tool usage
    - Type-safe handling of different content types
    - Structured logging and monitoring of streaming responses
    - Fine-grained control over UI rendering of different content types
    
    DeltaEvents are enabled via the TIMBAL_DELTA_EVENTS environment variable.
    When disabled, the system falls back to simpler ChunkEvents for text content.
    
    Example:
        ```python
        async for event in agent(input="Hello"):
            if isinstance(event, DeltaEvent):
                if isinstance(event.item, TextItem):
                    print(event.item.delta, end="", flush=True)
                elif isinstance(event.item, ToolItem):
                    print(f"\nCalling tool: {event.item.name}")
        ```
    
    Attributes:
        type: Event type discriminator, always "DELTA"
        item: The specific delta item containing the streamed content
    """
    type: Literal["DELTA"] = "DELTA"
    item: Annotated[
        ToolItem | ToolInputItem | ToolResultItem | TextItem | ThinkingItem | CustomItem,
        Field(discriminator="type")
    ]
    """The structured delta item containing typed streaming output."""
