from typing import Any

from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ConfigDict

# Create a type alias for Anthropic events
AnthropicEvent = (
    RawContentBlockStartEvent |
    RawContentBlockDeltaEvent |
    RawContentBlockStopEvent |
    RawMessageStartEvent |
    RawMessageDeltaEvent |
    RawMessageStopEvent
)


# Create a type alias for OpenAI events (in the future we may have more than one)
OpenAIEvent = ChatCompletionChunk


class TimbalEvent(BaseModel):
    """Base class for all timbal events yielded during flow execution."""
    # Allow storing extra fields in the model.
    model_config = ConfigDict(extra="allow")

    type: str
    """The type of the event. This will be very useful for serializing and deserializing events."""
    run_id: str | None = None
    """The id of the run this event was emitted from."""
    parent_step_id: str | None = None
    """The id of the parent step that yielded this event."""
    step_id: str | None = None
    """The id of the step that yielded this event."""


class StepStartEvent(TimbalEvent):
    """Event emitted when a step starts execution."""
    type: str = "STEP_START"


class StepChunkEvent(TimbalEvent):
    """Event emitted for streaming output chunks from a step."""
    type: str = "STEP_CHUNK"

    step_chunk: Any
    """The chunk of output from the step."""


class StepOutputEvent(TimbalEvent):
    """Event emitted when a step completes with its full output."""
    type: str = "STEP_OUTPUT"

    input: Any
    """The input arguments passed to the step."""
    output: Any | None = None
    """The result of the step (if any)."""
    error: Any | None = None
    """The error that occurred during the step (if any)."""
    t0: int 
    """The start time of the step in milliseconds."""
    t1: int 
    """The end time of the step in milliseconds."""
    usage: dict[str, int] = {}
    """The usage of the step."""


class FlowOutputEvent(TimbalEvent):
    """Event emitted when a flow completes with its full output."""
    type: str = "FLOW_OUTPUT"

    output: Any | None = None
    """The output of the flow."""
