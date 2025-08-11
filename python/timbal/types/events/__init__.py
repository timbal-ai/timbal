# ruff: noqa: F401
from typing import Annotated

from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from openai.types.chat import ChatCompletionChunk
from pydantic import Field

from .base import BaseEvent
from .chunk import ChunkEvent
from .output import OutputEvent
from .start import StartEvent

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

# Create a discriminated union of all possible event types.
# Pydantic will use the 'type' field to determine which model to use.
Event = Annotated[
    StartEvent | OutputEvent | ChunkEvent,
    Field(discriminator="type"),
]
