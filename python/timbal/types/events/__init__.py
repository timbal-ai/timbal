# ruff: noqa: F401
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from openai.types.chat import ChatCompletionChunk

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


from .chunk import ChunkEvent
from .output import OutputEvent
from .start import StartEvent
