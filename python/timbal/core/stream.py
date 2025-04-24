import asyncio
import copy
from collections.abc import AsyncGenerator, Generator
from typing import Any

import structlog
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    TextDelta,
    ToolUseBlock,
)
from pydantic import BaseModel, ConfigDict
from uuid_extensions import uuid7

from ..types.events import AnthropicEvent, OpenAIEvent
from ..types.events.base import BaseEvent as TimbalBaseEvent
from ..types.events.chunk import ChunkEvent as TimbalChunkEvent
from ..types.events.output import OutputEvent as TimbalOutputEvent
from ..types.events.start import StartEvent as TimbalStartEvent

logger = structlog.get_logger("timbal.core.stream")


async def sync_to_async_gen(gen: Generator[Any, None, None], loop: asyncio.AbstractEventLoop) -> AsyncGenerator[Any, None]:
    """Auxiliary function to convert a sync generator to an async generator."""
    while True:
        # StopIteration is special in Python. It's used to implement generator protocol and can't
        # be pickled/transferred across threads properly. By catching it explicitly in the executor 
        # function and converting it to a sentinel value, we avoid problematic exception propagation.
        def _next():
            try:
                return next(gen)
            except StopIteration: 
                return None
        value = await loop.run_in_executor(None, _next)
        if value is None:
            break
        yield value


class AsyncGenState(BaseModel):
    """Tracks state for async generators during flow execution.

    Allow for arbitrary types not properly handled by Pydantic.
    Also allow for extra fields to be added to the model.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    gen: AsyncGenerator[Any, None] | None = None
    """The async generator being processed."""
    input: Any | None = None
    """The input for the step."""
    collections: Any = []
    """Accumulated results from generator yields (e.g. LLM message chunks)."""
    usage: dict[str, int] = {}
    """Resource usage stats (e.g. token counts for LLM calls)."""
    events_source: str | None = None # Allow None for late initializations.
    """The source of the events (e.g. 'openai', 'anthropic', 'timbal', 'str', ...)."""

    def collect(self) -> Any:
        return self.collections


def handle_openai_event(
    openai_event: OpenAIEvent,
    async_gen_state: AsyncGenState,
) -> Any:
    """Process OpenAI streaming events and update the async generator state.

    Handles different types of OpenAI events including:
    - Usage statistics (token counts)
    - Tool calls (function calling)
    - Text content deltas

    Args:
        openai_event: The OpenAI event to process
        async_gen_state: State object tracking the async generator execution

    Returns:
        Text chunk if the event contains content or tool call info, None otherwise
    """

    # TODO Review. Gemini sends usage stats in every chunk. It doesn't send a separate chunk with the total usage.
    # ? We guess they send the revised number in the last chunk.
    if openai_event.usage:
        openai_model = openai_event.model
        openai_usage = openai_event.usage

        input_tokens = int(openai_usage.prompt_tokens)
        input_tokens_details = openai_usage.prompt_tokens_details
        if hasattr(input_tokens_details, "cached_tokens"):
            input_cached_tokens = int(input_tokens_details.cached_tokens)
            if input_cached_tokens:
                input_tokens -= input_cached_tokens
                async_gen_state.usage[f"{openai_model}:input_cached_tokens"] = input_cached_tokens
        if hasattr(input_tokens_details, "audio_tokens"):
            input_audio_tokens = int(input_tokens_details.audio_tokens)
            if input_audio_tokens:
                input_tokens -= input_audio_tokens
                async_gen_state.usage[f"{openai_model}:input_audio_tokens"] = input_audio_tokens
        # ? We've seen this in some responses. Usually they return the image tokens as regular text tokens.
        # if hasattr(input_tokens_details, "image_tokens"):
        #     input_image_tokens = int(input_tokens_details.image_tokens)
        #     if input_image_tokens:
        #         input_tokens -= input_image_tokens
        #         async_gen_state.usage[f"{openai_model}:input_image_tokens"] = input_image_tokens
        # Text tokens are used as the default.
        # if hasattr(input_tokens_details, "text_tokens"):

        async_gen_state.usage[f"{openai_model}:input_text_tokens"] = input_tokens

        output_tokens = int(openai_usage.completion_tokens)
        output_tokens_details = openai_usage.completion_tokens_details
        if hasattr(output_tokens_details, "audio_tokens"):
            output_audio_tokens = int(output_tokens_details.audio_tokens)
            if output_audio_tokens:
                output_tokens -= output_audio_tokens
                async_gen_state.usage[f"{openai_model}:output_audio_tokens"] = output_audio_tokens

        async_gen_state.usage[f"{openai_model}:output_text_tokens"] = output_tokens

    if not len(openai_event.choices):
        return None

    if openai_event.choices[0].delta.tool_calls:
        tool_call = copy.deepcopy(openai_event.choices[0].delta.tool_calls[0])
        if tool_call.id:
            chunk_result = {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": ""
            }
            async_gen_state.collections.append(chunk_result)
        else:
            # ? Gemini doesn't add an id to the tool call.
            if not async_gen_state.collections:
                tool_use_id = f"call_{uuid7(as_type='str').replace('-', '')}"
                async_gen_state.collections.append({
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_call.function.name,
                    "input": ""
                })
            async_gen_state.collections[-1]["input"] += tool_call.function.arguments
        return None

    if openai_event.choices[0].delta.content:
        text_chunk = openai_event.choices[0].delta.content
        chunk_result = {
            "type": "text",
            "text": text_chunk,
        }

        # Extra is allowed in the async gen pydantic model.
        if hasattr(openai_event, "citations"):
            async_gen_state.citations = openai_event.citations

        if async_gen_state.collections:
            async_gen_state.collections[-1]["text"] += text_chunk
        else:
            async_gen_state.collections.append(chunk_result)
        return text_chunk


def handle_anthropic_event(
    anthropic_event: AnthropicEvent,
    async_gen_state: AsyncGenState,
) -> Any:
    """Process Anthropic streaming events and update the async generator state.

    Handles different types of Anthropic events including:
    - Message start/usage info
    - Content block starts (text or tool use)
    - Content deltas (text or JSON)
    - Message completion with usage stats

    Args:
        anthropic_event: The Anthropic event to process
        async_gen_state: State object tracking the async generator execution

    Returns:
        Text chunk if the event contains content or tool input, None otherwise
    """

    # The raw content block stop event is sent when the LLM sends multiple blocks.
    # The raw message stop event is sent when the LLM is actually done.
    # if isinstance(anthropic_event, RawContentBlockStopEvent | RawMessageStopEvent):
    #     pass

    # TODO Handle other types of tokens.
    if isinstance(anthropic_event, RawMessageStartEvent):
        if anthropic_event.message.usage:
            # Output and delta messages do not include the model. 
            # We store this temp attribute to create the correct usage key underneath.
            anthropic_model = anthropic_event.message.model
            async_gen_state.anthropic_model = anthropic_model
            input_tokens_key = f"{anthropic_model}:input_tokens"
            input_tokens = anthropic_event.message.usage.input_tokens
            existing_input_tokens = async_gen_state.usage.get(input_tokens_key, 0)
            async_gen_state.usage[input_tokens_key] = existing_input_tokens + input_tokens
        return None

    if isinstance(anthropic_event, RawContentBlockStartEvent):
        if isinstance(anthropic_event.content_block, ToolUseBlock):
            chunk_result = {
                "type": "tool_use",
                "id": anthropic_event.content_block.id,
                "name": anthropic_event.content_block.name,
                "input": ""
            }
            async_gen_state.collections.append(chunk_result)
        else:
            chunk_result = {
                "type": "text",
                "text": anthropic_event.content_block.text
            }
            async_gen_state.collections.append(chunk_result)
        return None

    if isinstance(anthropic_event, RawContentBlockDeltaEvent):
        if isinstance(anthropic_event.delta, InputJSONDelta):
            text_chunk = anthropic_event.delta.partial_json
            async_gen_state.collections[-1]["input"] += text_chunk
            return None
        elif isinstance(anthropic_event.delta, TextDelta):
            text_chunk = anthropic_event.delta.text
            async_gen_state.collections[-1]["text"] += text_chunk
            return text_chunk

    if isinstance(anthropic_event, RawMessageDeltaEvent):
        if anthropic_event.usage:
            anthropic_model = async_gen_state.anthropic_model
            output_tokens_key = f"{anthropic_model}:output_tokens"
            output_tokens = anthropic_event.usage.output_tokens
            existing_output_tokens = async_gen_state.usage.get(output_tokens_key, 0)
            async_gen_state.usage[output_tokens_key] = existing_output_tokens + output_tokens
        return None


def handle_timbal_event(
    timbal_event: TimbalBaseEvent,
    async_gen_state: AsyncGenState,
) -> dict[str, Any] | None:
    """Process Timbal flow events and update the async generator state.

    Handles different types of Timbal events including:
    - Step start events (filtered out for parent flows)
    - Step chunk events (currently filtered)
    - Step output events (captures usage statistics)
    - Flow output events (captures final results)

    Args:
        timbal_event: The Timbal event to process
        async_gen_state: State object tracking the async generator execution

    Returns:
        None - Timbal events currently don't return streaming content
    """

    # For a parent flow, we don't need to stream the subflow step start events.
    if isinstance(timbal_event, TimbalStartEvent):
        return None

    # TODO Think how should we stream partial flow outputs results or chunks. 
    # Perhaps we should only stream up if the step output is selected as a flow output.
    if isinstance(timbal_event, TimbalChunkEvent):
        return None
    
    # TODO Think how should we stream partial flow outputs results or chunks. 
    # Perhaps we should only stream up if the step output is selected as a flow output.
    if isinstance(timbal_event, TimbalOutputEvent):
        if timbal_event.usage:
            for k, v in timbal_event.usage.items():
                current_key_value = async_gen_state.usage.get(k, 0)
                async_gen_state.usage[k] = current_key_value + v

        async_gen_state.collections = timbal_event.output
        return None

    return None


def handle_event(
    event: Any,
    async_gen_state: AsyncGenState,
) -> dict[str, Any] | None:
    """Route events to their appropriate handlers based on type.

    Dispatches events to specialized handlers for OpenAI, Anthropic, and Timbal events.
    Provides basic handling for string events and default fallback for other types.

    Args:
        event: The event to process
        async_gen_state: State object tracking the async generator execution

    Returns:
        Processed event content if available, None otherwise
    """
    if isinstance(event, OpenAIEvent):
        if async_gen_state.events_source is None:
            async_gen_state.events_source = "openai"
        if async_gen_state.events_source != "openai":
            raise ValueError("We cannot handle mixed events sources.")
        return handle_openai_event(event, async_gen_state)

    if isinstance(event, AnthropicEvent):
        if async_gen_state.events_source is None:
            async_gen_state.events_source = "anthropic"
        if async_gen_state.events_source != "anthropic":
            raise ValueError("We cannot handle mixed events sources.")
        return handle_anthropic_event(event, async_gen_state)

    if isinstance(event, TimbalBaseEvent):
        if async_gen_state.events_source is None:
            async_gen_state.events_source = "timbal"
        if async_gen_state.events_source != "timbal":
            raise ValueError("We cannot handle mixed events sources.")
        return handle_timbal_event(event, async_gen_state)

    if isinstance(event, str):
        if async_gen_state.events_source is None:
            async_gen_state.events_source = "str"
        if async_gen_state.events_source != "str":
            raise ValueError("We cannot handle mixed events sources.")
        if async_gen_state.collections:
            async_gen_state.collections += event
        else: 
            async_gen_state.collections = event
        return event

    # ? Implement custom behavior for other event types.

    # Default behavior for any other event type.
    async_gen_state.events_source = "any"
    async_gen_state.collections.append(event)
    return event
