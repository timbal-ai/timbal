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

from ..types.events import (
    AnthropicEvent,
    FlowOutputEvent,
    OpenAIEvent,
    StepChunkEvent,
    StepOutputEvent,
    StepStartEvent,
    TimbalEvent,
)

logger = structlog.get_logger("timbal.graph.stream")


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

    Attributes:
        gen (AsyncGenerator[Any, None]): The async generator being processed
        inputs (BaseModel): The inputs for the step (after being resolved and validated)
        results (list[Any]): Accumulated results from generator yields (e.g. LLM message chunks)
        usage (dict[str, Any]): Resource usage stats (e.g. token counts for LLM calls)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    gen: AsyncGenerator[Any, None]
    inputs: BaseModel
    results: list[Any] = []
    usage: dict[str, Any] = {}


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

    # OpenAI SDKs use a message with empty choices to indicate the end of a stream and send the usage.
    if not len(openai_event.choices):
        if openai_event.usage:
            output_tokens = async_gen_state.usage.get("output_tokens", 0)
            output_tokens += openai_event.usage.completion_tokens
            input_tokens = async_gen_state.usage.get("input_tokens", 0)
            input_tokens += openai_event.usage.prompt_tokens
            async_gen_state.usage["output_tokens"] = output_tokens
            async_gen_state.usage["input_tokens"] = input_tokens
        return None

    if openai_event.choices[0].delta.tool_calls:
        tool_call = copy.deepcopy(openai_event.choices[0].delta.tool_calls[0])
        text_chunk = ""
        if tool_call.id:
            chunk_result = {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": ""
            }
            if len(async_gen_state.results):
                text_chunk = ".\n"
            text_chunk += f"Using tool '{tool_call.function.name}' with input: "
            async_gen_state.results.append(chunk_result)
        else:
            text_chunk += tool_call.function.arguments
            async_gen_state.results[-1]["input"] += tool_call.function.arguments
        return text_chunk

    if openai_event.choices[0].delta.content:
        text_chunk = openai_event.choices[0].delta.content
        chunk_result = {
            "type": "text",
            "text": text_chunk,
        }

        # Extra is allowed in the async gen pydantic model.
        if hasattr(openai_event, "citations"):
            async_gen_state.citations = openai_event.citations

        if len(async_gen_state.results):
            async_gen_state.results[-1]["text"] += text_chunk
        else:
            async_gen_state.results.append(chunk_result)
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

    # # The raw content block stop event is sent when the LLM sends multiple blocks.
    # # The raw message stop event is sent when the LLM is actually done.
    # if isinstance(anthropic_event, RawContentBlockStopEvent | RawMessageStopEvent):
    #     pass

    # 
    if isinstance(anthropic_event, RawMessageStartEvent):
        if anthropic_event.message.usage:
            input_tokens = anthropic_event.message.usage.input_tokens
            async_gen_state.usage["input_tokens"] = input_tokens
        return None

    #
    if isinstance(anthropic_event, RawContentBlockStartEvent):
        if isinstance(anthropic_event.content_block, ToolUseBlock):
            chunk_result = {
                "type": "tool_use",
                "id": anthropic_event.content_block.id,
                "name": anthropic_event.content_block.name,
                "input": ""
            }
            async_gen_state.results.append(chunk_result)
        else:
            chunk_result = {
                "type": "text",
                "text": anthropic_event.content_block.text
            }
            async_gen_state.results.append(chunk_result)
        return None

    # 
    if isinstance(anthropic_event, RawContentBlockDeltaEvent):
        if isinstance(anthropic_event.delta, InputJSONDelta):
            text_chunk = anthropic_event.delta.partial_json
            async_gen_state.results[-1]["input"] += text_chunk
            return text_chunk
        elif isinstance(anthropic_event.delta, TextDelta):
            text_chunk = anthropic_event.delta.text
            async_gen_state.results[-1]["text"] += text_chunk
            return text_chunk

    # TODO Check if there is something in here (besides usage)
    if isinstance(anthropic_event, RawMessageDeltaEvent):
        if anthropic_event.usage:
            output_tokens = anthropic_event.usage.output_tokens
            async_gen_state.usage["output_tokens"] = output_tokens
        return None


def handle_timbal_event(
    timbal_event: TimbalEvent,
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
    if isinstance(timbal_event, StepStartEvent):
        return None

    # TODO Think how should we stream partial flow outputs results or chunks. 
    # Perhaps we should only stream up if the step output is selected as a flow output.
    if isinstance(timbal_event, StepChunkEvent):
        return None
    
    if isinstance(timbal_event, StepOutputEvent):
        async_gen_state.usage[timbal_event.step_id] = timbal_event.step_usage
        # Time is not aggregated here, since we might have parallel executions, so the final adding up would be wrong.
        return None

    # TODO Here probably we should inject properties and convert to StepOutputEvent, no?
    if isinstance(timbal_event, FlowOutputEvent):
        async_gen_state.results = timbal_event.outputs
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
        return handle_openai_event(event, async_gen_state)

    if isinstance(event, AnthropicEvent):
        return handle_anthropic_event(event, async_gen_state)

    if isinstance(event, TimbalEvent):
        return handle_timbal_event(event, async_gen_state)

    if isinstance(event, str):
        async_gen_state.results += event
        return event

    # ? Implement custom behavior for other event types.

    # Default behavior for any other event type.
    async_gen_state.results.append(event)
    return event
