import time
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from uuid_extensions import uuid7

from ...state import get_run_context
from ...types.content.text import TextContent
from ...types.content.thinking import ThinkingContent
from ...types.content.tool_use import ToolUseContent
from ...types.message import Message
from .. import register_collector
from ..base import BaseCollector

# Create type aliases for OpenAI events
ChatCompletionEvent = ChatCompletionChunk
ResponseEvent = (
    ResponseCreatedEvent |
    ResponseInProgressEvent |
    ResponseOutputItemAddedEvent |
    ResponseWebSearchCallInProgressEvent |
    ResponseWebSearchCallSearchingEvent |
    ResponseWebSearchCallCompletedEvent |
    ResponseOutputItemDoneEvent |
    ResponseContentPartAddedEvent |
    ResponseOutputTextAnnotationAddedEvent |
    ResponseTextDeltaEvent |
    ResponseTextDoneEvent |
    ResponseContentPartDoneEvent |
    ResponseCompletedEvent
)

logger = structlog.get_logger("timbal.collectors.impl.openai")


@register_collector
class ChatCompletionCollector(BaseCollector):
    """Collector for OpenAI chat completions streaming events."""
    
    def __init__(self, start: float, **kwargs: Any):
        super().__init__(**kwargs)
        self._start = start
        self._content: str = ""
        self._tool_calls: list[dict[str, Any]] = []
        self._current_tool_call: dict[str, Any] | None = None
        self._first_token: float | None = None
        self._output_tokens: int = 0
    
    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, ChatCompletionEvent)
    
    @override
    def process(self, event: ChatCompletionEvent) -> Any:
        """Processes OpenAI streaming events."""
        # Handle usage statistics
        if event.usage:
            self._handle_usage(event)
        if not len(event.choices):
            return None
        # Calculate TTFT (Time To First Token) on first token
        if self._first_token is None:
            self._first_token = time.perf_counter()
        # Handle tool calls
        if event.choices[0].delta.tool_calls:
            return self._handle_tool_calls(event)
        # Handle text content
        if event.choices[0].delta.content:
            return self._handle_text_content(event)
    
    def _handle_usage(self, event: ChatCompletionEvent) -> None:
        """Handle usage statistics from OpenAI events."""
        run_context = get_run_context()
        openai_model = event.model
        openai_usage = event.usage
        input_tokens = int(openai_usage.prompt_tokens)
        input_tokens_details = openai_usage.prompt_tokens_details
        if hasattr(input_tokens_details, "cached_tokens") and input_tokens_details.cached_tokens is not None:
            input_cached_tokens = int(input_tokens_details.cached_tokens)
            if input_cached_tokens:
                input_tokens -= input_cached_tokens
                run_context.update_usage(f"{openai_model}:input_cached_tokens", input_cached_tokens)
        if hasattr(input_tokens_details, "audio_tokens") and input_tokens_details.audio_tokens is not None:
            input_audio_tokens = int(input_tokens_details.audio_tokens)
            if input_audio_tokens:
                input_tokens -= input_audio_tokens
                run_context.update_usage(f"{openai_model}:input_audio_tokens", input_audio_tokens)
        run_context.update_usage(f"{openai_model}:input_text_tokens", input_tokens)
        output_tokens = int(openai_usage.completion_tokens)
        self._output_tokens += output_tokens
        output_tokens_details = openai_usage.completion_tokens_details
        if hasattr(output_tokens_details, "audio_tokens") and output_tokens_details.audio_tokens is not None:
            output_audio_tokens = int(output_tokens_details.audio_tokens)
            if output_audio_tokens:
                output_tokens -= output_audio_tokens
                run_context.update_usage(f"{openai_model}:output_audio_tokens", output_audio_tokens)
        run_context.update_usage(f"{openai_model}:output_text_tokens", output_tokens)
    
    def _handle_tool_calls(self, event: ChatCompletionEvent) -> None:
        """Handle tool call events from OpenAI."""
        tool_call = event.choices[0].delta.tool_calls[0]
        # TODO Review this for parallel tool calls
        if tool_call.id:
            # Start new tool call
            self._current_tool_call = {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": tool_call.function.arguments
            }
            self._tool_calls.append(self._current_tool_call)
        else:
            self._current_tool_call["input"] += tool_call.function.arguments
    
    def _handle_text_content(self, event: ChatCompletionEvent) -> str:
        """Handle text content from OpenAI events."""
        text_chunk = event.choices[0].delta.content
        # Handle citations if present
        if hasattr(event, "citations"):
            self.citations = event.citations
        self._content += text_chunk
        return text_chunk
    
    @override
    def result(self) -> Message:
        """Returns structured OpenAI response."""
        span = get_run_context().current_span()
        ttft = self._first_token - self._start
        span.metadata["ttft"] = ttft
        tps = self._output_tokens / (time.perf_counter() - self._first_token)
        span.metadata["tps"] = tps

        content = []
        if self._content:
            content.append(self._content)

        if self._tool_calls:
            # Check if this is an output_model_tool call and intercept it
            for tool_call in self._tool_calls:
                if tool_call.get("name") == "output_model_tool":
                    # Convert to text message instead of tool call
                    return Message.validate({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": tool_call.get("input", "{}")
                        }]
                    })
            # Openai allows the use of custom IDs for tool calls. 
            # We choose to generate our own random IDs for consistency and to make sure they don't collide
            # (they are not transparent with the algs being used)
            tool_calls = [
                {**tc, "id": uuid7(as_type="str").replace("-", "")}
                for tc in self._tool_calls
            ]
            content.extend(tool_calls)

        return Message.validate({
            "role": "assistant",
            "content": content
        })


@register_collector
class ResponseCollector(BaseCollector):
    """Collector for OpenAI responses streaming events."""
    
    def __init__(self, start: float, **kwargs: Any):
        super().__init__(**kwargs)
        self._start = start
        self._first_token: float | None = None
        self._output_tokens: int = 0
        #
        self.content: dict[str, dict[str, Any]] = {}
    
    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, ResponseEvent)
    
    @override
    def process(self, event: ResponseEvent) -> Any:
        """Processes OpenAI responses streaming events."""
        if self._first_token is None:
            self._first_token = time.perf_counter()
        if isinstance(event, ResponseCreatedEvent):
            return self._handle_created(event)
        elif isinstance(event, ResponseInProgressEvent):
            return None
        elif isinstance(event, ResponseOutputItemAddedEvent):
            return self._handle_output_item_added(event)
        elif isinstance(event, ResponseContentPartAddedEvent):
            return self._handle_content_part_added(event)
        elif isinstance(event, ResponseContentPartDoneEvent):
            return None
        elif isinstance(event, ResponseTextDeltaEvent):
            return self._handle_text_delta(event)
        elif isinstance(event, ResponseTextDoneEvent):
            return None
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            return self._handle_function_call_arguments_delta(event)
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            return None
        elif isinstance(event, ResponseWebSearchCallInProgressEvent):
            return None
        elif isinstance(event, ResponseWebSearchCallSearchingEvent):
            return None
        elif isinstance(event, ResponseWebSearchCallCompletedEvent):
            return None
        elif isinstance(event, ResponseOutputItemDoneEvent):
            return self._handle_output_item_done(event)
        elif isinstance(event, ResponseCompletedEvent):
            return self._handle_completed(event)
        elif isinstance(event, ResponseOutputTextAnnotationAddedEvent):
            return self._handle_output_text_annotation_added(event)
        elif isinstance(event, ResponseReasoningSummaryPartAddedEvent):
            return self._handle_reasoning_summary_part_added(event)
        elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            return self._handle_reasoning_summary_text_delta(event)
        elif isinstance(event, ResponseReasoningSummaryTextDoneEvent):
            return None
        elif isinstance(event, ResponseReasoningSummaryPartDoneEvent):
            return None
        else:
            logger.warning("Unhandled response event", response_event=event)

    def _handle_created(self, event: ResponseCreatedEvent) -> None:
        """Handle created events from OpenAI."""
        self.model = event.response.model
    
    def _handle_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
        """Handle output item added events from OpenAI."""
        if isinstance(event.item, ResponseFunctionToolCall):
            self.content[event.item.id] = {
                "type": "tool_use",
                "id": event.item.call_id,
                "name": event.item.name,
                "input": event.item.arguments, # openai sends an empty string here
            }
        elif isinstance(event.item, ResponseOutputMessage):
            return None
        elif isinstance(event.item, ResponseFunctionWebSearch):
            return None
        elif isinstance(event.item, ResponseReasoningItem):
            return None
        else:
            logger.warning("Unhandled output item added event", response_output_item_added_event=event)
    
    def _handle_content_part_added(self, event: ResponseContentPartAddedEvent) -> None:
        """Handle content part added events from OpenAI."""
        self.content[event.item_id] = {
            "type": "text",
            "citations": [],
            "text": event.part.text,
        }

    def _handle_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        """Handle text delta events from OpenAI."""
        self.content[event.item_id]["text"] += event.delta

    def _handle_output_text_annotation_added(self, event: ResponseOutputTextAnnotationAddedEvent) -> None:
        """Handle output text annotation added events from OpenAI."""
        self.content[event.item_id]["citations"].append(event.annotation)
    
    def _handle_function_call_arguments_delta(self, event: ResponseFunctionCallArgumentsDeltaEvent) -> None:
        """Handle function call arguments delta events from OpenAI."""
        self.content[event.item_id]["input"] += event.delta

    def _handle_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        """Handle output item done events from OpenAI."""
        if isinstance(event.item, ResponseFunctionWebSearch):
            get_run_context().update_usage(f"{self.model}:web_search_requests", 1)
    
    def _handle_reasoning_summary_part_added(self, event: ResponseReasoningSummaryPartAddedEvent) -> None:
        """Handle reasoning summary part added events from OpenAI."""
        self.content[event.item_id] = {
            "type": "thinking",
            "thinking": event.part.text, # Usually empty string from the beginning
        }

    def _handle_reasoning_summary_text_delta(self, event: ResponseReasoningSummaryTextDeltaEvent) -> None:
        """Handle reasoning summary text delta events from OpenAI."""
        self.content[event.item_id]["thinking"] += event.delta
    
    def _handle_completed(self, event: ResponseCompletedEvent) -> None:
        """Handle completed events from OpenAI."""
        run_context = get_run_context()
        usage = event.response.usage
        input_tokens = int(usage.input_tokens)
        input_tokens_details = usage.input_tokens_details
        if hasattr(input_tokens_details, "cached_tokens"):
            input_cached_tokens = int(input_tokens_details.cached_tokens)
            if input_cached_tokens:
                input_tokens -= input_cached_tokens
                run_context.update_usage(f"{self.model}:input_cached_tokens", input_cached_tokens)
        if hasattr(input_tokens_details, "audio_tokens"):
            input_audio_tokens = int(input_tokens_details.audio_tokens)
            if input_audio_tokens:
                input_tokens -= input_audio_tokens
                run_context.update_usage(f"{self.model}:input_audio_tokens", input_audio_tokens)
        run_context.update_usage(f"{self.model}:input_text_tokens", input_tokens)
        output_tokens = int(usage.output_tokens)
        self._output_tokens += output_tokens
        output_tokens_details = usage.output_tokens_details
        if hasattr(output_tokens_details, "audio_tokens"):
            output_audio_tokens = int(output_tokens_details.audio_tokens)
            if output_audio_tokens:
                output_tokens -= output_audio_tokens
                run_context.update_usage(f"{self.model}:output_audio_tokens", output_audio_tokens)
        run_context.update_usage(f"{self.model}:output_text_tokens", output_tokens)
    
    @override
    def result(self) -> Message:
        """Returns structured OpenAI response."""
        span = get_run_context().current_span()
        ttft = self._first_token - self._start
        span.metadata["ttft"] = ttft
        tps = self._output_tokens / (time.perf_counter() - self._first_token)
        span.metadata["tps"] = tps

        content = []
        for content_block in self.content.values(): # Python dicts are ordered
            if content_block["type"] == "tool_use":
                # Convert to text message and early return instead of tool call
                if content_block["name"] == "output_model_tool":
                    return Message(role="assistant", content=[TextContent(text=content_block.get("input", "{}"))])
                content.append(ToolUseContent(
                    id=content_block["id"],
                    name=content_block["name"],
                    input=content_block["input"]
                ))
            elif content_block["type"] == "server_tool_use":
                continue
            elif content_block["type"] == "server_tool_result":
                continue
            elif content_block["type"] == "thinking":
                content.append(ThinkingContent(thinking=content_block["thinking"]))
            elif content_block["type"] == "text":
                text = content_block["text"]
                # e.g. {'type': 'url_citation', 'end_index': 2538, 'start_index': 2403, 'title': 'Weather Forecast and Conditions for Barcelona, Barcelona, Spain - The Weather Channel | Weather.com', 'url': 'https://weather.com/weather/today/l/b3b13a74649dd0a2a0aada41e1bf764de39e5dacf21d062ef18ecdeb09796ba0?utm_source=openai'}
                # Openai annotations are already formatted into the text
                content.append(TextContent(text=text))
            else:
                # Unreachable
                raise AssertionError(f"Unknown content block type: {content_block['type']}")

        message = Message(role="assistant", content=content)
        return message
