import time
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextDelta,
    ToolUseBlock,
)

from ...state.context import RunContext
from ...types.message import Message
from .. import register_collector
from ..base import EventCollector

# Create a type alias for Anthropic events
AnthropicEvent = (
    RawContentBlockStartEvent |
    RawContentBlockDeltaEvent |
    RawContentBlockStopEvent |
    RawMessageStartEvent |
    RawMessageDeltaEvent |
    RawMessageStopEvent
)


@register_collector
class AnthropicCollector(EventCollector):
    """Collector for Anthropic streaming events."""
    
    def __init__(self, run_context: RunContext, start: float):
        super().__init__(run_context, start)
        self._content: str = ""
        self._tool_calls: list[dict[str, Any]] = []
        self._current_block: dict[str, Any] | None = None
        self._first_token: float | None = None
        self._output_tokens: int = 0
    
    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, AnthropicEvent)
    
    @override
    def process(self, event: AnthropicEvent) -> Any:
        """Processes Anthropic streaming events."""
        if isinstance(event, RawMessageStartEvent):
            return self._handle_message_start(event)
        
        if isinstance(event, RawContentBlockStartEvent):
            return self._handle_content_block_start(event)
        
        if isinstance(event, RawContentBlockDeltaEvent):
            if self._first_token is None:
                self._first_token = time.perf_counter()
            return self._handle_content_block_delta(event)
        
        if isinstance(event, RawMessageDeltaEvent):
            return self._handle_message_delta(event)
        
        return None
    
    def _handle_message_start(self, event: RawMessageStartEvent) -> None:
        """Handle message start events with usage information."""
        if event.message.usage:
            # Store model info for later usage tracking
            anthropic_model = event.message.model
            self.anthropic_model = anthropic_model
            
            input_tokens_key = f"{anthropic_model}:input_tokens"
            input_tokens = event.message.usage.input_tokens
            self._run_context.update_usage(input_tokens_key, input_tokens)
        
        return None
    
    def _handle_content_block_start(self, event: RawContentBlockStartEvent) -> None:
        """Handle content block start events."""
        if isinstance(event.content_block, ToolUseBlock):
            # Start new tool call
            self._current_block = {
                "type": "tool_use",
                "id": event.content_block.id,
                "name": event.content_block.name,
                "input": ""
            }
            self._tool_calls.append(self._current_block)
        else:
            # Handle text content
            if hasattr(event.content_block, 'text'):
                self._content += event.content_block.text
        
        return None
    
    def _handle_content_block_delta(self, event: RawContentBlockDeltaEvent) -> str | None:
        """Handle content block delta events."""
        if isinstance(event.delta, InputJSONDelta):
            # Add to current tool call input
            if self._current_block:
                self._current_block["input"] += event.delta.partial_json
            return None
        elif isinstance(event.delta, TextDelta):
            # Add to content
            text_chunk = event.delta.text
            self._content += text_chunk
            return text_chunk
        
        return None
    
    def _handle_message_delta(self, event: RawMessageDeltaEvent) -> None:
        """Handle message delta events with output usage information."""
        if event.usage:
            anthropic_model = self.anthropic_model
            output_tokens_key = f"{anthropic_model}:output_tokens"
            output_tokens = event.usage.output_tokens
            self._output_tokens += output_tokens
            self._run_context.update_usage(output_tokens_key, output_tokens)
        
        return None
    
    @override
    def collect(self) -> Message:
        """Returns structured Anthropic response."""
        ttft = self._first_token - self._start
        self._run_context.current_trace().metadata["ttft"] = ttft
        tps = self._output_tokens / (time.perf_counter() - self._first_token)
        self._run_context.current_trace().metadata["tps"] = tps

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
            content.extend(self._tool_calls)
        
        return Message.validate({
            "role": "assistant",
            "content": content
        })
