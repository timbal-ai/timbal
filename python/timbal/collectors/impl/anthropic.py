import time
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from urllib.parse import urlparse

import structlog
from anthropic.types import (
    CitationsDelta,
    CitationsWebSearchResultLocation,
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    #
    ServerToolUseBlock,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    WebSearchToolResultBlock,
)

from ...state import get_run_context
from ...types.content.custom import CustomContent
from ...types.content.text import TextContent
from ...types.content.thinking import ThinkingContent
from ...types.content.tool_use import ToolUseContent
from ...types.events.delta import *
from ...types.message import Message
from .. import register_collector
from ..base import BaseCollector

# Create a type alias for Anthropic events
AnthropicEvent = (
    RawContentBlockStartEvent |
    RawContentBlockDeltaEvent |
    RawContentBlockStopEvent |
    RawMessageStartEvent |
    RawMessageDeltaEvent |
    RawMessageStopEvent
)

logger = structlog.get_logger("timbal.collectors.impl.anthropic")


@register_collector
class AnthropicCollector(BaseCollector):
    """Collector for Anthropic streaming events."""
    
    def __init__(self, start: float, **kwargs: Any):
        super().__init__(**kwargs)
        self._start = start
        self._first_token: float | None = None
        self._output_tokens: int = 0
        #
        self.content: list[dict[str, Any]] = []
    
    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, AnthropicEvent)
    
    @override
    def process(self, event: AnthropicEvent) -> Any:
        """Processes Anthropic streaming events."""
        if self._first_token is None:
            self._first_token = time.perf_counter()
        if isinstance(event, RawMessageStartEvent):
            return self._handle_message_start(event)
        elif isinstance(event, RawContentBlockStartEvent):
            return self._handle_content_block_start(event)
        elif isinstance(event, RawContentBlockDeltaEvent):
            return self._handle_content_block_delta(event)
        elif isinstance(event, RawContentBlockStopEvent):
            return None
        elif isinstance(event, RawMessageDeltaEvent):
            return self._handle_message_delta(event)
        elif isinstance(event, RawMessageStopEvent):
            return None
        else:
            logger.warning("Unknown event type", anthropic_event=event)
    
    def _handle_message_start(self, event: RawMessageStartEvent) -> None:
        """Handle message start events with usage information."""
        self.anthropic_model = event.message.model
        self.content = event.message.content
    
    def _handle_content_block_start(self, event: RawContentBlockStartEvent) -> None:
        """Handle content block start events."""
        if isinstance(event.content_block, ToolUseBlock | ServerToolUseBlock):
            # Start new tool call
            self.content.append({
                "type": event.content_block.type,
                "id": event.content_block.id,
                "name": event.content_block.name,
                "input": "" # claude sends an empty object here {}
            })
            return ToolItem(
                id=event.content_block.id,
                name=event.content_block.name,
                input="", # claude sends an empty object here {}
                is_server_tool_use=event.content_block.type == "server_tool_use",
            )
        elif isinstance(event.content_block, ThinkingBlock):
            self.content.append({
                "type": "thinking",
                "thinking": event.content_block.thinking,
            })
        elif isinstance(event.content_block, RedactedThinkingBlock):
            self.content.append({
                "type": "thinking",
                "thinking": event.content_block.data,
            })
        elif isinstance(event.content_block, WebSearchToolResultBlock):
            if isinstance(event.content_block.content, list):
                content = [item.model_dump() for item in event.content_block.content]
            else:
                content = event.content_block.content.model_dump()
            self.content.append({
                "type": "web_search_tool_result",
                "tool_use_id": event.content_block.tool_use_id,
                "content": content,
            })
            return ToolResultItem(
                id=event.content_block.tool_use_id,
                result=content,
            )
        elif isinstance(event.content_block, TextBlock):
            self.content.append({
                "type": "text",
                "citations": [],
                "text": ""
            })
        else:
            logger.warning("Unhandled content block start event", raw_content_block_start_event=event)
    
    def _handle_content_block_delta(self, event: RawContentBlockDeltaEvent) -> str | None:
        """Handle content block delta events."""
        if isinstance(event.delta, InputJSONDelta):
            tool_delta = event.delta.partial_json
            self.content[-1]["input"] += tool_delta
            return ToolInputItem(
                id=self.content[-1]["id"],
                delta=tool_delta,
            )
        elif isinstance(event.delta, TextDelta):
            text_delta = event.delta.text
            self.content[-1]["text"] += text_delta
            # return text_delta
            return TextItem(delta=text_delta)
        elif isinstance(event.delta, ThinkingDelta):
            thinking_delta = event.delta.thinking
            self.content[-1]["thinking"] += thinking_delta
            return ThinkingItem(delta=thinking_delta)
        elif isinstance(event.delta, CitationsDelta):
            if isinstance(event.delta.citation, CitationsWebSearchResultLocation):
                self.content[-1]["citations"].append(event.delta.citation) # ? Parse this
            else:
                logger.warning("Unhandled citation delta event", citation_delta_event=event.delta.citation)
        elif isinstance(event.delta, SignatureDelta):
            return None
        else:
            logger.warning("Unhandled content block delta event", raw_content_block_delta_event=event)
    
    def _handle_message_delta(self, event: RawMessageDeltaEvent) -> None:
        """Handle message delta events with output usage information."""
        run_context = get_run_context()
        anthropic_model = self.anthropic_model # Resolved in RawMessageStartEvent
        def _update_usage(usage):
            for k, v in usage.items():
                if isinstance(v, dict):
                    _update_usage(v)
                elif isinstance(v, int) and v > 0:
                    run_context.update_usage(f"{anthropic_model}:{k}", v)
        _update_usage(event.usage.model_dump())
    
    @override
    def result(self) -> Message:
        """Returns structured Anthropic response."""
        span = get_run_context().current_span()
        ttft = self._first_token - self._start
        span.metadata["ttft"] = ttft
        tps = self._output_tokens / (time.perf_counter() - self._first_token)
        span.metadata["tps"] = tps

        # Check if this is an output_model_tool call and intercept it
        content = []
        for content_block in self.content:
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
                content.append(ToolUseContent(
                    id=content_block["id"],
                    name=content_block["name"],
                    input=content_block["input"],
                    is_server_tool_use=True
                ))
            elif content_block["type"] == "web_search_tool_result":
                content.append(CustomContent(value=content_block))
            elif content_block["type"] == "thinking":
                content.append(ThinkingContent(thinking=content_block["thinking"]))
            elif content_block["type"] == "text":
                text = content_block["text"]
                # TODO Make an effort to deduplicate these
                for citation in content_block["citations"]:
                    domain = urlparse(citation.url).netloc
                    domain = domain.lstrip("www.")
                    text += f" [[{domain}]({citation.url})]"
                if len(content) > 0 and isinstance(content[-1], TextContent):
                    content[-1].text += text
                else:
                    content.append(TextContent(text=text))
            else:
                # Unreachable
                raise AssertionError(f"Unknown content block type: {content_block['type']}")
        
        return Message(role="assistant", content=content)
