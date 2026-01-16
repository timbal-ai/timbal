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
    ServerToolUseBlock,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    WebSearchToolResultBlock,
)
from anthropic.types.beta import (
    BetaCitationsDelta,
    BetaInputJSONDelta,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRawMessageStopEvent,
    BetaServerToolUseBlock,
    BetaSignatureDelta,
    BetaTextBlock,
    BetaTextDelta,
    BetaThinkingBlock,
    BetaThinkingDelta,
    BetaToolUseBlock,
    BetaWebSearchToolResultBlock,
)

from ...state import get_run_context
from ...types.content.custom import CustomContent
from ...types.content.text import TextContent
from ...types.content.thinking import ThinkingContent
from ...types.content.tool_use import ToolUseContent
from ...types.events.delta import (
    ContentBlockStop as TimbalContentBlockStop,
)
from ...types.events.delta import DeltaItem as TimbalDeltaItem
from ...types.events.delta import Text as TimbalText
from ...types.events.delta import TextDelta as TimbalTextDelta
from ...types.events.delta import Thinking as TimbalThinking
from ...types.events.delta import ThinkingDelta as TimbalThinkingDelta
from ...types.events.delta import ToolUse as TimbalToolUse
from ...types.events.delta import ToolUseDelta as TimbalToolUseDelta
from ...types.message import Message
from .. import register_collector
from ..base import BaseCollector

# Create a type alias for Anthropic events
AnthropicEvent = (
    BetaRawContentBlockStartEvent
    | BetaRawContentBlockDeltaEvent
    | BetaRawContentBlockStopEvent
    | BetaRawMessageStartEvent
    | BetaRawMessageDeltaEvent
    | BetaRawMessageStopEvent
    | RawContentBlockStartEvent
    | RawContentBlockDeltaEvent
    | RawContentBlockStopEvent
    | RawMessageStartEvent
    | RawMessageDeltaEvent
    | RawMessageStopEvent
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
        self._stop_reason: str | None = None
        self.content_blocks: set[str] = set()
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
        if isinstance(event, RawMessageStartEvent | BetaRawMessageStartEvent):
            return self._handle_message_start(event)
        elif isinstance(event, RawContentBlockStartEvent | BetaRawContentBlockStartEvent):
            return self._handle_content_block_start(event)
        elif isinstance(event, RawContentBlockDeltaEvent | BetaRawContentBlockDeltaEvent):
            return self._handle_content_block_delta(event)
        elif isinstance(event, RawContentBlockStopEvent | BetaRawContentBlockStopEvent):
            return self._handle_content_block_stop(event)
        elif isinstance(event, RawMessageDeltaEvent | BetaRawMessageDeltaEvent):
            return self._handle_message_delta(event)
        elif isinstance(event, RawMessageStopEvent | BetaRawMessageStopEvent):
            return None
        else:
            logger.warning("Unknown event type", anthropic_event=event)

    def _handle_message_start(self, event: RawMessageStartEvent) -> None:
        """Handle message start events with usage information."""
        self.id = event.message.id
        self.anthropic_model = event.message.model
        self.content = []

    def _handle_content_block_start(self, event: RawContentBlockStartEvent) -> TimbalDeltaItem | None:
        """Handle content block start events."""
        content_block_id = f"{self.id}-{event.index}"
        if isinstance(
            event.content_block, ToolUseBlock | ServerToolUseBlock | BetaToolUseBlock | BetaServerToolUseBlock
        ):
            self.content.append(
                {
                    "type": event.content_block.type,
                    "id": event.content_block.id,
                    "name": event.content_block.name,
                    "input": "",  # claude sends an empty object here {}
                    "block_id": content_block_id,
                }
            )
            self.content_blocks.add(content_block_id)
            return TimbalToolUse(
                id=content_block_id,
                name=event.content_block.name,
                input="",  # claude sends an empty object here {}
                is_server_tool_use=event.content_block.type == "server_tool_use",
            )
        elif isinstance(event.content_block, ThinkingBlock | BetaThinkingBlock):
            self.content.append(
                {
                    "type": "thinking",
                    "thinking": event.content_block.thinking,
                    "block_id": content_block_id,
                }
            )
            self.content_blocks.add(content_block_id)
            return TimbalThinking(
                id=content_block_id,
                thinking=event.content_block.thinking,
            )
        elif isinstance(event.content_block, WebSearchToolResultBlock | BetaWebSearchToolResultBlock):
            if isinstance(event.content_block.content, list):
                content = [item.model_dump() for item in event.content_block.content]
            else:
                content = event.content_block.content.model_dump()
            # We need to append the web search tool result here so it can be used in subsequent requests
            self.content.append(
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": event.content_block.tool_use_id,
                    "content": content,
                    # We don't add block_id here. Anthropic API blocks unknown params.
                    # "block_id": content_block_id,
                }
            )
            # TODO Return something so we can print the web search results
            # self.content_blocks.add(content_block_id)
            # return TimbalToolResult(
            #     id=event.content_block.tool_use_id,
            #     result=content,
            # )
        elif isinstance(event.content_block, TextBlock | BetaTextBlock):
            self.content.append(
                {
                    "type": "text",
                    "citations": [],
                    "text": "",
                    "block_id": content_block_id,
                }
            )
            self.content_blocks.add(content_block_id)
            return TimbalText(
                id=content_block_id,
                text=event.content_block.text,
            )
        else:
            logger.warning("Unhandled content block start event", raw_content_block_start_event=event)

    def _handle_content_block_delta(self, event: RawContentBlockDeltaEvent) -> TimbalDeltaItem | None:
        """Handle content block delta events."""
        if isinstance(event.delta, InputJSONDelta | BetaInputJSONDelta):
            tool_delta = event.delta.partial_json
            content_block = self.content[-1]
            content_block["input"] += tool_delta
            return TimbalToolUseDelta(
                id=content_block["block_id"],
                input_delta=tool_delta,
            )
        elif isinstance(event.delta, TextDelta | BetaTextDelta):
            text_delta = event.delta.text
            content_block = self.content[-1]
            content_block["text"] += text_delta
            return TimbalTextDelta(
                id=content_block["block_id"],
                text_delta=text_delta,
            )
        elif isinstance(event.delta, ThinkingDelta | BetaThinkingDelta):
            thinking_delta = event.delta.thinking
            content_block = self.content[-1]
            content_block["thinking"] += thinking_delta
            return TimbalThinkingDelta(
                id=content_block["block_id"],
                thinking_delta=thinking_delta,
            )
        elif isinstance(event.delta, CitationsDelta | BetaCitationsDelta):
            if isinstance(event.delta.citation, CitationsWebSearchResultLocation):
                self.content[-1]["citations"].append(event.delta.citation)  # ? Parse this
            else:
                logger.warning("Unhandled citation delta event", citation_delta_event=event.delta.citation)
        elif isinstance(event.delta, SignatureDelta | BetaSignatureDelta):
            self.content[-1]["signature"] = event.delta.signature
            return None
        else:
            logger.warning("Unhandled content block delta event", raw_content_block_delta_event=event)

    def _handle_content_block_stop(self, event: RawContentBlockStopEvent) -> TimbalDeltaItem | None:
        """Handle content block stop events."""
        content_block_id = f"{self.id}-{event.index}"
        if content_block_id in self.content_blocks:
            return TimbalContentBlockStop(id=content_block_id)
        else:
            return None

    def _handle_message_delta(self, event: RawMessageDeltaEvent) -> None:
        """Handle message delta events with output usage information."""
        # Capture stop_reason from the delta event
        # Possible values: 'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'pause_turn', 'refusal'
        if event.delta.stop_reason:
            self._stop_reason = event.delta.stop_reason

        run_context = get_run_context()
        if not run_context:
            return None
        anthropic_model = self.anthropic_model  # Resolved in RawMessageStartEvent

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
        run_context = get_run_context()
        if run_context:
            if self._first_token:
                span = run_context.current_span()
                ttft = self._first_token - self._start
                span.metadata["ttft"] = ttft
                tps = self._output_tokens / (time.perf_counter() - self._first_token)
                span.metadata["tps"] = tps

        content = []
        for content_block in self.content:
            if content_block["type"] == "tool_use":
                content.append(
                    ToolUseContent(
                        id=content_block["id"],
                        name=content_block["name"],
                        input=content_block["input"],
                    )
                )
            elif content_block["type"] == "server_tool_use":
                content.append(
                    ToolUseContent(
                        id=content_block["id"],
                        name=content_block["name"],
                        input=content_block["input"],
                        is_server_tool_use=True,
                    )
                )
            elif content_block["type"] == "web_search_tool_result":
                content.append(CustomContent(value=content_block))
            elif content_block["type"] == "thinking":
                content.append(
                    ThinkingContent(thinking=content_block["thinking"], signature=content_block.get("signature"))
                )
            elif content_block["type"] == "text":
                text = content_block["text"]
                # TODO Make an effort to deduplicate these
                for citation in content_block["citations"]:
                    domain = urlparse(citation.url).netloc
                    domain = domain.removeprefix("www.")
                    text += f" [[{domain}]({citation.url})]"
                if len(content) > 0 and isinstance(content[-1], TextContent):
                    content[-1].text += text
                else:
                    content.append(TextContent(text=text))
            else:
                # Unreachable
                raise AssertionError(f"Unknown content block type: {content_block['type']}")

        return Message(role="assistant", content=content, stop_reason=self._stop_reason)
