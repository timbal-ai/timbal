import json
import time
from collections import deque
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
    ResponseCustomToolCall,
    ResponseCustomToolCallInputDeltaEvent,
    ResponseCustomToolCallInputDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from uuid_extensions import uuid7

from ...core.models import (
    LONG_CONTEXT_USAGE_SUFFIX,
    has_cache_write_pricing,
    service_tier_usage_suffix,
    uses_long_context_pricing,
)
from ...state import get_billing_id, get_run_context
from ...types.content.text import TextContent
from ..harmony_leak import LEAKED_TOOL_CALL_UNRECOVERED, contains_leak, leak_state, parse_leaked_tool_calls
from ...types.content.thinking import ThinkingContent
from ...types.content.tool_use import ToolUseContent
from ...types.events.delta import (
    ContentBlockStop as TimbalContentBlockStop,
)
from ...types.events.delta import (
    Text as TimbalText,
)
from ...types.events.delta import (
    TextDelta as TimbalTextDelta,
)
from ...types.events.delta import (
    Thinking as TimbalThinking,
)
from ...types.events.delta import (
    ThinkingDelta as TimbalThinkingDelta,
)
from ...types.events.delta import (
    ToolUse as TimbalToolUse,
)
from ...types.events.delta import (
    ToolUseDelta as TimbalToolUseDelta,
)
from ...types.message import Message
from .. import register_collector
from ..base import BaseCollector


def _usage_tier_suffix(billing_id: str, input_tokens: int, service_tier: str | None = None) -> str:
    """Return the usage-key suffix for this request's pricing tier.

    OpenAI (>272K on 1.05M-context models), xAI (>=200K) and BytePlus (>128K) reprice
    the *entire* request — input, cache reads/writes and output — once the prompt
    exceeds the model's threshold. Emitting distinct units (``input_text_tokens_long_context``
    etc.) lets cost tables bill each tier at its own rate instead of silently applying
    the short-context rate.

    ``input_tokens`` must be the raw prompt size as reported by the provider (cached and
    cache-write tokens included) — that is the number the threshold is defined against.
    """
    suffix = LONG_CONTEXT_USAGE_SUFFIX if uses_long_context_pricing(billing_id, input_tokens) else ""
    return f"{suffix}{service_tier_usage_suffix(billing_id, service_tier)}"


def _optional_int(obj: Any, attr: str) -> int:
    """Read a non-negative optional integer, tolerating malformed provider extras."""
    value = getattr(obj, attr, None) if obj is not None else None
    if value is None or isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return parsed if parsed > 0 else 0


def _bounded_usage_detail(obj: Any, attr: str, remaining: int) -> int:
    """Read a usage detail without allowing malformed telemetry to make a bucket negative."""
    return min(_optional_int(obj, attr), max(remaining, 0))


def _cache_write_tokens(billing_id: str, input_tokens_details: Any) -> int:
    """Cache-write tokens to split out of plain input, or 0 when the catalog cannot price them.

    OpenAI bills prompt-cache writes at a premium (1.25x input). The SDK does not declare
    ``cache_write_tokens`` yet, so it arrives as a pydantic extra. Only models with a
    ``cache_write_price`` get a dedicated unit — for anything else the tokens must stay in
    ``input_text_tokens`` so they are still billed (at the input rate) instead of vanishing
    into a unit with no cost row.
    """
    if not has_cache_write_pricing(billing_id):
        return 0
    return _optional_int(input_tokens_details, "cache_write_tokens")


# Create type aliases for OpenAI events
ChatCompletionEvent = ChatCompletionChunk
ResponseEvent = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseOutputItemAddedEvent
    | ResponseWebSearchCallInProgressEvent
    | ResponseWebSearchCallSearchingEvent
    | ResponseWebSearchCallCompletedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseOutputTextAnnotationAddedEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseContentPartDoneEvent
    | ResponseCompletedEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    | ResponseIncompleteEvent
    | ResponseCustomToolCallInputDeltaEvent
    | ResponseCustomToolCallInputDoneEvent
)

logger = structlog.get_logger("timbal.collectors.impl.openai")


def _delta_reasoning_content(delta: Any) -> str | None:
    """Extract chat-completions reasoning text (Fireworks / DeepSeek / etc.).

    OpenAI's typed ``ChoiceDelta`` does not declare ``reasoning_content``; providers
    put it on the delta as an extra field (kept in ``model_extra``).
    """
    rc = getattr(delta, "reasoning_content", None)
    if isinstance(rc, str) and rc:
        return rc
    extra = getattr(delta, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            val = extra.get(key)
            if isinstance(val, str) and val:
                return val
    return None


@register_collector
class ChatCompletionCollector(BaseCollector):
    """Collector for OpenAI chat completions streaming events."""

    # Content block ID for text content (chat completions only have one text block)
    TEXT_BLOCK_ID = "text_0"
    THINKING_BLOCK_ID = "thinking_0"

    def __init__(self, start: float, **kwargs: Any):
        super().__init__(**kwargs)
        self._start = start
        self._content: str = ""
        self._thinking: str = ""
        # `_current_tool_call` is appended to `_tool_calls` by reference (same dict).
        # Subsequent mutations of `_current_tool_call` propagate to the entry in
        # `_tool_calls` without needing to re-append.
        self._tool_calls: list[dict[str, Any]] = []
        self._current_tool_call: dict[str, Any] | None = None
        self._tool_use_header_emitted: bool = False
        self._first_token: float | None = None
        self._output_tokens: int = 0
        self._text_block_started: bool = False
        self._thinking_block_started: bool = False
        self._content_blocks: set[str] = set()
        self._stop_reason: str | None = None
        self._pending_usage: Any | None = None  # Last usage event, written once in result()
        # One source chunk can produce multiple stream items (e.g. reasoning + text).
        # process() returns the first; the rest drain via __anext__ / pop_pending_stream_item().
        self._pending_stream_items: deque[Any] = deque()

    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, ChatCompletionEvent)

    def pop_pending_stream_item(self) -> Any | None:
        """Return the next queued stream item, if any (used after process() on the peek chunk)."""
        if self._pending_stream_items:
            return self._pending_stream_items.popleft()
        return None

    async def __anext__(self):
        pending = self.pop_pending_stream_item()
        if pending is not None:
            return pending
        return await super().__anext__()

    def _emit_stream_items(self, items: list[Any]) -> Any:
        """Return the first stream item and queue any remaining for later yields."""
        filtered = [item for item in items if item is not None]
        if not filtered:
            return None
        first, *rest = filtered
        self._pending_stream_items.extend(rest)
        return first

    @override
    def process(self, event: ChatCompletionEvent) -> Any:
        """Processes OpenAI streaming events."""
        # Stash usage for deferred processing in result().
        # Some providers (e.g. Gemini) send cumulative usage on every chunk,
        # not just the final one. By always overwriting _pending_usage and
        # writing once in result(), we avoid double-counting.
        if event.usage:
            self._pending_usage = event
        if not len(event.choices):
            return None
        # Capture finish_reason from the choice
        # Possible values: 'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
        # 'length' indicates max_tokens was reached
        if event.choices[0].finish_reason:
            self._stop_reason = event.choices[0].finish_reason
        delta = event.choices[0].delta
        has_tool_calls = bool(delta.tool_calls)
        reasoning = _delta_reasoning_content(delta)
        has_text = bool(delta.content)
        # Calculate TTFT on first visible / reasoning / tool token
        if self._first_token is None and (has_tool_calls or reasoning or has_text):
            self._first_token = time.perf_counter()

        # Fireworks / Moonshot / DeepSeek-style: reasoning may co-arrive with
        # visible content or tool_calls on the same chunk. Emit all of them.
        items: list[Any] = []
        if reasoning:
            items.append(self._handle_reasoning_content(reasoning))
        if has_tool_calls:
            items.append(self._handle_tool_calls(event))
        elif has_text:
            items.append(self._handle_text_content(event))
        return self._emit_stream_items(items)

    @staticmethod
    def _usage_billing_id(api_model: str) -> str:
        return get_billing_id() or api_model

    def _handle_usage(self, event: ChatCompletionEvent) -> None:
        """Handle usage statistics from OpenAI events.

        Output rule of thumb: every "billed-as-output" token (visible
        completion + OpenAI o-series reasoning + Gemini hidden thinking) is
        collapsed into a single `output_text_tokens` bucket — they are all
        billed at the output rate and Gemini's hidden thinking is **not**
        in `completion_tokens` (only visible via `total_tokens`). We
        compute `output = total_tokens - prompt_tokens` after the input
        side has been split out, falling back to `completion_tokens` when
        `total_tokens` is missing/inconsistent.
        """
        run_context = get_run_context()
        billing_id = self._usage_billing_id(event.model)
        openai_usage = event.usage
        raw_input = int(openai_usage.prompt_tokens)
        raw_output = int(openai_usage.completion_tokens)
        total_tokens = int(getattr(openai_usage, "total_tokens", 0) or 0)
        # Long-context tier is decided on the raw prompt size (cache hits included)
        # and applies to every token bucket of this request.
        tier = _usage_tier_suffix(billing_id, raw_input, getattr(event, "service_tier", None))

        input_tokens = raw_input
        input_tokens_details = openai_usage.prompt_tokens_details
        input_cached_tokens = _bounded_usage_detail(input_tokens_details, "cached_tokens", input_tokens)
        if input_cached_tokens:
            input_tokens -= input_cached_tokens
            run_context.update_usage(f"{billing_id}:input_cached_tokens{tier}", input_cached_tokens)
        input_cache_write_tokens = min(_cache_write_tokens(billing_id, input_tokens_details), max(input_tokens, 0))
        if input_cache_write_tokens:
            input_tokens -= input_cache_write_tokens
            run_context.update_usage(f"{billing_id}:input_cache_write_tokens{tier}", input_cache_write_tokens)
        input_audio_tokens = _bounded_usage_detail(input_tokens_details, "audio_tokens", input_tokens)
        if input_audio_tokens:
            input_tokens -= input_audio_tokens
            run_context.update_usage(f"{billing_id}:input_audio_tokens", input_audio_tokens)
        run_context.update_usage(f"{billing_id}:input_text_tokens{tier}", input_tokens)

        # Total-derived output: visible completion + OpenAI o-series
        # reasoning (already inside completion_tokens) + Gemini hidden
        # thinking (only visible via total_tokens). max() with raw_output
        # is a safety net for providers that report total_tokens
        # inconsistently.
        output_tokens = max(total_tokens - raw_input, raw_output) if total_tokens > 0 else raw_output
        self._output_tokens += output_tokens
        output_audio_tokens = _bounded_usage_detail(openai_usage.completion_tokens_details, "audio_tokens", output_tokens)
        if output_audio_tokens:
            output_tokens -= output_audio_tokens
            run_context.update_usage(f"{billing_id}:output_audio_tokens", output_audio_tokens)
        run_context.update_usage(f"{billing_id}:output_text_tokens{tier}", output_tokens)

    def _handle_tool_calls(self, event: ChatCompletionEvent) -> TimbalToolUse | TimbalToolUseDelta | None:
        """Handle tool call events from OpenAI."""
        tool_call = event.choices[0].delta.tool_calls[0]
        fn = tool_call.function
        fn_name = fn.name if fn is not None else None
        fn_args_part = fn.arguments if fn is not None else None

        def _merge_same_id_stream() -> TimbalToolUse | TimbalToolUseDelta | None:
            """Continue the same tool_call when the provider resends the same id (e.g. Fireworks/Kimi)."""
            assert self._current_tool_call is not None
            if fn_name:
                self._current_tool_call["name"] = fn_name
            if fn_args_part is not None:
                self._current_tool_call["input"] += fn_args_part
            if not self._tool_use_header_emitted and self._current_tool_call["name"]:
                self._tool_calls.append(self._current_tool_call)
                self._tool_use_header_emitted = True
                return TimbalToolUse(
                    id=self._current_tool_call["id"],
                    name=self._current_tool_call["name"],
                    input=self._current_tool_call["input"],
                    is_server_tool_use=False,
                )
            if self._tool_use_header_emitted and fn_args_part is not None:
                return TimbalToolUseDelta(
                    id=self._current_tool_call["id"],
                    input_delta=fn_args_part,
                )
            return None

        # TODO Review this for parallel tool calls
        if tool_call.id:
            same_stream = (
                self._current_tool_call is not None
                and self._current_tool_call.get("id") == tool_call.id
            )
            if same_stream:
                return _merge_same_id_stream()

            # New tool call stream. Some providers (e.g. Fireworks / Kimi) send tool_call.id before
            # function.name is set; defer emitting TimbalToolUse until we have a name.
            self._tool_use_header_emitted = bool(fn_name)
            self._current_tool_call = {
                "type": "tool_use",
                "id": tool_call.id,
                "name": fn_name or "",
                "input": fn_args_part if fn_args_part is not None else "",
            }

            # Check for extra_content (Google Gemini thought signature)
            extra_content = getattr(tool_call, "extra_content", None)
            if extra_content:
                google_extra = extra_content.get("google")
                if google_extra:
                    self._current_tool_call["thought_signature"] = google_extra.get("thought_signature")

            self._content_blocks.add(tool_call.id)
            if fn_name:
                self._tool_calls.append(self._current_tool_call)
                return TimbalToolUse(
                    id=tool_call.id,
                    name=fn_name,
                    input=self._current_tool_call["input"],
                    is_server_tool_use=False,
                )
            return None

        if self._current_tool_call is None:
            return None

        if fn_name:
            self._current_tool_call["name"] = fn_name
        if fn_args_part is not None:
            self._current_tool_call["input"] += fn_args_part

        if not self._tool_use_header_emitted and self._current_tool_call["name"]:
            self._tool_calls.append(self._current_tool_call)
            self._tool_use_header_emitted = True
            return TimbalToolUse(
                id=self._current_tool_call["id"],
                name=self._current_tool_call["name"],
                input=self._current_tool_call["input"],
                is_server_tool_use=False,
            )
        if self._tool_use_header_emitted and fn_args_part is not None:
            return TimbalToolUseDelta(
                id=self._current_tool_call["id"],
                input_delta=fn_args_part,
            )
        return None

    def _handle_reasoning_content(self, reasoning_chunk: str) -> TimbalThinking | TimbalThinkingDelta:
        """Handle ``delta.reasoning_content`` from OpenAI-compatible reasoning models."""
        self._thinking += reasoning_chunk
        if not self._thinking_block_started:
            self._thinking_block_started = True
            self._content_blocks.add(self.THINKING_BLOCK_ID)
            return TimbalThinking(
                id=self.THINKING_BLOCK_ID,
                thinking=reasoning_chunk,
            )
        return TimbalThinkingDelta(
            id=self.THINKING_BLOCK_ID,
            thinking_delta=reasoning_chunk,
        )

    def _handle_text_content(self, event: ChatCompletionEvent) -> TimbalText | TimbalTextDelta:
        """Handle text content from OpenAI events."""
        text_chunk = event.choices[0].delta.content
        # Handle citations if present
        if hasattr(event, "citations"):
            self.citations = event.citations
        self._content += text_chunk

        if not self._text_block_started:
            self._text_block_started = True
            self._content_blocks.add(self.TEXT_BLOCK_ID)
            return TimbalText(
                id=self.TEXT_BLOCK_ID,
                text=text_chunk,
            )
        else:
            return TimbalTextDelta(
                id=self.TEXT_BLOCK_ID,
                text_delta=text_chunk,
            )

    @override
    def result(self) -> Message:
        """Returns structured OpenAI response."""
        # Write usage once from the last chunk that carried it.
        if self._pending_usage is not None:
            self._handle_usage(self._pending_usage)

        span = get_run_context().current_span()
        # finish_reason-only chunks never set _first_token — keep metrics defined.
        first = self._first_token if self._first_token is not None else time.perf_counter()
        ttft = first - self._start
        span.metadata["ttft"] = ttft
        elapsed = time.perf_counter() - first
        tps = self._output_tokens / elapsed if elapsed > 0 else 0.0
        span.metadata["tps"] = tps

        content: list[Any] = []
        if self._thinking:
            content.append({"type": "thinking", "thinking": self._thinking})
        if self._content:
            content.append(self._content)

        if self._tool_calls:
            # Openai allows the use of custom IDs for tool calls.
            # We choose to generate our own random IDs for consistency and to make sure they don't collide
            # (they are not transparent with the algs being used)
            tool_calls = [{**tc, "id": uuid7(as_type="hex")} for tc in self._tool_calls]
            content.extend(tool_calls)

        return Message.validate({"role": "assistant", "content": content, "stop_reason": self._stop_reason})


@register_collector
class ResponseCollector(BaseCollector):
    """Collector for OpenAI responses streaming events."""

    def __init__(self, start: float, **kwargs: Any):
        super().__init__(**kwargs)
        self._start = start
        self._first_token: float | None = None
        self._output_tokens: int = 0
        self._stop_reason: str | None = None
        self.content_blocks: set[str] = set()
        self.content: dict[str, dict[str, Any]] = {}
        # `phase` per message item id (`commentary` / `final_answer`), from the
        # item's `.added` / `.done`; the text block is keyed by the same id.
        self._message_phase: dict[str, str | None] = {}
        # Once a text block is a leaked tool call, every later text block of this
        # response is the model narrating work that never ran ("Invoice Triage is
        # built…"). It is neither streamed nor kept; the real answer comes after
        # the recovered call has actually run.
        self._leak_seen: bool = False
        # A `.done` that has to flush held text emits two items; see `_emit_stream_items`.
        self._pending_stream_items: deque[Any] = deque()

    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, ResponseEvent)

    def pop_pending_stream_item(self) -> Any | None:
        if self._pending_stream_items:
            return self._pending_stream_items.popleft()
        return None

    async def __anext__(self):
        pending = self.pop_pending_stream_item()
        if pending is not None:
            return pending
        return await super().__anext__()

    def _emit_stream_items(self, items: list[Any]) -> Any:
        filtered = [item for item in items if item is not None]
        if not filtered:
            return None
        first, *rest = filtered
        self._pending_stream_items.extend(rest)
        return first

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
        elif isinstance(event, ResponseCompletedEvent | ResponseIncompleteEvent):
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
        elif isinstance(event, ResponseReasoningTextDeltaEvent):
            return self._handle_reasoning_text_delta(event)
        elif isinstance(event, ResponseReasoningTextDoneEvent):
            return None
        elif isinstance(event, ResponseCustomToolCallInputDeltaEvent | ResponseCustomToolCallInputDoneEvent):
            return None
        else:
            logger.warning("Unhandled response event", response_event=event)

    def _handle_created(self, event: ResponseCreatedEvent) -> None:
        """Handle created events from OpenAI."""
        self.model = event.response.model

    def _usage_billing_id(self, *, response_fallback: str | None = None) -> str:
        bid = get_billing_id()
        if bid:
            return bid
        m = getattr(self, "model", None)
        if m:
            return m
        if response_fallback:
            return response_fallback
        return ""

    def _handle_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
        """Handle output item added events from OpenAI."""
        if isinstance(event.item, ResponseFunctionToolCall):
            self.content[event.item.id] = {
                "type": "tool_use",
                "id": event.item.call_id,
                "name": event.item.name,
                "input": event.item.arguments,  # openai sends an empty string here
            }
            content_block_id = event.item.id
            self.content_blocks.add(content_block_id)
            return TimbalToolUse(
                id=content_block_id,
                name=event.item.name,
                input=event.item.arguments,
                is_server_tool_use=False,
            )
        elif isinstance(event.item, ResponseOutputMessage):
            self._message_phase[event.item.id] = getattr(event.item, "phase", None)
            return None
        elif isinstance(event.item, ResponseFunctionWebSearch):
            # TODO We should add this to the messages history
            content_block_id = event.item.id
            self.content_blocks.add(content_block_id)
            return TimbalToolUse(
                id=content_block_id,
                name="web_search",
                input="",  # openai gives the query param at the response
                is_server_tool_use=True,
            )
        elif isinstance(event.item, ResponseReasoningItem):
            content_block_id = event.item.id
            # Register the block now so it keeps its position ahead of the function_call
            # it produced; the encrypted payload (and any summary) land on `.done`.
            self._reasoning_entry(event.item)
            self.content_blocks.add(content_block_id)
            return TimbalThinking(
                id=content_block_id,
                thinking="",
            )
        elif isinstance(event.item, ResponseCustomToolCall):
            # Server-side custom tools (e.g. xAI's x_keyword_search, web_fetch).
            # Track as server tool use but don't emit to the stream.
            content_block_id = event.item.id
            self.content_blocks.add(content_block_id)
            return TimbalToolUse(
                id=content_block_id,
                name=event.item.name,
                input=event.item.input,
                is_server_tool_use=True,
            )
        else:
            logger.warning("Unhandled output item added event", response_output_item_added_event=event)

    def _handle_content_part_added(self, event: ResponseContentPartAddedEvent) -> None:
        """Handle content part added events from OpenAI.

        The text block is *held* until its first characters show whether it is prose
        or a leaked tool call (`` to=functions.…``, see ``harmony_leak``). Prose is
        released as one ``Text`` carrying what accumulated meanwhile; a leak is never
        streamed — it is recovered as a tool call in ``result()``.
        """
        if isinstance(event.part, ResponseOutputText):
            self.content[event.item_id] = {
                "type": "text",
                "citations": [],
                "text": event.part.text,
                "phase": self._message_phase.get(event.item_id),
                "held": True,
                "leak": False,
            }
            return self._release_text_if_decided(event.item_id)
        else:
            logger.warning("Unhandled content part added event", response_content_part_added_event=event)

    def _release_text_if_decided(self, item_id: str, *, force: bool = False) -> TimbalText | None:
        """Start the stream block for a held text once it is known not to be a leak.

        ``force`` settles an undecided block (the item ended while still a prefix of
        the marker, e.g. a message that is just ``"to"``) as prose.
        """
        entry = self.content[item_id]
        if not entry.get("held"):
            return None
        if self._leak_seen:
            entry["after_leak"] = True
            entry["held"] = False
            return None
        state = leak_state(entry["text"])
        if state == "leak":
            entry["leak"] = True
            entry["held"] = False
            self._leak_seen = True
            return None
        if state == "undecided" and not force:
            return None
        entry["held"] = False
        self.content_blocks.add(item_id)
        return TimbalText(id=item_id, text=entry["text"])

    def _handle_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        """Handle text delta events from OpenAI."""
        entry = self.content[event.item_id]
        entry["text"] += event.delta
        content_block_id = event.item_id
        if entry.get("held"):
            return self._release_text_if_decided(content_block_id)
        if entry.get("leak") or entry.get("after_leak"):
            return None
        assert content_block_id in self.content_blocks, "Text delta event without content block start event"
        return TimbalTextDelta(
            id=content_block_id,
            text_delta=event.delta,
        )

    def _handle_output_text_annotation_added(self, event: ResponseOutputTextAnnotationAddedEvent) -> None:
        """Handle output text annotation added events from OpenAI."""
        self.content[event.item_id]["citations"].append(event.annotation)

    def _handle_function_call_arguments_delta(self, event: ResponseFunctionCallArgumentsDeltaEvent) -> None:
        """Handle function call arguments delta events from OpenAI."""
        self.content[event.item_id]["input"] += event.delta
        content_block_id = event.item_id
        assert content_block_id in self.content_blocks, (
            "Function call arguments delta event without content block start event"
        )
        return TimbalToolUseDelta(
            id=content_block_id,
            input_delta=event.delta,
        )

    def _reasoning_entry(self, item: ResponseReasoningItem | None = None, item_id: str | None = None) -> dict[str, Any]:
        """The content block for a reasoning item, created on first sight.

        Carries the item id and, once `.done` delivers it, the `encrypted_content` the
        request asked for via `include: ["reasoning.encrypted_content"]`. Both are what
        `ThinkingContent` needs to replay this step's chain of thought on the next call.
        """
        key = item.id if item is not None else item_id
        entry = self.content.get(key)
        if entry is None:
            entry = {"type": "thinking", "thinking": "", "id": key, "encrypted_content": None}
            self.content[key] = entry
        if item is not None:
            entry["id"] = item.id
            if item.encrypted_content:
                entry["encrypted_content"] = item.encrypted_content
            # The final item carries the complete summary; prefer it over what streamed
            # in (identical when parts streamed, filled in when they did not).
            summary = "\n\n".join(part.text for part in (item.summary or []) if getattr(part, "text", ""))
            if summary:
                entry["thinking"] = summary
        return entry

    def _handle_reasoning_summary_part_added(self, event: ResponseReasoningSummaryPartAddedEvent) -> None:
        """Handle reasoning summary part added events from OpenAI."""
        entry = self._reasoning_entry(item_id=event.item_id)
        # Several summary parts make one thinking block; separate them like paragraphs.
        if entry["thinking"]:
            entry["thinking"] += "\n\n"
        entry["thinking"] += event.part.text  # Usually empty string from the beginning
        content_block_id = event.item_id
        self.content_blocks.add(content_block_id)
        return TimbalThinking(
            id=content_block_id,
            thinking=event.part.text,
        )

    def _handle_reasoning_summary_text_delta(self, event: ResponseReasoningSummaryTextDeltaEvent) -> None:
        """Handle reasoning summary text delta events from OpenAI."""
        self._reasoning_entry(item_id=event.item_id)["thinking"] += event.delta
        content_block_id = event.item_id
        assert content_block_id in self.content_blocks, (
            "Reasoning summary text delta event without content block start event"
        )
        return TimbalThinkingDelta(
            id=content_block_id,
            thinking_delta=event.delta,
        )

    def _handle_reasoning_text_delta(self, event: ResponseReasoningTextDeltaEvent) -> None:
        """Handle raw reasoning text delta events (e.g. from xAI)."""
        self._reasoning_entry(item_id=event.item_id)["thinking"] += event.delta
        content_block_id = event.item_id
        if content_block_id not in self.content_blocks:
            self.content_blocks.add(content_block_id)
            return TimbalThinking(
                id=content_block_id,
                thinking=event.delta,
            )
        return TimbalThinkingDelta(
            id=content_block_id,
            thinking_delta=event.delta,
        )

    def _handle_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        """Handle output item done events from OpenAI."""
        if isinstance(event.item, ResponseFunctionWebSearch):
            bid = self._usage_billing_id()
            if bid:
                get_run_context().update_usage(
                    f"{bid}:web_search_requests", 1
                )  # TODO Review. Do they only perform one query?
            # TODO Grab the query and return the result
            content_block_id = event.item.id
            if content_block_id in self.content_blocks:
                return TimbalContentBlockStop(id=content_block_id)
            else:
                return None
        elif isinstance(event.item, ResponseReasoningItem):
            # `.done` is where `encrypted_content` (and the full summary) arrive.
            self._reasoning_entry(event.item)
            content_block_id = event.item.id
            if content_block_id in self.content_blocks:
                return TimbalContentBlockStop(id=content_block_id)
            else:
                return None
        elif isinstance(event.item, ResponseOutputMessage):
            content_block_id = event.item.id
            phase = getattr(event.item, "phase", None)
            if phase:
                self._message_phase[content_block_id] = phase
                if content_block_id in self.content:
                    self.content[content_block_id]["phase"] = phase
            released = None
            if content_block_id in self.content:
                # A block still held at the end (too short to decide) is prose.
                released = self._release_text_if_decided(content_block_id, force=True)
            if content_block_id in self.content_blocks:
                return self._emit_stream_items([released, TimbalContentBlockStop(id=content_block_id)])
            return None
        elif isinstance(event.item, ResponseFunctionToolCall):
            content_block_id = event.item.id
            if content_block_id in self.content_blocks:
                return TimbalContentBlockStop(id=content_block_id)
            else:
                return None
        elif isinstance(event.item, ResponseCustomToolCall):
            # Track server-side custom tool requests for cost tracking.
            tool_name = event.item.name
            bid = self._usage_billing_id()
            if bid:
                get_run_context().update_usage(f"{bid}:{tool_name}_requests", 1)
            content_block_id = event.item.id
            if content_block_id in self.content_blocks:
                return TimbalContentBlockStop(id=content_block_id)
            else:
                return None
        else:
            logger.warning("Unhandled output item done event", response_output_item_done_event=event)

    def _handle_completed(self, event: ResponseCompletedEvent | ResponseIncompleteEvent) -> None:
        """Handle completed events from OpenAI."""
        # Capture stop reason from the response
        # status can be: 'completed', 'failed', 'in_progress', 'cancelled', 'queued', 'incomplete'
        # incomplete_details.reason can be: 'max_output_tokens', 'content_filter'
        if event.response.status == "incomplete" and event.response.incomplete_details:
            self._stop_reason = event.response.incomplete_details.reason  # 'max_output_tokens' or 'content_filter'
        else:
            self._stop_reason = event.response.status  # 'completed', 'failed', etc.

        run_context = get_run_context()
        usage = event.response.usage
        billing_id = self._usage_billing_id(response_fallback=event.response.model)
        raw_input = int(usage.input_tokens)
        raw_output = int(usage.output_tokens)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        # Long-context tier is decided on the raw prompt size (cache hits included)
        # and applies to every token bucket of this request.
        tier = _usage_tier_suffix(billing_id, raw_input, getattr(event.response, "service_tier", None))

        input_tokens = raw_input
        input_tokens_details = usage.input_tokens_details
        input_cached_tokens = _bounded_usage_detail(input_tokens_details, "cached_tokens", input_tokens)
        if input_cached_tokens:
            input_tokens -= input_cached_tokens
            run_context.update_usage(f"{billing_id}:input_cached_tokens{tier}", input_cached_tokens)
        input_cache_write_tokens = min(_cache_write_tokens(billing_id, input_tokens_details), max(input_tokens, 0))
        if input_cache_write_tokens:
            input_tokens -= input_cache_write_tokens
            run_context.update_usage(f"{billing_id}:input_cache_write_tokens{tier}", input_cache_write_tokens)
        input_audio_tokens = _bounded_usage_detail(input_tokens_details, "audio_tokens", input_tokens)
        if input_audio_tokens:
            input_tokens -= input_audio_tokens
            run_context.update_usage(f"{billing_id}:input_audio_tokens", input_audio_tokens)
        run_context.update_usage(f"{billing_id}:input_text_tokens{tier}", input_tokens)

        # See `_handle_usage` in ChatCompletionCollector: collapse all
        # billed-as-output tokens (visible + reasoning + hidden thinking)
        # into a single bucket via `total - raw_input`.
        output_tokens = max(total_tokens - raw_input, raw_output) if total_tokens > 0 else raw_output
        self._output_tokens += output_tokens
        output_audio_tokens = _bounded_usage_detail(usage.output_tokens_details, "audio_tokens", output_tokens)
        if output_audio_tokens:
            output_tokens -= output_audio_tokens
            run_context.update_usage(f"{billing_id}:output_audio_tokens", output_audio_tokens)
        run_context.update_usage(f"{billing_id}:output_text_tokens{tier}", output_tokens)

    @override
    def result(self) -> Message:
        """Returns structured OpenAI response."""
        span = get_run_context().current_span()
        ttft = self._first_token - self._start
        span.metadata["ttft"] = ttft
        tps = self._output_tokens / (time.perf_counter() - self._first_token)
        span.metadata["tps"] = tps

        content = []
        # Leaked tool calls are recovered only when the response produced no structured
        # one: next to real function_calls the leaked text is a duplicate or a
        # hallucination, and running it would double the action.
        has_structured_tool_use = any(b["type"] == "tool_use" for b in self.content.values())
        recovered = 0
        recovered_keys: set[tuple[str, str]] = set()
        leak_recovered_here = False  # a leaked block has been seen in this response (content order)
        for content_block in self.content.values():  # Python dicts are ordered
            if content_block["type"] == "tool_use":
                content.append(
                    ToolUseContent(
                        id=content_block["id"],
                        name=content_block["name"],
                        input=content_block["input"],
                    )
                )
            elif content_block["type"] == "server_tool_use":
                continue
            elif content_block["type"] == "server_tool_result":
                continue
            elif content_block["type"] == "thinking":
                thinking = content_block.get("thinking") or ""
                encrypted = content_block.get("encrypted_content")
                if not thinking and not encrypted:
                    # A reasoning item with neither a summary nor an encrypted payload
                    # (`include` not requested) carries nothing to replay; an empty
                    # thinking block would only serialize as an empty text part.
                    continue
                content.append(
                    ThinkingContent(
                        thinking=thinking,
                        id=content_block.get("id") if encrypted else None,
                        encrypted_content=encrypted,
                    )
                )
            elif content_block["type"] == "text":
                text = content_block["text"]
                phase = content_block.get("phase")
                # e.g. {'type': 'url_citation', 'end_index': 2538, 'start_index': 2403, 'title': 'Weather Forecast and Conditions for Barcelona, Barcelona, Spain - The Weather Channel | Weather.com', 'url': 'https://weather.com/weather/today/l/b3b13a74649dd0a2a0aada41e1bf764de39e5dacf21d062ef18ecdeb09796ba0?utm_source=openai'}
                # Openai annotations are already formatted into the text
                if (content_block.get("after_leak") or leak_recovered_here) and not contains_leak(text):
                    # Prose after a leaked call in the same response: the model's
                    # account of work that never ran. Replaying it puts a "final
                    # answer" between the recovered function_call and its output,
                    # and the next turn answers as if it were already done.
                    logger.warning("Dropped assistant text following a leaked tool call", text=text[:200])
                    continue
                if content_block.get("leak") or contains_leak(text):
                    leak_recovered_here = True
                    # A tool call written as text (see harmony_leak). Keep the prose
                    # before it; turn what parses into real tool calls; drop the rest
                    # — surfacing it would hand the user a summary of work nobody did.
                    prefix, calls = parse_leaked_tool_calls(text)
                    if prefix:
                        content.append(TextContent(text=prefix, phase=phase))
                    if has_structured_tool_use:
                        logger.warning("Leaked tool-call text alongside structured tool calls; dropped",
                                       names=[n for n, _ in calls], text=text[:300])
                        continue
                    for name, args in calls:
                        # The model repeats the same call across message items; one run each.
                        key = (name, json.dumps(args, sort_keys=True))
                        if key in recovered_keys:
                            continue
                        recovered_keys.add(key)
                        recovered += 1
                        content.append(ToolUseContent(id=f"call_leak_{uuid7(as_type='hex')}", name=name, input=args))
                    if not calls:
                        logger.warning("Leaked tool-call text with no parseable call; dropped", text=text[:300])
                    continue
                content.append(TextContent(text=text, phase=phase))
            else:
                # Unreachable
                raise AssertionError(f"Unknown content block type: {content_block['type']}")
        if recovered:
            logger.warning("Recovered tool calls from leaked assistant text", count=recovered, model=getattr(self, "model", None))
            get_run_context().update_usage("recovered_tool_calls", recovered)
        metadata: dict[str, Any] | None = None
        if leak_recovered_here and not has_structured_tool_use:
            # Debris-adjacent empty text parts carry nothing; they would only replay as
            # empty `output_text` items around the recovered calls.
            content = [c for c in content if not (isinstance(c, TextContent) and not c.text)]
            if not recovered and not any(isinstance(c, TextContent) for c in content):
                # A leak whose call never got its arguments (`to=functions.builder (json…`
                # and then nothing): nothing to run, nothing to say. The agent loop
                # re-requests on this flag instead of ending the turn empty.
                metadata = {"source": "runtime", "kind": LEAKED_TOOL_CALL_UNRECOVERED}
                logger.warning("Leaked tool call with no recoverable arguments; turn needs a retry",
                               model=getattr(self, "model", None))
                get_run_context().update_usage("unrecovered_tool_call_leaks", 1)

        return Message(role="assistant", content=content, stop_reason=self._stop_reason, metadata=metadata)
