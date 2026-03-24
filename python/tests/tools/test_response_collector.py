"""Tests for ResponseCollector handling of xAI-specific and newer Responses API events."""

import time
from unittest.mock import MagicMock, patch

import pytest
from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseCustomToolCallInputDeltaEvent,
    ResponseCustomToolCallInputDoneEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
)

from timbal.collectors.impl.openai import ResponseCollector
from timbal.types.events.delta import (
    ContentBlockStop as TimbalContentBlockStop,
)
from timbal.types.events.delta import (
    Thinking as TimbalThinking,
)
from timbal.types.events.delta import (
    ThinkingDelta as TimbalThinkingDelta,
)
from timbal.types.events.delta import (
    ToolUse as TimbalToolUse,
)


@pytest.fixture
def collector():
    """Create a ResponseCollector with a mocked async generator."""
    mock_gen = MagicMock()
    c = ResponseCollector(start=time.perf_counter(), async_gen=mock_gen)
    c.model = "grok-4-fast-reasoning"
    return c


def _make_custom_tool_call(*, call_id="xs_call_1", name="x_keyword_search", input="", status="in_progress"):
    return ResponseCustomToolCall(
        call_id=call_id,
        input=input,
        name=name,
        type="custom_tool_call",
        id=f"ctc_{call_id}",
        status=status,
    )


class TestReasoningTextEvents:
    """Test handling of raw reasoning text events (e.g. from xAI)."""

    def test_first_delta_creates_thinking_block(self, collector):
        event = ResponseReasoningTextDeltaEvent(
            content_index=0,
            delta="Let me think",
            item_id="rs_1",
            output_index=0,
            sequence_number=1,
            type="response.reasoning_text.delta",
        )
        result = collector.process(event)
        assert isinstance(result, TimbalThinking)
        assert result.id == "rs_1"
        assert result.thinking == "Let me think"
        assert "rs_1" in collector.content_blocks

    def test_subsequent_delta_returns_thinking_delta(self, collector):
        # First delta creates the block.
        event1 = ResponseReasoningTextDeltaEvent(
            content_index=0, delta="First", item_id="rs_1",
            output_index=0, sequence_number=1, type="response.reasoning_text.delta",
        )
        collector.process(event1)

        # Second delta appends.
        event2 = ResponseReasoningTextDeltaEvent(
            content_index=0, delta=" second", item_id="rs_1",
            output_index=0, sequence_number=2, type="response.reasoning_text.delta",
        )
        result = collector.process(event2)
        assert isinstance(result, TimbalThinkingDelta)
        assert result.thinking_delta == " second"
        assert collector.content["rs_1"]["thinking"] == "First second"

    def test_done_event_returns_none(self, collector):
        event = ResponseReasoningTextDoneEvent(
            content_index=0, item_id="rs_1", output_index=0,
            sequence_number=3, text="Full reasoning text",
            type="response.reasoning_text.done",
        )
        result = collector.process(event)
        assert result is None


class TestCustomToolCallEvents:
    """Test handling of server-side custom tool calls (e.g. xAI x_keyword_search)."""

    def test_output_item_added_custom_tool(self, collector):
        tool_call = _make_custom_tool_call()
        event = ResponseOutputItemAddedEvent(
            item=tool_call, output_index=0, sequence_number=1,
            type="response.output_item.added",
        )
        result = collector._handle_output_item_added(event)
        assert isinstance(result, TimbalToolUse)
        assert result.name == "x_keyword_search"
        assert result.is_server_tool_use is True
        assert "ctc_xs_call_1" in collector.content_blocks

    def test_input_delta_returns_none(self, collector):
        event = ResponseCustomToolCallInputDeltaEvent(
            delta='{"query":"test"}', item_id="ctc_1",
            output_index=0, sequence_number=2,
            type="response.custom_tool_call_input.delta",
        )
        result = collector.process(event)
        assert result is None

    def test_input_done_returns_none(self, collector):
        event = ResponseCustomToolCallInputDoneEvent(
            input='{"query":"test"}', item_id="ctc_1",
            output_index=0, sequence_number=3,
            type="response.custom_tool_call_input.done",
        )
        result = collector.process(event)
        assert result is None

    @patch("timbal.collectors.impl.openai.get_run_context")
    def test_output_item_done_tracks_usage(self, mock_ctx, collector):
        # Register the content block first.
        collector.content_blocks.add("ctc_xs_call_1")

        tool_call = _make_custom_tool_call(status="completed")
        event = ResponseOutputItemDoneEvent(
            item=tool_call, output_index=0, sequence_number=5,
            type="response.output_item.done",
        )
        result = collector._handle_output_item_done(event)
        assert isinstance(result, TimbalContentBlockStop)
        mock_ctx().update_usage.assert_called_with(
            "grok-4-fast-reasoning:x_keyword_search_requests", 1,
        )


class TestIncompleteEvent:
    """Test handling of ResponseIncompleteEvent (e.g. max_output_tokens)."""

    @patch("timbal.collectors.impl.openai.get_run_context")
    def test_incomplete_sets_stop_reason(self, mock_ctx, collector):
        mock_response = MagicMock()
        mock_response.status = "incomplete"
        mock_response.incomplete_details.reason = "max_output_tokens"
        mock_response.usage.input_tokens = 100
        mock_response.usage.input_tokens_details.cached_tokens = 0
        mock_response.usage.output_tokens = 256
        mock_response.usage.output_tokens_details.audio_tokens = 0
        mock_response.model = "grok-4-fast-reasoning"
        # hasattr checks
        type(mock_response.usage.input_tokens_details).cached_tokens = 0
        type(mock_response.usage.input_tokens_details).audio_tokens = 0
        type(mock_response.usage.output_tokens_details).audio_tokens = 0

        event = MagicMock(spec=ResponseIncompleteEvent)
        event.response = mock_response

        collector._handle_completed(event)
        assert collector._stop_reason == "max_output_tokens"
