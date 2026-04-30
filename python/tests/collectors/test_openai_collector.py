"""Tests for ChatCompletionCollector and ResponseCollector."""
import time

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

from timbal.collectors.impl.openai import ChatCompletionCollector, ResponseCollector
from timbal.state import set_billing_id, set_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.span import Span
from timbal.types.content.text import TextContent
from timbal.types.content.thinking import ThinkingContent
from timbal.types.content.tool_use import ToolUseContent
from timbal.types.events.delta import (
    ContentBlockStop,
    Text,
    TextDelta as TimbalTextDelta,
    Thinking,
    ThinkingDelta as TimbalThinkingDelta,
    ToolUse,
    ToolUseDelta,
)


async def _empty_gen():
    return
    yield


@pytest.fixture(autouse=True)
def clean_context():
    """Reset context vars after each test to avoid state pollution."""
    from timbal.state import _run_context_var, _call_id, _billing_id
    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    token_bid = _billing_id.set(None)
    yield
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)
    _billing_id.reset(token_bid)


def _make_context():
    ctx = RunContext(tracing_provider=None)
    call_id = "test_call"
    span = Span(path="test", call_id=call_id, parent_call_id=None, t0=int(time.time() * 1000))
    ctx._trace[call_id] = span
    set_run_context(ctx)
    set_call_id(call_id)
    return ctx


def _make_cc_chunk(content=None, tool_calls=None, finish_reason=None, usage=None, model="gpt-4o"):
    delta = ChoiceDelta(content=content, tool_calls=tool_calls, role="assistant")
    choice = Choice(index=0, delta=delta, finish_reason=finish_reason, logprobs=None)
    return ChatCompletionChunk(
        id="chatcmpl_123",
        choices=[choice],
        created=int(time.time()),
        model=model,
        object="chat.completion.chunk",
        usage=usage,
    )


def _make_response(model="gpt-4o", status="in_progress"):
    return Response(
        id="resp_001",
        created_at=1234567890,
        model=model,
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status=status,
        usage=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        error=None,
        temperature=1.0,
        top_p=1.0,
        max_output_tokens=None,
        text=None,
        truncation="disabled",
    )


def _make_response_usage(input_tokens=10, output_tokens=5, cached_tokens=0):
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


class TestChatCompletionCollectorCanHandle:
    def test_handles_chat_completion_chunk(self):
        chunk = _make_cc_chunk(content="hi")
        assert ChatCompletionCollector.can_handle(chunk) is True

    def test_does_not_handle_other(self):
        assert ChatCompletionCollector.can_handle("str") is False
        assert ChatCompletionCollector.can_handle(None) is False


class TestChatCompletionCollectorText:
    def test_first_chunk_returns_text(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        item = collector.process(_make_cc_chunk(content="Hello"))
        assert isinstance(item, Text)
        assert item.text == "Hello"
        assert item.id == ChatCompletionCollector.TEXT_BLOCK_ID

    def test_subsequent_chunks_return_text_delta(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        collector.process(_make_cc_chunk(content="Hello"))
        item = collector.process(_make_cc_chunk(content=" world"))
        assert isinstance(item, TimbalTextDelta)
        assert item.text_delta == " world"

    def test_stop_reason_captured(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(_make_cc_chunk(content="hi"))
        collector.process(_make_cc_chunk(content=None, finish_reason="stop"))
        assert collector._stop_reason == "stop"

    def test_result_builds_message(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(_make_cc_chunk(content="Hello"))
        collector.process(_make_cc_chunk(content=" world"))
        collector.process(_make_cc_chunk(content=None, finish_reason="stop"))

        msg = collector.result()
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "Hello world"
        assert msg.stop_reason == "stop"

    def test_empty_choices_returns_none(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())
        chunk = ChatCompletionChunk(
            id="chatcmpl_456",
            choices=[],
            created=int(time.time()),
            model="gpt-4o",
            object="chat.completion.chunk",
            usage=None,
        )
        # Before any tokens, _first_token is None so early-return happens
        result = collector.process(chunk)
        assert result is None

    def test_no_content_no_tool_calls_returns_none(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())
        item = collector.process(_make_cc_chunk(content=None, tool_calls=None))
        assert item is None


class TestChatCompletionCollectorToolCalls:
    def test_tool_call_start(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        tc_start = ChoiceDeltaToolCall(
            index=0,
            id="call_001",
            type="function",
            function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=""),
        )
        item = collector.process(_make_cc_chunk(tool_calls=[tc_start]))
        assert isinstance(item, ToolUse)
        assert item.name == "get_weather"
        assert item.is_server_tool_use is False

    def test_tool_call_delta(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        tc_start = ChoiceDeltaToolCall(
            index=0,
            id="call_001",
            type="function",
            function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=""),
        )
        collector.process(_make_cc_chunk(tool_calls=[tc_start]))

        tc_delta = ChoiceDeltaToolCall(
            index=0,
            id=None,
            type=None,
            function=ChoiceDeltaToolCallFunction(name=None, arguments='{"city": "NYC"}'),
        )
        item = collector.process(_make_cc_chunk(tool_calls=[tc_delta]))
        assert isinstance(item, ToolUseDelta)
        assert item.input_delta == '{"city": "NYC"}'

    def test_result_has_tool_use_content(self):
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        tc_start = ChoiceDeltaToolCall(
            index=0,
            id="call_001",
            type="function",
            function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=""),
        )
        collector.process(_make_cc_chunk(tool_calls=[tc_start]))
        tc_delta = ChoiceDeltaToolCall(
            index=0, id=None, type=None,
            function=ChoiceDeltaToolCallFunction(name=None, arguments='{"city": "NYC"}'),
        )
        collector.process(_make_cc_chunk(tool_calls=[tc_delta]))
        collector.process(_make_cc_chunk(finish_reason="tool_calls"))

        msg = collector.result()
        assert len(msg.content) == 1
        tc = msg.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.name == "get_weather"
        assert tc.input == {"city": "NYC"}

    def test_tool_call_name_arrives_in_chunk_after_id(self):
        """Fireworks / Kimi can stream tool_call.id before function.name is set."""
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        tc_id_only = ChoiceDeltaToolCall(
            index=0,
            id="call_fw_1",
            type="function",
            function=ChoiceDeltaToolCallFunction(name=None, arguments=""),
        )
        assert collector.process(_make_cc_chunk(tool_calls=[tc_id_only])) is None

        tc_name_args = ChoiceDeltaToolCall(
            index=0,
            id=None,
            type=None,
            function=ChoiceDeltaToolCallFunction(name="get_user", arguments='{"id": 5}'),
        )
        item = collector.process(_make_cc_chunk(tool_calls=[tc_name_args]))
        assert isinstance(item, ToolUse)
        assert item.name == "get_user"
        assert item.id == "call_fw_1"

        collector.process(_make_cc_chunk(finish_reason="tool_calls"))
        msg = collector.result()
        assert len(msg.content) == 1
        tc = msg.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.name == "get_user"
        assert tc.input == {"id": 5}

    def test_tool_call_same_id_then_arguments(self):
        """Fireworks/Kimi may resend the same tool_call id with arguments after name+empty args."""
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        tc1 = ChoiceDeltaToolCall(
            index=0,
            id="call_fw_repeat",
            type="function",
            function=ChoiceDeltaToolCallFunction(name="get_user", arguments=""),
        )
        first = collector.process(_make_cc_chunk(tool_calls=[tc1]))
        assert isinstance(first, ToolUse)

        tc2 = ChoiceDeltaToolCall(
            index=0,
            id="call_fw_repeat",
            type="function",
            function=ChoiceDeltaToolCallFunction(name=None, arguments='{"id": 5}'),
        )
        second = collector.process(_make_cc_chunk(tool_calls=[tc2]))
        assert isinstance(second, ToolUseDelta)

        collector.process(_make_cc_chunk(finish_reason="tool_calls"))
        msg = collector.result()
        tc = msg.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.input == {"id": 5}


class TestResponseCollectorCanHandle:
    def test_handles_response_in_progress_event(self):
        event = ResponseInProgressEvent(
            type="response.in_progress",
            response=_make_response(),
            sequence_number=0,
        )
        assert ResponseCollector.can_handle(event) is True

    def test_does_not_handle_chat_completion_chunk(self):
        chunk = _make_cc_chunk(content="hi")
        assert ResponseCollector.can_handle(chunk) is False


class TestResponseCollectorText:
    def test_created_event_sets_model(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created",
            response=_make_response(model="gpt-4o"),
            sequence_number=0,
        ))
        assert collector.model == "gpt-4o"

    def test_in_progress_returns_none(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        result = collector.process(ResponseInProgressEvent(
            type="response.in_progress",
            response=_make_response(),
            sequence_number=0,
        ))
        assert result is None

    def test_text_message_flow(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))

        item_id = "item_001"
        text_part = ResponseOutputText(type="output_text", text="", annotations=[])

        item = collector.process(ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id=item_id,
            output_index=0,
            content_index=0,
            part=text_part,
            sequence_number=1,
        ))
        assert isinstance(item, Text)
        assert item.id == item_id

        item = collector.process(ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=item_id,
            output_index=0,
            content_index=0,
            delta="Hello",
            logprobs=[],
            sequence_number=2,
        ))
        assert isinstance(item, TimbalTextDelta)
        assert item.text_delta == "Hello"

        result = collector.process(ResponseTextDoneEvent(
            type="response.output_text.done",
            item_id=item_id,
            output_index=0,
            content_index=0,
            text="Hello",
            logprobs=[],
            sequence_number=3,
        ))
        assert result is None

        result = collector.process(ResponseContentPartDoneEvent(
            type="response.content_part.done",
            item_id=item_id,
            output_index=0,
            content_index=0,
            part=text_part,
            sequence_number=4,
        ))
        assert result is None

    def test_completed_event_sets_stop_reason(self):
        ctx = _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))

        item_id = "item_002"
        collector.process(ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part=ResponseOutputText(type="output_text", text="", annotations=[]),
            sequence_number=1,
        ))
        collector.process(ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=item_id, output_index=0, content_index=0, delta="done", logprobs=[],
            sequence_number=2,
        ))

        completed_response = _make_response(status="completed")
        completed_response = completed_response.model_copy(update={"usage": _make_response_usage()})
        collector.process(ResponseCompletedEvent(
            type="response.completed", response=completed_response, sequence_number=1,
        ))
        assert collector._stop_reason == "completed"

        msg = collector.result()
        assert msg.stop_reason == "completed"
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "done"

    def test_incomplete_event_sets_max_tokens_reason(self):
        ctx = _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))

        item_id = "item_003"
        collector.process(ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part=ResponseOutputText(type="output_text", text="", annotations=[]),
            sequence_number=1,
        ))
        collector.process(ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=item_id, output_index=0, content_index=0, delta="truncated", logprobs=[],
            sequence_number=2,
        ))

        from openai.types.responses.response import IncompleteDetails as ResponseIncompleteDetails
        incomplete_response = _make_response(status="incomplete")
        incomplete_details = ResponseIncompleteDetails(reason="max_output_tokens")
        incomplete_response = incomplete_response.model_copy(update={
            "incomplete_details": incomplete_details,
            "usage": _make_response_usage(output_tokens=100),
        })
        collector.process(ResponseIncompleteEvent(
            type="response.incomplete", response=incomplete_response, sequence_number=1,
        ))
        assert collector._stop_reason == "max_output_tokens"


class TestResponseCollectorToolCalls:
    def test_function_tool_call_start(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))

        fn_tool = ResponseFunctionToolCall(
            type="function_call",
            id="item_fn_001",
            call_id="call_001",
            name="search",
            arguments="",
            status="in_progress",
        )
        item = collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=fn_tool, sequence_number=1,
        ))
        assert isinstance(item, ToolUse)
        assert item.name == "search"
        assert item.is_server_tool_use is False

    def test_function_call_arguments_delta(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))

        fn_tool = ResponseFunctionToolCall(
            type="function_call", id="item_fn_001", call_id="call_001",
            name="search", arguments="", status="in_progress",
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=fn_tool, sequence_number=1,
        ))

        item = collector.process(ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            item_id="item_fn_001", output_index=0, delta='{"q": "test"}', sequence_number=2,
        ))
        assert isinstance(item, ToolUseDelta)
        assert item.input_delta == '{"q": "test"}'

    def test_function_call_arguments_done_returns_none(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))
        fn_tool = ResponseFunctionToolCall(
            type="function_call", id="item_fn_001", call_id="call_001",
            name="search", arguments="", status="in_progress",
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=fn_tool, sequence_number=1,
        ))
        result = collector.process(ResponseFunctionCallArgumentsDoneEvent(
            type="response.function_call_arguments.done",
            item_id="item_fn_001", output_index=0, arguments='{"q": "test"}', name="search", sequence_number=3,
        ))
        assert result is None

    def test_output_item_done_returns_content_block_stop(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))
        fn_tool = ResponseFunctionToolCall(
            type="function_call", id="item_fn_001", call_id="call_001",
            name="search", arguments="", status="in_progress",
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=fn_tool, sequence_number=1,
        ))
        fn_tool_done = ResponseFunctionToolCall(
            type="function_call", id="item_fn_001", call_id="call_001",
            name="search", arguments='{"q": "test"}', status="completed",
        )
        item = collector.process(ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=fn_tool_done, sequence_number=2,
        ))
        assert isinstance(item, ContentBlockStop)


class TestResponseCollectorReasoning:
    def test_reasoning_summary_part_added(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="o3"), sequence_number=0,
        ))

        reasoning_item = ResponseReasoningItem(
            type="reasoning", id="item_r_001", status="in_progress", summary=[],
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=reasoning_item, sequence_number=1,
        ))

        from openai.types.responses.response_reasoning_summary_part_added_event import Part as ReasoningSummaryPart
        part = ReasoningSummaryPart(type="summary_text", text="")
        item = collector.process(ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            item_id="item_r_001", output_index=0, summary_index=0, part=part, sequence_number=2,
        ))
        assert isinstance(item, Thinking)

    def test_reasoning_summary_text_delta(self):
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="o3"), sequence_number=0,
        ))

        reasoning_item = ResponseReasoningItem(
            type="reasoning", id="item_r_002", status="in_progress", summary=[],
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=reasoning_item, sequence_number=1,
        ))

        from openai.types.responses.response_reasoning_summary_part_added_event import Part as ReasoningSummaryPart
        part = ReasoningSummaryPart(type="summary_text", text="")
        collector.process(ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            item_id="item_r_002", output_index=0, summary_index=0, part=part, sequence_number=2,
        ))

        item = collector.process(ResponseReasoningSummaryTextDeltaEvent(
            type="response.reasoning_summary_text.delta",
            item_id="item_r_002", output_index=0, summary_index=0, delta="reasoning step", sequence_number=3,
        ))
        assert isinstance(item, TimbalThinkingDelta)
        assert item.thinking_delta == "reasoning step"


# ---------------------------------------------------------------------------
# New tests to cover remaining gaps
# ---------------------------------------------------------------------------

class TestChatCompletionCollectorEarlyReturn:
    """Line 134: early return when _first_token is None and choices is empty after usage chunk."""

    def test_empty_choices_after_usage_chunk_returns_none(self):
        """Sending a usage-only chunk (no choices) before any text should return None."""
        from openai.types.completion_usage import CompletionUsage

        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        # Build a usage-carrying chunk with no choices
        usage = CompletionUsage(
            completion_tokens=5,
            prompt_tokens=10,
            total_tokens=15,
        )
        chunk = ChatCompletionChunk(
            id="chatcmpl_usage",
            choices=[],
            created=int(time.time()),
            model="gpt-4o",
            object="chat.completion.chunk",
            usage=usage,
        )
        # _first_token is None because no text/tool chunk has been seen yet
        assert collector._first_token is None
        result = collector.process(chunk)
        assert result is None
        # usage was stashed
        assert collector._pending_usage is chunk


class TestChatCompletionCollectorHandleUsage:
    """Lines 154-178: _handle_usage with cached and audio tokens deduction."""

    def test_usage_with_cached_input_tokens(self):
        """Cached input tokens are deducted from plain input and tracked separately."""
        from openai.types.completion_usage import CompletionUsage, PromptTokensDetails, CompletionTokensDetails

        ctx = _make_context()
        set_billing_id("openai/gpt-4o")
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        # Fire a text chunk so _first_token is set (needed for result())
        collector.process(_make_cc_chunk(content="hi"))

        usage = CompletionUsage(
            completion_tokens=5,
            prompt_tokens=20,
            total_tokens=25,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=8, audio_tokens=None),
            completion_tokens_details=CompletionTokensDetails(audio_tokens=None, reasoning_tokens=0),
        )
        chunk = ChatCompletionChunk(
            id="chatcmpl_usage2",
            choices=[],
            created=int(time.time()),
            model="gpt-4o",
            object="chat.completion.chunk",
            usage=usage,
        )
        collector.process(chunk)

        # Trigger _handle_usage via result()
        msg = collector.result()

        span = ctx._trace["test_call"]
        # cached tokens should be tracked, plain input should be 20-8=12
        assert span.usage.get("openai/gpt-4o:input_cached_tokens") == 8
        assert span.usage.get("openai/gpt-4o:input_text_tokens") == 12

    def test_usage_with_audio_input_tokens(self):
        """Audio input tokens are deducted from plain input and tracked separately."""
        from openai.types.completion_usage import CompletionUsage, PromptTokensDetails, CompletionTokensDetails

        ctx = _make_context()
        set_billing_id("openai/gpt-4o")
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(_make_cc_chunk(content="hi"))

        usage = CompletionUsage(
            completion_tokens=5,
            prompt_tokens=20,
            total_tokens=25,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=None, audio_tokens=6),
            completion_tokens_details=CompletionTokensDetails(audio_tokens=None, reasoning_tokens=0),
        )
        chunk = ChatCompletionChunk(
            id="chatcmpl_audio",
            choices=[],
            created=int(time.time()),
            model="gpt-4o",
            object="chat.completion.chunk",
            usage=usage,
        )
        collector.process(chunk)
        collector.result()

        span = ctx._trace["test_call"]
        assert span.usage.get("openai/gpt-4o:input_audio_tokens") == 6
        assert span.usage.get("openai/gpt-4o:input_text_tokens") == 14  # 20 - 6

    def test_usage_with_audio_output_tokens(self):
        """Audio output tokens are deducted from plain output and tracked separately."""
        from openai.types.completion_usage import CompletionUsage, PromptTokensDetails, CompletionTokensDetails

        ctx = _make_context()
        set_billing_id("openai/gpt-4o")
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(_make_cc_chunk(content="hi"))

        usage = CompletionUsage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=None, audio_tokens=None),
            completion_tokens_details=CompletionTokensDetails(audio_tokens=4, reasoning_tokens=0),
        )
        chunk = ChatCompletionChunk(
            id="chatcmpl_audio_out",
            choices=[],
            created=int(time.time()),
            model="gpt-4o",
            object="chat.completion.chunk",
            usage=usage,
        )
        collector.process(chunk)
        collector.result()

        span = ctx._trace["test_call"]
        assert span.usage.get("openai/gpt-4o:output_audio_tokens") == 4
        assert span.usage.get("openai/gpt-4o:output_text_tokens") == 6  # 10 - 4


class TestChatCompletionCollectorGoogleThoughtSignature:
    """Lines 196-198: Google Gemini thought_signature stored in tool call extra_content."""

    def test_thought_signature_captured_in_tool_call(self):
        """extra_content with google.thought_signature is stored on the tool call dict."""
        _make_context()
        collector = ChatCompletionCollector(async_gen=_empty_gen(), start=time.perf_counter())

        from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

        tc = ChoiceDeltaToolCall.model_construct(
            index=0,
            id="call_gemini_001",
            type="function",
            function=ChoiceDeltaToolCallFunction(name="my_tool", arguments=""),
            extra_content={"google": {"thought_signature": "sig_abc123"}},
        )
        item = collector.process(_make_cc_chunk(tool_calls=[tc]))
        assert isinstance(item, ToolUse)
        # The tool call dict should carry the thought_signature
        assert collector._tool_calls[0]["thought_signature"] == "sig_abc123"


class TestResponseCollectorWebSearch:
    """Lines 305,307,309,315,320-331: web search call events return None; web search output_item_added."""

    def test_web_search_in_progress_returns_none(self):
        from openai.types.responses import ResponseWebSearchCallInProgressEvent

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))
        result = collector.process(ResponseWebSearchCallInProgressEvent(
            type="response.web_search_call.in_progress",
            item_id="ws_001", output_index=0, sequence_number=1,
        ))
        assert result is None

    def test_web_search_searching_returns_none(self):
        from openai.types.responses import ResponseWebSearchCallSearchingEvent

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))
        result = collector.process(ResponseWebSearchCallSearchingEvent(
            type="response.web_search_call.searching",
            item_id="ws_001", output_index=0, sequence_number=1,
        ))
        assert result is None

    def test_web_search_completed_returns_none(self):
        from openai.types.responses import ResponseWebSearchCallCompletedEvent

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))
        result = collector.process(ResponseWebSearchCallCompletedEvent(
            type="response.web_search_call.completed",
            item_id="ws_001", output_index=0, sequence_number=1,
        ))
        assert result is None

    def test_web_search_output_item_added_returns_tool_use(self):
        """ResponseFunctionWebSearch in output_item_added emits a server ToolUse."""
        from openai.types.responses import ResponseFunctionWebSearch
        from openai.types.responses.response_function_web_search import ActionSearch

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))

        ws_item = ResponseFunctionWebSearch(
            id="ws_item_001",
            type="web_search_call",
            action=ActionSearch(type="search", query="latest news"),
            status="in_progress",
        )
        item = collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=ws_item, sequence_number=1,
        ))
        assert isinstance(item, ToolUse)
        assert item.name == "web_search"
        assert item.is_server_tool_use is True
        assert "ws_item_001" in collector.content_blocks

    def test_output_message_item_added_returns_none(self):
        """ResponseOutputMessage in output_item_added returns None."""
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))

        out_msg = ResponseOutputMessage(
            id="msg_001", type="message", role="assistant", content=[], status="in_progress",
        )
        result = collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=out_msg, sequence_number=1,
        ))
        assert result is None


class TestResponseCollectorAnnotationAndReasoning:
    """Lines 355,358-360: output text annotation added; reasoning summary part/text done."""

    def test_output_text_annotation_added_returns_none(self):
        """ResponseOutputTextAnnotationAddedEvent appends annotation and returns None."""
        from openai.types.responses import ResponseOutputTextAnnotationAddedEvent
        from openai.types.responses.response_output_text import AnnotationURLCitation

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))

        item_id = "ann_item_001"
        text_part = ResponseOutputText(type="output_text", text="", annotations=[])
        collector.process(ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id=item_id, output_index=0, content_index=0, part=text_part, sequence_number=1,
        ))

        annotation = AnnotationURLCitation(
            type="url_citation", url="https://example.com", title="Example", start_index=0, end_index=5,
        )
        result = collector.process(ResponseOutputTextAnnotationAddedEvent(
            type="response.output_text.annotation.added",
            item_id=item_id, output_index=0, content_index=0,
            annotation_index=0, annotation=annotation, sequence_number=2,
        ))
        assert result is None
        assert len(collector.content[item_id]["citations"]) == 1

    def test_reasoning_summary_part_done_returns_none(self):
        """ResponseReasoningSummaryPartDoneEvent returns None."""
        from openai.types.responses import ResponseReasoningSummaryPartDoneEvent
        from openai.types.responses.response_reasoning_summary_part_added_event import Part as ReasoningSummaryPart

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="o3"), sequence_number=0,
        ))
        reasoning_item = ResponseReasoningItem(
            type="reasoning", id="item_r_done_001", status="in_progress", summary=[],
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=reasoning_item, sequence_number=1,
        ))
        part_added = ReasoningSummaryPart(type="summary_text", text="")
        collector.process(ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            item_id="item_r_done_001", output_index=0, summary_index=0, part=part_added, sequence_number=2,
        ))

        from openai.types.responses.response_reasoning_summary_part_done_event import Part as ReasoningSummaryPartDone
        part_done = ReasoningSummaryPartDone(type="summary_text", text="full text")
        result = collector.process(ResponseReasoningSummaryPartDoneEvent(
            type="response.reasoning_summary_part.done",
            item_id="item_r_done_001", output_index=0, summary_index=0, part=part_done, sequence_number=3,
        ))
        assert result is None

    def test_reasoning_summary_text_done_returns_none(self):
        """ResponseReasoningSummaryTextDoneEvent returns None."""
        from openai.types.responses import ResponseReasoningSummaryTextDoneEvent
        from openai.types.responses.response_reasoning_summary_part_added_event import Part as ReasoningSummaryPart

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="o3"), sequence_number=0,
        ))
        reasoning_item = ResponseReasoningItem(
            type="reasoning", id="item_r_done_002", status="in_progress", summary=[],
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=reasoning_item, sequence_number=1,
        ))
        part = ReasoningSummaryPart(type="summary_text", text="")
        collector.process(ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            item_id="item_r_done_002", output_index=0, summary_index=0, part=part, sequence_number=2,
        ))
        result = collector.process(ResponseReasoningSummaryTextDoneEvent(
            type="response.reasoning_summary_text.done",
            item_id="item_r_done_002", output_index=0, summary_index=0, text="full reasoning", sequence_number=3,
        ))
        assert result is None


class TestResponseCollectorCustomToolCall:
    """Lines 373-385: ResponseCustomToolCall in output_item_added."""

    def test_custom_tool_call_output_item_added_returns_server_tool_use(self):
        """ResponseCustomToolCall emits a server ToolUse and is tracked."""
        from openai.types.responses import ResponseCustomToolCall

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="xai/grok-3"), sequence_number=0,
        ))

        custom_tool = ResponseCustomToolCall(
            id="custom_001",
            call_id="ccall_001",
            name="x_keyword_search",
            input='{"query": "ai news"}',
            type="custom_tool_call",
        )
        item = collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=custom_tool, sequence_number=1,
        ))
        assert isinstance(item, ToolUse)
        assert item.name == "x_keyword_search"
        assert item.is_server_tool_use is True
        assert item.input == '{"query": "ai news"}'
        assert "custom_001" in collector.content_blocks


class TestResponseCollectorOutputItemDone:
    """Lines 402,416,478-503: output item done for web search, reasoning, and custom tool call."""

    def test_web_search_done_with_content_block_tracks_usage_and_returns_stop(self):
        """Web search done tracks usage and returns ContentBlockStop when block registered."""
        from openai.types.responses import ResponseFunctionWebSearch
        from openai.types.responses.response_function_web_search import ActionSearch

        ctx = _make_context()
        set_billing_id("openai/gpt-4o")
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="gpt-4o"), sequence_number=0,
        ))

        ws_item = ResponseFunctionWebSearch(
            id="ws_done_001",
            type="web_search_call",
            action=ActionSearch(type="search", query="test"),
            status="in_progress",
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=ws_item, sequence_number=1,
        ))

        ws_done = ResponseFunctionWebSearch(
            id="ws_done_001",
            type="web_search_call",
            action=ActionSearch(type="search", query="test"),
            status="completed",
        )
        item = collector.process(ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=ws_done, sequence_number=2,
        ))
        assert isinstance(item, ContentBlockStop)
        assert item.id == "ws_done_001"
        span = ctx._trace["test_call"]
        assert span.usage.get("openai/gpt-4o:web_search_requests") == 1

    def test_web_search_done_without_content_block_returns_none(self):
        """Web search done returns None when content_block_id was never registered."""
        from openai.types.responses import ResponseFunctionWebSearch
        from openai.types.responses.response_function_web_search import ActionSearch

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="gpt-4o"), sequence_number=0,
        ))
        # Don't add an output_item_added for this ID — block never registered
        ws_done = ResponseFunctionWebSearch(
            id="ws_orphan_001",
            type="web_search_call",
            action=ActionSearch(type="search", query="test"),
            status="completed",
        )
        item = collector.process(ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=ws_done, sequence_number=1,
        ))
        assert item is None

    def test_reasoning_item_done_returns_content_block_stop(self):
        """ResponseReasoningItem done returns ContentBlockStop."""
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="o3"), sequence_number=0,
        ))
        reasoning_item = ResponseReasoningItem(
            type="reasoning", id="item_r_stop_001", status="in_progress", summary=[],
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=reasoning_item, sequence_number=1,
        ))
        reasoning_done = ResponseReasoningItem(
            type="reasoning", id="item_r_stop_001", status="completed", summary=[],
        )
        item = collector.process(ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=reasoning_done, sequence_number=2,
        ))
        assert isinstance(item, ContentBlockStop)
        assert item.id == "item_r_stop_001"

    def test_output_message_done_without_content_block_returns_none(self):
        """ResponseOutputMessage done returns None when block not registered."""
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))
        # output_item_added for OutputMessage returns None and doesn't register in content_blocks
        out_msg = ResponseOutputMessage(
            id="msg_orphan_001", type="message", role="assistant", content=[], status="in_progress",
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=out_msg, sequence_number=1,
        ))
        out_msg_done = ResponseOutputMessage(
            id="msg_orphan_001", type="message", role="assistant", content=[], status="completed",
        )
        item = collector.process(ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=out_msg_done, sequence_number=2,
        ))
        assert item is None

    def test_custom_tool_call_done_tracks_usage_and_returns_stop(self):
        """ResponseCustomToolCall done tracks usage and returns ContentBlockStop."""
        from openai.types.responses import ResponseCustomToolCall

        ctx = _make_context()
        set_billing_id("xai/grok-3")
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="xai/grok-3"), sequence_number=0,
        ))
        custom_tool = ResponseCustomToolCall(
            id="custom_done_001",
            call_id="ccall_done_001",
            name="x_keyword_search",
            input='{"query": "ai"}',
            type="custom_tool_call",
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=custom_tool, sequence_number=1,
        ))
        custom_tool_done = ResponseCustomToolCall(
            id="custom_done_001",
            call_id="ccall_done_001",
            name="x_keyword_search",
            input='{"query": "ai"}',
            type="custom_tool_call",
        )
        item = collector.process(ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=custom_tool_done, sequence_number=2,
        ))
        assert isinstance(item, ContentBlockStop)
        assert item.id == "custom_done_001"
        span = ctx._trace["test_call"]
        assert span.usage.get("xai/grok-3:x_keyword_search_requests") == 1

    def test_custom_tool_call_done_without_content_block_returns_none(self):
        """ResponseCustomToolCall done returns None when block not registered."""
        from openai.types.responses import ResponseCustomToolCall

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="xai/grok-3"), sequence_number=0,
        ))
        # Don't call output_item_added — block never registered
        custom_tool_done = ResponseCustomToolCall(
            id="custom_orphan_001",
            call_id="ccall_orphan_001",
            name="x_keyword_search",
            input='{"query": "ai"}',
            type="custom_tool_call",
        )
        item = collector.process(ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=custom_tool_done, sequence_number=1,
        ))
        assert item is None


class TestResponseCollectorReasoningTextDelta:
    """Line 513: _handle_reasoning_text_delta first-delta path (item_id not yet in content)."""

    def test_reasoning_text_delta_first_delta_creates_block(self):
        """First raw reasoning text delta creates a new thinking block and returns Thinking."""
        from openai.types.responses import ResponseReasoningTextDeltaEvent

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="xai/grok-3"), sequence_number=0,
        ))
        # Send delta without any output_item_added — raw xAI reasoning path
        item = collector.process(ResponseReasoningTextDeltaEvent(
            type="response.reasoning_text.delta",
            item_id="raw_r_001", output_index=0, content_index=0, delta="step one", sequence_number=1,
        ))
        assert isinstance(item, Thinking)
        assert item.id == "raw_r_001"
        assert item.thinking == "step one"
        assert "raw_r_001" in collector.content_blocks

    def test_reasoning_text_delta_subsequent_returns_thinking_delta(self):
        """Subsequent raw reasoning text deltas return ThinkingDelta."""
        from openai.types.responses import ResponseReasoningTextDeltaEvent

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="xai/grok-3"), sequence_number=0,
        ))
        collector.process(ResponseReasoningTextDeltaEvent(
            type="response.reasoning_text.delta",
            item_id="raw_r_002", output_index=0, content_index=0, delta="first", sequence_number=1,
        ))
        item = collector.process(ResponseReasoningTextDeltaEvent(
            type="response.reasoning_text.delta",
            item_id="raw_r_002", output_index=0, content_index=0, delta=" second", sequence_number=2,
        ))
        assert isinstance(item, TimbalThinkingDelta)
        assert item.thinking_delta == " second"


class TestResponseCollectorHandleCompleted:
    """Lines 522-537: _handle_completed with cached tokens, audio input/output tokens."""

    def test_completed_with_cached_input_tokens(self):
        """Cached tokens are tracked and deducted from plain input tokens."""
        from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

        ctx = _make_context()
        set_billing_id("openai/gpt-4o")
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="gpt-4o"), sequence_number=0,
        ))
        item_id = "cached_item_001"
        collector.process(ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part=ResponseOutputText(type="output_text", text="", annotations=[]), sequence_number=1,
        ))
        collector.process(ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=item_id, output_index=0, content_index=0, delta="hi", logprobs=[], sequence_number=2,
        ))

        completed_response = _make_response(model="gpt-4o", status="completed")
        input_details = InputTokensDetails(cached_tokens=7)
        output_details = OutputTokensDetails(reasoning_tokens=0)
        usage = ResponseUsage(
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )
        completed_response = completed_response.model_copy(update={"usage": usage})
        collector.process(ResponseCompletedEvent(
            type="response.completed", response=completed_response, sequence_number=3,
        ))

        span = ctx._trace["test_call"]
        assert span.usage.get("openai/gpt-4o:input_cached_tokens") == 7
        assert span.usage.get("openai/gpt-4o:input_text_tokens") == 13  # 20 - 7

    def test_completed_with_audio_input_tokens(self):
        """Audio input tokens are tracked and deducted from plain input tokens."""
        from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

        ctx = _make_context()
        set_billing_id("openai/gpt-4o")
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="gpt-4o"), sequence_number=0,
        ))
        item_id = "audio_in_item_001"
        collector.process(ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part=ResponseOutputText(type="output_text", text="", annotations=[]), sequence_number=1,
        ))
        collector.process(ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=item_id, output_index=0, content_index=0, delta="hi", logprobs=[], sequence_number=2,
        ))

        completed_response = _make_response(model="gpt-4o", status="completed")
        input_details = InputTokensDetails.model_construct(cached_tokens=0, audio_tokens=5)
        output_details = OutputTokensDetails(reasoning_tokens=0)
        usage = ResponseUsage(
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )
        completed_response = completed_response.model_copy(update={"usage": usage})
        collector.process(ResponseCompletedEvent(
            type="response.completed", response=completed_response, sequence_number=3,
        ))

        span = ctx._trace["test_call"]
        assert span.usage.get("openai/gpt-4o:input_audio_tokens") == 5
        assert span.usage.get("openai/gpt-4o:input_text_tokens") == 15  # 20 - 5

    def test_completed_with_audio_output_tokens(self):
        """Audio output tokens are tracked and deducted from plain output tokens."""
        from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

        ctx = _make_context()
        set_billing_id("openai/gpt-4o")
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="gpt-4o"), sequence_number=0,
        ))
        item_id = "audio_out_item_001"
        collector.process(ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part=ResponseOutputText(type="output_text", text="", annotations=[]), sequence_number=1,
        ))
        collector.process(ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=item_id, output_index=0, content_index=0, delta="hi", logprobs=[], sequence_number=2,
        ))

        completed_response = _make_response(model="gpt-4o", status="completed")
        input_details = InputTokensDetails(cached_tokens=0)
        output_details = OutputTokensDetails.model_construct(reasoning_tokens=0, audio_tokens=3)
        usage = ResponseUsage(
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )
        completed_response = completed_response.model_copy(update={"usage": usage})
        collector.process(ResponseCompletedEvent(
            type="response.completed", response=completed_response, sequence_number=3,
        ))

        span = ctx._trace["test_call"]
        assert span.usage.get("openai/gpt-4o:output_audio_tokens") == 3
        assert span.usage.get("openai/gpt-4o:output_text_tokens") == 7  # 10 - 3


class TestResponseCollectorResult:
    """Lines 457-470, 552,560,562,564,572: result() branches for all content block types."""

    def test_result_with_tool_use_content_block(self):
        """result() builds ToolUseContent for function_call blocks."""
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))
        fn_tool = ResponseFunctionToolCall(
            type="function_call", id="item_fn_res_001", call_id="call_res_001",
            name="do_thing", arguments="", status="in_progress",
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=fn_tool, sequence_number=1,
        ))
        collector.process(ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            item_id="item_fn_res_001", output_index=0, delta='{"x": 1}', sequence_number=2,
        ))
        fn_tool_done = ResponseFunctionToolCall(
            type="function_call", id="item_fn_res_001", call_id="call_res_001",
            name="do_thing", arguments='{"x": 1}', status="completed",
        )
        collector.process(ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=fn_tool_done, sequence_number=3,
        ))

        completed_response = _make_response(status="completed")
        completed_response = completed_response.model_copy(update={"usage": _make_response_usage()})
        collector.process(ResponseCompletedEvent(
            type="response.completed", response=completed_response, sequence_number=4,
        ))

        msg = collector.result()
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolUseContent)
        assert msg.content[0].name == "do_thing"

    def test_result_with_thinking_content_block(self):
        """result() builds ThinkingContent for reasoning blocks."""
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="o3"), sequence_number=0,
        ))
        reasoning_item = ResponseReasoningItem(
            type="reasoning", id="item_r_res_001", status="in_progress", summary=[],
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=reasoning_item, sequence_number=1,
        ))

        from openai.types.responses.response_reasoning_summary_part_added_event import Part as ReasoningSummaryPart
        part = ReasoningSummaryPart(type="summary_text", text="")
        collector.process(ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            item_id="item_r_res_001", output_index=0, summary_index=0, part=part, sequence_number=2,
        ))
        collector.process(ResponseReasoningSummaryTextDeltaEvent(
            type="response.reasoning_summary_text.delta",
            item_id="item_r_res_001", output_index=0, summary_index=0, delta="my thoughts", sequence_number=3,
        ))

        completed_response = _make_response(model="o3", status="completed")
        completed_response = completed_response.model_copy(update={"usage": _make_response_usage()})
        collector.process(ResponseCompletedEvent(
            type="response.completed", response=completed_response, sequence_number=4,
        ))

        msg = collector.result()
        thinking_blocks = [c for c in msg.content if isinstance(c, ThinkingContent)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "my thoughts"

    def test_result_with_text_content_block(self):
        """result() builds TextContent for text blocks."""
        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(), sequence_number=0,
        ))
        item_id = "text_res_001"
        collector.process(ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part=ResponseOutputText(type="output_text", text="", annotations=[]), sequence_number=1,
        ))
        collector.process(ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=item_id, output_index=0, content_index=0, delta="hello world", logprobs=[], sequence_number=2,
        ))

        completed_response = _make_response(status="completed")
        completed_response = completed_response.model_copy(update={"usage": _make_response_usage()})
        collector.process(ResponseCompletedEvent(
            type="response.completed", response=completed_response, sequence_number=3,
        ))

        msg = collector.result()
        text_blocks = [c for c in msg.content if isinstance(c, TextContent)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "hello world"

    def test_result_skips_server_tool_use_blocks(self):
        """result() skips server_tool_use content blocks (web search)."""
        from openai.types.responses import ResponseFunctionWebSearch
        from openai.types.responses.response_function_web_search import ActionSearch

        _make_context()
        collector = ResponseCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(ResponseCreatedEvent(
            type="response.created", response=_make_response(model="gpt-4o"), sequence_number=0,
        ))

        ws_item = ResponseFunctionWebSearch(
            id="ws_res_001",
            type="web_search_call",
            action=ActionSearch(type="search", query="test"),
            status="in_progress",
        )
        collector.process(ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=0, item=ws_item, sequence_number=1,
        ))
        # Manually inject server_tool_use block to test result() skip branch
        collector.content["ws_res_001"] = {"type": "server_tool_use", "name": "web_search"}

        completed_response = _make_response(model="gpt-4o", status="completed")
        completed_response = completed_response.model_copy(update={"usage": _make_response_usage()})
        collector.process(ResponseCompletedEvent(
            type="response.completed", response=completed_response, sequence_number=2,
        ))

        msg = collector.result()
        assert len(msg.content) == 0  # server_tool_use skipped
