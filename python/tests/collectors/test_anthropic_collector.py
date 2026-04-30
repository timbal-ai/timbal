"""Tests for AnthropicCollector event processing."""
import time

import pytest
from anthropic.types import (
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
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRawMessageStopEvent,
    BetaTextBlock,
    BetaTextDelta,
    BetaThinkingBlock,
    BetaThinkingDelta,
    BetaToolUseBlock,
)

from timbal.collectors.impl.anthropic import AnthropicCollector
from timbal.state import set_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.span import Span
from timbal.types.content.custom import CustomContent
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
    yield  # make it an async generator


@pytest.fixture(autouse=True)
def clean_context():
    """Reset context vars after each test to avoid state pollution."""
    from timbal.state import _run_context_var, _call_id, set_call_id
    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    yield
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)


def _make_context():
    """Set up a RunContext with a current span."""
    ctx = RunContext(tracing_provider=None)
    call_id = "test_call"
    span = Span(path="test", call_id=call_id, parent_call_id=None, t0=int(time.time() * 1000))
    ctx._trace[call_id] = span
    set_run_context(ctx)
    set_call_id(call_id)
    return ctx


def _make_collector():
    return AnthropicCollector(async_gen=_empty_gen(), start=time.perf_counter())


def _make_message_start(msg_id="msg_abc123", model="claude-sonnet-4-6"):
    return RawMessageStartEvent(**{
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    })


def _make_beta_message_start(msg_id="msg_beta", model="claude-sonnet-4-6"):
    return BetaRawMessageStartEvent(**{
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 0},
        },
    })


class TestAnthropicCollectorCanHandle:
    def test_handles_raw_message_start(self):
        event = _make_message_start()
        assert AnthropicCollector.can_handle(event) is True

    def test_handles_beta_raw_message_start(self):
        event = _make_beta_message_start()
        assert AnthropicCollector.can_handle(event) is True

    def test_does_not_handle_arbitrary_object(self):
        assert AnthropicCollector.can_handle("not an event") is False
        assert AnthropicCollector.can_handle(42) is False
        assert AnthropicCollector.can_handle(None) is False


class TestAnthropicCollectorTextMessage:
    def test_text_message_end_to_end(self):
        _make_context()
        collector = _make_collector()

        collector.process(_make_message_start())
        assert collector.id == "msg_abc123"

        block_start = RawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": TextBlock(type="text", text=""),
        })
        item = collector.process(block_start)
        assert isinstance(item, Text)
        assert item.id == "msg_abc123-0"

        item = collector.process(RawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": TextDelta(type="text_delta", text="Hello"),
        }))
        assert isinstance(item, TimbalTextDelta)
        assert item.text_delta == "Hello"

        item = collector.process(RawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": TextDelta(type="text_delta", text=" world"),
        }))
        assert item.text_delta == " world"

        item = collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": 0}))
        assert isinstance(item, ContentBlockStop)
        assert item.id == "msg_abc123-0"

        msg = collector.result()
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "Hello world"

    def test_consecutive_text_blocks_merged(self):
        """Two text blocks should be merged into a single TextContent."""
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())

        for i in range(2):
            collector.process(RawContentBlockStartEvent(**{
                "type": "content_block_start",
                "index": i,
                "content_block": TextBlock(type="text", text=""),
            }))
            collector.process(RawContentBlockDeltaEvent(**{
                "type": "content_block_delta",
                "index": i,
                "delta": TextDelta(type="text_delta", text=f"part{i}"),
            }))
            collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": i}))

        msg = collector.result()
        assert len(msg.content) == 1
        assert msg.content[0].text == "part0part1"

    def test_message_stop_returns_none(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())
        result = collector.process(RawMessageStopEvent(**{"type": "message_stop"}))
        assert result is None


class TestAnthropicCollectorToolUse:
    def test_tool_use_end_to_end(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())

        item = collector.process(RawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": ToolUseBlock(type="tool_use", id="tool_001", name="get_weather", input={}),
        }))
        assert isinstance(item, ToolUse)
        assert item.name == "get_weather"
        assert item.is_server_tool_use is False

        item = collector.process(RawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": InputJSONDelta(type="input_json_delta", partial_json='{"city": "NYC"}'),
        }))
        assert isinstance(item, ToolUseDelta)
        assert item.input_delta == '{"city": "NYC"}'

        collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": 0}))

        msg = collector.result()
        assert len(msg.content) == 1
        tc = msg.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.name == "get_weather"
        assert tc.input == {"city": "NYC"}

    def test_server_tool_use(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())

        item = collector.process(RawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": ServerToolUseBlock(type="server_tool_use", id="svu_001", name="web_search", input={}),
        }))
        assert isinstance(item, ToolUse)
        assert item.is_server_tool_use is True

        msg = collector.result()
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolUseContent)
        assert msg.content[0].is_server_tool_use is True


class TestAnthropicCollectorThinking:
    def test_thinking_block_end_to_end(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())

        item = collector.process(RawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": ThinkingBlock(type="thinking", thinking="", signature=""),
        }))
        assert isinstance(item, Thinking)

        item = collector.process(RawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": ThinkingDelta(type="thinking_delta", thinking="Let me reason..."),
        }))
        assert isinstance(item, TimbalThinkingDelta)
        assert item.thinking_delta == "Let me reason..."

        collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": 0}))

        msg = collector.result()
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ThinkingContent)
        assert msg.content[0].thinking == "Let me reason..."

    def test_signature_delta_returns_none(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())

        collector.process(RawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": ThinkingBlock(type="thinking", thinking="", signature=""),
        }))
        item = collector.process(RawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": SignatureDelta(type="signature_delta", signature="sig_abc"),
        }))
        assert item is None
        assert collector.content[0]["signature"] == "sig_abc"


class TestAnthropicCollectorWebSearch:
    def test_web_search_tool_result_returns_none_from_process(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())

        item = collector.process(RawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": WebSearchToolResultBlock(
                type="web_search_tool_result",
                tool_use_id="tu_001",
                content=[],
            ),
        }))
        assert item is None

        msg = collector.result()
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], CustomContent)


class TestAnthropicCollectorMessageDelta:
    def test_stop_reason_end_turn(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())

        collector.process(RawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": TextBlock(type="text", text=""),
        }))
        collector.process(RawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": TextDelta(type="text_delta", text="done"),
        }))

        collector.process(RawMessageDeltaEvent(**{
            "type": "message_delta",
            "delta": {"type": "message_delta", "stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        }))
        assert collector._stop_reason == "end_turn"

        msg = collector.result()
        assert msg.stop_reason == "end_turn"

    def test_stop_reason_tool_use(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())

        collector.process(RawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": ToolUseBlock(type="tool_use", id="t1", name="fn", input={}),
        }))
        collector.process(RawMessageDeltaEvent(**{
            "type": "message_delta",
            "delta": {"type": "message_delta", "stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        }))

        msg = collector.result()
        assert msg.stop_reason == "tool_use"


class TestAnthropicCollectorBetaVariants:
    def test_beta_text_message(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_beta_message_start("msg_beta_001"))

        item = collector.process(BetaRawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": BetaTextBlock(type="text", text=""),
        }))
        assert isinstance(item, Text)

        item = collector.process(BetaRawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": BetaTextDelta(type="text_delta", text="hi"),
        }))
        assert isinstance(item, TimbalTextDelta)

        item = collector.process(BetaRawContentBlockStopEvent(**{"type": "content_block_stop", "index": 0}))
        assert isinstance(item, ContentBlockStop)

        msg = collector.result()
        assert msg.content[0].text == "hi"

    def test_beta_tool_use(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_beta_message_start("msg_beta_002"))

        item = collector.process(BetaRawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": BetaToolUseBlock(type="tool_use", id="t_beta", name="search", input={}),
        }))
        assert isinstance(item, ToolUse)

    def test_beta_thinking(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_beta_message_start("msg_beta_003"))

        item = collector.process(BetaRawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": BetaThinkingBlock(type="thinking", thinking="", signature=""),
        }))
        assert isinstance(item, Thinking)

        item = collector.process(BetaRawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": BetaThinkingDelta(type="thinking_delta", thinking="reasoning..."),
        }))
        assert isinstance(item, TimbalThinkingDelta)

    def test_beta_message_delta_stop_reason(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_beta_message_start("msg_beta_004"))
        collector.process(BetaRawContentBlockStartEvent(**{
            "type": "content_block_start",
            "index": 0,
            "content_block": BetaTextBlock(type="text", text=""),
        }))
        collector.process(BetaRawContentBlockDeltaEvent(**{
            "type": "content_block_delta",
            "index": 0,
            "delta": BetaTextDelta(type="text_delta", text="ok"),
        }))

        collector.process(BetaRawMessageDeltaEvent(**{
            "type": "message_delta",
            "delta": {"type": "message_delta", "stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        }))
        assert collector._stop_reason == "end_turn"

    def test_beta_message_stop_returns_none(self):
        _make_context()
        collector = _make_collector()
        collector.process(_make_beta_message_start("msg_beta_005"))
        result = collector.process(BetaRawMessageStopEvent(**{"type": "message_stop"}))
        assert result is None


class TestAnthropicCollectorContentBlockStop:
    def test_stop_for_unknown_block_returns_none(self):
        """ContentBlockStop for a block not in content_blocks returns None."""
        _make_context()
        collector = _make_collector()
        collector.process(_make_message_start())
        item = collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": 99}))
        assert item is None
