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
from timbal.state import set_billing_id, set_call_id, set_run_context
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
    from timbal.state import _billing_id, _run_context_var, _call_id, set_call_id
    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    token_bid = _billing_id.set(None)
    yield
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)
    _billing_id.reset(token_bid)


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


def _usage_message_start(usage: dict, *, model: str = "claude-sonnet-5", msg_id: str = "msg_usage") -> RawMessageStartEvent:
    """message_start with a caller-supplied usage payload (shape copied from live API captures)."""
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
            "usage": usage,
        },
    })


def _usage_message_delta(usage: dict) -> RawMessageDeltaEvent:
    return RawMessageDeltaEvent(**{
        "type": "message_delta",
        "delta": {"type": "message_delta", "stop_reason": "end_turn", "stop_sequence": None},
        "usage": usage,
    })


class TestAnthropicCollectorUsageAccounting:
    """Billing units derived from message_start + message_delta.

    Payload shapes below are copied from live captures against api.anthropic.com
    (2026-09-06): the per-TTL cache breakdown and ``speed`` ship only on
    ``message_start``; ``message_delta`` repeats the aggregate input-side totals and
    carries the final ``output_tokens`` (+ ``output_tokens_details.thinking_tokens``,
    a subset of ``output_tokens``).
    """

    def _run(self, start_usage: dict, delta_usage: dict | None, *, billing_id: str = "anthropic/claude-sonnet-5"):
        ctx = _make_context()
        set_billing_id(billing_id)
        collector = _make_collector()
        collector.process(_usage_message_start(start_usage, model=billing_id.split("/", 1)[1]))
        if delta_usage is not None:
            collector.process(_usage_message_delta(delta_usage))
        collector.result()
        return dict(ctx._trace["test_call"].usage)

    def test_five_minute_cache_write_bills_per_ttl_unit(self):
        usage = self._run(
            {"input_tokens": 18, "output_tokens": 1, "cache_creation_input_tokens": 13315, "cache_read_input_tokens": 0,
             "cache_creation": {"ephemeral_5m_input_tokens": 13315, "ephemeral_1h_input_tokens": 0},
             "service_tier": "standard", "inference_geo": "not_available"},
            {"input_tokens": 18, "output_tokens": 4, "cache_creation_input_tokens": 13315, "cache_read_input_tokens": 0},
        )
        assert usage == {
            "anthropic/claude-sonnet-5:input_tokens": 18,
            "anthropic/claude-sonnet-5:ephemeral_5m_input_tokens": 13315,
            "anthropic/claude-sonnet-5:output_tokens": 4,
        }

    def test_one_hour_cache_write_bills_at_its_own_unit(self):
        """1h writes are 2x input (vs 1.25x for 5m); the aggregate unit would under-bill them."""
        usage = self._run(
            {"input_tokens": 9, "output_tokens": 1, "cache_creation_input_tokens": 13320, "cache_read_input_tokens": 0,
             "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 13320}},
            {"input_tokens": 9, "output_tokens": 5, "cache_creation_input_tokens": 13320, "cache_read_input_tokens": 0},
        )
        assert usage["anthropic/claude-sonnet-5:ephemeral_1h_input_tokens"] == 13320
        assert "anthropic/claude-sonnet-5:ephemeral_5m_input_tokens" not in usage
        assert "anthropic/claude-sonnet-5:cache_creation_input_tokens" not in usage

    def test_mixed_ttl_writes_split_without_double_billing(self):
        usage = self._run(
            {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 248, "cache_read_input_tokens": 1000,
             "cache_creation": {"ephemeral_5m_input_tokens": 148, "ephemeral_1h_input_tokens": 100}},
            {"input_tokens": 5, "output_tokens": 7, "cache_creation_input_tokens": 248, "cache_read_input_tokens": 1000},
        )
        assert usage["anthropic/claude-sonnet-5:ephemeral_5m_input_tokens"] == 148
        assert usage["anthropic/claude-sonnet-5:ephemeral_1h_input_tokens"] == 100
        assert usage["anthropic/claude-sonnet-5:cache_read_input_tokens"] == 1000
        assert "anthropic/claude-sonnet-5:cache_creation_input_tokens" not in usage

    def test_missing_breakdown_falls_back_to_aggregate_unit(self):
        usage = self._run(
            {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 0},
            {"input_tokens": 5, "output_tokens": 7, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 0},
        )
        assert usage["anthropic/claude-sonnet-5:cache_creation_input_tokens"] == 500
        assert not any("ephemeral" in k for k in usage)

    def test_breakdown_that_does_not_reconcile_falls_back_to_aggregate_unit(self):
        usage = self._run(
            {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 0,
             "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 100}},
            {"input_tokens": 5, "output_tokens": 7, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 0},
        )
        assert usage["anthropic/claude-sonnet-5:cache_creation_input_tokens"] == 500
        assert not any("ephemeral" in k for k in usage)

    def test_thinking_tokens_are_not_billed_on_top_of_output(self):
        """output_tokens_details.thinking_tokens ⊂ output_tokens (live: 524 output / 519 thinking / 4 visible)."""
        usage = self._run(
            {"input_tokens": 66, "output_tokens": 2, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}},
            {"input_tokens": 66, "output_tokens": 524, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "output_tokens_details": {"thinking_tokens": 519}},
        )
        assert usage == {
            "anthropic/claude-sonnet-5:input_tokens": 66,
            "anthropic/claude-sonnet-5:output_tokens": 524,
        }

    def test_delta_without_input_fields_falls_back_to_message_start(self):
        usage = self._run(
            {"input_tokens": 40, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 300},
            {"output_tokens": 12},
        )
        assert usage == {
            "anthropic/claude-sonnet-5:input_tokens": 40,
            "anthropic/claude-sonnet-5:cache_read_input_tokens": 300,
            "anthropic/claude-sonnet-5:output_tokens": 12,
        }

    def test_server_tool_requests_are_billed_unsuffixed(self):
        usage = self._run(
            {"input_tokens": 40, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            {"input_tokens": 40, "output_tokens": 12, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0}},
        )
        assert usage["anthropic/claude-sonnet-5:web_search_requests"] == 2
        assert "anthropic/claude-sonnet-5:web_fetch_requests" not in usage

    def test_fast_mode_suffixes_token_units_when_catalog_prices_it(self):
        """Opus 5 fast mode (2x): ``usage.speed == "fast"`` ships on message_start only."""
        usage = self._run(
            {"input_tokens": 11, "output_tokens": 1, "cache_creation_input_tokens": 200, "cache_read_input_tokens": 50,
             "cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 0},
             "service_tier": "standard", "inference_geo": "global", "speed": "fast"},
            {"input_tokens": 11, "output_tokens": 5, "cache_creation_input_tokens": 200, "cache_read_input_tokens": 50,
             "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0}},
            billing_id="anthropic/claude-opus-5",
        )
        assert usage == {
            "anthropic/claude-opus-5:input_tokens_fast": 11,
            "anthropic/claude-opus-5:cache_read_input_tokens_fast": 50,
            "anthropic/claude-opus-5:ephemeral_5m_input_tokens_fast": 200,
            "anthropic/claude-opus-5:output_tokens_fast": 5,
            "anthropic/claude-opus-5:web_search_requests": 1,
        }

    def test_fast_mode_is_ignored_for_models_without_a_fast_tier(self):
        usage = self._run(
            {"input_tokens": 11, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "speed": "fast"},
            {"input_tokens": 11, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            billing_id="anthropic/claude-haiku-4-5",
        )
        assert usage == {
            "anthropic/claude-haiku-4-5:input_tokens": 11,
            "anthropic/claude-haiku-4-5:output_tokens": 5,
        }

    def test_standard_speed_never_suffixes(self):
        usage = self._run(
            {"input_tokens": 11, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "speed": "standard"},
            {"input_tokens": 11, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            billing_id="anthropic/claude-opus-5",
        )
        assert set(usage) == {"anthropic/claude-opus-5:input_tokens", "anthropic/claude-opus-5:output_tokens"}

    def test_interrupted_stream_bills_prompt_from_message_start(self):
        """No message_delta (cancelled mid-stream): Anthropic still charges the prompt; output is unknown."""
        usage = self._run(
            {"input_tokens": 40, "output_tokens": 1, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 300,
             "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1000}},
            None,
        )
        assert usage == {
            "anthropic/claude-sonnet-5:input_tokens": 40,
            "anthropic/claude-sonnet-5:cache_read_input_tokens": 300,
            "anthropic/claude-sonnet-5:ephemeral_1h_input_tokens": 1000,
        }

    def test_usage_is_recorded_exactly_once(self):
        ctx = _make_context()
        set_billing_id("anthropic/claude-sonnet-5")
        collector = _make_collector()
        collector.process(_usage_message_start(
            {"input_tokens": 40, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ))
        collector.process(_usage_message_delta(
            {"input_tokens": 40, "output_tokens": 12, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ))
        collector.result()
        collector.result()
        assert ctx._trace["test_call"].usage == {
            "anthropic/claude-sonnet-5:input_tokens": 40,
            "anthropic/claude-sonnet-5:output_tokens": 12,
        }

    def test_malformed_undeclared_extras_are_ignored(self):
        """``speed`` / ``output_tokens_details`` are pydantic extras (undeclared by the SDK), so
        nothing validates them upstream; garbage there must not crash or skew billing."""
        usage = self._run(
            {"input_tokens": 40, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "cache_creation": None, "speed": 42},
            {"input_tokens": 40, "output_tokens": 12, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "output_tokens_details": {"thinking_tokens": "bad"}, "server_tool_use": None},
        )
        assert usage == {
            "anthropic/claude-sonnet-5:input_tokens": 40,
            "anthropic/claude-sonnet-5:output_tokens": 12,
        }
