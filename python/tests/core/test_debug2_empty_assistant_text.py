"""Regression tests for Benito DEBUG2 (empty assistant text bricking Anthropic memory).

Incident: assistant turns with thinking + ``TextContent(text="")`` caused
``messages: text content blocks must be non-empty`` on the next LLM call.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageStartEvent,
    TextBlock,
    ThinkingBlock,
    ThinkingDelta,
)
from timbal import Agent
from timbal.collectors.impl.anthropic import AnthropicCollector
from timbal.core.llm_router import _llm_router
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.span import Span
from timbal.types.content import TextContent, ThinkingContent
from timbal.types.message import Message


def _debug2_poison_assistant_message() -> Message:
    return Message(
        role="assistant",
        content=[
            ThinkingContent(
                thinking="motivo_rechazo_clave '#' = Sin asignar → ofertas pendientes en Vallès Oriental",
                signature="sig_debug2",
            ),
            TextContent(text=""),
        ],
        stop_reason="end_turn",
    )


def _debug2_memory_after_partial_sql_turn() -> list[Message]:
    return [
        Message(role="user", content=[TextContent(text="Hola")]),
        Message(role="assistant", content=[TextContent(text="Hola, ¿en qué puedo ayudarte?")]),
        Message(
            role="user",
            content=[
                TextContent(text="Ruta comercial Vallès Oriental: municipios por actividad y presupuestos pendientes")
            ],
        ),
        _debug2_poison_assistant_message(),
        Message(
            role="user",
            content=[
                TextContent(text="Dels municipis del valles oriental diguem quins sin els que tenem temes pendents")
            ],
        ),
    ]


def _anthropic_messages_have_empty_text_block(messages: list[dict]) -> bool:
    for msg in messages:
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text") == "":
                return True
    return False


def _benito_style_only_select_allowed(sql: str) -> str | None:
    """Replicates Benito execute_sql rejection described in DEBUG2 (not in timbal core)."""
    stripped = sql.lstrip()
    if stripped.startswith("--"):
        return "only_select_allowed"
    upper = stripped.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return "only_select_allowed"
    return None


async def _empty_gen():
    return
    yield


async def _empty_async_stream():
    return
    yield


@pytest.fixture(autouse=True)
def _clean_context():
    from timbal.state import _call_id, _run_context_var

    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    yield
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)


def _make_collector_context() -> None:
    ctx = RunContext(tracing_provider=None)
    call_id = "debug2_call"
    span = Span(path="test", call_id=call_id, parent_call_id=None, t0=int(time.time() * 1000))
    ctx._trace[call_id] = span
    set_run_context(ctx)
    set_call_id(call_id)


def _make_message_start(msg_id: str = "msg_debug2") -> RawMessageStartEvent:
    return RawMessageStartEvent(
        **{
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-6",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        }
    )


class TestDebug2AnthropicCollectorFix:
    def test_empty_text_block_start_stop_omitted_from_result(self):
        _make_collector_context()
        collector = AnthropicCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(_make_message_start())
        collector.process(
            RawContentBlockStartEvent(
                **{
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": TextBlock(type="text", text=""),
                }
            )
        )
        collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": 0}))
        assert not any(isinstance(c, TextContent) for c in collector.result().content)

    def test_thinking_without_visible_text_keeps_thinking_only(self):
        _make_collector_context()
        collector = AnthropicCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(_make_message_start())
        collector.process(
            RawContentBlockStartEvent(
                **{
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": ThinkingBlock(type="thinking", thinking="", signature=""),
                }
            )
        )
        collector.process(
            RawContentBlockDeltaEvent(
                **{
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": ThinkingDelta(type="thinking_delta", thinking="Analyzing # Sin asignar offers..."),
                }
            )
        )
        collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": 0}))
        collector.process(
            RawContentBlockStartEvent(
                **{
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": TextBlock(type="text", text=""),
                }
            )
        )
        collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": 1}))
        msg = collector.result()
        assert any(isinstance(c, ThinkingContent) for c in msg.content)
        assert not any(isinstance(c, TextContent) for c in msg.content)

    def test_text_on_block_start_seeded_without_deltas(self):
        _make_collector_context()
        collector = AnthropicCollector(async_gen=_empty_gen(), start=time.perf_counter())
        collector.process(_make_message_start())
        collector.process(
            RawContentBlockStartEvent(
                **{
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": TextBlock(type="text", text="Visible on start only"),
                }
            )
        )
        collector.process(RawContentBlockStopEvent(**{"type": "content_block_stop", "index": 0}))
        text_blocks = [c for c in collector.result().content if isinstance(c, TextContent)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Visible on start only"


class TestDebug2MemorySerializationFix:
    def test_to_anthropic_input_omits_empty_text_block(self):
        payload = _debug2_poison_assistant_message().to_anthropic_input()
        assert {"type": "text", "text": ""} not in payload["content"]
        assert any(b.get("type") == "thinking" for b in payload["content"])

    def test_full_debug2_memory_payload_has_no_empty_text(self):
        anthropic_messages = [m.to_anthropic_input() for m in _debug2_memory_after_partial_sql_turn()]
        assert not _anthropic_messages_have_empty_text_block(anthropic_messages)

    def test_without_empty_text_blocks_strips_poison(self):
        cleaned = _debug2_poison_assistant_message().without_empty_text_blocks()
        assert cleaned is not None
        assert not any(isinstance(c, TextContent) for c in cleaned.content)


class TestDebug2LlmRouterFix:
    @pytest.mark.asyncio
    async def test_anthropic_create_never_receives_empty_text(self):
        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        set_run_context(RunContext(tracing_provider=None))
        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="anthropic/claude-sonnet-4-6",
                        max_tokens=256,
                        messages=_debug2_memory_after_partial_sql_turn(),
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass
        assert not _anthropic_messages_have_empty_text_block(captured["messages"])


class TestDebug2AgentSessionFix:
    @pytest.mark.asyncio
    async def test_chained_turn_succeeds_after_poison_stripped_from_memory(self):
        agent1 = Agent(
            name="Benito_AI",
            model=TestModel(responses=[_debug2_poison_assistant_message()]),
            tools=[],
        )
        turn1 = await agent1(prompt="Ruta Vallès Oriental presupuestos pendientes").collect()
        assert turn1.status.code == "success"

        captured: dict = {}

        async def rejecting_create(**kwargs):
            captured.update(kwargs)
            if _anthropic_messages_have_empty_text_block(kwargs.get("messages", [])):
                raise RuntimeError("messages: text content blocks must be non-empty")
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = rejecting_create
        agent2 = Agent(
            name="Benito_AI",
            model="anthropic/claude-sonnet-4-6",
            max_tokens=1024,
            tools=[],
        )
        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
                turn2 = await agent2(
                    prompt="Dels municipis del valles oriental diguem quins sin els que tenem temes pendents",
                    parent_id=turn1.run_id,
                ).collect()

        assert "text content blocks must be non-empty" not in (turn2.error or {}).get("message", "")
        assert not _anthropic_messages_have_empty_text_block(captured.get("messages", []))


class TestDebug2SqlCommentReplication:
    @pytest.mark.parametrize(
        "sql",
        [
            "-- Activitat per municipi\nSELECT 1",
            "--\nSELECT cl.poblacion FROM clientes cl",
        ],
    )
    def test_leading_line_comment_rejected_as_only_select_allowed(self, sql: str) -> None:
        assert _benito_style_only_select_allowed(sql) == "only_select_allowed"

    def test_same_query_without_comment_is_allowed(self) -> None:
        sql = "SELECT cl.poblacion FROM clientes cl WHERE cl.region = 'Vallès Oriental'"
        assert _benito_style_only_select_allowed(sql) is None
