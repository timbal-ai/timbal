"""Prompt-cache request shape for OpenAI, Groq, and Anthropic, including Bedrock."""

import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails
from timbal.collectors.impl.openai import ChatCompletionCollector
from timbal.core.llm.chat_completions import prepare_chat_completions_request
from timbal.core.llm.messages import prepare_messages_request
from timbal.core.llm.prompt_cache import (
    ANTHROPIC_MIN_CACHE_TOKENS,
    OPENAI_MIN_CACHE_TOKENS,
    PROMPT_CACHE_KEY_HEADER,
    SystemPrompt,
    bedrock_converse_system,
    bind_prompt_cache_key,
    groq_prompt_cache_model,
    normalize_cache_key,
    reset_prompt_cache_key,
    supports_explicit_prompt_cache_breakpoint,
)
from timbal.core.llm.registry import _PROVIDERS
from timbal.core.llm.responses import prepare_responses_request
from timbal.state import get_billing_id, set_billing_id, set_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.span import Span
from timbal.voice.prompt import compose_voice_system_prompt


@pytest.fixture(autouse=True)
def clean_context():
    """Drop any run context these tests install so later tests start clean."""
    from timbal.state import _call_id, _run_context_var

    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    cache_token = bind_prompt_cache_key(None)
    yield
    reset_prompt_cache_key(cache_token)
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)


def _make_run_context():
    ctx = RunContext(tracing_provider=None)
    set_run_context(ctx)
    return ctx


async def _empty_async_stream():
    return
    yield


def _capture_create():
    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _empty_async_stream()

    return captured, fake_create


async def _drive(create_stream):
    async for _ in create_stream():
        pass


class TestCacheKeyAndMinima:
    def test_documented_minimums(self):
        assert OPENAI_MIN_CACHE_TOKENS == 1024
        assert ANTHROPIC_MIN_CACHE_TOKENS["claude-haiku-5-5"] == 512
        assert ANTHROPIC_MIN_CACHE_TOKENS["claude-haiku-4-5"] == 4096

    def test_explicit_breakpoints_start_at_gpt_5_6(self):
        assert not supports_explicit_prompt_cache_breakpoint("gpt-5.4-mini")
        assert not supports_explicit_prompt_cache_breakpoint("gpt-5.5")
        assert not supports_explicit_prompt_cache_breakpoint("gpt-4o")
        assert supports_explicit_prompt_cache_breakpoint("gpt-5.6")
        assert supports_explicit_prompt_cache_breakpoint("gpt-5.6-luna")
        assert supports_explicit_prompt_cache_breakpoint("gpt-6")
        assert supports_explicit_prompt_cache_breakpoint("gpt-6-luna")
        assert supports_explicit_prompt_cache_breakpoint("gpt-6.1-sol")

    def test_groq_cache_is_gpt_oss_only(self):
        assert groq_prompt_cache_model("openai/gpt-oss-20b")
        assert groq_prompt_cache_model("openai/gpt-oss-120b")
        assert not groq_prompt_cache_model("qwen/qwen3.8-27b")

    def test_long_cache_key_is_stable(self):
        raw = "k" * 300
        assert normalize_cache_key(raw) == normalize_cache_key(raw)
        assert len(normalize_cache_key(raw) or "") == 64


class TestVoicePromptOrder:
    def test_static_prefix_survives_a_new_clock(self):
        first = compose_voice_system_prompt(
            "Be brief.",
            call_context={"from": "+15551212", "account": "42"},
            now=datetime(2026, 11, 4, 9, 40, tzinfo=UTC),
        )
        second = compose_voice_system_prompt(
            "Be brief.",
            call_context={"account": "42", "from": "+15551212"},
            now=datetime(2026, 11, 4, 9, 41, tzinfo=UTC),
            greeting_note="You already opened this call.",
        )
        assert first.static == second.static == "Be brief."
        assert first.dynamic.startswith("Caller and customer:\naccount: 42\nfrom: +15551212")
        assert "Current date and time (UTC): 2026-11-04T09:40:00Z" in first.dynamic
        assert first.dynamic.index("from: +15551212") < first.dynamic.index("Current date and time")
        assert second.dynamic.endswith("You already opened this call.")
        # The clock changed; the caller block did not.
        assert first.dynamic.split("Current date and time")[0] == second.dynamic.split("Current date and time")[0]

    def test_user_dynamic_stays_after_call_facts(self):
        prompt = compose_voice_system_prompt(
            SystemPrompt(static="Rules.", dynamic="Plan: gold"),
            call_context={"from": "+1"},
            now=datetime(2026, 11, 4, 9, 40, tzinfo=UTC),
        )
        assert prompt.static == "Rules."
        assert prompt.dynamic.index("from: +1") < prompt.dynamic.index("Plan: gold")
        assert prompt.dynamic.index("Plan: gold") < prompt.dynamic.index("Current date and time")


class TestAnthropicAndBedrock:
    async def test_split_prompt_breakpoints_the_static_block_only(self):
        captured, fake_create = _capture_create()
        client = MagicMock()
        client.messages.create = fake_create
        create_stream, _ = prepare_messages_request(
            client=client,
            model_name="claude-haiku-5-5",
            request_headers={},
            system_prompt=SystemPrompt(static="Stable rules.", dynamic="Caller: +1"),
            messages=[],
            tools=None,
            max_tokens=100,
            temperature=None,
            output_model=None,
            provider_params={},
        )
        await _drive(create_stream)
        assert captured["cache_control"] == {"type": "ephemeral"}
        assert captured["system"] == [
            {"type": "text", "text": "Stable rules.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Caller: +1"},
        ]

    async def test_stable_string_stays_a_string(self):
        captured, fake_create = _capture_create()
        client = MagicMock()
        client.messages.create = fake_create
        create_stream, _ = prepare_messages_request(
            client=client,
            model_name="claude-haiku-4-5",
            request_headers={},
            system_prompt="Stable rules.",
            messages=[],
            tools=None,
            max_tokens=100,
            temperature=None,
            output_model=None,
            provider_params={},
        )
        await _drive(create_stream)
        assert captured["system"] == "Stable rules."
        assert captured["cache_control"] == {"type": "ephemeral"}

    async def test_opt_out_drops_breakpoints(self):
        captured, fake_create = _capture_create()
        client = MagicMock()
        client.messages.create = fake_create
        create_stream, _ = prepare_messages_request(
            client=client,
            model_name="claude-haiku-4-5",
            request_headers={},
            system_prompt=SystemPrompt(static="Stable rules.", dynamic="Caller: +1"),
            messages=[],
            tools=None,
            max_tokens=100,
            temperature=None,
            output_model=None,
            provider_params={"cache_control": None},
        )
        await _drive(create_stream)
        assert "cache_control" not in captured
        assert captured["system"] == "Stable rules.\n\nCaller: +1"

    async def test_ttl_override_is_the_block_breakpoint(self):
        captured, fake_create = _capture_create()
        client = MagicMock()
        client.messages.create = fake_create
        create_stream, _ = prepare_messages_request(
            client=client,
            model_name="claude-haiku-5-5",
            request_headers={},
            system_prompt=SystemPrompt(static="Stable rules.", dynamic="Caller: +1"),
            messages=[],
            tools=None,
            max_tokens=100,
            temperature=None,
            output_model=None,
            provider_params={"cache_control": {"type": "ephemeral", "ttl": "1h"}},
        )
        await _drive(create_stream)
        assert captured["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert captured["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_converse_cache_point_follows_the_static_text(self):
        blocks = bedrock_converse_system(SystemPrompt(static="Stable rules.", dynamic="Caller: +1"))
        assert blocks == [
            {"text": "Stable rules."},
            {"cachePoint": {"type": "default"}},
            {"text": "Caller: +1"},
        ]
        hour = bedrock_converse_system(
            SystemPrompt(static="Stable rules.", dynamic="Caller: +1"),
            ttl="1h",
        )
        assert hour[1] == {"cachePoint": {"type": "default", "ttl": "1h"}}
        stable = bedrock_converse_system("Stable rules.")
        assert stable == [
            {"text": "Stable rules."},
            {"cachePoint": {"type": "default"}},
        ]


class TestOpenAIAndGroqRequests:
    async def _prepare_chat(self, provider: str, model_name: str, system_prompt):
        captured, fake_create = _capture_create()
        client = MagicMock()
        client.chat.completions.create = fake_create
        create_stream, _ = prepare_chat_completions_request(
            provider=provider,
            config=_PROVIDERS[provider],
            client=client,
            model_name=model_name,
            request_headers={"x-timbal-run-id": "run"},
            system_prompt=system_prompt,
            messages=[],
            tools=None,
            max_tokens=None,
            temperature=None,
            output_model=None,
            provider_params={},
        )
        await _drive(create_stream)
        return captured

    async def _prepare_responses(self, model_name: str, system_prompt, provider_params=None):
        captured, fake_create = _capture_create()
        client = MagicMock()
        client.responses.create = fake_create
        create_stream, _ = prepare_responses_request(
            provider="openai",
            client=client,
            model_name=model_name,
            request_headers={},
            system_prompt=system_prompt,
            messages=[],
            tools=None,
            max_tokens=None,
            temperature=None,
            output_model=None,
            provider_params=provider_params or {},
        )
        await _drive(create_stream)
        return captured

    async def test_pre_5_6_keeps_a_stable_instructions_prefix(self):
        token = bind_prompt_cache_key("sess-1")
        try:
            captured = await self._prepare_responses(
                "gpt-5.4-mini",
                SystemPrompt(static="Stable rules.", dynamic="Caller: +1"),
            )
        finally:
            reset_prompt_cache_key(token)
        assert captured["instructions"] == "Stable rules.\n\nCaller: +1"
        assert captured["instructions"].startswith("Stable rules.")
        assert captured["prompt_cache_key"] == "sess-1"
        assert captured["input"] == []

    async def test_gpt_5_6_breakpoints_the_static_developer_item(self):
        captured = await self._prepare_responses(
            "gpt-5.6-luna",
            SystemPrompt(static="Stable rules.", dynamic="Caller: +1"),
        )
        assert "instructions" not in captured
        assert captured["input"][0] == {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "Stable rules.",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        }
        assert captured["input"][1]["role"] == "developer"
        assert captured["input"][1]["content"].startswith("Caller: +1")
        assert "prompt_cache_options" not in captured

    async def test_explicit_key_wins_and_empty_opts_out(self):
        token = bind_prompt_cache_key("sess-1")
        try:
            won = await self._prepare_responses("gpt-5.4-mini", "Hi", {"prompt_cache_key": "override"})
            opted = await self._prepare_responses("gpt-5.4-mini", "Hi", {"prompt_cache_key": ""})
        finally:
            reset_prompt_cache_key(token)
        assert won["prompt_cache_key"] == "override"
        assert "prompt_cache_key" not in opted

    async def test_groq_gets_the_prefix_and_no_session_key(self):
        token = bind_prompt_cache_key("sess-1")
        try:
            captured = await self._prepare_chat(
                "groq",
                "openai/gpt-oss-120b",
                SystemPrompt(static="Stable rules.", dynamic="Caller: +1"),
            )
        finally:
            reset_prompt_cache_key(token)
        assert captured["messages"][0] == {
            "role": "system",
            "content": "Stable rules.\n\nCaller: +1",
        }
        assert "prompt_cache_key" not in captured

    async def test_openai_chat_completions_sends_the_session_key(self):
        token = bind_prompt_cache_key("sess-1")
        try:
            captured = await self._prepare_chat("openai", "gpt-5.4-mini", "Stable rules.")
        finally:
            reset_prompt_cache_key(token)
        assert captured["messages"][0]["content"] == "Stable rules."
        assert captured["prompt_cache_key"] == "sess-1"


class TestRouterSessionHeader:
    @pytest.mark.asyncio
    async def test_header_rides_every_provider(self):
        from timbal.core.llm import _llm_router

        _make_run_context()
        captured, fake_create = _capture_create()
        client = MagicMock()
        client.chat.completions.create = fake_create
        token = bind_prompt_cache_key("sess-9")
        try:
            with (
                patch("timbal.core.llm.clients._get_client", return_value=client),
                patch.dict(os.environ, {"GROQ_API_KEY": "key"}),
                patch("timbal.core.llm.router.TIMBAL_OPENAI_API", "chat_completions"),
            ):
                try:
                    async for _ in _llm_router(model="groq/openai/gpt-oss-120b"):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass
        finally:
            reset_prompt_cache_key(token)
        assert captured["extra_headers"][PROMPT_CACHE_KEY_HEADER] == "sess-9"
        assert "prompt_cache_key" not in captured


class TestGroqCachedTokens:
    def test_cached_tokens_land_on_input_cached_tokens(self):
        import time

        ctx = _make_run_context()
        call_id = "test_call"
        ctx._trace[call_id] = Span(path="test", call_id=call_id, parent_call_id=None, t0=1)
        set_call_id(call_id)
        set_billing_id("groq/openai/gpt-oss-120b")
        assert get_billing_id() == "groq/openai/gpt-oss-120b"
        collector = ChatCompletionCollector(async_gen=_empty_async_stream(), start=time.perf_counter())
        collector._first_token = time.perf_counter()
        usage = CompletionUsage(
            completion_tokens=5,
            prompt_tokens=4641,
            total_tokens=4646,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=4608, audio_tokens=None),
            completion_tokens_details=CompletionTokensDetails(audio_tokens=None, reasoning_tokens=0),
        )
        chunk = ChatCompletionChunk(
            id="chatcmpl_groq",
            choices=[],
            created=int(time.time()),
            model="openai/gpt-oss-120b",
            object="chat.completion.chunk",
            usage=usage,
        )
        collector.process(chunk)
        collector.result()
        span = next(iter(ctx._trace.values()))
        assert span.usage["groq/openai/gpt-oss-120b:input_cached_tokens"] == 4608
        assert span.usage["groq/openai/gpt-oss-120b:input_text_tokens"] == 33
