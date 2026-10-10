"""``timbal/auto``: platform-served provider, request metadata, served-model relabel."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from timbal.core.llm import _PROVIDERS, _resolve_client
from timbal.core.llm.auto import auto_metadata
from timbal.core.llm.chat_completions import prepare_chat_completions_request
from timbal.core.llm.router import _llm_router
from timbal.errors import APIKeyNotFoundError
from timbal.state import set_run_context
from timbal.state.context import RunContext
from timbal.types.content import FileContent, TextContent
from timbal.types.file import File
from timbal.types.message import Message


async def _empty_async_stream():
    return
    yield


async def _one_chunk_stream():
    # `_retry_on_error` treats an empty stream as a failure; one chunk is a success.
    yield MagicMock()


@pytest.fixture(autouse=True)
def clean_context():
    # The billing id is reset too: the relabel test sets it, and a stale one
    # leaks into later collector tests (they prefer it over the API model).
    from timbal.state import _billing_id, _call_id, _run_context_var

    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    token_bid = _billing_id.set(None)
    yield
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)
    _billing_id.reset(token_bid)


def _platform_config(org_id="org_42", app_id="app_7"):
    from timbal.state.config import PlatformConfig

    pc = MagicMock(spec=PlatformConfig)
    pc.host = "api.timbal.ai"
    pc.subject = MagicMock()
    pc.subject.org_id = org_id
    pc.subject.app_id = app_id
    pc.subject.project_id = None
    pc.auth = MagicMock()
    pc.auth.header_value = "Bearer platform_token"
    return pc


def _user_with_files(tmp_path, text, *names):
    content = [TextContent(text=text)]
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"x")
        content.append(FileContent(file=File.validate(str(path))))
    return Message(role="user", content=content)


class TestProviderConfig:
    def test_timbal_provider_is_platform_only_chat_completions(self):
        cfg = _PROVIDERS["timbal"]
        assert cfg.platform_only is True
        assert cfg.client_type == "openai"
        assert cfg.proxy_name == "openai-completions"
        assert cfg.proxy_suffix == "/v1"
        assert cfg.supports_platform_proxy is True

    def test_other_providers_unchanged(self):
        assert _PROVIDERS["openai"].platform_only is False
        assert _PROVIDERS["anthropic"].platform_only is False


class TestResolveClient:
    def test_timbal_api_key_in_env_is_not_a_vendor_key(self):
        """``TIMBAL_API_KEY`` is the provider's env_key, but it must never be
        sent to api.openai.com as if it were an OpenAI key."""
        ctx = RunContext(tracing_provider=None, platform_config=None)
        with patch.dict(os.environ, {"TIMBAL_API_KEY": "tk_test"}):
            with pytest.raises(APIKeyNotFoundError, match="served by the Timbal platform"):
                _resolve_client("timbal", _PROVIDERS["timbal"], None, None, ctx)

    def test_resolves_to_the_chat_completions_proxy(self):
        ctx = RunContext(tracing_provider=None, platform_config=_platform_config())
        with patch.dict(os.environ, {"TIMBAL_API_KEY": "tk_test", "OPENAI_API_KEY": "sk_vendor"}):
            client, base_url = _resolve_client("timbal", _PROVIDERS["timbal"], None, None, ctx)
        assert base_url == "https://api.timbal.ai/orgs/org_42/proxies/openai-completions/v1"
        assert str(client.base_url).startswith("https://api.timbal.ai/orgs/org_42/proxies/openai-completions/v1")
        assert client.api_key == "Bearer platform_token"

    def test_explicit_credentials_still_win(self):
        ctx = RunContext(tracing_provider=None, platform_config=None)
        client, base_url = _resolve_client(
            "timbal", _PROVIDERS["timbal"], "k", "https://llm.example.com/v1", ctx
        )
        assert base_url == "https://llm.example.com/v1"
        assert client.api_key == "k"


class TestAutoMetadata:
    def test_attachments_detected_on_last_user_message_only(self, tmp_path):
        messages = [
            _user_with_files(tmp_path, "first", "old.pdf"),
            Message(role="assistant", content=[TextContent(text="ok")]),
            _user_with_files(tmp_path, "now this", "deck.pptx", "costes.xlsx"),
        ]
        assert auto_metadata(messages) == {"timbal_attachments": "deck.pptx,costes.xlsx"}

    def test_text_only_turn_yields_nothing(self):
        messages = [Message(role="user", content=[TextContent(text="hola")])]
        assert auto_metadata(messages) == {}
        assert auto_metadata([]) == {}


class TestChatCompletionsRequest:
    @staticmethod
    def _kwargs_for(provider, messages, provider_params=None):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_empty_async_stream())
        create_stream, _ = prepare_chat_completions_request(
            provider=provider,
            config=_PROVIDERS[provider],
            client=client,
            model_name="auto" if provider == "timbal" else "gpt-6-luna",
            request_headers={},
            system_prompt=None,
            messages=messages,
            tools=None,
            max_tokens=None,
            temperature=None,
            output_model=None,
            provider_params=provider_params or {},
        )
        return create_stream, client

    async def test_metadata_injected_for_timbal_only(self, tmp_path):
        messages = [_user_with_files(tmp_path, "què hi veus?", "foto.png")]
        create_stream, client = self._kwargs_for("timbal", messages, {"metadata": {"trace": "t1"}})
        async for _ in create_stream():
            pass
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "auto"
        # Caller's own metadata is kept; the routing key is added next to it.
        assert kwargs["metadata"] == {"trace": "t1", "timbal_attachments": "foto.png"}

        create_stream, client = self._kwargs_for("openai", messages)
        async for _ in create_stream():
            pass
        assert "metadata" not in client.chat.completions.create.call_args.kwargs

    async def test_no_metadata_key_when_nothing_to_declare(self):
        messages = [Message(role="user", content=[TextContent(text="hola")])]
        create_stream, client = self._kwargs_for("timbal", messages)
        async for _ in create_stream():
            pass
        assert "metadata" not in client.chat.completions.create.call_args.kwargs


class TestServedModelRelabel:
    def test_served_model_id_infers_provider(self):
        from timbal.core.llm.auto import served_model_id

        assert served_model_id("claude-haiku-5-5") == "anthropic/claude-haiku-5-5"
        assert served_model_id("gpt-6.1-sol") == "openai/gpt-6.1-sol"
        assert served_model_id("gemini-3.1-flash-lite") == "google/gemini-3.1-flash-lite"
        assert served_model_id("openai/gpt-6-luna") == "openai/gpt-6-luna"
        assert served_model_id("mystery-7b") == "timbal/mystery-7b"

    def test_collector_relabels_billing_and_span_once(self):
        """Usage and span metadata must name the model that served the turn,
        not `timbal/auto`; the platform proxy already bills by served model,
        this keeps SDK traces and cost estimates consistent with it."""
        import time

        from openai.types.chat import ChatCompletionChunk
        from openai.types.completion_usage import CompletionUsage
        from timbal.collectors.impl.openai import ChatCompletionCollector
        from timbal.state import get_billing_id, set_billing_id, set_call_id
        from timbal.state.tracing.span import Span

        ctx = RunContext(tracing_provider=None, platform_config=None)
        set_run_context(ctx)
        span = Span(path="assistant.llm", call_id="call_1", parent_call_id=None, t0=0, t1=None)
        span.metadata.update(model_provider="timbal", model_name="auto")
        ctx._trace["call_1"] = span
        set_call_id("call_1")
        set_billing_id("timbal/auto")

        async def _gen():
            return
            yield

        collector = ChatCompletionCollector(async_gen=_gen(), start=time.perf_counter())
        chunk = ChatCompletionChunk(
            id="c1", choices=[], created=0, model="claude-haiku-5-5", object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=51, completion_tokens=166, total_tokens=217),
        )
        collector.process(chunk)
        assert get_billing_id() == "anthropic/claude-haiku-5-5"
        assert span.metadata["model_provider"] == "anthropic"
        assert span.metadata["model_name"] == "claude-haiku-5-5"
        # Second chunk: already relabelled, nothing changes even if it echoes "auto".
        collector.process(ChatCompletionChunk(id="c2", choices=[], created=0, model="auto", object="chat.completion.chunk"))
        assert get_billing_id() == "anthropic/claude-haiku-5-5"

        # Not an auto call → untouched.
        set_billing_id("openai/gpt-6-luna")
        collector.process(chunk)
        assert get_billing_id() == "openai/gpt-6-luna"


class TestRouterDispatch:
    async def test_timbal_auto_goes_to_chat_completions_with_auto_model(self, tmp_path):
        """End to end through ``_llm_router``: ``timbal/auto`` is dispatched to the
        chat-completions adapter (never Responses, whatever ``TIMBAL_OPENAI_API`` is),
        with ``model="auto"``, the routing metadata and the platform headers."""
        set_run_context(RunContext(tracing_provider=None, platform_config=_platform_config()))
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=lambda **_: _one_chunk_stream())
        client.responses.create = AsyncMock(side_effect=AssertionError("Responses API must not be used for timbal/auto"))

        messages = [_user_with_files(tmp_path, "què hi veus?", "foto.png")]
        with patch("timbal.core.llm.router._resolve_client", return_value=(client, "https://api.timbal.ai/orgs/org_42/proxies/openai-completions/v1")):
            async for _ in _llm_router(model="timbal/auto", messages=messages):
                pass

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "auto"
        assert kwargs["metadata"] == {"timbal_attachments": "foto.png"}
        headers = kwargs["extra_headers"]
        assert headers["x-timbal-app-id"] == "app_7"
        assert "x-timbal-run-id" in headers

    @pytest.mark.parametrize("model_id", ["timbal/auto-cost", "timbal/auto-intelligence", "timbal/auto-balanced"])
    async def test_profiles_pass_through_as_bare_model_names(self, model_id):
        """Profiles are plain model strings; the proxy parses the suffix."""
        from timbal.core.models import Model

        set_run_context(RunContext(tracing_provider=None, platform_config=_platform_config()))
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=lambda **_: _one_chunk_stream())
        messages = [Message(role="user", content=[TextContent(text="hola")])]
        with patch("timbal.core.llm.router._resolve_client", return_value=(client, "https://api.timbal.ai/orgs/org_42/proxies/openai-completions/v1")):
            async for _ in _llm_router(model=model_id, messages=messages):
                pass
        assert client.chat.completions.create.call_args.kwargs["model"] == model_id.split("/", 1)[1]
        if model_id != "timbal/auto-balanced":  # alias is accepted by the proxy but not listed
            assert model_id in Model.__args__


class TestMemoryCompaction:
    """``timbal/auto*`` needs a known context window: with an unknown one the Agent
    compacts on every run (its "safe fallback") and skips attachment bounding."""

    @pytest.mark.parametrize(
        ("model_id", "window"),
        [("timbal/auto", 500_000), ("timbal/auto-cost", 500_000), ("timbal/auto-intelligence", 1_000_000)],
    )
    def test_context_window_is_known(self, model_id, window):
        from timbal.core.models import get_context_window

        assert get_context_window(model_id) == window

    @staticmethod
    def _span(memory):
        class _FakeSpan:
            def __init__(self) -> None:
                self.input = {}
                self.memory = memory
                self.metadata = {}

        return _FakeSpan()

    @staticmethod
    def _agent(calls):
        from timbal.core.agent import Agent

        def compactor(memory):
            calls.append(len(memory))
            return memory

        return Agent(name="assistant", model="timbal/auto", memory_compaction=compactor, memory_compaction_ratio=0.75)

    async def test_short_conversation_does_not_compact(self):
        calls = []
        agent = self._agent(calls)
        memory = [Message(role="user", content=[TextContent(text="hola")])]
        await agent._maybe_compact_memory(self._span(memory), measurement=None)
        assert calls == []

    async def test_conversation_past_the_ratio_compacts(self):
        calls = []
        agent = self._agent(calls)
        # ~400K tokens by content estimate (1 token ≈ 4 chars) ≥ 0.75 × 500K.
        memory = [Message(role="user", content=[TextContent(text="x" * 1_600_000)])]
        await agent._maybe_compact_memory(self._span(memory), measurement=None)
        assert calls == [1]

    async def test_measured_usage_of_the_served_model_drives_compaction(self):
        """Usage arrives relabelled to the model that served the turn; its token
        units still measure the context against the ``timbal/auto`` window."""
        calls = []
        agent = self._agent(calls)
        memory = [Message(role="user", content=[TextContent(text="hola")])]

        near_full = {"usage": {"anthropic/claude-haiku-5-5:input_tokens": 400_000}, "messages": 1}
        await agent._maybe_compact_memory(self._span(memory), measurement=near_full)
        assert calls == [1]

        calls.clear()
        light = {"usage": {"anthropic/claude-haiku-5-5:input_tokens": 10_000}, "messages": 1}
        await agent._maybe_compact_memory(self._span(memory), measurement=light)
        assert calls == []
