"""Unit tests for newly registered frontier models."""

import os
from unittest.mock import MagicMock, patch

import pytest
from timbal.core.models import Model, get_context_window
from timbal.state import set_run_context
from timbal.state.context import RunContext

NEW_MODEL_IDS = [
    "google/gemini-4-argon",
    "anthropic/claude-haiku-5-5",
    "anthropic/claude-sonnet-5-5",
    "anthropic/claude-opus-5-5",
    "xai/grok-4.7",
    "xai/grok-4.6",
    "xai/grok-4.5",
    "xai/grok-4.3",
    "fireworks/accounts/fireworks/models/minimax-m3",
    "togetherai/Qwen/Qwen3.8-2.4T-A95B",
    "fireworks/accounts/fireworks/models/deepseek-v4-flash-0731",
    "fireworks/accounts/fireworks/models/kimi-k2p7-code",
    "openai/gpt-6-astra",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-sol",
    "openai/gpt-6-luna",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.5-pro",
    "openai/gpt-5.4-pro",
    "fireworks/accounts/fireworks/models/glm-5p2",
    "groq/qwen/qwen3.8-27b",
    "togetherai/moonshotai/Kimi-K2.7-Code",
    "togetherai/zai-org/GLM-5.2",
    "togetherai/MiniMaxAI/MiniMax-M3",
    "moonshot/kimi-k3",
    "moonshot/kimi-k2.6",
    "moonshot/kimi-k2.7-code",
    "moonshot/kimi-k2.7-code-highspeed",
]


@pytest.fixture(autouse=True)
def clean_context():
    from timbal.state import _call_id, _run_context_var

    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    yield
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)


def _make_run_context():
    ctx = RunContext(tracing_provider=None, platform_config=None)
    set_run_context(ctx)
    return ctx


class TestFrontierModelRegistry:
    def test_new_models_are_in_model_literal(self):
        hints = Model.__args__  # type: ignore[attr-defined]
        for model_id in NEW_MODEL_IDS:
            assert model_id in hints, f"{model_id} missing from Model Literal"

    @pytest.mark.parametrize("model_id", NEW_MODEL_IDS)
    def test_context_window_registered(self, model_id: str):
        assert get_context_window(model_id) is not None

    @pytest.mark.parametrize("model_id", NEW_MODEL_IDS)
    def test_provider_prefix_matches_yaml_provider(self, model_id: str):
        from timbal.codegen.model_discovery import get_models

        by_id = {m["id"]: m for m in get_models()}
        assert model_id in by_id
        provider, _ = model_id.split("/", 1)
        assert by_id[model_id]["provider"] == provider


async def _empty_async_stream():
    return
    yield


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
async def test_argon_uses_google_chat_route(effort):
    from pydantic import BaseModel
    from timbal.core.llm import _llm_router
    from timbal.core.llm.registry import _PROVIDERS
    from timbal.types.content import TextContent
    from timbal.types.message import Message

    class Answer(BaseModel):
        answer: str

    captured = {}

    async def stream():
        yield {"choices": []}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return stream()

    client = MagicMock()
    client.chat.completions.create = fake_create
    _make_run_context()
    with patch("timbal.core.llm.clients._get_client", return_value=client) as get_client:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            async for _ in _llm_router(
                model="google/gemini-4-argon",
                messages=[Message(role="user", content=[TextContent(text="Reply with ok")])],
                max_tokens=32768,
                output_model=Answer,
                provider_params={"reasoning_effort": effort},
            ):
                pass

    assert captured["model"] == "gemini-4-argon"
    assert captured["max_completion_tokens"] == 32768
    assert captured["reasoning_effort"] == effort
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert "temperature" not in captured
    assert get_context_window("google/gemini-4-argon") == 1048576
    assert _PROVIDERS["google"].default_base_url in str(get_client.call_args)


class TestXaiRouterDispatch:
    def _make_mock_client_and_capturer(self):
        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.responses.create = fake_create
        return mock_client, captured_kwargs

    @pytest.mark.parametrize(
        "model_id,api_name",
        [
            ("xai/grok-4.7", "grok-4.7"),
            ("xai/grok-4.6", "grok-4.6"),
            ("xai/grok-4.5", "grok-4.5"),
            ("xai/grok-4.3", "grok-4.3"),
        ],
    )
    @pytest.mark.asyncio
    async def test_grok_models_use_responses_path(self, model_id: str, api_name: str):
        from timbal.core.llm import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm.clients._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"XAI_API_KEY": "key"}):
                with patch("timbal.core.llm.router.TIMBAL_OPENAI_API", "responses"):
                    try:
                        async for _ in _llm_router(
                            model=model_id,
                            max_tokens=32,
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert captured_kwargs.get("model") == api_name
        assert captured_kwargs.get("max_output_tokens") == 32


class TestFireworksRouterDispatch:
    def _make_mock_client_and_capturer(self):
        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        return mock_client, captured_kwargs

    @pytest.mark.parametrize(
        "model_id,api_name",
        [
            ("fireworks/accounts/fireworks/models/minimax-m3", "accounts/fireworks/models/minimax-m3"),
            ("fireworks/accounts/fireworks/models/qwen3p8-max", "accounts/fireworks/models/qwen3p8-max"),
            ("fireworks/accounts/fireworks/models/deepseek-v4-flash-0731", "accounts/fireworks/models/deepseek-v4-flash-0731"),
        ],
    )
    @pytest.mark.asyncio
    async def test_fireworks_models_use_chat_completions(self, model_id: str, api_name: str):
        from timbal.core.llm import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm.clients._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"FIREWORKS_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(model=model_id, max_tokens=16):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert captured_kwargs.get("model") == api_name
        assert captured_kwargs.get("max_completion_tokens") == 16


class TestMoonshotRouterDispatch:
    def _make_mock_client_and_capturer(self):
        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        return mock_client, captured_kwargs

    @pytest.mark.parametrize(
        "model_id,api_name",
        [
            ("moonshot/kimi-k3", "kimi-k3"),
            ("moonshot/kimi-k2.6", "kimi-k2.6"),
            ("moonshot/kimi-k2.7-code", "kimi-k2.7-code"),
            ("moonshot/kimi-k2.7-code-highspeed", "kimi-k2.7-code-highspeed"),
        ],
    )
    @pytest.mark.asyncio
    async def test_moonshot_models_use_chat_completions(self, model_id: str, api_name: str):
        from timbal.core.llm import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm.clients._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"MOONSHOT_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model=model_id,
                        max_tokens=16,
                        provider_params={"reasoning_effort": "max"},
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert captured_kwargs.get("model") == api_name
        assert captured_kwargs.get("max_completion_tokens") == 16
        assert captured_kwargs.get("reasoning_effort") == "max"


class TestHaiku55RouterDispatch:
    @pytest.mark.asyncio
    async def test_haiku_55_uses_messages_with_effort_and_automatic_caching(self):
        from anthropic.types import RawMessageStopEvent
        from timbal.core.llm import _llm_router

        _make_run_context()
        captured = {}

        async def response_stream():
            yield RawMessageStopEvent(type="message_stop")

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return response_stream()

        client = MagicMock()
        client.messages.create = fake_create
        with patch("timbal.core.llm.clients._get_client", return_value=client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                async for _ in _llm_router(
                    model="anthropic/claude-haiku-5-5",
                    max_tokens=4096,
                    provider_params={"output_config": {"effort": "medium"}},
                ):
                    pass

        assert captured["model"] == "claude-haiku-5-5"
        assert captured["max_tokens"] == 4096
        assert captured["cache_control"] == {"type": "ephemeral"}
        assert captured["output_config"] == {"effort": "medium"}
        assert "temperature" not in captured
        assert get_context_window("anthropic/claude-haiku-5-5") == 1_000_000
