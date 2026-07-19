"""Unit tests for newly registered frontier models."""

import os
from unittest.mock import MagicMock, patch

import pytest

from timbal.core.models import Model, get_context_window
from timbal.state import set_run_context
from timbal.state.context import RunContext

NEW_MODEL_IDS = [
    "xai/grok-4.5",
    "fireworks/accounts/fireworks/models/minimax-m2p7",
    "fireworks/accounts/fireworks/models/qwen3p7-plus",
    "fireworks/accounts/fireworks/models/deepseek-v4-flash",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "fireworks/accounts/fireworks/models/glm-5p2",
    "moonshot/kimi-k3",
    "moonshot/kimi-k2.6",
    "moonshot/kimi-k2.5",
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


class TestXaiRouterDispatch:
    def _make_mock_client_and_capturer(self):
        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.responses.create = fake_create
        return mock_client, captured_kwargs

    @pytest.mark.asyncio
    async def test_grok_45_uses_responses_path(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"XAI_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "responses"):
                    try:
                        async for _ in _llm_router(
                            model="xai/grok-4.5",
                            max_tokens=32,
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert captured_kwargs.get("model") == "grok-4.5"
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
            ("fireworks/accounts/fireworks/models/minimax-m2p7", "accounts/fireworks/models/minimax-m2p7"),
            ("fireworks/accounts/fireworks/models/qwen3p7-plus", "accounts/fireworks/models/qwen3p7-plus"),
            ("fireworks/accounts/fireworks/models/deepseek-v4-flash", "accounts/fireworks/models/deepseek-v4-flash"),
        ],
    )
    @pytest.mark.asyncio
    async def test_fireworks_models_use_chat_completions(self, model_id: str, api_name: str):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
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
            ("moonshot/kimi-k2.5", "kimi-k2.5"),
            ("moonshot/kimi-k2.7-code", "kimi-k2.7-code"),
            ("moonshot/kimi-k2.7-code-highspeed", "kimi-k2.7-code-highspeed"),
        ],
    )
    @pytest.mark.asyncio
    async def test_moonshot_models_use_chat_completions(self, model_id: str, api_name: str):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
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
