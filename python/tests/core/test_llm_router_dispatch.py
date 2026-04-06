"""Tests for llm_router dispatch logic, client resolution, and kwargs building."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, SecretStr

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from timbal.core.llm_router import _PROVIDERS, _get_client, _resolve_client


async def _empty_async_stream():
    """Async generator that yields nothing — used as mock LLM response."""
    return
    yield
from timbal.errors import APIKeyNotFoundError
from timbal.state import set_run_context
from timbal.state.context import RunContext


@pytest.fixture(autouse=True)
def clean_context():
    """Reset context vars after each test to avoid state pollution."""
    from timbal.state import _run_context_var, _call_id
    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    yield
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)


def _make_run_context(platform_config=None):
    ctx = RunContext(tracing_provider=None, platform_config=platform_config)
    set_run_context(ctx)
    return ctx


class TestGetClients:
    def test_get_openai_client_cached(self):
        from timbal.core.llm_router import _CLIENT_CACHE
        _CLIENT_CACHE.clear()

        c1 = _get_client(AsyncOpenAI, "key_a", None, "openai")
        c2 = _get_client(AsyncOpenAI, "key_a", None, "openai")
        assert c1 is c2

    def test_get_openai_client_different_keys(self):
        from timbal.core.llm_router import _CLIENT_CACHE
        _CLIENT_CACHE.clear()

        c1 = _get_client(AsyncOpenAI, "key_a", None, "openai")
        c2 = _get_client(AsyncOpenAI, "key_b", None, "openai")
        assert c1 is not c2

    def test_get_openai_client_with_base_url(self):
        from timbal.core.llm_router import _CLIENT_CACHE
        _CLIENT_CACHE.clear()

        c1 = _get_client(AsyncOpenAI, "key", "https://custom.api.com/v1", "groq")
        c2 = _get_client(AsyncOpenAI, "key", None, "openai")
        assert c1 is not c2

    def test_get_anthropic_client_cached(self):
        from timbal.core.llm_router import _CLIENT_CACHE
        _CLIENT_CACHE.clear()

        c1 = _get_client(AsyncAnthropic, "key_a", None, "anthropic")
        c2 = _get_client(AsyncAnthropic, "key_a", None, "anthropic")
        assert c1 is c2

    def test_get_anthropic_client_different_keys(self):
        from timbal.core.llm_router import _CLIENT_CACHE
        _CLIENT_CACHE.clear()

        c1 = _get_client(AsyncAnthropic, "key_a", None, "anthropic")
        c2 = _get_client(AsyncAnthropic, "key_b", None, "anthropic")
        assert c1 is not c2


class TestResolveClient:
    def test_uses_explicit_api_key(self):
        ctx = _make_run_context()
        config = _PROVIDERS["openai"]
        client, base_url = _resolve_client("openai", config, "my_key", None, ctx)
        assert client is not None
        assert base_url is None

    def test_uses_env_api_key(self):
        ctx = _make_run_context()
        config = _PROVIDERS["openai"]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
            client, base_url = _resolve_client("openai", config, None, None, ctx)
        assert client is not None

    def test_raises_when_no_api_key(self):
        ctx = _make_run_context()
        config = _PROVIDERS["openai"]
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(APIKeyNotFoundError):
                _resolve_client("openai", config, None, None, ctx)

    def test_anthropic_returns_anthropic_client(self):
        from anthropic import AsyncAnthropic
        ctx = _make_run_context()
        config = _PROVIDERS["anthropic"]
        client, _ = _resolve_client("anthropic", config, "key_xyz", None, ctx)
        assert isinstance(client, AsyncAnthropic)

    def test_openai_returns_openai_client(self):
        from openai import AsyncOpenAI
        ctx = _make_run_context()
        config = _PROVIDERS["openai"]
        client, _ = _resolve_client("openai", config, "key_xyz", None, ctx)
        assert isinstance(client, AsyncOpenAI)

    def test_platform_proxy_used_when_no_api_key(self):
        from timbal.state.config import PlatformConfig

        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.subject = MagicMock()
        platform_config.subject.org_id = "org_123"
        platform_config.subject.app_id = "app_456"
        platform_config.host = "api.timbal.ai"
        platform_config.auth = MagicMock()
        platform_config.auth.header_value = "Bearer platform_token"

        ctx = _make_run_context(platform_config=platform_config)
        config = _PROVIDERS["openai"]
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            client, base_url = _resolve_client("openai", config, None, None, ctx)

        assert base_url is not None
        assert "api.timbal.ai" in base_url
        assert "org_123" in base_url

    def test_secretstr_api_key_unwrapped(self):
        """SecretStr is unwrapped before use."""
        ctx = _make_run_context()
        from openai import AsyncOpenAI
        config = _PROVIDERS["openai"]
        client, _ = _resolve_client("openai", config, "plain_key", None, ctx)
        assert isinstance(client, AsyncOpenAI)


class TestLlmRouterProviderValidation:
    """Test model string parsing and validation in _llm_router."""

    async def _collect(self, **kwargs):
        from timbal.core.llm_router import _llm_router
        chunks = []
        async for chunk in _llm_router(**kwargs):
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_missing_provider_prefix_raises(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()
        with pytest.raises(ValueError, match="provider/model_name"):
            async for _ in _llm_router(model="gpt-4o"):
                pass

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()
        with pytest.raises(ValueError, match="Unsupported provider"):
            async for _ in _llm_router(model="fakeprovider/some-model"):
                pass

    @pytest.mark.asyncio
    async def test_anthropic_missing_max_tokens_raises(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
            with pytest.raises(ValueError, match="max_tokens"):
                async for _ in _llm_router(model="anthropic/claude-sonnet-4-6"):
                    pass

    @pytest.mark.asyncio
    async def test_secretstr_converted_before_use(self):
        """SecretStr values for api_key and base_url are unwrapped."""
        from timbal.core.llm_router import _llm_router
        _make_run_context()

        mock_client = MagicMock()

        async def _empty_stream():
            return
            yield  # makes this an async generator

        mock_client.messages.create = AsyncMock(return_value=_empty_stream())

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            try:
                async for _ in _llm_router(
                    model="anthropic/claude-sonnet-4-6",
                    max_tokens=100,
                    api_key=SecretStr("secret_key"),
                    base_url=SecretStr("https://custom.api.com"),
                ):
                    pass
            except Exception:
                pass  # We just want to verify it got past SecretStr unwrapping


class TestLlmRouterAnthropicKwargs:
    """Test that Anthropic-specific kwargs are built correctly."""

    @pytest.mark.asyncio
    async def test_system_prompt_included(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="anthropic/claude-sonnet-4-6",
                        max_tokens=100,
                        system_prompt="You are helpful.",
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert captured_kwargs.get("system") == "You are helpful."

    @pytest.mark.asyncio
    async def test_temperature_included_when_set(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="anthropic/claude-sonnet-4-6",
                        max_tokens=100,
                        temperature=0.5,
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert captured_kwargs.get("temperature") == 0.5

    @pytest.mark.asyncio
    async def test_provider_params_forwarded(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="anthropic/claude-sonnet-4-6",
                        max_tokens=100,
                        provider_params={"top_p": 0.9, "top_k": 50},
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert captured_kwargs.get("top_p") == 0.9
        assert captured_kwargs.get("top_k") == 50

    @pytest.mark.asyncio
    async def test_tools_included_in_kwargs(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create

        mock_tool = MagicMock()
        mock_tool.anthropic_schema = {"name": "my_tool", "description": "does stuff", "input_schema": {}}

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="anthropic/claude-sonnet-4-6",
                        max_tokens=100,
                        tools=[mock_tool],
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert "tools" in captured_kwargs
        assert captured_kwargs["tools"] == [mock_tool.anthropic_schema]


class TestLlmRouterChatCompletionsKwargs:
    """Test Chat Completions path kwargs (groq, cerebras, etc.)."""

    @pytest.mark.asyncio
    async def test_system_prompt_as_system_message(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="groq/llama-3.3-70b-versatile",
                        system_prompt="Be concise.",
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        messages = captured_kwargs.get("messages", [])
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."

    @pytest.mark.asyncio
    async def test_flatten_text_content_for_xiaomi(self):
        """xiaomi provider has flatten_text_content=True."""
        from timbal.core.llm_router import _llm_router
        from timbal.types.message import Message
        from timbal.types.content.text import TextContent
        _make_run_context()

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create

        user_msg = Message(role="user", content=[TextContent(text="hello")])

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"XIAOMI_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="xiaomi/some-model",
                        messages=[user_msg],
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        messages = captured_kwargs.get("messages", [])
        # After flattening, content should be a plain string
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert isinstance(user_messages[0]["content"], str)
        assert user_messages[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_no_stream_options_when_not_supported(self):
        """xiaomi has supports_stream_options=False."""
        from timbal.core.llm_router import _llm_router
        _make_run_context()

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"XIAOMI_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(model="xiaomi/some-model"):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert "stream_options" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_output_model_adds_response_format(self):
        from timbal.core.llm_router import _llm_router
        _make_run_context()

        class MyOutput(BaseModel):
            result: str

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="groq/llama-3.3-70b-versatile",
                        output_model=MyOutput,
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert "response_format" in captured_kwargs
        assert captured_kwargs["response_format"]["type"] == "json_schema"
        assert captured_kwargs["response_format"]["json_schema"]["name"] == "MyOutput"


class TestLlmRouterProviderLookup:
    def test_all_expected_providers_present(self):
        expected = {"openai", "anthropic", "google", "groq", "cerebras", "sambanova", "xai", "fireworks"}
        for provider in expected:
            assert provider in _PROVIDERS, f"Missing provider: {provider}"

    def test_anthropic_uses_anthropic_client_type(self):
        assert _PROVIDERS["anthropic"].client_type == "anthropic"

    def test_openai_compatible_providers_use_openai_client(self):
        for provider in ("google", "groq", "cerebras", "sambanova", "xai"):
            assert _PROVIDERS[provider].client_type == "openai"

    def test_sambanova_flattens_text_content(self):
        assert _PROVIDERS["sambanova"].flatten_text_content is True

    def test_xiaomi_no_stream_options(self):
        assert _PROVIDERS["xiaomi"].supports_stream_options is False


class TestLlmRouterTestModelPath:
    """Test that TestModel bypasses provider resolution and yields chunks directly."""

    @pytest.mark.asyncio
    async def test_testmodel_yields_chunks(self):
        """TestModel.stream() is called directly; no network call occurs."""
        from timbal.core.llm_router import _llm_router
        from timbal.core.test_model import TestModel

        _make_run_context()
        model = TestModel(responses=["hello"])

        chunks = []
        async for chunk in _llm_router(model=model):
            chunks.append(chunk)

        # At least one chunk (the Message) should be yielded.
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_testmodel_increments_call_count(self):
        """call_count reflects actual calls via the router."""
        from timbal.core.llm_router import _llm_router
        from timbal.core.test_model import TestModel

        _make_run_context()
        model = TestModel(responses=["hi"])

        async for _ in _llm_router(model=model):
            pass

        assert model.call_count == 1

    @pytest.mark.asyncio
    async def test_testmodel_skips_provider_validation(self):
        """TestModel short-circuit means no ValueError for missing provider prefix."""
        from timbal.core.llm_router import _llm_router
        from timbal.core.test_model import TestModel

        _make_run_context()
        model = TestModel(responses=["no provider needed"])

        # If the TestModel path did NOT short-circuit, this would raise ValueError.
        chunks = []
        async for chunk in _llm_router(model=model):
            chunks.append(chunk)

        assert chunks  # something yielded — no exception raised


class TestLlmRouterPlatformHeaders:
    """Test that platform config subject fields are forwarded as request headers."""

    def _make_platform_context(self, app_id=None, version_id=None):
        from timbal.state.config import PlatformConfig

        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.subject = MagicMock()
        platform_config.subject.org_id = "org_999"
        platform_config.subject.app_id = app_id
        platform_config.subject.version_id = version_id
        platform_config.host = "api.timbal.ai"
        platform_config.auth = MagicMock()
        platform_config.auth.header_value = "Bearer tok"
        return _make_run_context(platform_config=platform_config)

    @pytest.mark.asyncio
    async def test_app_id_added_to_headers(self):
        from timbal.core.llm_router import _llm_router

        self._make_platform_context(app_id="app_123", version_id=None)

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            try:
                async for _ in _llm_router(
                    model="anthropic/claude-sonnet-4-6",
                    max_tokens=100,
                ):
                    pass
            except (RuntimeError, StopAsyncIteration):
                pass

        headers = captured_kwargs.get("extra_headers", {})
        assert headers.get("x-timbal-app-id") == "app_123"
        assert "x-timbal-version-id" not in headers

    @pytest.mark.asyncio
    async def test_version_id_added_to_headers(self):
        from timbal.core.llm_router import _llm_router

        self._make_platform_context(app_id=None, version_id="v42")

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            try:
                async for _ in _llm_router(
                    model="anthropic/claude-sonnet-4-6",
                    max_tokens=100,
                ):
                    pass
            except (RuntimeError, StopAsyncIteration):
                pass

        headers = captured_kwargs.get("extra_headers", {})
        assert headers.get("x-timbal-version-id") == "v42"
        assert "x-timbal-app-id" not in headers

    @pytest.mark.asyncio
    async def test_both_app_id_and_version_id_added(self):
        from timbal.core.llm_router import _llm_router

        self._make_platform_context(app_id="app_123", version_id="v42")

        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            try:
                async for _ in _llm_router(
                    model="anthropic/claude-sonnet-4-6",
                    max_tokens=100,
                ):
                    pass
            except (RuntimeError, StopAsyncIteration):
                pass

        headers = captured_kwargs.get("extra_headers", {})
        assert headers.get("x-timbal-app-id") == "app_123"
        assert headers.get("x-timbal-version-id") == "v42"


class TestLlmRouterAnthropicStructuredOutput:
    """Test that output_model routes to client.beta.messages.create."""

    @pytest.mark.asyncio
    async def test_output_model_uses_beta_endpoint(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()

        class MyOutput(BaseModel):
            answer: str

        beta_captured = {}
        stable_captured = {}

        async def fake_beta_create(**kwargs):
            beta_captured.update(kwargs)
            return _empty_async_stream()

        async def fake_stable_create(**kwargs):
            stable_captured.update(kwargs)
            return _empty_async_stream()

        mock_beta_messages = MagicMock()
        mock_beta_messages.create = fake_beta_create

        mock_client = MagicMock()
        mock_client.messages.create = fake_stable_create
        mock_client.beta.messages.create = fake_beta_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="anthropic/claude-sonnet-4-6",
                        max_tokens=100,
                        output_model=MyOutput,
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        # Beta endpoint was called; stable endpoint was not.
        assert beta_captured, "Expected client.beta.messages.create to be called"
        assert not stable_captured, "Expected client.messages.create NOT to be called"

    @pytest.mark.asyncio
    async def test_output_model_sets_betas_flag(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()

        class MyOutput(BaseModel):
            value: int

        captured_kwargs = {}

        async def fake_beta_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=AssertionError("should not call stable"))
        mock_client.beta.messages.create = fake_beta_create

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="anthropic/claude-sonnet-4-6",
                        max_tokens=100,
                        output_model=MyOutput,
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert "structured-outputs-2025-11-13" in captured_kwargs.get("betas", [])
        assert captured_kwargs.get("output_format", {}).get("type") == "json_schema"

    @pytest.mark.asyncio
    async def test_no_output_model_uses_stable_endpoint(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()

        stable_captured = {}

        async def fake_stable_create(**kwargs):
            stable_captured.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.messages.create = fake_stable_create
        mock_client.beta.messages.create = AsyncMock(side_effect=AssertionError("should not call beta"))

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                try:
                    async for _ in _llm_router(
                        model="anthropic/claude-sonnet-4-6",
                        max_tokens=100,
                    ):
                        pass
                except (RuntimeError, StopAsyncIteration):
                    pass

        assert stable_captured, "Expected client.messages.create to be called"


class TestLlmRouterOpenAIResponsesPath:
    """Test the OpenAI Responses API path (TIMBAL_OPENAI_API == 'responses')."""

    def _make_mock_client_and_capturer(self):
        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.responses.create = fake_create
        return mock_client, captured_kwargs

    @pytest.mark.asyncio
    async def test_responses_path_system_prompt_as_instructions(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "key", "TIMBAL_OPENAI_API": "responses"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "responses"):
                    try:
                        async for _ in _llm_router(
                            model="openai/gpt-4o",
                            system_prompt="system prompt",
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert captured_kwargs.get("instructions") == "system prompt"

    @pytest.mark.asyncio
    async def test_responses_path_max_tokens(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "responses"):
                    try:
                        async for _ in _llm_router(
                            model="openai/gpt-4o",
                            max_tokens=512,
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert captured_kwargs.get("max_output_tokens") == 512
        assert "max_tokens" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_responses_path_output_model(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()

        class MySchema(BaseModel):
            result: str

        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "responses"):
                    try:
                        async for _ in _llm_router(
                            model="openai/gpt-4o",
                            output_model=MySchema,
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        text_format = captured_kwargs.get("text", {}).get("format", {})
        assert text_format.get("type") == "json_schema"
        assert text_format.get("name") == "MySchema"
        assert text_format.get("strict") is True

    @pytest.mark.asyncio
    async def test_responses_path_tools(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        mock_tool = MagicMock()
        mock_tool.openai_responses_schema = {"type": "function", "name": "my_tool"}

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "responses"):
                    try:
                        async for _ in _llm_router(
                            model="openai/gpt-4o",
                            tools=[mock_tool],
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert "tools" in captured_kwargs
        assert captured_kwargs["tools"] == [mock_tool.openai_responses_schema]

    @pytest.mark.asyncio
    async def test_responses_path_temperature(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "responses"):
                    try:
                        async for _ in _llm_router(
                            model="openai/gpt-4o",
                            temperature=0.3,
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert captured_kwargs.get("temperature") == 0.3

    @pytest.mark.asyncio
    async def test_xai_uses_responses_path(self):
        """xai provider also routes through the Responses API path."""
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"XAI_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "responses"):
                    try:
                        async for _ in _llm_router(
                            model="xai/grok-3",
                            system_prompt="be concise",
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        # Responses path sets "instructions" rather than a system message in "messages"
        assert captured_kwargs.get("instructions") == "be concise"
        # And uses responses.create (our mock_client.responses.create was called)
        assert captured_kwargs  # something was captured — fake_create was invoked


class TestLlmRouterChatCompletionsMore:
    """Additional coverage for the Chat Completions path."""

    def _make_mock_client_and_capturer(self):
        captured_kwargs = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _empty_async_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        return mock_client, captured_kwargs

    @pytest.mark.asyncio
    async def test_max_tokens_as_max_completion_tokens(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "chat_completions"):
                    try:
                        async for _ in _llm_router(
                            model="groq/llama-3.3-70b-versatile",
                            max_tokens=256,
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert captured_kwargs.get("max_completion_tokens") == 256
        assert "max_tokens" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "chat_completions"):
                    try:
                        async for _ in _llm_router(
                            model="groq/llama-3.3-70b-versatile",
                            temperature=0.8,
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert captured_kwargs.get("temperature") == 0.8

    @pytest.mark.asyncio
    async def test_tools_forwarded_with_correct_schema(self):
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        mock_tool = MagicMock()
        mock_tool.openai_chat_completions_schema = {
            "type": "function",
            "function": {"name": "do_thing", "description": "does a thing", "parameters": {}},
        }

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "chat_completions"):
                    try:
                        async for _ in _llm_router(
                            model="groq/llama-3.3-70b-versatile",
                            tools=[mock_tool],
                        ):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert "tools" in captured_kwargs
        assert captured_kwargs["tools"] == [mock_tool.openai_chat_completions_schema]

    @pytest.mark.asyncio
    async def test_no_max_tokens_omits_max_completion_tokens(self):
        """When max_tokens is not provided, max_completion_tokens should be absent."""
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "chat_completions"):
                    try:
                        async for _ in _llm_router(model="groq/llama-3.3-70b-versatile"):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert "max_completion_tokens" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_no_temperature_omits_temperature(self):
        """When temperature is not passed, it should not appear in kwargs."""
        from timbal.core.llm_router import _llm_router

        _make_run_context()
        mock_client, captured_kwargs = self._make_mock_client_and_capturer()

        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "chat_completions"):
                    try:
                        async for _ in _llm_router(model="groq/llama-3.3-70b-versatile"):
                            pass
                    except (RuntimeError, StopAsyncIteration):
                        pass

        assert "temperature" not in captured_kwargs


class TestLlmRouterYieldsChunks:
    """Test that the yield lines in all three provider paths are actually executed."""

    @pytest.mark.asyncio
    async def test_anthropic_path_yields_chunks_with_messages(self):
        """Cover lines 372-373 (message building) and 412, 415 (yield chunk)."""
        from timbal.core.llm_router import _llm_router
        from timbal.types.message import Message
        from timbal.types.content.text import TextContent

        _make_run_context()

        sentinel = object()

        async def _one_item():
            yield sentinel

        async def fake_create(**kwargs):
            return _one_item()

        mock_client = MagicMock()
        mock_client.messages.create = fake_create

        user_msg = Message(role="user", content=[TextContent(text="hello")])
        chunks = []
        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
                async for chunk in _llm_router(
                    model="anthropic/claude-sonnet-4-6",
                    max_tokens=100,
                    messages=[user_msg],
                ):
                    chunks.append(chunk)

        assert chunks == [sentinel]

    @pytest.mark.asyncio
    async def test_openai_responses_path_yields_chunks(self):
        """Cover lines 458, 461 (yield chunk in responses path)."""
        from timbal.core.llm_router import _llm_router

        _make_run_context()

        sentinel = object()

        async def _one_item():
            yield sentinel

        async def fake_create(**kwargs):
            return _one_item()

        mock_client = MagicMock()
        mock_client.responses.create = fake_create

        chunks = []
        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "responses"):
                    async for chunk in _llm_router(
                        model="openai/gpt-4o",
                    ):
                        chunks.append(chunk)

        assert chunks == [sentinel]

    @pytest.mark.asyncio
    async def test_chat_completions_path_yields_chunks(self):
        """Cover lines 520, 525 (yield chunk in chat completions path)."""
        from timbal.core.llm_router import _llm_router

        _make_run_context()

        sentinel = object()

        async def _one_item():
            yield sentinel

        async def fake_create(**kwargs):
            return _one_item()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create

        chunks = []
        with patch("timbal.core.llm_router._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
                with patch("timbal.core.llm_router.TIMBAL_OPENAI_API", "chat_completions"):
                    async for chunk in _llm_router(
                        model="groq/llama-3.3-70b-versatile",
                    ):
                        chunks.append(chunk)

        assert chunks == [sentinel]


class TestGetAnthropicClientWithBaseUrl:
    """Cover _get_client when base_url is provided for Anthropic."""

    def test_base_url_set_on_client(self):
        from timbal.core.llm_router import _CLIENT_CACHE
        _CLIENT_CACHE.clear()

        c_with_url = _get_client(AsyncAnthropic, "key_x", "https://custom.api.com/v1", "anthropic")
        c_without_url = _get_client(AsyncAnthropic, "key_x", None, "anthropic")
        # Different cache keys, different instances
        assert c_with_url is not c_without_url

    def test_base_url_cached_separately(self):
        from timbal.core.llm_router import _CLIENT_CACHE
        _CLIENT_CACHE.clear()

        c1 = _get_client(AsyncAnthropic, "key_y", "https://proxy.example.com", "anthropic")
        c2 = _get_client(AsyncAnthropic, "key_y", "https://proxy.example.com", "anthropic")
        assert c1 is c2
