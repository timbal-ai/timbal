from unittest.mock import MagicMock

import pytest
from openai import APIStatusError as OpenAIAPIStatusError
from timbal import Agent
from timbal.core.fallback_model import FallbackModel, ModelEntry
from timbal.core.llm_router import _llm_router
from timbal.errors import FallbackExhausted


def _status_error(status_code: int) -> OpenAIAPIStatusError:
    response = MagicMock()
    response.status_code = status_code
    return OpenAIAPIStatusError(message=f"HTTP {status_code}", response=response, body=None)


class TestFallbackModel:
    def test_exported_from_top_level_package(self):
        from timbal import FallbackModel as ExportedFallbackModel
        from timbal import ModelEntry as ExportedModelEntry

        assert ExportedFallbackModel is FallbackModel
        assert ExportedModelEntry is ModelEntry

    def test_agent_accepts_fallback_model(self):
        fallback = FallbackModel("openai/primary", "openai/backup")

        agent = Agent(name="fallback_agent", model=fallback)

        assert agent.model is fallback
        assert agent._llm.metadata["model_provider"] == "fallback"
        assert agent._llm.metadata["model_name"] == "openai/primary -> openai/backup"

    @pytest.mark.asyncio
    async def test_falls_back_after_retryable_provider_error(self):
        model = FallbackModel("openai/primary", "openai/backup")
        calls = []

        async def router(**kwargs):
            calls.append(kwargs)
            if kwargs["model"] == "openai/primary":
                raise _status_error(503)
            yield kwargs["model"]

        chunks = []
        async for chunk in model.route(router, temperature=0.2):
            chunks.append(chunk)

        assert chunks == ["openai/backup"]
        assert [call["model"] for call in calls] == ["openai/primary", "openai/backup"]
        assert all(call["temperature"] == 0.2 for call in calls)

    @pytest.mark.asyncio
    async def test_uses_per_entry_retry_and_auth_overrides(self):
        model = FallbackModel(
            ModelEntry("openai/primary", max_retries=4, retry_delay=0.5, api_key="entry_key", base_url="https://entry"),
        )
        calls = []

        async def router(**kwargs):
            calls.append(kwargs)
            yield "ok"

        chunks = [chunk async for chunk in model.route(router, api_key="global_key", base_url="https://global")]

        assert chunks == ["ok"]
        assert calls[0]["max_retries"] == 4
        assert calls[0]["retry_delay"] == 0.5
        assert calls[0]["api_key"] == "entry_key"
        assert calls[0]["base_url"] == "https://entry"

    @pytest.mark.asyncio
    async def test_non_retryable_error_does_not_fallback(self):
        model = FallbackModel("openai/primary", "openai/backup")
        calls = []

        async def router(**kwargs):
            calls.append(kwargs["model"])
            raise ValueError("bad request")
            yield

        with pytest.raises(ValueError, match="bad request"):
            async for _ in model.route(router):
                pass

        assert calls == ["openai/primary"]

    @pytest.mark.asyncio
    async def test_custom_fallback_exception_type(self):
        model = FallbackModel("openai/primary", "openai/backup", fallback_on=ValueError)

        async def router(**kwargs):
            if kwargs["model"] == "openai/primary":
                raise ValueError("try next")
            yield kwargs["model"]

        chunks = [chunk async for chunk in model.route(router)]

        assert chunks == ["openai/backup"]

    @pytest.mark.asyncio
    async def test_error_after_first_chunk_does_not_fallback(self):
        model = FallbackModel("openai/primary", "openai/backup")
        calls = []
        chunks = []

        async def router(**kwargs):
            calls.append(kwargs["model"])
            yield "partial"
            raise _status_error(503)

        with pytest.raises(OpenAIAPIStatusError):
            async for chunk in model.route(router):
                chunks.append(chunk)

        assert chunks == ["partial"]
        assert calls == ["openai/primary"]

    @pytest.mark.asyncio
    async def test_exhaustion_raises_bundled_errors(self):
        model = FallbackModel("openai/primary", "openai/backup")

        async def router(**_kwargs):
            raise _status_error(503)
            yield

        with pytest.raises(FallbackExhausted) as exc_info:
            async for _ in model.route(router):
                pass

        assert [model for model, _ in exc_info.value.errors] == ["openai/primary", "openai/backup"]
        assert "All 2 fallback models failed" in str(exc_info.value)


class TestFallbackRouterIntegration:
    @pytest.mark.asyncio
    async def test_llm_router_delegates_to_fallback_model(self):
        model = FallbackModel("openai/primary")
        captured = {}

        async def route(router, **kwargs):
            captured["router"] = router
            captured["kwargs"] = kwargs
            yield "delegated"

        model.route = route  # type: ignore[method-assign]

        chunks = []
        async for chunk in _llm_router(model=model, temperature=0.4):
            chunks.append(chunk)

        assert chunks == ["delegated"]
        assert captured["router"] is _llm_router
        assert captured["kwargs"]["temperature"] == 0.4
