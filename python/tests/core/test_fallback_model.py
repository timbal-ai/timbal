import json
from unittest.mock import MagicMock

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import APIStatusError as OpenAIAPIStatusError
from openai import AsyncOpenAI
from timbal import Agent
from timbal.core.fallback_model import FallbackModel, ModelEntry
from timbal.core.llm import _llm_router
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
    async def test_fail_fast_rate_limit_set_for_all_but_last_entry(self):
        """Every entry with a fallback behind it must fail over on 429 instead of
        sleeping through Retry-After in place; the last entry retries normally."""
        model = FallbackModel("openai/primary", "openai/middle", "openai/last")
        calls = []

        async def router(**kwargs):
            calls.append(kwargs)
            if kwargs["model"] != "openai/last":
                raise _status_error(429)
            yield "ok"

        chunks = [chunk async for chunk in model.route(router)]

        assert chunks == ["ok"]
        assert [(call["model"], call["fail_fast_rate_limit"]) for call in calls] == [
            ("openai/primary", True),
            ("openai/middle", True),
            ("openai/last", False),
        ]

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
    async def test_default_falls_back_on_any_exception(self):
        """Default behavior: any pre-stream Exception triggers fallback."""
        model = FallbackModel("openai/primary", "openai/backup")
        calls = []

        async def router(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "openai/primary":
                raise ValueError("bad request")
            yield "ok"

        chunks = [chunk async for chunk in model.route(router)]

        assert chunks == ["ok"]
        assert calls == ["openai/primary", "openai/backup"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("primary,other", [("openai", "anthropic"), ("anthropic", "openai")])
    async def test_scopes_shared_auth_to_primary_provider(self, primary, other):
        model = FallbackModel(
            f"{primary}/primary",
            f"{other}/backup",
            ModelEntry(f"{other}/custom", api_key="custom_key", base_url="https://custom"),
            f"{primary}/last",
        )
        calls = []

        async def router(**kwargs):
            calls.append(kwargs)
            if kwargs["model"] != f"{primary}/last":
                raise _status_error(401)
            yield "ok"

        chunks = [chunk async for chunk in model.route(router, api_key="primary_key", base_url="https://primary")]

        assert chunks == ["ok"]
        assert [(call.get("api_key"), call.get("base_url")) for call in calls] == [
            ("primary_key", "https://primary"),
            (None, None),
            ("custom_key", "https://custom"),
            ("primary_key", "https://primary"),
        ]

    @pytest.mark.asyncio
    async def test_default_falls_back_on_auth_error(self):
        """401/403 should trigger fallback by default — conservative behavior is opt-in."""
        from timbal.core.fallback_model import is_retryable_provider_error

        model = FallbackModel("openai/primary", "openai/backup")
        calls = []

        async def router(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "openai/primary":
                raise _status_error(401)
            yield "ok"

        chunks = [chunk async for chunk in model.route(router)]

        assert chunks == ["ok"]
        assert calls == ["openai/primary", "openai/backup"]
        # And the conservative helper still classifies 401 as non-retryable
        assert not is_retryable_provider_error(_status_error(401))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "primary,other",
        [("openai", "anthropic"), ("anthropic", "openai"), ("openai", "xai"), ("xai", "openai")],
    )
    async def test_provider_params_inheritance_and_replacement(self, primary, other):
        shared = {"shared": {"value": 1}}
        override = {"override": {"value": 2}}
        model = FallbackModel(
            ModelEntry(f"{primary}/primary", provider_params=override),
            f"{primary}/inherited",
            ModelEntry(f"{primary}/cleared", provider_params={}),
            f"{other}/default",
            ModelEntry(f"{other}/custom", provider_params=override),
            f"{other}/next",
            f"{primary}/last",
        )
        calls = []

        async def router(**kwargs):
            calls.append(kwargs.get("provider_params"))
            if kwargs["model"] != f"{primary}/last":
                raise _status_error(503)
            yield "ok"

        assert [chunk async for chunk in model.route(router, provider_params=shared)] == ["ok"]
        assert calls == [override, shared, {}, None, override, None, shared]
        assert shared == {"shared": {"value": 1}}
        assert override == {"override": {"value": 2}}

    @pytest.mark.asyncio
    async def test_conservative_predicate_skips_non_provider_errors(self):
        """Opt-in conservative mode: only transient provider errors trigger fallback."""
        from timbal.core.fallback_model import is_retryable_provider_error

        model = FallbackModel(
            "openai/primary",
            "openai/backup",
            fallback_on=is_retryable_provider_error,
        )
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
    async def test_base_exceptions_propagate_without_fallback(self):
        """KeyboardInterrupt / CancelledError must not be swallowed by fallback."""
        import asyncio

        model = FallbackModel("openai/primary", "openai/backup")
        calls = []

        async def router(**kwargs):
            calls.append(kwargs["model"])
            raise asyncio.CancelledError()
            yield

        with pytest.raises(asyncio.CancelledError):
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
    @pytest.mark.parametrize(
        "primary,backup,shared,native",
        [
            ("openai", "anthropic", {"reasoning": {"effort": "low"}}, {"thinking": {"type": "adaptive"}}),
            ("xai", "anthropic", {"reasoning": {"effort": "low"}}, {"thinking": {"type": "adaptive"}}),
            ("anthropic", "openai", {"thinking": {"type": "adaptive"}}, {"reasoning": {"effort": "low"}}),
            ("anthropic", "xai", {"thinking": {"type": "adaptive"}}, {"reasoning": {"effort": "low"}}),
            ("openai", "google", {"reasoning": {"effort": "low"}}, {"top_p": 0.8}),
            ("anthropic", "google", {"thinking": {"type": "adaptive"}}, {"top_p": 0.8}),
        ],
    )
    @pytest.mark.parametrize("override", ["inherit", "native", "empty"])
    async def test_cross_provider_params_with_real_sdks(self, primary, backup, shared, native, override, monkeypatch):
        """Keep real SDK argument validation and serialization, mocking only HTTP."""
        from timbal.core.llm import router as router_module
        from timbal.state import _call_id, _run_context_var
        from timbal.state.context import RunContext

        monkeypatch.setattr(router_module, "TIMBAL_OPENAI_API", "responses")
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            if body["model"] == "primary":
                return httpx.Response(
                    404, json={"error": {"type": "not_found_error", "message": "Model does not exist"}},
                )
            if backup == "anthropic":
                data = 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            elif backup in ("openai", "xai"):
                data = (
                    'event: response.output_text.delta\n'
                    'data: {"type":"response.output_text.delta","delta":"OK"}\n\n'
                )
            else:
                data = (
                    'data: {"id":"test","object":"chat.completion.chunk","created":0,"model":"backup",'
                    '"choices":[{"index":0,"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
                )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)

        async with (
            AsyncAnthropic(
                api_key="test-key", max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            ) as anthropic,
            AsyncOpenAI(
                api_key="test-key", max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            ) as openai,
        ):
            def resolve_client(provider, *_args):
                return (anthropic if provider == "anthropic" else openai), None

            monkeypatch.setattr(router_module, "_resolve_client", resolve_client)
            entry_params = {"inherit": None, "native": native, "empty": {}}[override]
            model = FallbackModel(
                ModelEntry(f"{primary}/primary", max_retries=0),
                ModelEntry(f"{backup}/backup", max_retries=0, provider_params=entry_params),
            )
            token_ctx = _run_context_var.set(RunContext(tracing_provider=None))
            token_cid = _call_id.set(None)
            try:
                chunks = [chunk async for chunk in _llm_router(model=model, max_tokens=1024, provider_params=shared)]
            finally:
                _run_context_var.reset(token_ctx)
                _call_id.reset(token_cid)

        assert len(chunks) == 1
        assert [request["model"] for request in requests] == ["primary", "backup"]
        assert all(requests[0][key] == value for key, value in shared.items())
        assert not shared.keys() & requests[1].keys()
        if override == "native":
            assert all(requests[1][key] == value for key, value in native.items())
        else:
            assert not native.keys() & requests[1].keys()

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
