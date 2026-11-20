"""The Anthropic adapter must not send a temperature that the model's API rejects (400)."""

from types import SimpleNamespace

import pytest
from timbal.core.llm.messages import prepare_messages_request, rejects_temperature


async def _request_kwargs(model_name: str, temperature: float | None, provider_params: dict | None = None) -> dict:
    received: dict = {}

    async def empty():
        if False:
            yield None

    async def create(**kwargs):
        received.update(kwargs)
        return empty()

    stream, _ = prepare_messages_request(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model_name=model_name, request_headers={}, system_prompt=None, messages=[], tools=None,
        max_tokens=64, temperature=temperature, output_model=None, provider_params=provider_params or {},
    )
    async for _ in stream():
        pass
    return received


@pytest.mark.parametrize("model_name", [
    "claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-opus-5-5", "claude-opus-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5-2", "claude-fable-5-1", "claude-fable-5", "claude-mythos-5-2",
])
@pytest.mark.asyncio
async def test_temperature_dropped_where_rejected(model_name):
    assert rejects_temperature(model_name)
    assert "temperature" not in await _request_kwargs(model_name, 0.0)


@pytest.mark.parametrize("model_name", [
    "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5", "claude-opus-4-5-20251101",
    "claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-sonnet-4-5",
])
@pytest.mark.asyncio
async def test_temperature_kept_where_supported(model_name):
    assert not rejects_temperature(model_name)
    assert (await _request_kwargs(model_name, 0.0))["temperature"] == 0.0


@pytest.mark.asyncio
async def test_no_temperature_when_unset_and_explicit_model_params_pass_through():
    assert "temperature" not in await _request_kwargs("claude-sonnet-4-6", None)
    assert (await _request_kwargs("claude-haiku-5-5", 0.0, {"temperature": 1.0}))["temperature"] == 1.0
