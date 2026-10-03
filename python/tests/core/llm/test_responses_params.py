"""Regression: a Groq -> OpenAI model switch must not crash before the request."""

import copy
import inspect
from types import SimpleNamespace

import pytest
from openai.resources.responses.responses import AsyncResponses
from timbal.core.llm.responses import normalize_responses_params, prepare_responses_request


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-6-luna", "gpt-6.1-sol"])
async def test_legacy_effort_reaches_responses_in_native_shape(model_name):
    received = {}

    async def empty():
        if False:
            yield None

    async def create(**kwargs):
        # A permissive **kwargs mock alone would miss the original SDK error.
        inspect.signature(AsyncResponses.create).bind_partial(None, **kwargs)
        received.update(kwargs)
        return empty()

    params = {"reasoning_effort": "low", "reasoning": {"summary": "auto"}}
    original = copy.deepcopy(params)
    stream, _ = prepare_responses_request(
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        model_name=model_name, request_headers={}, system_prompt=None,
        messages=[], tools=None, max_tokens=2048, temperature=None,
        output_model=None, provider_params=params,
    )
    async for _ in stream():
        pass
    assert received["reasoning"] == {"effort": "low", "summary": "auto"}
    assert received["model"] == model_name
    assert "reasoning_effort" not in received
    assert received["max_output_tokens"] == 2048
    assert params == original


def test_equal_aliases_and_null_legacy_effort():
    assert normalize_responses_params({"reasoning_effort": "low", "reasoning": {"effort": "low"}}) == {
        "reasoning": {"effort": "low"}
    }
    assert normalize_responses_params({"reasoning_effort": None}) == {}
    assert normalize_responses_params({"reasoning_effort": None, "reasoning": {"effort": "high"}}) == {
        "reasoning": {"effort": "high"}
    }


@pytest.mark.parametrize("params", [
    {"reasoning_effort": "low", "reasoning": {"effort": "high"}},
    {"reasoning_effort": "low", "reasoning": "high"},
    {"reasoning_effort": ""},
    {"reasoning_effort": 2},
])
def test_conflicts_are_explicit_and_do_not_mutate_config(params):
    original = copy.deepcopy(params)
    with pytest.raises(ValueError, match="model_params"):
        normalize_responses_params(params)
    assert params == original
