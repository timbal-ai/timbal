"""Live integration tests for newly registered frontier models.

Run explicitly (requires provider API keys in the environment or ``.env``):

    uv run pytest python/tests/core/test_frontier_models_integration.py -m integration -v

Each test performs a minimal Agent round-trip and asserts structural success
(non-empty output, status success). Text content is not matched exactly.
"""

from __future__ import annotations

import os

import pytest
from timbal import Agent
from timbal.types.events import OutputEvent

PROMPT = "Reply with exactly one word: ok"

LIVE_MODELS = [
    pytest.param(
        "google/gemini-4-argon", "GEMINI_API_KEY", None, id="google-gemini-4-argon",
    ),
    pytest.param(
        "anthropic/claude-haiku-5-5", "ANTHROPIC_API_KEY", None, id="anthropic-claude-haiku-5-5",
    ),
    pytest.param(
        "anthropic/claude-sonnet-5-5", "ANTHROPIC_API_KEY", None, id="anthropic-claude-sonnet-5-5",
    ),
    pytest.param(
        "openai/gpt-6-sol", "OPENAI_API_KEY", None, id="openai-gpt-6-sol",
    ),
    pytest.param(
        "openai/gpt-6-luna", "OPENAI_API_KEY", None, id="openai-gpt-6-luna",
    ),
    pytest.param(
        "anthropic/claude-opus-5-5", "ANTHROPIC_API_KEY", None, id="anthropic-claude-opus-5-5",
    ),
    pytest.param(
        "xai/grok-4.7", "XAI_API_KEY", None, id="xai-grok-4.7",
    ),
    pytest.param(
        "openai/gpt-6-astra",
        "OPENAI_API_KEY",
        None,
        id="openai-gpt-6-astra",
    ),
    pytest.param(
        "anthropic/claude-opus-5",
        "ANTHROPIC_API_KEY",
        None,
        id="anthropic-claude-opus-5",
    ),
    pytest.param(
        "google/gemini-3.6-flash",
        "GEMINI_API_KEY",
        None,
        id="google-gemini-3.6-flash",
    ),
    pytest.param(
        "google/gemini-3.5-flash-lite",
        "GEMINI_API_KEY",
        None,
        id="google-gemini-3.5-flash-lite",
    ),
    pytest.param(
        "xai/grok-4.6",
        "XAI_API_KEY",
        None,
        id="xai-grok-4.6",
    ),
    pytest.param(
        "xai/grok-4.5",
        "XAI_API_KEY",
        None,
        id="xai-grok-4.5",
    ),
    pytest.param(
        "fireworks/accounts/fireworks/models/minimax-m3",
        "FIREWORKS_API_KEY",
        None,
        id="fireworks-minimax-m3",
    ),
    pytest.param(
        "fireworks/accounts/fireworks/models/qwen3p8-max",
        "FIREWORKS_API_KEY",
        None,
        id="fireworks-qwen3p8-max",
    ),
    pytest.param(
        "fireworks/accounts/fireworks/models/deepseek-v4-flash-0731",
        "FIREWORKS_API_KEY",
        None,
        id="fireworks-deepseek-v4-flash",
    ),
]


def _has_key(primary: str, fallback: str | None) -> bool:
    if os.getenv(primary):
        return True
    return bool(fallback and os.getenv(fallback))


def _skip_if_no_key(primary: str, fallback: str | None) -> None:
    if not _has_key(primary, fallback):
        hint = primary if not fallback else f"{primary} or {fallback}"
        pytest.skip(f"Frontier integration: set {hint} in the environment or .env")


@pytest.mark.integration
@pytest.mark.parametrize("model,env_key,fallback_env", LIVE_MODELS)
async def test_frontier_model_agent_collect(model: str, env_key: str, fallback_env: str | None):
    _skip_if_no_key(env_key, fallback_env)

    # Reasoning models (MiniMax / Qwen) spend many tokens on reasoning_content
    # before visible text, and Opus 5 thinks by default — a small budget is
    # often exhausted with an empty content array.
    agent = Agent(name=f"probe_{model.replace('/', '_')}", model=model, max_tokens=2048, tools=[])
    result: OutputEvent = await agent(prompt=PROMPT).collect()

    assert result.status.code == "success", result.error
    text = result.output.collect_text() if hasattr(result.output, "collect_text") else str(result.output)
    assert text.strip(), "expected non-empty model output"


@pytest.mark.integration
@pytest.mark.parametrize("model,env_key,fallback_env", LIVE_MODELS)
async def test_frontier_model_llm_router_streams(model: str, env_key: str, fallback_env: str | None):
    """Direct router smoke test — ensures streaming path works, not only Agent.collect."""
    _skip_if_no_key(env_key, fallback_env)

    from timbal.core.llm import _llm_router
    from timbal.types.content import TextContent
    from timbal.types.message import Message

    chunks = []
    async for chunk in _llm_router(
        model=model,
        messages=[Message(role="user", content=[TextContent(text=PROMPT)])],
        max_tokens=2048,
    ):
        chunks.append(chunk)

    assert chunks, f"expected at least one stream chunk from {model}"


@pytest.mark.integration
async def test_argon_tool_round_trip():
    """Exercise Google thought-signature preservation across a real tool round trip."""
    _skip_if_no_key("GEMINI_API_KEY", None)
    calls = []

    def lookup_code() -> str:
        """Return the verification code required for the answer."""
        calls.append(True)
        return "argon-round-trip-47"

    agent = Agent(
        name="argon_tools",
        model="google/gemini-4-argon",
        max_tokens=32768,
        model_params={"reasoning_effort": "low"},
        tools=[lookup_code],
    )
    result = await agent(prompt="Call lookup_code and return its exact code.").collect()
    assert result.status.code == "success", result.error
    assert calls, "the model must actually call the tool"
    assert "argon-round-trip-47" in result.output.collect_text()


@pytest.mark.integration
async def test_argon_structured_output():
    from pydantic import BaseModel

    _skip_if_no_key("GEMINI_API_KEY", None)

    class Answer(BaseModel):
        code: str

    agent = Agent(
        name="argon_json",
        model="google/gemini-4-argon",
        max_tokens=32768,
        model_params={"reasoning_effort": "low"},
        output_model=Answer,
    )
    result = await agent(prompt="Return code argon-round-trip-47.").collect()
    assert result.status.code == "success", result.error
    assert isinstance(result.output, Answer)
    assert result.output.code == "argon-round-trip-47"
