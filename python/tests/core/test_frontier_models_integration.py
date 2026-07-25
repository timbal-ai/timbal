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
        "xai/grok-4.5",
        "XAI_API_KEY",
        None,
        id="xai-grok-4.5",
    ),
    pytest.param(
        "fireworks/accounts/fireworks/models/minimax-m2p7",
        "FIREWORKS_API_KEY",
        None,
        id="fireworks-minimax-m2p7",
    ),
    pytest.param(
        "fireworks/accounts/fireworks/models/qwen3p7-plus",
        "FIREWORKS_API_KEY",
        None,
        id="fireworks-qwen3p7-plus",
    ),
    pytest.param(
        "fireworks/accounts/fireworks/models/deepseek-v4-flash",
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
    # before visible text — 64 is often exhausted with an empty content array.
    agent = Agent(name=f"probe_{model.replace('/', '_')}", model=model, max_tokens=512, tools=[])
    result: OutputEvent = await agent(prompt=PROMPT).collect()

    assert result.status.code == "success", result.error
    text = result.output.collect_text() if hasattr(result.output, "collect_text") else str(result.output)
    assert text.strip(), "expected non-empty model output"


@pytest.mark.integration
@pytest.mark.parametrize("model,env_key,fallback_env", LIVE_MODELS)
async def test_frontier_model_llm_router_streams(model: str, env_key: str, fallback_env: str | None):
    """Direct router smoke test — ensures streaming path works, not only Agent.collect."""
    _skip_if_no_key(env_key, fallback_env)

    from timbal.core.llm_router import _llm_router
    from timbal.types.message import Message
    from timbal.types.content import TextContent

    chunks = []
    async for chunk in _llm_router(
        model=model,
        messages=[Message(role="user", content=[TextContent(text=PROMPT)])],
        max_tokens=512,
    ):
        chunks.append(chunk)

    assert chunks, f"expected at least one stream chunk from {model}"
