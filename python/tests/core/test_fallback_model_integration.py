"""Live cross-provider failover probes using real SDKs and provider endpoints.

Run with keys in .env (loaded by conftest with override=True):
    uv run pytest python/tests/core/test_fallback_model_integration.py -m integration -v

Every ordered provider pair is tested with entry-level and agent-level bad
credentials. Four longer chains also exercise three failures before success.
Only the final provider needs a valid key; no provider requests are mocked.
"""

import asyncio
import os
from itertools import permutations

import pytest
from timbal import Agent, FallbackModel, ModelEntry
from timbal.core.llm import clients

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

PROVIDERS = {
    "openai": ("openai/gpt-4.1-mini", "OPENAI_API_KEY"),
    "anthropic": ("anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    "google": ("google/gemini-2.5-flash-lite", "GEMINI_API_KEY"),
    "xai": ("xai/grok-4.3", "XAI_API_KEY"),
}
_ORDER = tuple(PROVIDERS)
CHAINS = [
    pytest.param(chain, id="-to-".join(chain))
    for chain in [
        *permutations(PROVIDERS, 2),
        *(_ORDER[index:] + _ORDER[:index] for index in range(len(_ORDER))),
    ]
]


@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("key_scope", ["entry", "agent"])
async def test_live_cross_provider_auth_fallback(chain, key_scope, monkeypatch):
    backup_env = PROVIDERS[chain[-1]][1]
    if not os.getenv(backup_env):
        pytest.skip(f"Requires {backup_env} in the environment or .env")

    # SDK clients belong to this test's event loop and are closed afterward.
    monkeypatch.setattr(clients, "_CLIENT_CACHE", {})
    invalid_key = "invalid-fallback-test-key"
    failures = []

    def record_failure(exc):
        failures.append(exc)
        return True

    entries = [
        ModelEntry(
            PROVIDERS[provider][0],
            max_retries=0,
            api_key=(
                invalid_key
                if index < len(chain) - 1 and (index > 0 or key_scope == "entry")
                else None
            ),
        )
        for index, provider in enumerate(chain)
    ]
    agent = Agent(
        name="fallback_probe",
        model=FallbackModel(*entries, fallback_on=record_failure),
        api_key=invalid_key if key_scope == "agent" else None,
        max_tokens=1024,
        tracing_provider=None,
    )
    try:
        result = await asyncio.wait_for(agent(prompt="Reply only with OK").collect(), timeout=90)
        assert result.error is None, result.error
        assert result.status.code == "success"
        assert "ok" in result.output.collect_text().lower()
        assert len(failures) == len(chain) - 1
        # Google reports invalid API keys as 400; other providers use 401/403.
        # Connection errors or invalid model names must not masquerade as auth probes.
        for exc in failures:
            assert getattr(exc, "status_code", None) in (400, 401, 403), type(exc).__name__
            assert any(word in str(exc).lower() for word in ("key", "auth", "credential")), str(exc)
        assert [key[3] for key in clients._CLIENT_CACHE] == list(chain)
    finally:
        for client in clients._CLIENT_CACHE.values():
            await client.close()
