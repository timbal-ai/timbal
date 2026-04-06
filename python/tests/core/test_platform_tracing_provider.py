"""Integration tests for PlatformTracingProvider.

These tests run live against the Timbal API and require a populated
.env.test_platform_tracing file in this directory with:

    TIMBAL_API_HOST=api.timbal.ai
    TIMBAL_API_KEY=<your key>
    TIMBAL_ORG_ID=<your org id>
    TIMBAL_APP_ID=<app id to record traces under>
    TIMBAL_VERSION_ID=<version id>   # optional

All tests are skipped automatically if the file is absent.
Run with:  uv run pytest python/tests/core/test_platform_tracing_provider.py -v
"""

import os
from pathlib import Path

import pytest

TEST_ENV_FILE = Path(__file__).parent / ".env.test_platform_tracing"
SKIP_REASON = (
    "Platform tracing integration tests require "
    ".env.test_platform_tracing in tests/core/. "
    "See the module docstring for required variables."
)


def _load_env() -> bool:
    if not TEST_ENV_FILE.exists():
        return False
    with open(TEST_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()
    return True


@pytest.mark.skipif(not TEST_ENV_FILE.exists(), reason=SKIP_REASON)
@pytest.mark.integration
class TestPlatformTracingProviderIntegration:
    """Live tests against the Timbal platform API.

    Each test method resets the environment so imports that read env vars at
    module load time are not affected by other test modules.
    """

    def setup_method(self):
        self._saved_env = os.environ.copy()
        os.environ.clear()
        _load_env()
        # Reset the config_loader module-level cache so it re-reads env vars.
        import timbal.state.config_loader as _cl
        _cl._cached_default_config = None
        _cl._default_config_resolved = False

    def teardown_method(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        import timbal.state.config_loader as _cl
        _cl._cached_default_config = None
        _cl._default_config_resolved = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_agent(name: str, **kwargs):
        from timbal import Agent
        from timbal.core.test_model import TestModel
        from timbal.state.tracing.providers.platform import PlatformTracingProvider
        model = kwargs.pop("model", TestModel(responses=["ok"]))
        return Agent(name=name, model=model, tracing_provider=PlatformTracingProvider.configured(), **kwargs)

    @staticmethod
    async def _get_trace(run_id: str):
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers.platform import PlatformTracingProvider
        child_ctx = RunContext(tracing_provider=PlatformTracingProvider.configured())
        object.__setattr__(child_ctx, "parent_id", run_id)
        return await PlatformTracingProvider.get(child_ctx)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_single_turn_trace_written(self):
        """A completed agent run appears in the platform as a retrievable trace."""
        agent = self._make_agent("platform_trace_agent")
        result = await agent(prompt="hello").collect()
        assert result.status.code == "success"

        trace = await self._get_trace(result.run_id)
        assert trace is not None, "Platform did not return a trace for the completed run"
        root_span = trace[trace._root_call_id]
        assert root_span.path == "platform_trace_agent"

    @pytest.mark.asyncio
    async def test_multi_turn_memory_survives_round_trip(self):
        """Two-turn session: turn 2 loads turn 1 memory from the platform.

        Verifies the full path: write → platform API → read back → agent uses
        prior conversation in its LLM call.
        """
        from timbal.core.test_model import TestModel

        turn_count = 0

        def _handler(messages):
            nonlocal turn_count
            turn_count += 1
            return f"response {turn_count}"

        agent = self._make_agent("platform_memory_agent", model=TestModel(handler=_handler))

        out1 = await agent(prompt="turn 1").collect()
        assert out1.status.code == "success"

        out2 = await agent(prompt="turn 2", parent_id=out1.run_id).collect()
        assert out2.status.code == "success"

        # A larger input token count on turn 2 confirms prior conversation was in context.
        model_key = next(k for k in out2.usage if "input" in k)
        assert out2.usage[model_key] > out1.usage[model_key], (
            "Turn 2 input token count should exceed turn 1 — prior conversation must be in context"
        )

    @pytest.mark.asyncio
    async def test_intermediate_spans_overwrite_not_accumulate(self):
        """Repeated _store calls for the same run_id update the record, not duplicate it.

        An agent run triggers at least 2 _store calls (LLM span + agent span).
        The platform's PATCH semantics mean the final write wins — the retrieved
        trace must have the complete agent span memory (user + assistant), not
        just the partial snapshot written before _append_memory ran.
        """
        from timbal.core.test_model import TestModel

        agent = self._make_agent("overwrite_test_agent", model=TestModel(responses=["done"]))
        result = await agent(prompt="test overwrite").collect()
        assert result.status.code == "success"

        trace = await self._get_trace(result.run_id)
        assert trace is not None

        agent_span = next(
            (s for s in trace.as_records() if s.path == "overwrite_test_agent"),
            None,
        )
        assert agent_span is not None
        memory = agent_span.memory
        assert memory is not None and len(memory) == 2, (
            f"Expected [user, assistant] in agent span memory, got {len(memory) if memory else 0}"
        )
