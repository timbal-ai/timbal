import pytest
from timbal.state.context import RunContext
from timbal.state.tracing.providers.base import Exporter
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider
from timbal.state.tracing.span import Span


def _make_run_context(provider, run_id: str, parent_id: str | None = None) -> RunContext:
    """Create a minimal RunContext with a populated trace."""
    ctx = RunContext(tracing_provider=provider)
    object.__setattr__(ctx, "id", run_id)
    object.__setattr__(ctx, "parent_id", parent_id)
    span = Span(path="test.step", call_id="c1", parent_call_id=None, t0=0, t1=10)
    span._input_dump = {"x": 1}
    span._output_dump = {"result": 2}
    span._memory_dump = None
    span._session_dump = None
    ctx._trace["c1"] = span
    return ctx


class TestInMemoryTracingProvider:

    @pytest.mark.asyncio
    async def test_put_and_get_roundtrip(self):
        provider = InMemoryTracingProvider.configured(_storage={})
        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)

        child = _make_run_context(provider, "child", parent_id="run-1")
        trace = await provider.get(child)
        assert trace is not None
        assert "c1" in trace

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_run_id(self):
        provider = InMemoryTracingProvider.configured(_storage={})
        child = _make_run_context(provider, "child", parent_id="nonexistent")
        assert await provider.get(child) is None

    @pytest.mark.asyncio
    async def test_get_returns_none_when_no_parent_id(self):
        provider = InMemoryTracingProvider.configured(_storage={})
        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)
        assert await provider.get(ctx) is None  # ctx.parent_id is None

    @pytest.mark.asyncio
    async def test_same_run_id_put_twice_last_write_wins(self):
        """InMemory uses a dict — second put overwrites the first."""
        provider = InMemoryTracingProvider.configured(_storage={})
        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)

        # Mutate the live span to simulate a later, more-complete snapshot
        span = ctx._trace[list(ctx._trace.keys())[0]]
        span.output = {"updated": True}
        await provider.put(ctx)

        child = _make_run_context(provider, "child", parent_id="run-1")
        trace = await provider.get(child)
        retrieved_span = trace[list(trace.keys())[0]]
        assert retrieved_span.output == {"updated": True}

    @pytest.mark.asyncio
    async def test_storage_isolated_between_configured_instances(self):
        a = InMemoryTracingProvider.configured(_storage={})
        b = InMemoryTracingProvider.configured(_storage={})
        await a.put(_make_run_context(a, "run-a"))
        assert await b.get(_make_run_context(b, "child", parent_id="run-a")) is None

    @pytest.mark.asyncio
    async def test_multiple_runs_coexist_in_storage(self):
        provider = InMemoryTracingProvider.configured(_storage={})
        for i in range(5):
            await provider.put(_make_run_context(provider, f"run-{i}"))
        assert len(provider._storage) == 5

    @pytest.mark.asyncio
    async def test_real_agent_run_intermediate_snapshots_last_wins(self):
        """An Agent run emits multiple _save_trace calls; InMemory must end up with
        the final complete snapshot (user + assistant in memory), not an intermediate one.
        """
        from timbal import Agent
        from timbal.core.test_model import TestModel

        provider = InMemoryTracingProvider.configured(_storage={})
        agent = Agent(
            name="im_agent",
            model=TestModel(responses=["ok"]),
            tracing_provider=provider,
        )
        out = await agent(prompt="hello").collect()
        assert out.status.code == "success"

        trace = provider._storage[out.run_id]
        agent_span = next(s for s in trace.as_records() if s.path == "im_agent")
        memory = agent_span.memory
        assert memory is not None and len(memory) == 2, (
            f"Expected [user, assistant] in final snapshot, got {len(memory) if memory else 0}"
        )

    @pytest.mark.asyncio
    async def test_multi_turn_memory_correct(self):
        """Two-turn session: turn 2 must load turn 1 history correctly."""
        from timbal import Agent
        from timbal.core.test_model import TestModel

        provider = InMemoryTracingProvider.configured(_storage={})
        agent = Agent(
            name="im_multiturn",
            model=TestModel(responses=["resp1", "resp2"]),
            tracing_provider=provider,
        )

        out1 = await agent(prompt="msg1").collect()
        out2 = await agent(prompt="msg2", parent_id=out1.run_id).collect()
        assert out2.status.code == "success"

        trace2 = provider._storage[out2.run_id]
        agent_span = next(s for s in trace2.as_records() if s.path == "im_multiturn")
        memory = agent_span.memory
        assert len(memory) == 4  # user1, assistant1, user2, assistant2

    @pytest.mark.asyncio
    async def test_exporter_fires_on_put(self):
        calls = []

        class Rec(Exporter):
            async def export(self, run_context):
                calls.append(run_context.id)

        provider = InMemoryTracingProvider.configured(_storage={}, _exporters=[Rec()])
        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)
        assert calls == ["run-1"]
