import asyncio
import json

import pytest
from timbal.state.context import RunContext
from timbal.state.tracing.providers.base import Exporter, TracingProvider
from timbal.state.tracing.providers.sqlite import SqliteTracingProvider
from timbal.state.tracing.span import Span
from timbal.state.tracing.trace import Trace


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


class TestTracingProviderConfigured:
    """Tests for the base-class configured() factory (exercised via SqliteTracingProvider)."""

    def test_returns_a_subclass(self):
        sub = SqliteTracingProvider.configured()
        assert issubclass(sub, SqliteTracingProvider)
        assert issubclass(sub, TracingProvider)

    def test_sets_class_level_attributes(self, tmp_path):
        from pathlib import Path
        sub = SqliteTracingProvider.configured(_path=Path("my.db"), _lock=None)
        assert sub._path == Path("my.db")
        assert sub._lock is None

    def test_does_not_mutate_original(self):
        from pathlib import Path
        original_path = SqliteTracingProvider._path
        SqliteTracingProvider.configured(_path=Path("other.db"))
        assert SqliteTracingProvider._path == original_path

    def test_two_subclasses_are_independent(self):
        from pathlib import Path
        a = SqliteTracingProvider.configured(_path=Path("a.db"))
        b = SqliteTracingProvider.configured(_path=Path("b.db"))
        assert a._path == Path("a.db")
        assert b._path == Path("b.db")

    def test_subclass_name_preserved(self):
        sub = SqliteTracingProvider.configured()
        assert sub.__name__ == "SqliteTracingProvider"

    def test_subclass_inherits_methods(self):
        sub = SqliteTracingProvider.configured()
        assert hasattr(sub, "put")
        assert hasattr(sub, "get")
        assert hasattr(sub, "configured")


class TestExporter:
    @pytest.mark.asyncio
    async def test_exporter_called_after_store(self, tmp_path):
        calls = []

        class RecordingExporter(Exporter):
            async def export(self, run_context):
                calls.append(run_context.id)

        provider = SqliteTracingProvider.configured(
            _path=tmp_path / "traces.db",
            _exporters=[RecordingExporter()],
        )
        await provider.put(_make_run_context(provider, "run-1"))

        assert calls == ["run-1"]
        assert (tmp_path / "traces.db").exists()

    @pytest.mark.asyncio
    async def test_multiple_exporters_all_called(self, tmp_path):
        calls = []

        class RecordingExporter(Exporter):
            def __init__(self, name):
                self.name = name

            async def export(self, run_context):
                calls.append(self.name)

        provider = SqliteTracingProvider.configured(
            _path=tmp_path / "traces.db",
            _exporters=[RecordingExporter("a"), RecordingExporter("b")],
        )
        await provider.put(_make_run_context(provider, "run-1"))

        assert calls == ["a", "b"]

    @pytest.mark.asyncio
    async def test_exporter_exception_does_not_break_run(self, tmp_path):
        class BrokenExporter(Exporter):
            async def export(self, run_context):
                raise RuntimeError("exporter blew up")

        provider = SqliteTracingProvider.configured(
            _path=tmp_path / "traces.db",
            _exporters=[BrokenExporter()],
        )
        await provider.put(_make_run_context(provider, "run-1"))
        assert (tmp_path / "traces.db").exists()

    @pytest.mark.asyncio
    async def test_exporter_receives_run_context(self, tmp_path):
        received = []

        class CapturingExporter(Exporter):
            async def export(self, run_context):
                received.append((run_context.id, list(run_context._trace.keys())))

        provider = SqliteTracingProvider.configured(
            _path=tmp_path / "traces.db",
            _exporters=[CapturingExporter()],
        )
        await provider.put(_make_run_context(provider, "run-42"))

        assert received == [("run-42", ["c1"])]

    @pytest.mark.asyncio
    async def test_no_exporters_by_default(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        await provider.put(_make_run_context(provider, "run-1"))
        assert (tmp_path / "traces.db").exists()

    @pytest.mark.asyncio
    async def test_exporters_isolated_between_providers(self, tmp_path):
        calls = []

        class RecordingExporter(Exporter):
            async def export(self, run_context):
                calls.append(run_context.id)

        provider_a = SqliteTracingProvider.configured(
            _path=tmp_path / "a.db",
            _exporters=[RecordingExporter()],
        )
        provider_b = SqliteTracingProvider.configured(_path=tmp_path / "b.db")

        await provider_a.put(_make_run_context(provider_a, "run-a"))
        await provider_b.put(_make_run_context(provider_b, "run-b"))

        assert calls == ["run-a"]


class TestSqliteTracingProviderPut:
    @pytest.mark.asyncio
    async def test_creates_db_and_writes_record(self, tmp_path):
        path = tmp_path / "traces.db"
        provider = SqliteTracingProvider.configured(_path=path)
        ctx = _make_run_context(provider, "run-1")

        await provider.put(ctx)

        assert path.exists()
        child = _make_run_context(provider, "child", parent_id="run-1")
        trace = await provider.get(child)
        assert isinstance(trace, Trace)
        assert trace["c1"].call_id == "c1"

    @pytest.mark.asyncio
    async def test_multiple_runs_coexist(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        for i in range(3):
            await provider.put(_make_run_context(provider, f"run-{i}"))

        for i in range(3):
            trace = await provider.get(_make_run_context(provider, "child", parent_id=f"run-{i}"))
            assert trace is not None

    @pytest.mark.asyncio
    async def test_parent_id_is_stored(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        await provider.put(_make_run_context(provider, "child", parent_id="parent"))

        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "traces.db"))
        row = conn.execute("SELECT parent_id FROM runs WHERE run_id = 'child'").fetchone()
        conn.close()
        assert row[0] == "parent"

    @pytest.mark.asyncio
    async def test_raises_without_path(self):
        provider = SqliteTracingProvider.configured()
        ctx = _make_run_context(provider, "run-x")
        with pytest.raises(RuntimeError, match="_path is not set"):
            await provider.put(ctx)

    @pytest.mark.asyncio
    async def test_two_providers_write_to_separate_files(self, tmp_path):
        provider_a = SqliteTracingProvider.configured(_path=tmp_path / "a.db")
        provider_b = SqliteTracingProvider.configured(_path=tmp_path / "b.db")

        await provider_a.put(_make_run_context(provider_a, "run-a"))
        await provider_b.put(_make_run_context(provider_b, "run-b"))

        trace_a = await provider_a.get(_make_run_context(provider_a, "c", parent_id="run-a"))
        trace_b = await provider_b.get(_make_run_context(provider_b, "c", parent_id="run-b"))
        assert trace_a is not None
        assert trace_b is not None
        # Cross-provider isolation
        assert await provider_a.get(_make_run_context(provider_a, "c", parent_id="run-b")) is None

    @pytest.mark.asyncio
    async def test_concurrent_puts_no_corruption(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")

        await asyncio.gather(*[
            provider.put(_make_run_context(provider, f"run-{i}"))
            for i in range(20)
        ])

        for i in range(20):
            trace = await provider.get(_make_run_context(provider, "c", parent_id=f"run-{i}"))
            assert trace is not None, f"run-{i} missing after concurrent writes"


class TestSqliteTracingProviderGet:
    @pytest.mark.asyncio
    async def test_returns_none_when_db_missing(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "nonexistent.db")
        assert await provider.get(_make_run_context(provider, "child", parent_id="parent")) is None

    @pytest.mark.asyncio
    async def test_returns_none_when_parent_id_not_in_db(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        await provider.put(_make_run_context(provider, "run-a"))
        assert await provider.get(_make_run_context(provider, "child", parent_id="run-missing")) is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_parent_id(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        await provider.put(_make_run_context(provider, "run-a"))
        assert await provider.get(_make_run_context(provider, "child", parent_id=None)) is None

    @pytest.mark.asyncio
    async def test_retrieves_correct_trace(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        for run_id in ["run-a", "run-b", "run-c"]:
            await provider.put(_make_run_context(provider, run_id))

        result = await provider.get(_make_run_context(provider, "child", parent_id="run-b"))

        assert isinstance(result, Trace)
        assert result["c1"].call_id == "c1"

    @pytest.mark.asyncio
    async def test_roundtrip_put_then_get(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        await provider.put(_make_run_context(provider, "parent-run"))

        retrieved = await provider.get(_make_run_context(provider, "child-run", parent_id="parent-run"))

        assert retrieved is not None
        assert retrieved["c1"].path == "test.step"
        assert retrieved["c1"].elapsed == 10


class TestSqliteUpdateInPlace:
    """Tests for the upsert (_store) behaviour.

    Providers emit intermediate snapshots on each span completion; _store must
    overwrite the existing row for a run_id rather than inserting duplicates.
    """

    @pytest.mark.asyncio
    async def test_same_run_id_twice_produces_one_row(self, tmp_path):
        import sqlite3 as _sqlite3

        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)
        await provider.put(ctx)

        conn = _sqlite3.connect(str(tmp_path / "t.db"))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 1

    @pytest.mark.asyncio
    async def test_same_run_id_many_times_produces_one_row(self, tmp_path):
        import sqlite3 as _sqlite3

        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        ctx = _make_run_context(provider, "run-1")
        for _ in range(10):
            await provider.put(ctx)

        conn = _sqlite3.connect(str(tmp_path / "t.db"))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 1

    @pytest.mark.asyncio
    async def test_update_does_not_disturb_other_run_ids(self, tmp_path):
        import sqlite3 as _sqlite3

        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        for rid in ["run-a", "run-b", "run-c"]:
            await provider.put(_make_run_context(provider, rid))
        await provider.put(_make_run_context(provider, "run-b"))
        await provider.put(_make_run_context(provider, "run-b"))

        conn = _sqlite3.connect(str(tmp_path / "t.db"))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 3

    @pytest.mark.asyncio
    async def test_updated_row_has_latest_content(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")

        ctx1 = _make_run_context(provider, "run-1")
        await provider.put(ctx1)

        ctx2 = _make_run_context(provider, "run-1")
        span2 = ctx2._trace[list(ctx2._trace.keys())[0]]
        span2._input_dump = {"x": 99}
        await provider.put(ctx2)

        trace = await provider.get(_make_run_context(provider, "child", parent_id="run-1"))
        assert trace is not None
        assert trace["c1"].input["x"] == 99

    @pytest.mark.asyncio
    async def test_concurrent_puts_same_run_id_exactly_one_row(self, tmp_path):
        import sqlite3 as _sqlite3

        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        ctx = _make_run_context(provider, "run-concurrent")

        await asyncio.gather(*[provider.put(ctx) for _ in range(20)])

        conn = _sqlite3.connect(str(tmp_path / "t.db"))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 1

    @pytest.mark.asyncio
    async def test_concurrent_mixed_new_and_update(self, tmp_path):
        import sqlite3 as _sqlite3

        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        for i in range(5):
            await provider.put(_make_run_context(provider, f"run-{i}"))

        await asyncio.gather(*[
            provider.put(_make_run_context(provider, f"run-{i}"))
            for _ in range(10)
            for i in range(5)
        ])

        conn = _sqlite3.connect(str(tmp_path / "t.db"))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 5

    @pytest.mark.asyncio
    async def test_real_agent_run_produces_one_row_per_run(self, tmp_path):
        import sqlite3 as _sqlite3

        from timbal import Agent
        from timbal.core.test_model import TestModel

        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        agent = Agent(
            name="sqlite_agent",
            model=TestModel(responses=["ok"]),
            tracing_provider=provider,
        )

        out = await agent(prompt="hello").collect()
        assert out.status.code == "success"

        conn = _sqlite3.connect(str(tmp_path / "t.db"))
        rows = conn.execute("SELECT run_id FROM runs").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == out.run_id

    @pytest.mark.asyncio
    async def test_get_returns_correct_data_after_multiple_puts(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")

        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)
        span = ctx._trace[list(ctx._trace.keys())[0]]
        span._output_dump = {"final": True}
        await provider.put(ctx)

        child_ctx = _make_run_context(provider, "child", parent_id="run-1")
        trace = await provider.get(child_ctx)
        assert trace is not None
        assert trace[list(trace.keys())[0]].output == {"final": True}


class TestSqliteMemoryIntegration:
    """Integration tests for multi-turn session memory via SQLite."""

    @pytest.mark.asyncio
    async def test_memory_grows_correctly_across_turns(self, tmp_path):
        import sqlite3 as _sqlite3

        from timbal import Agent
        from timbal.core.test_model import TestModel

        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        agent = Agent(
            name="sqlite_multiturn",
            model=TestModel(responses=["ok"]),
            tracing_provider=provider,
        )

        run_id = None
        for i in range(4):
            out = await agent(prompt=f"msg {i}", parent_id=run_id).collect()
            run_id = out.run_id

        conn = _sqlite3.connect(str(tmp_path / "traces.db"))
        rows = conn.execute("SELECT run_id, spans FROM runs ORDER BY stored_at").fetchall()
        conn.close()

        for turn, (_, spans_json) in enumerate(rows):
            spans = json.loads(spans_json)
            agent_span = next(s for s in spans if s["path"] == "sqlite_multiturn")
            memory = agent_span.get("memory", [])
            expected = (turn + 1) * 2  # user + assistant per turn
            assert len(memory) == expected, (
                f"Turn {turn + 1}: expected {expected} messages, got {len(memory)}"
            )

    @pytest.mark.asyncio
    async def test_roundtrip_with_parent_id_session_chain(self, tmp_path):
        from timbal import Agent
        from timbal.core.test_model import TestModel

        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        agent = Agent(
            name="chain_agent",
            model=TestModel(responses=["resp1", "resp2"]),
            tracing_provider=provider,
        )

        out1 = await agent(prompt="first").collect()
        out2 = await agent(prompt="second", parent_id=out1.run_id).collect()

        assert out2.status.code == "success"
        # Turn 2 should have loaded turn 1's memory (2 msgs) and added 2 more
        trace2 = provider
        child_ctx = _make_run_context(provider, "verify", parent_id=out2.run_id)
        trace = await provider.get(child_ctx)
        assert trace is not None
        agent_span = next(s for s in trace.as_records() if s.path == "chain_agent")
        assert len(agent_span.memory) == 4
