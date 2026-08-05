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

    @pytest.mark.asyncio
    async def test_sqlite_reloaded_root_span_uses_dict_status_for_memory_chain(self, tmp_path):
        """Regression: SQLite get() builds Span with status as a dict, not RunStatus;
        parent_id + turn 2 must not raise (AttributeError) and must succeed.

        Mirrors the JSONL regression (resolve_memory previously read
        previous_span.status.code and crashed when status was a dict).
        Same risk applies to SQLite: spans are stored as JSON in the spans
        column, so the reload path also produces dict-shaped status."""
        from timbal import Agent
        from timbal.core.test_model import TestModel

        provider = SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
        agent = Agent(
            name="chain_agent",
            model=TestModel(responses=["r1", "r2"]),
            tracing_provider=provider,
        )
        out1 = await agent(prompt="m0").collect()
        out2 = await agent(prompt="m1", parent_id=out1.run_id).collect()
        assert out2.status.code == "success", out2.error
        assert out2.error is None

    @pytest.mark.asyncio
    async def test_memory_dump_correct_after_sqlite_reload(self, tmp_path):
        """_memory_dump on turn 2 must equal a fresh dump of the full memory.

        Mirrors the JSONL ``test_memory_dump_correct_after_jsonl_reload``:
        proves the incremental ``_prev_memory_dump`` optimisation in
        ``Agent.resolve_memory`` is transparent when spans are reconstructed
        from the SQLite ``spans`` JSON column rather than carried in memory.
        """
        import sqlite3 as _sqlite3

        from timbal import Agent
        from timbal.core.test_model import TestModel
        from timbal.types.message import Message
        from timbal.utils import dump

        db_path = tmp_path / "traces.db"
        provider = SqliteTracingProvider.configured(_path=db_path)

        turn_count = 0

        def counting_handler(messages):  # noqa: ARG001 — TestModel handler signature
            nonlocal turn_count
            turn_count += 1
            return f"response {turn_count}"

        agent = Agent(
            name="sqlite_memdump",
            model=TestModel(handler=counting_handler),
            tracing_provider=provider,
        )

        out1 = await agent(prompt="message 0").collect()
        assert db_path.exists(), "SQLite db should be created after turn 1"

        out2 = await agent(prompt="message 1", parent_id=out1.run_id).collect()

        conn = _sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT spans FROM runs WHERE run_id = ?",
                (out2.run_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None, "turn 2 row should exist in SQLite"
        spans = json.loads(row[0])
        agent_span = next(s for s in spans if s["path"] == "sqlite_memdump")
        stored_memory = agent_span.get("memory")

        assert stored_memory is not None, "memory should be persisted in turn 2 span"
        assert len(stored_memory) == 4, (
            f"Expected 4 messages (user0, assistant0, user1, assistant1), got {len(stored_memory)}"
        )

        expected = await dump([Message.validate(m) for m in stored_memory])
        assert stored_memory == expected, "_memory_dump stored in SQLite does not match full re-dump"
