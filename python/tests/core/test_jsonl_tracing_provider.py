import asyncio
import json
from pathlib import Path

import pytest
from timbal.state.context import RunContext
from timbal.state.tracing.providers.base import Exporter, TracingProvider
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
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
    """Tests for the base-class configured() factory."""

    def test_returns_a_subclass(self):
        sub = JsonlTracingProvider.configured()
        assert issubclass(sub, JsonlTracingProvider)
        assert issubclass(sub, TracingProvider)

    def test_sets_class_level_attributes(self):
        sub = JsonlTracingProvider.configured(_path=Path("my.jsonl"), _lock=None)
        assert sub._path == Path("my.jsonl")
        assert sub._lock is None

    def test_does_not_mutate_original(self):
        original_path = JsonlTracingProvider._path
        JsonlTracingProvider.configured(_path=Path("other.jsonl"))
        assert JsonlTracingProvider._path == original_path

    def test_two_subclasses_are_independent(self):
        a = JsonlTracingProvider.configured(_path=Path("a.jsonl"))
        b = JsonlTracingProvider.configured(_path=Path("b.jsonl"))
        assert a._path == Path("a.jsonl")
        assert b._path == Path("b.jsonl")

    def test_subclass_name_preserved(self):
        sub = JsonlTracingProvider.configured(_path=Path("x.jsonl"))
        assert sub.__name__ == "JsonlTracingProvider"

    def test_subclass_inherits_methods(self):
        sub = JsonlTracingProvider.configured(_path=Path("x.jsonl"))
        assert hasattr(sub, "put")
        assert hasattr(sub, "get")
        assert hasattr(sub, "configured")


class TestJsonlTracingProviderPut:
    @pytest.mark.asyncio
    async def test_creates_file_and_writes_record(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=path)
        ctx = _make_run_context(provider, "run-1")

        await provider.put(ctx)

        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["run_id"] == "run-1"
        assert record["parent_id"] is None
        assert isinstance(record["spans"], list)
        assert len(record["spans"]) == 1
        assert record["spans"][0]["call_id"] == "c1"

    @pytest.mark.asyncio
    async def test_appends_multiple_runs(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=path)

        for i in range(3):
            await provider.put(_make_run_context(provider, f"run-{i}"))

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(l)["run_id"] for l in lines] == ["run-0", "run-1", "run-2"]

    @pytest.mark.asyncio
    async def test_parent_id_is_written(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        await provider.put(_make_run_context(provider, "child", parent_id="parent"))

        record = json.loads((tmp_path / "traces.jsonl").read_text().strip())
        assert record["parent_id"] == "parent"

    @pytest.mark.asyncio
    async def test_raises_without_path(self):
        provider = JsonlTracingProvider.configured()  # no _path
        ctx = _make_run_context(provider, "run-x")
        with pytest.raises(RuntimeError, match="_path is not set"):
            await provider.put(ctx)

    @pytest.mark.asyncio
    async def test_two_providers_write_to_separate_files(self, tmp_path):
        provider_a = JsonlTracingProvider.configured(_path=tmp_path / "a.jsonl")
        provider_b = JsonlTracingProvider.configured(_path=tmp_path / "b.jsonl")

        await provider_a.put(_make_run_context(provider_a, "run-a"))
        await provider_b.put(_make_run_context(provider_b, "run-b"))

        assert json.loads((tmp_path / "a.jsonl").read_text())["run_id"] == "run-a"
        assert json.loads((tmp_path / "b.jsonl").read_text())["run_id"] == "run-b"

    @pytest.mark.asyncio
    async def test_concurrent_puts_no_corruption(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=path)

        await asyncio.gather(*[
            provider.put(_make_run_context(provider, f"run-{i}"))
            for i in range(20)
        ])

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 20
        for line in lines:
            record = json.loads(line)
            assert "run_id" in record
            assert "spans" in record
        assert {json.loads(l)["run_id"] for l in lines} == {f"run-{i}" for i in range(20)}


class TestJsonlTracingProviderGet:
    @pytest.mark.asyncio
    async def test_returns_none_when_file_missing(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "nonexistent.jsonl")
        assert await provider.get(_make_run_context(provider, "child", parent_id="parent")) is None

    @pytest.mark.asyncio
    async def test_returns_none_when_parent_id_not_in_file(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        await provider.put(_make_run_context(provider, "run-a"))
        assert await provider.get(_make_run_context(provider, "child", parent_id="run-missing")) is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_parent_id(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        await provider.put(_make_run_context(provider, "run-a"))
        assert await provider.get(_make_run_context(provider, "child", parent_id=None)) is None

    @pytest.mark.asyncio
    async def test_retrieves_correct_trace(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        for run_id in ["run-a", "run-b", "run-c"]:
            await provider.put(_make_run_context(provider, run_id))

        result = await provider.get(_make_run_context(provider, "child", parent_id="run-b"))

        assert isinstance(result, Trace)
        assert result["c1"].call_id == "c1"

    @pytest.mark.asyncio
    async def test_roundtrip_put_then_get(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        await provider.put(_make_run_context(provider, "parent-run"))

        retrieved = await provider.get(_make_run_context(provider, "child-run", parent_id="parent-run"))

        assert retrieved is not None
        assert retrieved["c1"].path == "test.step"
        assert retrieved["c1"].elapsed == 10


class TestExporter:
    """Tests for the Exporter attachment mechanism."""

    @pytest.mark.asyncio
    async def test_exporter_called_after_store(self, tmp_path):
        calls = []

        class RecordingExporter(Exporter):
            async def export(self, run_context):
                calls.append(run_context.id)

        provider = JsonlTracingProvider.configured(
            _path=tmp_path / "traces.jsonl",
            _exporters=[RecordingExporter()],
        )
        await provider.put(_make_run_context(provider, "run-1"))

        assert calls == ["run-1"]
        assert (tmp_path / "traces.jsonl").exists()  # store also ran

    @pytest.mark.asyncio
    async def test_multiple_exporters_all_called(self, tmp_path):
        calls = []

        class RecordingExporter(Exporter):
            def __init__(self, name):
                self.name = name

            async def export(self, run_context):
                calls.append(self.name)

        provider = JsonlTracingProvider.configured(
            _path=tmp_path / "traces.jsonl",
            _exporters=[RecordingExporter("a"), RecordingExporter("b")],
        )
        await provider.put(_make_run_context(provider, "run-1"))

        assert calls == ["a", "b"]

    @pytest.mark.asyncio
    async def test_exporter_exception_does_not_break_run(self, tmp_path):
        class BrokenExporter(Exporter):
            async def export(self, run_context):
                raise RuntimeError("exporter blew up")

        provider = JsonlTracingProvider.configured(
            _path=tmp_path / "traces.jsonl",
            _exporters=[BrokenExporter()],
        )
        # Should not raise
        await provider.put(_make_run_context(provider, "run-1"))
        assert (tmp_path / "traces.jsonl").exists()

    @pytest.mark.asyncio
    async def test_exporter_receives_run_context(self, tmp_path):
        received = []

        class CapturingExporter(Exporter):
            async def export(self, run_context):
                received.append((run_context.id, list(run_context._trace.keys())))

        provider = JsonlTracingProvider.configured(
            _path=tmp_path / "traces.jsonl",
            _exporters=[CapturingExporter()],
        )
        await provider.put(_make_run_context(provider, "run-42"))

        assert received == [("run-42", ["c1"])]

    @pytest.mark.asyncio
    async def test_no_exporters_by_default(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        # Just verify put() works fine with the default empty list
        await provider.put(_make_run_context(provider, "run-1"))
        assert (tmp_path / "traces.jsonl").exists()

    @pytest.mark.asyncio
    async def test_exporters_isolated_between_providers(self, tmp_path):
        calls = []

        class RecordingExporter(Exporter):
            async def export(self, run_context):
                calls.append(run_context.id)

        provider_a = JsonlTracingProvider.configured(
            _path=tmp_path / "a.jsonl",
            _exporters=[RecordingExporter()],
        )
        provider_b = JsonlTracingProvider.configured(_path=tmp_path / "b.jsonl")

        await provider_a.put(_make_run_context(provider_a, "run-a"))
        await provider_b.put(_make_run_context(provider_b, "run-b"))

        assert calls == ["run-a"]  # provider_b has no exporters


class TestMemoryDumpJsonlIntegration:
    """Integration tests for the _memory_dump incremental optimization with JSONL provider.

    These tests exercise the full serialize-to-disk → reload → re-run cycle, which is
    the critical path for the _prev_memory_dump optimization. Unlike InMemoryTracingProvider
    (where previous_span._memory_dump is the live PrivateAttr), the JSONL reload path
    reconstructs Span objects from JSON — _memory_dump is not a PrivateAttr after reload,
    so span.memory (the stored dicts) is used instead.
    """

    @pytest.mark.asyncio
    async def test_memory_dump_correct_after_jsonl_reload(self, tmp_path):
        """Full cycle: run turn 1 → persist to JSONL → reload → run turn 2.

        Verifies that _memory_dump on turn 2 equals a fresh dump of the full memory,
        proving the incremental optimization is transparent when spans are reloaded
        from disk rather than kept in memory.
        """
        from timbal import Agent
        from timbal.core.test_model import TestModel
        from timbal.types.message import Message
        from timbal.utils import dump

        traces_path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces_path)

        turn_count = 0

        def counting_handler(messages):
            nonlocal turn_count
            turn_count += 1
            return f"response {turn_count}"

        agent = Agent(
            name="jsonl_agent",
            model=TestModel(handler=counting_handler),
            tracing_provider=provider,
        )

        # Turn 1: no prior history
        out1 = await agent(prompt="message 0").collect()
        assert traces_path.exists(), "JSONL file should be created after turn 1"

        # Turn 2: provider loads turn 1 from disk (JSONL reload path)
        out2 = await agent(prompt="message 1", parent_id=out1.run_id).collect()

        # Read back the turn 2 trace from disk and verify _memory_dump integrity
        records = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
        turn2_record = next(r for r in records if r["run_id"] == out2.run_id)

        agent_span = next(
            s for s in turn2_record["spans"] if s["path"] == "jsonl_agent"
        )

        stored_memory = agent_span.get("memory")
        assert stored_memory is not None, "memory should be persisted in turn 2 span"
        assert len(stored_memory) == 4, (
            f"Expected 4 messages (user0, assistant0, user1, assistant1), got {len(stored_memory)}"
        )

        # Re-dump from scratch and compare — must match what was stored
        expected = await dump([Message.validate(m) for m in stored_memory])
        assert stored_memory == expected, "_memory_dump stored on disk does not match full re-dump"

    @pytest.mark.asyncio
    async def test_jsonl_reloaded_root_span_uses_dict_status_for_memory_chain(self, tmp_path):
        """Regression: JSONL get() builds Span with status as a dict, not RunStatus;
        parent_id + turn 2 must not raise (AttributeError) and must succeed.

        (Previously resolve_memory read previous_span.status.code and crashed.)"""
        from timbal import Agent
        from timbal.core.test_model import TestModel

        traces_path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces_path)
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
    async def test_memory_grows_correctly_across_turns(self, tmp_path):
        """Each turn adds exactly 2 messages (user + assistant) to the stored memory."""
        from timbal import Agent
        from timbal.core.test_model import TestModel

        traces_path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces_path)

        agent = Agent(
            name="grow_agent",
            model=TestModel(responses=["ok"]),
            tracing_provider=provider,
        )

        run_id = None
        for i in range(4):
            out = await agent(prompt=f"msg {i}", parent_id=run_id).collect()
            run_id = out.run_id

        records = {
            json.loads(line)["run_id"]: json.loads(line)
            for line in traces_path.read_text().splitlines()
            if line.strip()
        }

        for turn, run_id in enumerate(records):
            agent_span = next(
                s for s in records[run_id]["spans"] if s["path"] == "grow_agent"
            )
            memory = agent_span.get("memory", [])
            expected_count = (turn + 1) * 2  # user + assistant per turn
            assert len(memory) == expected_count, (
                f"Turn {turn + 1}: expected {expected_count} messages, got {len(memory)}"
            )


class TestJsonlUpdateInPlace:
    """Tests for the update-in-place _store logic.

    Providers emit intermediate snapshots as each span completes; _store must
    overwrite the existing line for a run_id rather than appending duplicates.
    """

    @pytest.mark.asyncio
    async def test_same_run_id_twice_produces_one_line(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "t.jsonl")
        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)
        await provider.put(ctx)  # second intermediate snapshot

        lines = (tmp_path / "t.jsonl").read_text().splitlines()
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_same_run_id_many_times_produces_one_line(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "t.jsonl")
        ctx = _make_run_context(provider, "run-1")
        for _ in range(10):
            await provider.put(ctx)

        lines = (tmp_path / "t.jsonl").read_text().splitlines()
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_update_does_not_disturb_other_run_ids(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "t.jsonl")
        # Write three distinct runs
        for rid in ["run-a", "run-b", "run-c"]:
            await provider.put(_make_run_context(provider, rid))
        # Now update run-b twice
        await provider.put(_make_run_context(provider, "run-b"))
        await provider.put(_make_run_context(provider, "run-b"))

        lines = (tmp_path / "t.jsonl").read_text().splitlines()
        assert len(lines) == 3
        run_ids = [json.loads(l)["run_id"] for l in lines]
        assert run_ids == ["run-a", "run-b", "run-c"]

    @pytest.mark.asyncio
    async def test_updated_line_has_latest_content(self, tmp_path):
        """The stored record reflects the most recent put(), not the first one."""
        path = tmp_path / "t.jsonl"
        provider = JsonlTracingProvider.configured(_path=path)

        ctx1 = _make_run_context(provider, "run-1")
        await provider.put(ctx1)

        # Build a second context for the same run_id but with a different span
        ctx2 = _make_run_context(provider, "run-1")
        span2 = ctx2._trace[list(ctx2._trace.keys())[0]]
        span2._input_dump = {"x": 99}
        await provider.put(ctx2)

        record = json.loads(path.read_text().strip())
        assert record["run_id"] == "run-1"
        # Only one line and it has the span from ctx2 (input x=99)
        assert record["spans"][0]["input"]["x"] == 99

    @pytest.mark.asyncio
    async def test_update_skips_malformed_json_lines(self, tmp_path):
        """Corrupt lines in the file are skipped without crashing; update still works."""
        path = tmp_path / "t.jsonl"
        # Seed file with a bad line followed by a valid run-2 line
        path.write_text('NOT_JSON\n{"run_id": "run-2", "parent_id": null, "spans": []}\n')

        provider = JsonlTracingProvider.configured(_path=path)
        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)  # Should append (not found) without crashing

        lines = path.read_text().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[2])["run_id"] == "run-1"

    @pytest.mark.asyncio
    async def test_concurrent_puts_same_run_id_exactly_one_line(self, tmp_path):
        """Concurrent intermediate snapshots for the same run_id must not produce duplicates."""
        path = tmp_path / "t.jsonl"
        provider = JsonlTracingProvider.configured(_path=path)
        ctx = _make_run_context(provider, "run-concurrent")

        await asyncio.gather(*[provider.put(ctx) for _ in range(20)])

        lines = path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["run_id"] == "run-concurrent"

    @pytest.mark.asyncio
    async def test_concurrent_mixed_new_and_update(self, tmp_path):
        """Concurrent puts where some run_ids are new and some are updates stays consistent."""
        path = tmp_path / "t.jsonl"
        provider = JsonlTracingProvider.configured(_path=path)

        # First pass: create 5 run_ids
        for i in range(5):
            await provider.put(_make_run_context(provider, f"run-{i}"))

        # Now concurrently update all 5 ten times each
        await asyncio.gather(*[
            provider.put(_make_run_context(provider, f"run-{i}"))
            for _ in range(10)
            for i in range(5)
        ])

        lines = path.read_text().splitlines()
        assert len(lines) == 5
        assert {json.loads(l)["run_id"] for l in lines} == {f"run-{i}" for i in range(5)}

    @pytest.mark.asyncio
    async def test_real_agent_run_produces_one_line_per_run(self, tmp_path):
        """An actual Agent run triggers multiple _save_trace calls (one per span).
        The JSONL file must have exactly one line per run_id at the end.
        """
        from timbal import Agent
        from timbal.core.test_model import TestModel

        path = tmp_path / "t.jsonl"
        provider = JsonlTracingProvider.configured(_path=path)
        agent = Agent(
            name="snapshot_agent",
            model=TestModel(responses=["ok"]),
            tracing_provider=provider,
        )

        out = await agent(prompt="hello").collect()
        assert out.status.code == "success"

        lines = path.read_text().splitlines()
        # Despite multiple intermediate _save_trace calls, exactly one line per run
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["run_id"] == out.run_id

        # And the agent span must have the complete memory (user + assistant)
        agent_span = next(s for s in record["spans"] if s["path"] == "snapshot_agent")
        assert agent_span.get("memory") is not None
        assert len(agent_span["memory"]) == 2

    @pytest.mark.asyncio
    async def test_get_returns_correct_data_after_multiple_puts(self, tmp_path):
        """get() returns the final (most complete) snapshot, not an earlier one."""
        path = tmp_path / "t.jsonl"
        provider = JsonlTracingProvider.configured(_path=path)

        # Simulate two intermediate writes: first with partial spans, then complete
        ctx = _make_run_context(provider, "run-1")
        await provider.put(ctx)
        # Modify span to simulate a later more-complete snapshot
        span = ctx._trace[list(ctx._trace.keys())[0]]
        span._output_dump = {"final": True}
        await provider.put(ctx)

        # Now retrieve as a parent
        child_ctx = _make_run_context(provider, "child", parent_id="run-1")
        trace = await provider.get(child_ctx)
        assert trace is not None
        # Should be the last write (output has final=True)
        retrieved_span = trace[list(trace.keys())[0]]
        assert retrieved_span.output == {"final": True}


