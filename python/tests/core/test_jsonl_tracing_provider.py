import asyncio
import json

import pytest
from timbal.state.context import RunContext
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.state.tracing.span import Span
from timbal.state.tracing.trace import Trace


def _make_run_context(run_id: str, parent_id: str | None = None) -> RunContext:
    """Create a minimal RunContext with a populated trace."""
    ctx = RunContext(tracing_provider=JsonlTracingProvider)
    # Override auto-generated ids
    object.__setattr__(ctx, "id", run_id)
    object.__setattr__(ctx, "parent_id", parent_id)
    span = Span(path="test.step", call_id="c1", parent_call_id=None, t0=0, t1=10)
    span._input_dump = {"x": 1}
    span._output_dump = {"result": 2}
    span._memory_dump = None
    span._session_dump = None
    ctx._trace["c1"] = span
    return ctx


class TestJsonlTracingProviderPut:
    @pytest.mark.asyncio
    async def test_creates_file_and_writes_record(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        JsonlTracingProvider.configure(path)
        ctx = _make_run_context("run-1")

        await JsonlTracingProvider.put(ctx)

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
        JsonlTracingProvider.configure(path)

        for i in range(3):
            ctx = _make_run_context(f"run-{i}")
            await JsonlTracingProvider.put(ctx)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3
        ids = [json.loads(l)["run_id"] for l in lines]
        assert ids == ["run-0", "run-1", "run-2"]

    @pytest.mark.asyncio
    async def test_parent_id_is_written(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        JsonlTracingProvider.configure(path)
        ctx = _make_run_context("run-child", parent_id="run-parent")

        await JsonlTracingProvider.put(ctx)

        record = json.loads(path.read_text().strip())
        assert record["parent_id"] == "run-parent"

    @pytest.mark.asyncio
    async def test_raises_without_configure(self):
        JsonlTracingProvider._path = None
        ctx = _make_run_context("run-x")
        with pytest.raises(RuntimeError, match="not configured"):
            await JsonlTracingProvider.put(ctx)

    @pytest.mark.asyncio
    async def test_concurrent_puts_no_corruption(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        JsonlTracingProvider.configure(path)

        contexts = [_make_run_context(f"run-{i}") for i in range(20)]
        await asyncio.gather(*[JsonlTracingProvider.put(ctx) for ctx in contexts])

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 20
        # Every line must be valid JSON
        for line in lines:
            record = json.loads(line)
            assert "run_id" in record
            assert "spans" in record
        # All run_ids must be present (no duplicates, no missing)
        ids = {json.loads(l)["run_id"] for l in lines}
        assert ids == {f"run-{i}" for i in range(20)}


class TestJsonlTracingProviderGet:
    @pytest.mark.asyncio
    async def test_returns_none_when_file_missing(self, tmp_path):
        JsonlTracingProvider.configure(tmp_path / "nonexistent.jsonl")
        ctx = _make_run_context("child", parent_id="parent")
        result = await JsonlTracingProvider.get(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_parent_id_not_in_file(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        JsonlTracingProvider.configure(path)
        await JsonlTracingProvider.put(_make_run_context("run-a"))

        ctx = _make_run_context("child", parent_id="run-missing")
        result = await JsonlTracingProvider.get(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_parent_id(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        JsonlTracingProvider.configure(path)
        await JsonlTracingProvider.put(_make_run_context("run-a"))

        ctx = _make_run_context("child", parent_id=None)
        result = await JsonlTracingProvider.get(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieves_correct_trace(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        JsonlTracingProvider.configure(path)

        await JsonlTracingProvider.put(_make_run_context("run-a"))
        await JsonlTracingProvider.put(_make_run_context("run-b"))
        await JsonlTracingProvider.put(_make_run_context("run-c"))

        ctx = _make_run_context("child", parent_id="run-b")
        result = await JsonlTracingProvider.get(ctx)

        assert isinstance(result, Trace)
        assert "c1" in result
        assert result["c1"].call_id == "c1"

    @pytest.mark.asyncio
    async def test_roundtrip_put_then_get(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        JsonlTracingProvider.configure(path)

        parent_ctx = _make_run_context("parent-run")
        await JsonlTracingProvider.put(parent_ctx)

        child_ctx = _make_run_context("child-run", parent_id="parent-run")
        retrieved = await JsonlTracingProvider.get(child_ctx)

        assert retrieved is not None
        span = retrieved["c1"]
        assert span.path == "test.step"
        assert span.elapsed == 10
