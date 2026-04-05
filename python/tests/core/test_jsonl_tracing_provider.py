import asyncio
import json
from pathlib import Path

import pytest
from timbal.state.context import RunContext
from timbal.state.tracing.providers.base import TracingProvider
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider
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

    def test_in_memory_isolated_storage(self):
        a = InMemoryTracingProvider.configured(_storage={})
        b = InMemoryTracingProvider.configured(_storage={})
        a._storage["key"] = "value"
        assert "key" not in b._storage
        assert "key" not in InMemoryTracingProvider._storage


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
