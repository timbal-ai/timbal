"""``TracingProvider.get_strict``: ``None`` only for a run that is not stored; a failed read raises."""

import json
import pathlib
import sqlite3

import httpx
import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider
from timbal.state.tracing.providers.base import TracingProvider
from timbal.state.tracing.providers.platform import PlatformTracingProvider
from timbal.state.tracing.providers.sqlite import SqliteTracingProvider


async def stored_run(provider) -> str:
    agent = Agent(name="agent", model=TestModel(responses=["hi"]), tracing_provider=provider)
    return (await agent(prompt="hello").collect()).run_id


def lookup(provider, run_id: str) -> RunContext:
    return RunContext(parent_id=run_id, tracing_provider=provider)


class TestJsonl:
    async def test_finds_a_stored_run_and_none_for_a_missing_one(self, tmp_path):
        provider = JsonlTracingProvider.configured(path=tmp_path / "traces.jsonl")
        run_id = await stored_run(provider)
        assert await provider.get_strict(lookup(provider, run_id)) is not None
        assert await provider.get_strict(lookup(provider, "missing")) is None

    async def test_no_file_is_no_run(self, tmp_path):
        provider = JsonlTracingProvider.configured(path=tmp_path / "absent.jsonl")
        assert await provider.get_strict(lookup(provider, "any")) is None

    async def test_a_read_error_raises_where_get_returns_none(self, tmp_path, monkeypatch):
        provider = JsonlTracingProvider.configured(path=tmp_path / "traces.jsonl")
        run_id = await stored_run(provider)

        def broken_open(_self, *_args, **_kwargs):
            raise OSError("disk hiccup")

        monkeypatch.setattr(pathlib.Path, "open", broken_open)
        assert await provider.get(lookup(provider, run_id)) is None  # unchanged contract
        with pytest.raises(OSError):
            await provider.get_strict(lookup(provider, run_id))

    async def test_an_undecodable_line_raises_only_when_the_run_is_not_found(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(path=path)
        run_id = await stored_run(provider)
        with path.open("a", encoding="utf-8") as f:
            f.write('{"run_id": "torn", "spans": [\n')  # a record being written
        assert await provider.get_strict(lookup(provider, run_id)) is not None
        with pytest.raises(json.JSONDecodeError):
            await provider.get_strict(lookup(provider, "missing"))


class TestSqlite:
    async def test_finds_a_stored_run_and_none_for_a_missing_one(self, tmp_path):
        provider = SqliteTracingProvider.configured(path=tmp_path / "traces.db")
        run_id = await stored_run(provider)
        assert await provider.get_strict(lookup(provider, run_id)) is not None
        assert await provider.get_strict(lookup(provider, "missing")) is None

    async def test_a_database_error_raises_where_get_returns_none(self, tmp_path, monkeypatch):
        provider = SqliteTracingProvider.configured(path=tmp_path / "traces.db")
        run_id = await stored_run(provider)

        def broken_db():
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(provider, "_db", broken_db)
        assert await provider.get(lookup(provider, run_id)) is None  # unchanged contract
        with pytest.raises(sqlite3.OperationalError):
            await provider.get_strict(lookup(provider, run_id))


class TestPlatform:
    @pytest.mark.parametrize("status", [404, 500])
    async def test_a_404_is_no_run_and_other_errors_raise(self, monkeypatch, status):
        request = httpx.Request("GET", "https://platform.test/runs/x")
        response = httpx.Response(status, request=request)

        async def failing_get(_cls, _run_context):
            raise httpx.HTTPStatusError("error", request=request, response=response)

        monkeypatch.setattr(PlatformTracingProvider, "get", classmethod(failing_get))
        context = lookup(InMemoryTracingProvider, "x")
        if status == 404:
            assert await PlatformTracingProvider.get_strict(context) is None
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await PlatformTracingProvider.get_strict(context)


class TestCustomProviders:
    async def test_default_is_get(self):
        class Custom(TracingProvider):
            @classmethod
            async def get(cls, _run_context):
                return "trace"

            @classmethod
            async def _store(cls, _run_context):
                return None

        assert await Custom.get_strict(lookup(InMemoryTracingProvider, "x")) == "trace"
