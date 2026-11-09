"""``TracingProvider.get_strict``: ``None`` only for a run that is not stored; a failed read raises."""

import json
import os
import pathlib
import sqlite3

import httpx
import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.errors import PlatformError
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


def failing_stat(monkeypatch, path) -> None:
    """``os.stat`` of ``path`` fails with a permission error (``Path.exists()`` reads that as "no file" on 3.14)."""
    real_stat = os.stat

    def stat(target, *args, **kwargs):
        if os.fspath(target) == os.fspath(path):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)


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

    async def test_a_failed_stat_is_not_absence(self, tmp_path, monkeypatch):
        path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(path=path)
        run_id = await stored_run(provider)
        failing_stat(monkeypatch, path)
        assert await provider.get_strict(lookup(provider, run_id)) is not None

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

    async def test_a_failed_stat_raises(self, tmp_path, monkeypatch):
        path = tmp_path / "traces.db"
        provider = SqliteTracingProvider.configured(path=path)
        run_id = await stored_run(provider)
        failing_stat(monkeypatch, path)
        with pytest.raises(PermissionError):
            await provider.get_strict(lookup(provider, run_id))

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
    """Through the real ``get`` and ``_request``: only the transport is fake."""

    @pytest.fixture
    def transport(self, monkeypatch):
        import contextlib

        import timbal.platform.utils as platform_utils

        replies = {}

        def handler(request: httpx.Request) -> httpx.Response:
            reply = replies["reply"]
            if isinstance(reply, Exception):
                raise reply
            status, body = reply
            return httpx.Response(status, json=body, request=request)

        @contextlib.asynccontextmanager
        async def client(_timeout):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as fake:
                yield fake

        monkeypatch.setattr(platform_utils, "request_client", client)
        return replies

    @staticmethod
    def context(parent_id: str):
        from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject

        config = PlatformConfig(
            host="api.example.test",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_test"),
            subject=PlatformSubject(org_id="1", app_id="2"),
        )
        provider = PlatformTracingProvider.configured(platform_config=config)
        context = RunContext(tracing_provider=provider)
        object.__setattr__(context, "parent_id", parent_id)
        context.platform_config = config
        return provider, context

    async def test_a_404_is_no_run(self, transport):
        transport["reply"] = (404, {"detail": "not found"})
        provider, context = self.context("missing-run")
        assert await provider.get_strict(context) is None

    async def test_a_stored_run_is_found(self, transport):
        record = {"call_id": "c1", "parent_call_id": None, "path": "agent", "t0": 0}
        transport["reply"] = (200, {"parent_id": None, "trace": [record]})
        provider, context = self.context("stored-run")
        assert await provider.get_strict(context) is not None

    async def test_a_server_error_raises(self, transport):
        transport["reply"] = (500, {"detail": "boom"})
        provider, context = self.context("flaky-run")
        with pytest.raises(PlatformError) as raised:
            await provider.get_strict(context)
        assert raised.value.status_code == 500

    async def test_a_connection_error_raises(self, transport):
        transport["reply"] = httpx.ConnectError("unreachable")
        provider, context = self.context("unreachable-run")
        with pytest.raises(Exception):  # noqa: B017 - whatever _request surfaces, it is not "no run"
            await provider.get_strict(context)


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
