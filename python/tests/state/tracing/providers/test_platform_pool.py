"""Trace saves share one connection pool per event loop (one TLS handshake, not one per save)."""

import httpx
import pytest
from timbal.platform import _http_session
from timbal.state.tracing.providers import platform as platform_module
from timbal.state.tracing.providers.platform import PlatformTracingProvider

from .test_platform_deltas import run, tool_loop


@pytest.fixture
def clients(monkeypatch):
    made: list[httpx.AsyncClient] = []
    requests: list[httpx.Request] = []
    real = httpx.AsyncClient

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"trace": None})
        return httpx.Response(204, headers={"X-Timbal-Snapshot": request.headers.get("X-Timbal-Snapshot", "0")})

    class Counting(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(answer)
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(_http_session.httpx, "AsyncClient", Counting)
    platform_module._TRACE_POOLS.clear()
    yield made, requests
    platform_module._TRACE_POOLS.clear()


async def test_every_save_of_a_run_goes_through_one_client(clients):
    made, requests = clients
    agent, provider = tool_loop(4)
    await run(agent, provider)
    patches = [r for r in requests if r.method == "PATCH"]
    assert len(patches) > 4
    assert len(made) == 1 and not made[0].is_closed
    await PlatformTracingProvider.flush(timeout=5)
    assert made[0].is_closed  # the shutdown flush closes the pool


async def test_a_new_pool_after_the_flush_closed_one(clients):
    made, _ = clients
    agent, provider = tool_loop(1)
    await run(agent, provider)
    await PlatformTracingProvider.flush(timeout=5)
    agent, provider = tool_loop(1)
    await run(agent, provider)
    assert len(made) == 2 and made[0].is_closed and not made[1].is_closed


async def test_a_caller_scoped_session_is_kept(clients):
    made, _ = clients
    async with _http_session.platform_http_session():
        agent, provider = tool_loop(2)
        await run(agent, provider)
        scoped = _http_session._SESSION.get()
        assert len(made) == 1 and scoped.client is made[0]
    assert platform_module._TRACE_POOLS == {}  # the trace pool was not used
