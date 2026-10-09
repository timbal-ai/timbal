"""Connection lifetime/isolation through the actual buffered request helper."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from timbal.platform._http_session import platform_http_session
from timbal.platform.utils import _request
from timbal.state import RunContext, get_run_context, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject


def context(token="one", host="platform.test"):
    return RunContext(
        tracing_provider=None,
        platform_config=PlatformConfig(
            host=host,
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token=token),
            subject=PlatformSubject(org_id="1", app_id="2"),
        ),
    )


@pytest.fixture
def http(monkeypatch):
    original_client = httpx.AsyncClient
    state = SimpleNamespace(clients=[], requests=[], handler=None)
    original_context = get_run_context()
    set_run_context(context())

    async def handle(request):
        state.requests.append(request)
        if state.handler:
            return await state.handler(request)
        return httpx.Response(200, json={"ok": True}, headers={"set-cookie": "session=secret; Path=/"})

    def create(**kwargs):
        client = original_client(transport=httpx.MockTransport(handle), **kwargs)
        state.clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", create)
    yield state
    set_run_context(original_context)


async def test_lazy_and_nested_scopes_close_only_at_owner_exit(http):
    async with platform_http_session():
        assert not http.clients
        await _request("GET", "first")
        async with platform_http_session():
            await _request("GET", "second")
        assert len(http.clients) == 1
        assert not http.clients[0].is_closed
    assert http.clients[0].is_closed
    async with platform_http_session():
        pass
    assert len(http.clients) == 1


async def test_auth_timeouts_and_cookies_are_per_request(http):
    async with platform_http_session():
        await _request("GET", "first", timeout=httpx.Timeout(1, read=2))
        set_run_context(context(token="two"))
        await _request("GET", "second", timeout=httpx.Timeout(3, read=4))
    assert len(http.clients) == 1
    first, second = http.requests
    assert first.headers["Authorization"] == "Bearer one"
    assert second.headers["Authorization"] == "Bearer two"
    assert first.extensions["timeout"]["read"] == 2
    assert second.extensions["timeout"]["read"] == 4
    assert "cookie" not in second.headers


async def test_concurrent_scopes_do_not_share_clients_or_auth(http):
    async def run(token):
        set_run_context(context(token=token))
        async with platform_http_session():
            await _request("GET", token)
            await asyncio.sleep(0)
            await _request("GET", token)

    await asyncio.gather(run("alice"), run("bob"))
    assert len(http.clients) == 2
    assert all(client.is_closed for client in http.clients)
    assert all(r.headers["Authorization"] == f"Bearer {r.url.path[1:]}" for r in http.requests)


async def test_scope_retirement_waits_for_last_inflight_borrower(http):
    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    async def handler(request):
        index = int(request.url.path[1:])
        started[index].set()
        await release[index].wait()
        return httpx.Response(200)

    http.handler = handler
    async with platform_http_session():
        tasks = [asyncio.create_task(_request("GET", str(i))) for i in range(2)]
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
    assert len(http.clients) == 1
    assert not http.clients[0].is_closed
    release[0].set()
    await tasks[0]
    assert not http.clients[0].is_closed
    release[1].set()
    await tasks[1]
    assert http.clients[0].is_closed


async def test_detached_task_after_scope_exit_uses_fresh_client(http):
    release = asyncio.Event()

    async def later():
        await release.wait()
        await _request("GET", "later")

    async with platform_http_session():
        await _request("GET", "foreground")
        task = asyncio.create_task(later())
    assert http.clients[0].is_closed
    release.set()
    await task
    assert len(http.clients) == 2
    assert all(client.is_closed for client in http.clients)


async def test_cancelled_request_does_not_close_live_scope(http):
    started = asyncio.Event()

    async def handler(request):
        if request.url.path == "/blocked":
            started.set()
            await asyncio.Event().wait()
        return httpx.Response(200)

    http.handler = handler
    async with platform_http_session():
        task = asyncio.create_task(_request("GET", "blocked"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _request("GET", "working")
        assert len(http.clients) == 1
        assert not http.clients[0].is_closed
    assert http.clients[0].is_closed


async def test_owner_cancellation_closes_client(http):
    started = asyncio.Event()

    async def owner():
        async with platform_http_session():
            await _request("GET", "first")
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(owner())
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert http.clients[0].is_closed


async def test_inherited_scope_is_not_reused_on_another_event_loop(http):
    async with platform_http_session():
        await _request("GET", "parent-loop")
        # to_thread copies contextvars, including the parent's live session.
        await asyncio.to_thread(lambda: asyncio.run(_request("GET", "other-loop")))
        assert len(http.clients) == 2
        assert not http.clients[0].is_closed
        assert http.clients[1].is_closed
    assert http.clients[0].is_closed


async def test_retries_reuse_pool_and_keep_current_timeout(http):
    async def handler(_request):
        return httpx.Response(503 if len(http.requests) == 1 else 200)

    http.handler = handler
    async with platform_http_session():
        await _request("PATCH", "trace", json={"trace": []}, backoff=lambda _: 0, timeout=7)
    assert len(http.clients) == 1
    assert len(http.requests) == 2
    assert all(r.extensions["timeout"]["connect"] == 7 for r in http.requests)
    assert http.clients[0].is_closed


async def test_unscoped_requests_preserve_short_lived_clients(http):
    await _request("GET", "first")
    await _request("GET", "second")
    assert len(http.clients) == 2
    assert all(client.is_closed for client in http.clients)


@pytest.mark.parametrize("runner", ["codegen", "server"])
async def test_runners_reuse_connection_for_both_trace_writes(http, runner, capsys):
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.state.tracing.providers.platform import PlatformTracingProvider

    run_context = RunContext(platform_config=context().platform_config, tracing_provider=PlatformTracingProvider)
    agent = Agent(name="router", model=TestModel(responses=["ok"]), tracing_provider=PlatformTracingProvider)
    set_run_context(run_context)
    if runner == "codegen":
        from timbal.codegen.test import run_test

        await run_test(SimpleNamespace(load=lambda: agent), {"prompt": "hi"}, run_context=run_context)
        assert '"type": "OUTPUT"' in capsys.readouterr().out
    else:
        from timbal.server.jobs import JobStore

        _, job = JobStore().create_job(agent, {"prompt": "hi"})
        events = [event async for _, event in job.follow()]
        await job.task
        assert events[-1].type == "OUTPUT"
        assert events[-1].status.code == "success"
    assert len(http.requests) == 2
    assert all(r.method == "PATCH" for r in http.requests)
    assert len(http.clients) == 1
    assert http.clients[0].is_closed
