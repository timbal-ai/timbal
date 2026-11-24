"""Platform saves across event loops and threads: what reads get, the shared lock, saves that cannot be made, and
the shutdown flush. Each test is a defect a review reproduced."""

import asyncio
import gc
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from timbal.errors import PlatformError
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing import uploads
from timbal.state.tracing.providers import base
from timbal.state.tracing.providers import platform as platform_module
from timbal.state.tracing.providers.platform import PlatformTracingProvider

from .test_platform_deltas import CONFIG, FakePlatform, whole_root


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setenv(uploads.DURABILITY_ENV, "sync")
    for state in (
        PlatformTracingProvider._caps,
        PlatformTracingProvider._delta_runs,
        PlatformTracingProvider._recent,
        PlatformTracingProvider._flights,
        PlatformTracingProvider._unsaved,
    ):
        state.clear()
    platform_module._last_snapshot = 0
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    for task in PlatformTracingProvider._retries.values():
        task.cancel()
    PlatformTracingProvider._retries.clear()
    PlatformTracingProvider._unsaved.clear()
    PlatformTracingProvider._flights.clear()
    PlatformTracingProvider._recent.clear()
    for entry in PlatformTracingProvider._keepalives.values():
        entry.cancel()
    PlatformTracingProvider._keepalives.clear()


def context(provider, run_id=None, **root):
    ctx = RunContext(tracing_provider=provider) if run_id is None else RunContext(id=run_id, tracing_provider=provider)
    ctx.platform_config = CONFIG
    ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None, **root}
    return ctx


def read_on_another_loop(provider, run_id):
    reader = RunContext(parent_id=run_id, tracing_provider=provider)
    reader.platform_config = CONFIG
    out = []
    thread = threading.Thread(target=lambda: out.append(asyncio.run(provider.get(reader))))
    thread.start()
    thread.join(10)
    return out[0]


class TestReads:
    @pytest.mark.parametrize("case", ["the run's end", "the run id used again"])
    async def test_a_patch_the_host_dropped_is_not_what_reads_get(self, monkeypatch, case):
        """PATCH #1 is on its way; PATCH #2 (the run's end, or a fresh context of the run id) goes past it and is
        applied, its answer still out. #1 then reaches the host, which drops it as older than #2. A read of the run
        is answered with what the host holds, never with #1's body."""
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.0)
        server = FakePlatform()  # echoes the snapshot it holds, and whether it applied this one
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, deltas=False, compress=False)
        first = context(provider, output="old")
        if case == "the run's end":
            second = first
        else:
            first._trace["r"].t1 = 1
            second = context(provider, run_id=first.id, output="new", t1=2)
        in_transit, deliver_first = asyncio.Event(), asyncio.Event()
        second_applied, answer_second = asyncio.Event(), asyncio.Event()
        patches = []

        async def request(method, path, headers=None, **kwargs):
            if method != "PATCH":
                return await server.request(method, path, headers=headers, **kwargs)
            patches.append(None)
            if len(patches) == 1:
                in_transit.set()
                await deliver_first.wait()
                return await server.request(method, path, headers=headers, **kwargs)
            response = await server.request(method, path, headers=headers, **kwargs)
            if len(patches) == 2:
                second_applied.set()
                await answer_second.wait()
            return response

        with patch("timbal.platform.utils._request", request):
            one = asyncio.create_task(provider.put(first))
            await asyncio.wait_for(in_transit.wait(), 5)
            if case == "the run's end":
                second._trace["r"].t1 = 5
                second._trace["r"].output = "new"
            two = asyncio.create_task(provider.put(second))
            await asyncio.wait_for(second_applied.wait(), 5)
            deliver_first.set()
            await asyncio.wait_for(one, 5)  # dropped by the host as older
            try:
                read = await asyncio.wait_for(asyncio.to_thread(read_on_another_loop, provider, first.id), 15)
            finally:
                answer_second.set()
                await asyncio.wait_for(two, 5)
            after = await asyncio.wait_for(asyncio.to_thread(read_on_another_loop, provider, first.id), 15)
        assert whole_root(server, first.id)["output"] == "new"
        assert read["r"].output == "new" and after["r"].output == "new"

    async def test_a_read_while_another_loops_newer_save_is_unanswered_asks_the_host(self):
        """v1 landed on this loop and is kept. Another loop saves v2: the host applies it, its answer is still out.
        A read on this loop (which cannot wait for that loop's save) asks the host instead of answering v1."""
        server = FakePlatform(deltas=False)
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0)
        ctx = context(provider, output="v1", t1=1)
        main = threading.get_ident()
        applied, answer = threading.Event(), threading.Event()

        async def request(method, path, headers=None, **kwargs):
            response = await server.request(method, path, headers=headers, **kwargs)
            if method == "PATCH" and threading.get_ident() != main:
                applied.set()
                answer.wait(10)
            return response

        with patch("timbal.platform.utils._request", request):
            await asyncio.wait_for(provider.put(ctx), 5)
            ctx._trace["r"].output = "v2"  # a background child's late output, saved from a worker's loop
            other = threading.Thread(target=lambda: asyncio.run(provider.put(ctx)), daemon=True)
            other.start()
            try:
                assert await asyncio.to_thread(applied.wait, 10)
                reader = RunContext(parent_id=ctx.id, tracing_provider=provider)
                reader.platform_config = CONFIG
                read = await asyncio.wait_for(provider.get(reader), 5)
            finally:
                answer.set()
                await asyncio.wait_for(asyncio.to_thread(other.join, 10), 15)
        assert whole_root(server, ctx.id)["output"] == "v2"
        assert read["r"].output == "v2"


class TestSharedLock:
    def test_a_capture_and_a_save_on_two_threads_while_a_dead_save_is_collected(self, monkeypatch):
        """Thread A captures a redacted copy; a collection inside it finishes a save left suspended in a closed
        loop, whose cleanup takes the provider's lock. Thread B is making a PATCH body meanwhile. Neither waits for
        the other for good: there is one lock."""

        class Watched:
            """A re-entrant lock that says when a thread had to wait for it."""

            def __init__(self):
                self.lock = threading.RLock()
                self.waiting = threading.Event()

            def acquire(self, *args, **kwargs):
                return self.lock.acquire(*args, **kwargs)

            def release(self):
                self.lock.release()

            def __enter__(self):
                if not self.lock.acquire(blocking=False):
                    self.waiting.set()
                    self.lock.acquire()
                return self

            def __exit__(self, *_):
                self.lock.release()

        watched = Watched()
        monkeypatch.setattr(base, "_CAPTURE_LOCK", watched)
        monkeypatch.setattr(sys, "unraisablehook", lambda _info: None)  # "Task was destroyed but it is pending"
        never = []

        async def request(method, **_):
            assert method == "PATCH"
            if not never:
                never.append(None)
                await asyncio.get_running_loop().create_future()  # left on the wire: its loop closes
            return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})

        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0)
        left, other, copied = context(provider, output="x"), context(provider, output="x"), context(None, output="x")
        inside, go = threading.Event(), threading.Event()

        def slow(value):
            if not inside.is_set():
                inside.set()
                go.wait(10)
                gc.collect()  # what any allocation in the capture may do
            return value

        gc.collect()
        gc.disable()
        try:
            with patch("timbal.platform.utils._request", request):

                def leave_a_save():
                    loop = asyncio.new_event_loop()
                    loop.create_task(provider.put(left))

                    async def out():
                        while not never:
                            await asyncio.sleep(0)

                    loop.run_until_complete(asyncio.wait_for(out(), 5))
                    loop.close()

                first = threading.Thread(target=leave_a_save)
                first.start()
                first.join(10)
                flushed = []  # sent again from another loop: the closed loop's save is now only garbage
                flush = threading.Thread(target=lambda: flushed.append(asyncio.run(provider.flush(5))))
                flush.start()
                flush.join(10)
                assert flushed == [[]]
                capturing = threading.Thread(target=lambda: base._capture(copied, slow), daemon=True)
                capturing.start()
                assert inside.wait(10)
                saving = threading.Thread(target=lambda: asyncio.run(provider.put(other)), daemon=True)
                saving.start()
                assert watched.waiting.wait(10)  # the save waits for the capture
                go.set()
                capturing.join(10)
                saving.join(10)
                stuck = [name for name, thread in (("capture", capturing), ("save", saving)) if thread.is_alive()]
        finally:
            gc.enable()
        assert not stuck


class TestUnserializable:
    async def test_saves_that_cannot_be_made_are_not_kept_owed(self, monkeypatch):
        """Runs whose trace holds a value JSON cannot carry (a tool returned float('nan')): every save of them fails
        before a PATCH exists, and they are not kept owed (each would hold a place in ``_unsaved`` for good, which
        nothing could push out). A run that can be saved is unaffected."""
        monkeypatch.setattr(platform_module, "_UNSAVED_MAX", 2)
        server = FakePlatform(deltas=False)
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0)
        runs = [context(provider, output={"score": float("nan")}, t1=1) for _ in range(6)]
        with patch("timbal.platform.utils._request", server.request):
            for ctx in runs:
                with pytest.raises(ValueError):
                    await asyncio.wait_for(provider.put(ctx), 5)
            fine = context(provider, output="ok", t1=1)
            await asyncio.wait_for(provider.put(fine), 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert not PlatformTracingProvider._unsaved
        assert len(server.patches) == 1 and whole_root(server, fine.id)["output"] == "ok"

    async def test_they_do_not_push_out_a_save_that_can_be_retried(self, monkeypatch):
        """Two runs that cannot be serialized, then a run whose end fails on a 503 (owed, resent in the background),
        then an unrelated save: the failed end is still owed, and a flush stores it."""
        monkeypatch.setattr(platform_module, "_UNSAVED_MAX", 2)
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))
        server = FakePlatform(deltas=False)
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0)
        with patch("timbal.platform.utils._request", server.request):
            for ctx in [context(provider, output=float("nan"), t1=1) for _ in range(2)]:
                with pytest.raises(ValueError):
                    await asyncio.wait_for(provider.put(ctx), 5)
            ended = context(provider, output="the end", t1=1)
            server.fail_next = 1
            with pytest.raises(PlatformError):
                await asyncio.wait_for(provider.put(ended), 5)
            await asyncio.wait_for(provider.put(context(provider, output="unrelated", t1=1)), 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert whole_root(server, ended.id)["output"] == "the end"


async def test_a_shutdown_flush_after_everything_landed_sends_nothing():
    """Buffered uploads of a running run, all landed: the shutdown flush sends no PATCH."""
    server = FakePlatform(deltas=False)
    provider = PlatformTracingProvider.configured(
        platform_config=CONFIG, durability="buffered", flush_interval_s=0.01, keepalive_s=0
    )
    ctx = context(provider)

    async def until(predicate):
        while not predicate():
            await asyncio.sleep(0.005)

    try:
        with patch("timbal.platform.utils._request", server.request):
            assert uploads.schedule(ctx)
            await asyncio.wait_for(until(lambda: server.patches and not uploads.pending_runs()), 5)
            assert not PlatformTracingProvider._unsaved  # the newest save landed
            before = len(server.patches)
            await asyncio.wait_for(uploads.flush_all(timeout=2), 5)
            after = len(server.patches)
    finally:
        for run in list(uploads._RUNS.values()):
            if run.task is not None:
                run.task.cancel()
        uploads._reset_for_tests()
    assert after == before
