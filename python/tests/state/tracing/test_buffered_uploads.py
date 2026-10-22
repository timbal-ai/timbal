"""Buffered trace persistence: mid-run saves go to the background, coalesced; barriers store synchronously."""

import asyncio
import json
import time

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state import RunContext, get_run_context, set_run_context
from timbal.state.background import reset_background_depth, set_background_depth
from timbal.state.tracing import uploads
from timbal.state.tracing.providers.base import TracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


def make_storage(delay: float = 0.0, fail: int = 0):
    """A provider that copies each stored snapshot like remote storage would, after ``delay``.

    The first ``fail`` stores raise.
    """

    class Storage(TracingProvider):
        stored: list[dict] = []
        failures_left = fail

        @classmethod
        async def get(cls, _run_context):
            return None

        @classmethod
        async def _store(cls, run_context):
            await asyncio.sleep(delay)
            if cls.failures_left > 0:
                cls.failures_left -= 1
                raise RuntimeError("platform unavailable")
            spans = json.loads(json.dumps(run_context._trace.model_dump()))
            root = next(s for s in spans if s.get("parent_call_id") is None)
            cls.stored.append({"spans": len(spans), "finished": root.get("t1") is not None})

    Storage.stored = []
    return Storage


async def lookup(query: str) -> str:
    return f"result for {query}"


def tool_rounds(rounds: int, per_round: int) -> list:
    responses: list = [
        Message(
            role="assistant",
            content=[
                ToolUseContent(id=f"call_{r}_{i}", name="lookup", input={"query": f"{r}-{i}"}) for i in range(per_round)
            ],
        )
        for r in range(rounds)
    ]
    return [*responses, "done"]


async def run_agent(provider, rounds: int = 3, per_round: int = 2):
    original = get_run_context()
    try:
        set_run_context(RunContext(tracing_provider=provider))
        agent = Agent(
            name="worker",
            model=TestModel(responses=tool_rounds(rounds, per_round)),
            tools=[lookup],
            tracing_provider=provider,
        )
        started = time.monotonic()
        output = await agent(prompt="go").collect()
        return output, time.monotonic() - started
    finally:
        set_run_context(original)


@pytest.fixture(autouse=True)
def isolated_uploads(monkeypatch):
    monkeypatch.delenv(uploads.DURABILITY_ENV, raising=False)
    monkeypatch.delenv(uploads.METRICS_ENV, raising=False)
    previous = uploads._reset_for_tests()
    yield
    uploads._RUNS.clear()
    uploads._RUNS.update(previous)


async def test_sync_is_the_default_and_adds_no_usage_keys():
    storage = make_storage()
    output, _ = await run_agent(storage)
    assert output.status.code == "success"
    assert storage.stored[-1]["finished"]
    # One store per finished span: every save awaited.
    assert len(storage.stored) >= 7
    assert not [key for key in output.usage if key.startswith("tracing:")]


async def test_buffered_coalesces_and_the_end_of_the_run_lands_last():
    storage = make_storage(delay=0.02).configured(durability="buffered", flush_interval_s=0.05)
    output, _ = await run_agent(storage, rounds=4, per_round=3)
    assert output.status.code == "success"
    assert storage.stored[-1]["finished"]
    # Nothing unfinished is stored after the end of the run: no background save lands late.
    first_finished = next(i for i, snapshot in enumerate(storage.stored) if snapshot["finished"])
    assert all(snapshot["finished"] for snapshot in storage.stored[first_finished:])
    sync = make_storage(delay=0.02)
    await run_agent(sync, rounds=4, per_round=3)
    assert len(storage.stored) < len(sync.stored)
    assert output.usage["tracing:saves"] >= 1
    assert output.usage["tracing:saves_coalesced"] >= 1
    assert uploads.pending_runs() == []


async def test_a_slow_store_no_longer_holds_the_run():
    sync = make_storage(delay=0.1)
    _, sync_seconds = await run_agent(sync, rounds=3, per_round=2)
    buffered = make_storage(delay=0.1).configured(durability="buffered", flush_interval_s=0.5)
    output, buffered_seconds = await run_agent(buffered, rounds=3, per_round=2)
    assert buffered_seconds < sync_seconds / 2, (buffered_seconds, sync_seconds)
    assert buffered.stored[-1]["finished"]
    # The run waited only for its end barrier (and an upload in flight then), not per span.
    assert output.usage["tracing:wait_ms"] < sync_seconds * 1000 / 2


async def test_a_barrier_save_is_stored_before_it_returns():
    storage = make_storage(delay=0.05).configured(durability="buffered", flush_interval_s=10.0)
    seen = {}

    async def checkpoint() -> str:
        before = len(storage.stored)
        await get_run_context()._save_trace(barrier=True)
        seen["stored_during_call"] = len(storage.stored) - before
        return "ok"

    original = get_run_context()
    try:
        set_run_context(RunContext(tracing_provider=storage))
        agent = Agent(
            name="worker",
            model=TestModel(
                responses=[
                    Message(role="assistant", content=[ToolUseContent(id="c1", name="checkpoint", input={})]),
                    "done",
                ]
            ),
            tools=[checkpoint],
            tracing_provider=storage,
        )
        await agent(prompt="go").collect()
    finally:
        set_run_context(original)
    assert seen["stored_during_call"] == 1


async def test_background_work_after_the_root_ends_stays_buffered():
    storage = make_storage(delay=0.05).configured(durability="buffered", flush_interval_s=10.0)
    original = get_run_context()
    run_context = RunContext(tracing_provider=storage)
    try:
        set_run_context(run_context)
        agent = Agent(name="worker", model=TestModel(responses=["done"]), tracing_provider=storage)
        await agent(prompt="go").collect()
    finally:
        set_run_context(original)
    assert run_context.root_span().t1 is not None
    stored = len(storage.stored)

    # A span finishing in a background continuation, after the turn: scheduled, not awaited.
    token = set_background_depth(1)
    try:
        await run_context._save_trace()
        assert len(storage.stored) == stored
        assert uploads.pending_runs() == [run_context.id]
        # The continuation's end is a barrier: stored before it returns, nothing left pending.
        await run_context._save_trace(barrier=True)
        assert len(storage.stored) == stored + 1
        assert uploads.pending_runs() == []
    finally:
        reset_background_depth(token)

    # Outside background work, a save after the end stays synchronous.
    await run_context._save_trace()
    assert len(storage.stored) == stored + 2


async def test_failed_background_saves_are_retried_and_the_end_still_stores():
    storage = make_storage(delay=0.01, fail=1).configured(durability="buffered", flush_interval_s=0.01)
    output, _ = await run_agent(storage, rounds=4, per_round=2)
    assert output.status.code == "success"
    assert storage.stored and storage.stored[-1]["finished"]
    assert output.usage["tracing:save_failed"] >= 1


async def test_a_run_too_far_behind_saves_synchronously_again():
    storage = make_storage(delay=0.02, fail=100).configured(durability="buffered", flush_interval_s=0.01, max_lag_s=0.0)
    output, _ = await run_agent(storage, rounds=3, per_round=2)
    assert output.status.code == "success"
    assert output.usage.get("tracing:lag_exceeded", 0) >= 1


async def test_flush_all_stores_what_is_pending():
    stored = []

    class Recorder(TracingProvider):
        durability = "buffered"
        flush_interval_s = 60.0

        @classmethod
        async def get(cls, _run_context):
            return None

        @classmethod
        async def _store(cls, run_context):
            stored.append(run_context.id)

    class FakeRun:
        id = "run-1"
        _tracing_provider = Recorder

        def update_usage(self, key, value):
            pass

    run = FakeRun()
    assert uploads.schedule(run)  # the first save starts at once
    await asyncio.sleep(0.01)
    assert uploads.schedule(run)  # the next one waits out the 60 s interval
    assert uploads.pending_runs() == ["run-1"]
    await uploads.flush_all(timeout=1.0)
    assert stored == ["run-1", "run-1"]
    assert uploads.pending_runs() == []
