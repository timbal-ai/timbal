"""Buffered trace persistence: mid-run saves go to the background, coalesced; barriers store synchronously."""

import asyncio
import json
import time

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state import RunContext, get_run_context, list_background_tasks, set_run_context, wait_for_background
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


@pytest.mark.parametrize("by_env", [False, True])
async def test_end_stores_once_when_the_run_ends(monkeypatch, by_env):
    if by_env:
        monkeypatch.setenv(uploads.DURABILITY_ENV, "end")
        storage = make_storage()
    else:
        storage = make_storage().configured(durability="end")
    output, _ = await run_agent(storage, rounds=4, per_round=3)
    assert output.status.code == "success"
    assert storage.stored == [{"spans": storage.stored[0]["spans"], "finished": True}]
    assert output.usage["tracing:saves"] == 1
    assert output.usage["tracing:saves_coalesced"] >= 7
    assert uploads.pending_runs() == []


async def test_end_still_stores_a_barrier_mid_run():
    storage = make_storage().configured(durability="end")
    seen = {}

    async def checkpoint() -> str:
        await get_run_context()._save_trace(barrier=True)
        seen["stored"] = list(storage.stored)
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
    assert [snapshot["finished"] for snapshot in seen["stored"]] == [False]
    assert [snapshot["finished"] for snapshot in storage.stored] == [False, True]


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

    # A span finishing in background work on this context, after the turn: scheduled, not awaited.
    run_context._live_background += 1
    await run_context._save_trace()
    assert len(storage.stored) == stored
    assert uploads.pending_runs() == [run_context.id]
    # The background work's end stores it and releases the run's upload state.
    await run_context._end_background_work()
    assert len(storage.stored) == stored + 1
    assert not uploads.tracked(run_context)

    # With no background work left, a save after the end is synchronous again.
    await run_context._save_trace()
    assert len(storage.stored) == stored + 2
    assert not uploads.tracked(run_context)


async def test_a_spawned_subagent_that_outlives_the_turn_stores_its_end():
    """Saves after the turn are buffered; the spawned task's end stores the final trace and releases the run."""
    storage = make_storage(delay=0.01).configured(durability="buffered", flush_interval_s=10.0)

    async def slow_lookup(query: str) -> str:
        await asyncio.sleep(0.2)
        return f"result for {query}"

    sub = Agent(
        name="sub",
        model=TestModel(
            responses=[
                Message(role="assistant", content=[ToolUseContent(id="s1", name="slow_lookup", input={"query": "x"})]),
                "sub done",
            ]
        ),
        tools=[slow_lookup],
        background_mode="auto",
    )
    original = get_run_context()
    run_context = RunContext(tracing_provider=storage)
    try:
        set_run_context(run_context)
        agent = Agent(
            name="worker",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(id="c1", name="sub", input={"prompt": "go", "run_in_background": True})
                        ],
                    ),
                    "started",
                ]
            ),
            tools=[sub],
            tracing_provider=storage,
        )
        output = await agent(prompt="go").collect()
        assert output.status.code == "success"
        assert storage.stored[-1]["finished"]
        (task,) = list_background_tasks()
        await wait_for_background(task["task_id"], timeout=5.0)
        for _ in range(100):
            if run_context._live_background == 0:
                break
            await asyncio.sleep(0.01)
    finally:
        set_run_context(original)
    assert run_context._live_background == 0
    assert not uploads.tracked(run_context)
    # The last snapshot holds the sub-agent's spans, finished after the turn.
    assert storage.stored[-1]["spans"] == len(run_context._trace)


@pytest.mark.parametrize("durability", ["sync", "buffered", "end"])
async def test_a_spawned_tool_that_outlives_the_turn_stores_its_real_output(durability):
    """The spawned task writes its result over the placeholder after the turn; that write is stored."""
    snapshots: list[str] = []

    class Storage(TracingProvider):
        @classmethod
        async def get(cls, _run_context):
            return None

        @classmethod
        async def _store(cls, run_context):
            snapshots.append(json.dumps(run_context._trace.model_dump(), default=str))

    provider = Storage.configured(durability=durability, flush_interval_s=10.0)

    async def slow_job() -> str:
        await asyncio.sleep(0.2)
        return "the late result"

    original = get_run_context()
    run_context = RunContext(tracing_provider=provider)
    try:
        set_run_context(run_context)
        agent = Agent(
            name="worker",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="c1", name="slow_job", input={"run_in_background": True})],
                    ),
                    "started",
                ]
            ),
            tools=[Tool(name="slow_job", handler=slow_job, background_mode="auto")],
            tracing_provider=provider,
        )
        await agent(prompt="go").collect()
        assert "the late result" not in snapshots[-1]
        (task,) = list_background_tasks()
        await wait_for_background(task["task_id"], timeout=5.0)
        for _ in range(100):
            if run_context._live_background == 0:
                break
            await asyncio.sleep(0.01)
    finally:
        set_run_context(original)
    assert "the late result" in snapshots[-1]
    assert not uploads.tracked(run_context)


async def test_a_run_started_inside_background_work_still_stores_its_end():
    """An Agent forked from a background tool inherits the background depth, not the work: its end is a barrier."""
    storage = make_storage(delay=0.02).configured(durability="buffered", flush_interval_s=10.0)
    token = set_background_depth(1)
    try:
        output, _ = await run_agent(storage, rounds=2, per_round=1)
    finally:
        reset_background_depth(token)
    assert output.status.code == "success"
    assert storage.stored[-1]["finished"]
    assert uploads.pending_runs() == []


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


async def finished_run(provider) -> RunContext:
    """A run whose root has ended, on ``provider``."""
    original = get_run_context()
    run_context = RunContext(tracing_provider=provider)
    try:
        set_run_context(run_context)
        await Agent(name="worker", model=TestModel(responses=["done"]), tracing_provider=provider)(
            prompt="go"
        ).collect()
    finally:
        set_run_context(original)
    return run_context


async def until(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.005)


async def test_a_save_during_a_barrier_write_is_stored_after_it():
    """One write per run at a time: a background save while a barrier writes lands after it, never under it."""
    state = {"version": 0, "delays": []}
    persisted: list[int] = []

    class Storage(TracingProvider):
        @classmethod
        async def get(cls, _run_context):
            return None

        @classmethod
        async def _store(cls, _run_context):
            version = state["version"]  # the snapshot is taken here
            await asyncio.sleep(state["delays"].pop(0) if state["delays"] else 0)
            persisted.append(version)

    provider = Storage.configured(durability="buffered", flush_interval_s=0.01)
    run_context = await finished_run(provider)
    persisted.clear()
    run_context._live_background += 1

    state["version"], state["delays"] = 1, [0.1]
    barrier = asyncio.create_task(run_context._save_trace(barrier=True))
    await until(lambda: state["delays"] == [])  # the barrier is writing version 1
    state["version"] = 2
    await run_context._save_trace()  # background work saves meanwhile
    await barrier
    await until(lambda: not uploads.pending_runs())
    assert persisted == [1, 2]


async def test_a_finished_run_is_released_once_a_retry_stores_its_end():
    """The root's end barrier fails with a save pending; the drain's retry stores it and lets the run go."""
    state = {"fail_next": True}
    attempts: list[str] = []

    class Storage(TracingProvider):
        @classmethod
        async def get(cls, _run_context):
            return None

        @classmethod
        async def _store(cls, _run_context):
            failing, state["fail_next"] = state["fail_next"], False
            attempts.append("failed" if failing else "stored")
            if failing:
                raise RuntimeError("platform unavailable")

    run_context = await finished_run(Storage.configured(durability="buffered", flush_interval_s=0.01))
    await until(lambda: attempts == ["failed", "stored"] and not uploads.pending_runs())
    await until(lambda: not uploads.tracked(run_context))


async def test_a_cancelled_end_barrier_still_releases_the_run():
    """The end of background work is cancelled while an upload is in flight; that upload's drain lets the run go."""
    storage = make_storage(delay=0.1).configured(durability="buffered", flush_interval_s=0.01)
    run_context = await finished_run(storage)
    run_context._live_background += 1
    await run_context._save_trace()  # scheduled: the upload starts at once
    await until(lambda: uploads.tracked(run_context) and uploads._RUNS[uploads._key(run_context)].sending)
    end = asyncio.create_task(run_context._end_background_work())  # waits for that upload
    await asyncio.sleep(0.01)
    end.cancel()
    await asyncio.gather(end, return_exceptions=True)
    await until(lambda: not uploads.tracked(run_context))


@pytest.mark.parametrize("durability", ["buffered", "end"])
async def test_overlapping_barriers_store_in_order(durability):
    """The root's end and a background task's end can overlap: the barrier that starts later lands last."""
    state = {"version": 0, "delays": []}
    persisted: list[int] = []

    class Storage(TracingProvider):
        @classmethod
        async def get(cls, _run_context):
            return None

        @classmethod
        async def _store(cls, _run_context):
            version = state["version"]  # the snapshot is taken here
            await asyncio.sleep(state["delays"].pop(0) if state["delays"] else 0)
            persisted.append(version)

    provider = Storage.configured(durability=durability, flush_interval_s=0.01)
    run_context = await finished_run(provider)
    persisted.clear()

    state["version"], state["delays"] = 1, [0.1]
    first = asyncio.create_task(run_context._save_trace(barrier=True))
    await until(lambda: state["delays"] == [])  # the first barrier is writing version 1
    state["version"] = 2
    await run_context._save_trace(barrier=True)
    await first
    assert persisted == [1, 2]
    assert not uploads.tracked(run_context)


async def test_a_failed_upload_backing_off_does_not_hold_the_shutdown_flush():
    """A barrier waits for an upload in flight, not for a failed one's retry backoff."""
    state = {"fail_next": False}
    attempts: list[str] = []

    class Storage(TracingProvider):
        @classmethod
        async def get(cls, _run_context):
            return None

        @classmethod
        async def _store(cls, _run_context):
            failing, state["fail_next"] = state["fail_next"], False
            attempts.append("failed" if failing else "stored")
            if failing:
                await asyncio.sleep(0.02)
                raise RuntimeError("platform unavailable")

    # The failure backs off 2 s before its retry. Shutdown must store without
    # waiting that out. The flush budget is 1 s: long enough to finish the
    # in-flight write on a slow runner, short of the backoff.
    provider = Storage.configured(durability="buffered", flush_interval_s=1.0)
    run_context = await finished_run(provider)
    attempts.clear()
    run_context._live_background += 1

    state["fail_next"] = True
    await run_context._save_trace()  # the upload starts at once, and fails 0.02 s in
    await until(lambda: attempts == ["failed"])
    await uploads.flush_all(timeout=1.0)  # shutdown while that upload is in flight
    assert attempts == ["failed", "stored"]
    assert not uploads.tracked(run_context)


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
        _live_background = 0

        def update_usage(self, key, value):
            pass

        def root_span(self):
            return None  # still going

    run = FakeRun()
    assert uploads.schedule(run)  # the first save starts at once
    await asyncio.sleep(0.01)
    assert uploads.schedule(run)  # the next one waits out the 60 s interval
    assert uploads.pending_runs() == ["run-1"]
    await uploads.flush_all(timeout=1.0)
    assert stored == ["run-1", "run-1"]
    assert uploads.pending_runs() == []
