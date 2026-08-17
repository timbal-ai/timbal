import asyncio
import contextlib

import pytest
from timbal.server.jobs import Job, JobStore, RunIdInUse


class MockRunnable:
    """A mock runnable that tracks execution state."""

    def __init__(self, events: list, delay: float = 0):
        self.events = events
        self.delay = delay
        self.started = False
        self.completed = False
        self.events_emitted = []

    async def __call__(self, **kwargs):
        self.started = True
        for event in self.events:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.events_emitted.append(event)
            yield event
        self.completed = True


async def drain(job, after: int = 0) -> list:
    """Everything `job` yields from `after`, without the seqs."""
    return [event async for _, event in job.follow(after)]


class TestJob:
    def test_job_init(self):
        """Test Job initialization."""
        task = object()
        job = Job(task)

        assert job.task is task
        assert job.events == []
        assert job.last_seq == 0
        assert job.done is False
        assert job.finished_at is None

    def test_append_assigns_contiguous_seqs_from_one(self):
        job = Job()
        job.append("a")
        job.append("b")

        assert job.events == [(1, "a"), (2, "b")]
        assert job.last_seq == 2


class TestJobStore:
    def test_job_store_init(self):
        """Test JobStore initialization."""
        store = JobStore()
        assert store._jobs == {}

    @pytest.mark.asyncio
    async def test_create_job_generates_id(self):
        """Test that create_job generates a job ID if not provided."""
        store = JobStore()
        runnable = MockRunnable(events=["event1"])

        job_id, job = store.create_job(runnable, {})

        assert job_id is not None
        assert len(job_id) == 32  # UUID7 without dashes
        assert isinstance(job, Job)

    @pytest.mark.asyncio
    async def test_create_job_uses_provided_id(self):
        """Test that create_job uses the provided job ID."""
        store = JobStore()
        runnable = MockRunnable(events=["event1"])

        job_id, _ = store.create_job(runnable, {}, job_id="custom-id")

        assert job_id == "custom-id"

    @pytest.mark.asyncio
    async def test_job_added_to_store(self):
        """Test that created job is added to the store."""
        store = JobStore()
        runnable = MockRunnable(events=["event1"])

        job_id, job = store.create_job(runnable, {})

        assert store.get_job(job_id) is job

    @pytest.mark.asyncio
    async def test_get_job_returns_none_for_unknown_id(self):
        """Test that get_job returns None for unknown job ID."""
        store = JobStore()

        assert store.get_job("unknown-id") is None

    @pytest.mark.asyncio
    async def test_job_emits_events_to_the_log(self):
        """Test that job events are readable in order."""
        store = JobStore()
        events = ["event1", "event2", "event3"]
        runnable = MockRunnable(events=events)

        _, job = store.create_job(runnable, {})

        assert await drain(job) == events

    @pytest.mark.asyncio
    async def test_job_passes_params_to_runnable(self):
        """Test that job passes params to the runnable."""
        store = JobStore()
        received_params = {}

        async def capturing_runnable(**kwargs):
            received_params.update(kwargs)
            yield "done"

        _, job = store.create_job(capturing_runnable, {"x": "test", "y": 42})

        await job.task

        assert received_params == {"x": "test", "y": 42}

    @pytest.mark.asyncio
    async def test_job_runs_to_completion_without_consumer(self):
        """Test that job runs to completion even if no one consumes events."""
        store = JobStore()
        runnable = MockRunnable(events=["event1", "event2", "event3"])

        _, job = store.create_job(runnable, {})

        await job.task

        assert runnable.started is True
        assert runnable.completed is True
        assert runnable.events_emitted == ["event1", "event2", "event3"]

    @pytest.mark.asyncio
    async def test_job_runs_to_completion_with_slow_consumer(self):
        """Test that job continues running even with a slow consumer."""
        store = JobStore()
        events = ["event1", "event2", "event3", "event4", "event5"]
        runnable = MockRunnable(events=events, delay=0.01)

        _, job = store.create_job(runnable, {})

        stream = job.follow()
        await anext(stream)
        await asyncio.sleep(0.05)
        await anext(stream)

        await job.task

        assert runnable.completed is True
        assert runnable.events_emitted == events

    @pytest.mark.asyncio
    async def test_job_completes_when_consumer_disconnects(self):
        """The run outlives its reader — that is the point of the log."""
        store = JobStore()
        events = ["event1", "event2", "event3", "event4", "event5"]
        runnable = MockRunnable(events=events)

        _, job = store.create_job(runnable, {})

        stream = job.follow()
        assert await anext(stream) == (1, "event1")
        await stream.aclose()

        await job.task

        assert runnable.completed is True
        assert runnable.events_emitted == events

    @pytest.mark.asyncio
    async def test_job_completes_when_consumer_cancelled(self):
        """Test that job runs to completion even when consumer task is cancelled."""
        store = JobStore()
        events = ["event1", "event2", "event3", "event4", "event5"]
        runnable = MockRunnable(events=events, delay=0.02)

        _, job = store.create_job(runnable, {})

        async def consumer():
            async for _ in job.follow():
                await asyncio.sleep(1)  # Will be cancelled before this completes

        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        consumer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer_task

        await job.task

        assert runnable.completed is True
        assert runnable.events_emitted == events

    @pytest.mark.asyncio
    async def test_multiple_jobs_independent(self):
        """Test that multiple jobs run independently."""
        store = JobStore()
        runnable1 = MockRunnable(events=["a1", "a2"], delay=0.01)
        runnable2 = MockRunnable(events=["b1", "b2", "b3"], delay=0.01)

        job_id1, job1 = store.create_job(runnable1, {})
        job_id2, job2 = store.create_job(runnable2, {})

        assert store.get_job(job_id1) is not None
        assert store.get_job(job_id2) is not None

        await asyncio.gather(job1.task, job2.task)

        assert await drain(job1) == ["a1", "a2"]
        assert await drain(job2) == ["b1", "b2", "b3"]

    @pytest.mark.asyncio
    async def test_job_finishes_on_error(self):
        """A crashed runnable still terminates the log instead of hanging readers."""
        store = JobStore()

        async def failing_runnable(**kwargs):
            yield "event1"
            raise ValueError("Something went wrong")

        _, job = store.create_job(failing_runnable, {})

        # Would hang forever if the failure skipped `finish()`.
        assert await asyncio.wait_for(drain(job), timeout=1) == ["event1"]
        assert job.done is True

        with pytest.raises(ValueError, match="Something went wrong"):
            await job.task


class TestReconnect:
    """Reading a run's log after the fact — the reason it is a log."""

    @pytest.mark.asyncio
    async def test_a_reader_that_missed_everything_replays_from_zero(self):
        store = JobStore()
        events = ["event1", "event2", "event3"]

        _, job = store.create_job(MockRunnable(events=events), {})
        await job.task

        assert await drain(job) == events

    @pytest.mark.asyncio
    async def test_a_reader_resumes_from_its_cursor(self):
        store = JobStore()

        _, job = store.create_job(MockRunnable(events=["a", "b", "c", "d"]), {})
        await job.task

        assert await drain(job, after=2) == ["c", "d"]

    @pytest.mark.asyncio
    async def test_two_readers_do_not_race_for_events(self):
        """A queue hands each event to exactly one reader; a log does not."""
        store = JobStore()
        events = ["a", "b", "c"]

        _, job = store.create_job(MockRunnable(events=events, delay=0.01), {})

        both = await asyncio.gather(drain(job), drain(job))

        assert both == [events, events]

    @pytest.mark.asyncio
    async def test_read_reports_the_cursor_to_come_back_with(self):
        store = JobStore()

        _, job = store.create_job(MockRunnable(events=["a", "b", "c"]), {})
        await job.task

        events, next_cursor, done = job.read()

        assert [seq for seq, _ in events] == [1, 2, 3]
        assert next_cursor == 3
        assert done is True

    @pytest.mark.asyncio
    async def test_an_empty_batch_leaves_the_cursor_alone(self):
        store = JobStore()

        _, job = store.create_job(MockRunnable(events=["a"]), {})
        await job.task

        events, next_cursor, done = job.read(after=1)

        assert events == []
        assert next_cursor == 1
        assert done is True

    @pytest.mark.asyncio
    async def test_a_truncated_page_is_not_done_even_when_the_run_is(self):
        """`done` must mean "you have it all", or a paging client stops early."""
        store = JobStore()

        _, job = store.create_job(MockRunnable(events=["a", "b", "c"]), {})
        await job.task

        events, next_cursor, done = job.read(limit=2)

        assert [event for _, event in events] == ["a", "b"]
        assert next_cursor == 2
        assert done is False

        events, next_cursor, done = job.read(after=next_cursor, limit=2)

        assert [event for _, event in events] == ["c"]
        assert done is True

    @pytest.mark.asyncio
    async def test_wait_returns_once_an_event_lands(self):
        store = JobStore()

        _, job = store.create_job(MockRunnable(events=["a"], delay=0.05), {})

        await asyncio.wait_for(job.wait(after=0, timeout=5), timeout=1)

        assert job.last_seq >= 1

    @pytest.mark.asyncio
    async def test_wait_gives_up_quietly_on_timeout(self):
        """Timeout is not an error — the caller re-reads either way."""
        job = Job()

        await asyncio.wait_for(job.wait(after=0, timeout=0.01), timeout=1)

        assert job.last_seq == 0

    @pytest.mark.asyncio
    async def test_wait_does_not_block_when_the_run_is_over(self):
        store = JobStore()

        _, job = store.create_job(MockRunnable(events=["a"]), {})
        await job.task

        await asyncio.wait_for(job.wait(after=99, timeout=30), timeout=1)

    @pytest.mark.asyncio
    async def test_follow_does_not_miss_an_event_that_lands_mid_read(self):
        """The waiter has to be registered before the read, not after.

        Register after and an append landing in between resolves nothing, so the
        wake is lost — and when that append is a run's last event, `follow`
        waits for an event that is never coming. Patching `read` to produce the
        event as a side effect puts the append exactly in that window.
        """
        job = Job()
        real_read = job.read
        landed = False

        def read_then_produce(after=0, limit=500):
            nonlocal landed
            batch = real_read(after, limit)
            if not landed:
                landed = True
                job.append("a")
                job.finish()
            return batch

        job.read = read_then_produce

        assert await asyncio.wait_for(drain(job), timeout=1) == ["a"]


class TestNonReplayable:
    """`replayable=False` — the `/run` shape: one reader, no reconnection, no log."""

    @pytest.mark.asyncio
    async def test_events_are_forgotten_as_they_are_consumed(self):
        store = JobStore()
        events = ["a", "b", "c", "d", "e"]

        _, job = store.create_job(MockRunnable(events=events), {}, replayable=False)

        assert await drain(job) == events
        # The reader saw everything, and none of it is still held.
        assert job.events == []
        assert job.last_seq == 5

    @pytest.mark.asyncio
    async def test_a_finished_job_leaves_the_store_at_once(self):
        """Nothing can reconnect to it, so there is nothing to retain."""
        store = JobStore()

        job_id, job = store.create_job(MockRunnable(events=["a"]), {}, replayable=False)
        await drain(job)
        await job.task
        await asyncio.sleep(0)

        assert store.get_job(job_id) is None

    @pytest.mark.asyncio
    async def test_a_replayable_job_keeps_its_log_after_being_read(self):
        """The contrast: reading a `/stream` job must not consume it."""
        store = JobStore()

        _, job = store.create_job(MockRunnable(events=["a", "b"]), {})

        assert await drain(job) == ["a", "b"]
        assert job.events == [(1, "a"), (2, "b")]
        assert await drain(job) == ["a", "b"]


class TestRunIdCollision:
    """Run ids come from the client, so collisions are reachable from outside."""

    @pytest.mark.asyncio
    async def test_a_live_run_id_is_refused(self):
        """Overwriting would leave the first run going, with nothing pointing at it."""
        store = JobStore()

        _, job = store.create_job(MockRunnable(events=["a"], delay=0.05), {}, job_id="dup")

        with pytest.raises(RunIdInUse):
            store.create_job(MockRunnable(events=["b"]), {}, job_id="dup")

        assert store.get_job("dup") is job
        assert await drain(job) == ["a"]

    @pytest.mark.asyncio
    async def test_a_finished_run_id_can_be_reused(self):
        store = JobStore()

        _, first = store.create_job(MockRunnable(events=["a"]), {}, job_id="dup")
        await first.task

        _, second = store.create_job(MockRunnable(events=["b"]), {}, job_id="dup")

        assert store.get_job("dup") is second
        assert await drain(second) == ["b"]


class TestRetention:
    """A run stays readable for a while after it ends, so a late reconnect works."""

    @pytest.mark.asyncio
    async def test_a_finished_job_is_still_addressable(self):
        store = JobStore()

        job_id, job = store.create_job(MockRunnable(events=["a"]), {})
        await job.task
        await asyncio.sleep(0)

        assert store.get_job(job_id) is job

    @pytest.mark.asyncio
    async def test_reap_forgets_a_job_past_its_window(self):
        store = JobStore(retention_secs=0)

        job_id, job = store.create_job(MockRunnable(events=["a"]), {})
        await job.task

        store.reap()

        assert store.get_job(job_id) is None

    @pytest.mark.asyncio
    async def test_reap_keeps_a_job_inside_its_window(self):
        store = JobStore(retention_secs=300)

        job_id, job = store.create_job(MockRunnable(events=["a"]), {})
        await job.task

        store.reap()

        assert store.get_job(job_id) is job

    @pytest.mark.asyncio
    async def test_the_window_holds_without_a_sweep(self):
        """An idle server gets no `create_job` calls, so `reap` alone is not enough."""
        store = JobStore(retention_secs=0)

        job_id, job = store.create_job(MockRunnable(events=["a"]), {})
        await job.task

        assert store.get_job(job_id) is None

    @pytest.mark.asyncio
    async def test_reap_never_touches_a_running_job(self):
        store = JobStore(retention_secs=0)

        job_id, job = store.create_job(MockRunnable(events=["a", "b"], delay=0.05), {})
        store.reap()

        assert store.get_job(job_id) is job

        await job.task

    @pytest.mark.asyncio
    async def test_creating_a_job_sweeps_expired_ones(self):
        store = JobStore(retention_secs=0)

        old_id, old_job = store.create_job(MockRunnable(events=["a"]), {})
        await old_job.task

        store.create_job(MockRunnable(events=["b"]), {})

        assert store.get_job(old_id) is None


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_job_returns_true_for_running_job(self):
        """Test that cancel_job returns True for a running job."""
        store = JobStore()
        runnable = MockRunnable(events=["event1", "event2", "event3"], delay=0.1)

        job_id, job = store.create_job(runnable, {})

        assert store.cancel_job(job_id) is True

        with pytest.raises(asyncio.CancelledError):
            await job.task

    @pytest.mark.asyncio
    async def test_cancel_job_returns_false_for_unknown_id(self):
        """Test that cancel_job returns False for unknown job ID."""
        store = JobStore()

        assert store.cancel_job("unknown-id") is False

    @pytest.mark.asyncio
    async def test_cancel_job_returns_false_for_a_retained_finished_job(self):
        """Retention must not turn "already finished" into a successful cancel."""
        store = JobStore()

        job_id, job = store.create_job(MockRunnable(events=["a"]), {})
        await job.task
        await asyncio.sleep(0)

        assert store.get_job(job_id) is job
        assert store.cancel_job(job_id) is False

    @pytest.mark.asyncio
    async def test_cancel_job_stops_event_emission(self):
        """Test that cancelling a job stops further event emission."""
        store = JobStore()
        events = ["event1", "event2", "event3", "event4", "event5"]
        runnable = MockRunnable(events=events, delay=0.05)

        job_id, job = store.create_job(runnable, {})
        await asyncio.sleep(0.02)

        store.cancel_job(job_id)
        with contextlib.suppress(asyncio.CancelledError):
            await job.task

        assert len(runnable.events_emitted) < len(events)

    @pytest.mark.asyncio
    async def test_a_cancelled_job_terminates_its_log(self):
        """Readers of a cancelled run must be released, not left hanging."""
        store = JobStore()
        runnable = MockRunnable(events=["a", "b", "c"], delay=0.05)

        job_id, job = store.create_job(runnable, {})
        await asyncio.sleep(0.02)
        store.cancel_job(job_id)

        with contextlib.suppress(asyncio.CancelledError):
            await job.task

        assert job.done is True
        await asyncio.wait_for(drain(job), timeout=1)
