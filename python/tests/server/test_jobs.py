import asyncio

import pytest
from timbal.server.jobs import JOB_DONE_SENTINEL, Job, JobStore


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


class TestJob:
    def test_job_init(self):
        """Test Job initialization."""
        task = asyncio.Future()
        queue = asyncio.Queue()
        job = Job(task, queue)

        assert job.task is task
        assert job.queue is queue


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

        job_id, job = store.create_job(runnable, {}, job_id="custom-id")

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
    async def test_job_emits_events_to_queue(self):
        """Test that job emits events to the queue."""
        store = JobStore()
        events = ["event1", "event2", "event3"]
        runnable = MockRunnable(events=events)

        _, job = store.create_job(runnable, {})

        received_events = []
        while True:
            event = await job.queue.get()
            if event is JOB_DONE_SENTINEL:
                break
            received_events.append(event)

        assert received_events == events

    @pytest.mark.asyncio
    async def test_job_emits_done_sentinel(self):
        """Test that job emits JOB_DONE_SENTINEL when complete."""
        store = JobStore()
        runnable = MockRunnable(events=["event1"])

        _, job = store.create_job(runnable, {})

        # Consume events
        event1 = await job.queue.get()
        assert event1 == "event1"

        sentinel = await job.queue.get()
        assert sentinel is JOB_DONE_SENTINEL

    @pytest.mark.asyncio
    async def test_job_removed_from_store_on_completion(self):
        """Test that job is removed from store when task completes."""
        store = JobStore()
        runnable = MockRunnable(events=["event1"])

        job_id, job = store.create_job(runnable, {})

        # Job should be in store initially
        assert store.get_job(job_id) is not None

        # Consume all events
        while True:
            event = await job.queue.get()
            if event is JOB_DONE_SENTINEL:
                break

        # Wait for the task to fully complete and callback to fire
        await job.task

        # Give the event loop a chance to process the done callback
        await asyncio.sleep(0)

        # Job should be removed from store
        assert store.get_job(job_id) is None

    @pytest.mark.asyncio
    async def test_job_runs_to_completion_without_consumer(self):
        """Test that job runs to completion even if no one consumes events."""
        store = JobStore()
        runnable = MockRunnable(events=["event1", "event2", "event3"])

        job_id, job = store.create_job(runnable, {})

        # Don't consume any events, just wait for the task to complete
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

        # Consume only first 2 events slowly
        event1 = await job.queue.get()
        await asyncio.sleep(0.05)
        event2 = await job.queue.get()

        # Wait for the job to complete
        await job.task

        # Job should have completed regardless of consumer speed
        assert runnable.completed is True
        assert runnable.events_emitted == events

    @pytest.mark.asyncio
    async def test_job_runs_to_completion_when_consumer_stops(self):
        """Test that job runs to completion even if consumer stops reading."""
        store = JobStore()
        events = ["event1", "event2", "event3", "event4", "event5"]
        runnable = MockRunnable(events=events)

        job_id, job = store.create_job(runnable, {})

        # Consume only first event then "disconnect" (stop reading)
        first_event = await job.queue.get()
        assert first_event == "event1"

        # Simulate consumer disconnect by just waiting for task to complete
        await job.task

        # Job should have run to completion
        assert runnable.started is True
        assert runnable.completed is True
        assert runnable.events_emitted == events

        # Remaining events should still be in queue
        remaining = []
        while not job.queue.empty():
            event = job.queue.get_nowait()
            remaining.append(event)

        assert remaining == ["event2", "event3", "event4", "event5", JOB_DONE_SENTINEL]

    @pytest.mark.asyncio
    async def test_multiple_jobs_independent(self):
        """Test that multiple jobs run independently."""
        store = JobStore()
        runnable1 = MockRunnable(events=["a1", "a2"], delay=0.01)
        runnable2 = MockRunnable(events=["b1", "b2", "b3"], delay=0.01)

        job_id1, job1 = store.create_job(runnable1, {})
        job_id2, job2 = store.create_job(runnable2, {})

        # Both jobs should be in store
        assert store.get_job(job_id1) is not None
        assert store.get_job(job_id2) is not None

        # Wait for both to complete
        await asyncio.gather(job1.task, job2.task)

        assert runnable1.completed is True
        assert runnable2.completed is True

        # Both should be removed from store
        assert store.get_job(job_id1) is None
        assert store.get_job(job_id2) is None

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
    async def test_job_emits_sentinel_on_error(self):
        """Test that job emits JOB_DONE_SENTINEL even when runnable raises."""
        store = JobStore()

        async def failing_runnable(**kwargs):
            yield "event1"
            raise ValueError("Something went wrong")

        _, job = store.create_job(failing_runnable, {})

        # Should be able to get the first event
        event1 = await job.queue.get()
        assert event1 == "event1"

        # Should get sentinel even after error (not hang forever)
        sentinel = await job.queue.get()
        assert sentinel is JOB_DONE_SENTINEL

        # Task should have the exception
        with pytest.raises(ValueError, match="Something went wrong"):
            await job.task

    @pytest.mark.asyncio
    async def test_job_completes_when_consumer_cancelled(self):
        """Test that job runs to completion even when consumer task is cancelled."""
        store = JobStore()
        events = ["event1", "event2", "event3", "event4", "event5"]
        runnable = MockRunnable(events=events, delay=0.02)

        _, job = store.create_job(runnable, {})

        async def consumer():
            """Consumer that will be cancelled after first event."""
            event = await job.queue.get()
            assert event == "event1"
            # Simulate doing some work, then getting cancelled
            await asyncio.sleep(1)  # Will be cancelled before this completes

        consumer_task = asyncio.create_task(consumer())

        # Let consumer get first event
        await asyncio.sleep(0.01)

        # Cancel consumer (simulates HTTP disconnect)
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        # Wait for job to complete
        await job.task

        # Job should have completed fully
        assert runnable.completed is True
        assert runnable.events_emitted == events

    @pytest.mark.asyncio
    async def test_queue_accumulates_events_without_consumer(self):
        """Test that events accumulate in queue when no consumer is reading."""
        store = JobStore()
        events = ["event1", "event2", "event3"]
        runnable = MockRunnable(events=events)

        _, job = store.create_job(runnable, {})

        # Wait for job to complete without consuming anything
        await job.task

        # All events should be in the queue
        assert job.queue.qsize() == 4  # 3 events + sentinel

        # Can still consume them after job completion
        received = []
        while True:
            event = await job.queue.get()
            if event is JOB_DONE_SENTINEL:
                break
            received.append(event)

        assert received == events
