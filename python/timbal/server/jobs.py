import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from uuid_extensions import uuid7

# How long a finished job stays readable before the store forgets it. The window
# is what makes reconnection work at the edges: a consumer that dropped just
# before the run ended still has this long to come back, collect the tail, and
# see the terminal flag. Without it, "job not found" is the reward for
# reconnecting a second too late.
DEFAULT_RETENTION_SECS = 300.0

# Batch ceilings for `Job.read`, mirroring the platform's `/events` endpoints so
# a client written against one is written against both.
DEFAULT_READ_LIMIT = 500
MAX_READ_LIMIT = 2000


class Job:
    """A running (or recently finished) run, plus its replayable event log.

    Events are appended to a log and addressed by a 1-based monotonic ``seq``,
    not handed out through a queue. A queue can only be read once, so the first
    consumer to disconnect takes the events with it — fine for a script, useless
    when the consumer is a browser and the run is a ten-minute agent turn that
    keeps going after the socket drops. With a log every reader holds its own
    cursor, so reconnecting is just asking for everything after the last ``seq``
    it saw, and two readers (a tab and a task panel) can watch one run without
    racing each other for events.

    The log is unbounded for the run's lifetime and freed when the store reaps
    the job. A run that streams forever will grow it forever — the same exposure
    the undrained queue had, and it ends the same way, with the run.
    """

    def __init__(self, task: asyncio.Task | None = None) -> None:
        self.task = task
        # (seq, event), contiguous from seq 1, so the event with seq S always
        # sits at index S-1 — which is what lets `read` slice instead of scan.
        self.events: list[tuple[int, Any]] = []
        self.next_seq = 1
        self.done = False
        self.finished_at: float | None = None
        self._waiters: list[asyncio.Future] = []

    @property
    def last_seq(self) -> int:
        """Highest seq appended so far (0 before the first event)."""
        return self.next_seq - 1

    def append(self, event: Any) -> None:
        self.events.append((self.next_seq, event))
        self.next_seq += 1
        self._wake()

    def finish(self) -> None:
        self.done = True
        self.finished_at = time.monotonic()
        self._wake()

    def _wake(self) -> None:
        """Release every long-poll waiter.

        Deliberately synchronous, which is why `append` and `finish` are too:
        the producer calls `finish` from a ``finally`` that may already be
        unwinding a cancellation, and awaiting a lock there is how you get a
        cancelled run whose readers hang until their timeout.
        """
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()

    def read(
        self,
        after: int = 0,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> tuple[list[tuple[int, Any]], int, bool]:
        """``(events, next_cursor, done)`` for the events after ``after``.

        ``next_cursor`` is the last seq in the batch, or ``after`` untouched
        when the batch is empty, so it can always be fed straight back in.
        ``done`` means the run finished *and* this batch reached the end: a
        terminal run whose events don't fit in one page reports ``False`` so
        the caller keeps paging instead of stopping on the flag.
        """
        limit = max(1, min(limit, MAX_READ_LIMIT))
        start = max(after, 0)
        batch = self.events[start : start + limit]
        next_cursor = batch[-1][0] if batch else after
        return batch, next_cursor, self.done and next_cursor >= self.last_seq

    async def wait(self, after: int = 0, timeout: float = 0.0) -> None:
        """Block until something lands past ``after``, the run ends, or timeout.

        Returns rather than raising on timeout — the caller re-reads either way.
        """
        if timeout <= 0 or self.done or self.last_seq > after:
            return
        await self._sleep(timeout)

    async def follow(self, after: int = 0) -> AsyncGenerator[tuple[int, Any], None]:
        """Yield ``(seq, event)`` from ``after`` until the run is done.

        The streaming face of `read`, for SSE. Starting at a non-zero ``after``
        replays the backlog first and then continues live, so a reconnecting
        stream and a fresh one are the same code path.
        """
        cursor = after
        while True:
            batch, cursor, done = self.read(cursor, limit=MAX_READ_LIMIT)
            for entry in batch:
                yield entry
            if done:
                return
            if not batch:
                await self._sleep(None)

    async def _sleep(self, timeout: float | None) -> None:
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            if timeout is None:
                await waiter
            else:
                await asyncio.wait_for(waiter, timeout)
        except (TimeoutError, asyncio.TimeoutError):
            pass
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)


class JobStore:
    def __init__(self, retention_secs: float = DEFAULT_RETENTION_SECS) -> None:
        self._jobs: dict[str, Job] = {}
        self._retention_secs = retention_secs

    def create_job(self, runnable, params, job_id: str | None = None) -> tuple[str, Job]:
        _job_id: str = job_id if job_id is not None else uuid7(as_type="hex")  # type: ignore
        # Finished jobs are held for the retention window rather than dropped on
        # completion, so reaping happens here — one sweep per new job, no timer.
        self.reap()
        job = Job()
        job.task = asyncio.create_task(self._run(runnable, params, job))
        self._jobs[_job_id] = job
        return _job_id, job

    def get_job(self, job_id: str) -> Job | None:
        """The job, or ``None`` once it has been reaped (or never existed here).

        Callers must not read ``None`` as "finished cleanly" — it also covers a
        run this process never had. Both mean the log is unavailable, which is
        what the `/events` route reports as ``expired``.
        """
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job by its ID.

        Returns True if the job was found and cancelled, False otherwise —
        including for a job that is merely being retained after finishing,
        which is not cancellable however addressable it still is.
        """
        job = self._jobs.get(job_id)
        if job is None or job.done:
            return False
        return job.task.cancel() if job.task is not None else False

    def reap(self, now: float | None = None) -> None:
        """Forget jobs whose retention window has passed."""
        now = time.monotonic() if now is None else now
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None and now - job.finished_at >= self._retention_secs
        ]
        for job_id in expired:
            del self._jobs[job_id]

    async def _run(self, runnable, params, job: Job):
        try:
            async for event in runnable(**params):
                job.append(event)
        finally:
            job.finish()
