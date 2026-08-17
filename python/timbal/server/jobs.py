import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from uuid_extensions import uuid7

# How long a finished job stays readable before the store forgets it. The window
# is what makes reconnection work at the edges: a consumer that dropped just
# before the run ended still has this long to come back for the tail.
DEFAULT_RETENTION_SECS = 300.0

# Batch ceilings for `Job.read`, mirroring the platform's other `/events`
# endpoints so a client written against one is written against both.
DEFAULT_READ_LIMIT = 500
MAX_READ_LIMIT = 2000

# Ring-buffer cap on a replayable log. `None` (or 0) means unlimited. A single
# event larger than `max_bytes` is kept — dropping the tip would lose the live
# stream, not just the reconnect backlog.
DEFAULT_MAX_EVENTS = 50_000
DEFAULT_MAX_BYTES = 32 * 1024 * 1024


class RunIdInUse(Exception):
    """A caller supplied a run id that a still-running job already owns.

    Run ids come from the client (`context.id`), so a collision is reachable
    from the outside. Overwriting the entry would leave the first run executing
    with nothing pointing at it: unreadable, uncancellable, and invisible to
    the reaper.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(f"run id {job_id!r} is already in use by a running job")
        self.job_id = job_id


class Job:
    """A running (or recently finished) run, plus its replayable event log.

    Events are appended to a log and addressed by a 1-based monotonic ``seq``,
    not handed out through a queue. A queue can only be read once, so the first
    consumer to disconnect takes the events with it. With a log every reader
    holds its own cursor, so reconnecting is just asking for everything after
    the last ``seq`` it saw, and two readers can watch one run without racing
    each other for events.

    ``replayable=False`` opts out of keeping the log: events are forgotten as
    the single attached reader passes them. That is what `/run` wants — it reads
    to completion in one pass, discards everything but the final event, and
    nothing can reconnect to it — and it keeps that path O(unread) rather than
    holding an entire run's deltas until the retention window lapses.

    A replayable log is a ring: once ``max_events`` / ``max_bytes`` is exceeded
    the oldest entries are dropped and ``forgotten_through`` advances. Reconnect
    from a cursor still inside the window works; ``after < forgotten_through``
    is a gap — the same class of lie ``expired`` exists to catch — not a silent
    skip to the new head.
    """

    def __init__(
        self,
        task: asyncio.Task | None = None,
        *,
        replayable: bool = True,
        max_events: int | None = DEFAULT_MAX_EVENTS,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None:
        self.task = task
        self.replayable = replayable
        self._max_events = max_events or 0
        self._max_bytes = max_bytes or 0
        # (seq, event) for every seq past `forgotten_through`, contiguous, so
        # seq S sits at index S - forgotten_through - 1 — which is what lets
        # `read` slice instead of scan.
        self.events: list[tuple[int, Any]] = []
        self._sizes: list[int] = []
        self._nbytes = 0
        self.forgotten_through = 0
        self.next_seq = 1
        self.done = False
        self.finished_at: float | None = None
        self._waiters: list[asyncio.Future] = []

    @property
    def last_seq(self) -> int:
        """Highest seq appended so far (0 before the first event)."""
        return self.next_seq - 1

    def append(self, event: Any) -> None:
        nbytes = _event_nbytes(event)
        self.events.append((self.next_seq, event))
        self._sizes.append(nbytes)
        self._nbytes += nbytes
        self.next_seq += 1
        self._trim()
        self._wake()

    def finish(self) -> None:
        self.done = True
        self.finished_at = time.monotonic()
        self._wake()

    def expired(self, retention_secs: float, now: float | None = None) -> bool:
        """Whether this job's retention window has lapsed."""
        if self.finished_at is None:
            return False
        now = time.monotonic() if now is None else now
        return now - self.finished_at >= retention_secs

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
    ) -> tuple[list[tuple[int, Any]], int, bool, bool]:
        """``(events, next_cursor, done, gapped)`` for the events after ``after``.

        ``next_cursor`` is the last seq in the batch, or ``after`` untouched
        when the batch is empty, so it can always be fed straight back in.
        ``done`` means the run finished *and* this batch reached the end: a
        terminal run whose events don't fit in one page reports ``False`` so
        the caller keeps paging instead of stopping on the flag.
        ``gapped`` means ``after`` is behind the retained window — the events
        between the cursor and the floor were dropped. The caller must treat
        that as ``expired``, not skip to the new head.
        """
        if after < self.forgotten_through:
            return [], after, True, True
        limit = max(1, min(limit, MAX_READ_LIMIT))
        start = max(after, 0)
        offset = start - self.forgotten_through
        batch = self.events[offset : offset + limit]
        next_cursor = batch[-1][0] if batch else after
        return batch, next_cursor, self.done and next_cursor >= self.last_seq, False

    async def wait(self, after: int = 0, timeout: float = 0.0) -> None:
        """Block until something lands past ``after``, the run ends, or timeout.

        Returns rather than raising on timeout — the caller re-reads either way.
        """
        if timeout <= 0 or after < self.forgotten_through:
            return
        waiter = self._waiter()
        try:
            if self.done or self.last_seq > after:
                return
            await asyncio.wait_for(waiter, timeout)
        except TimeoutError:
            pass
        finally:
            self._discard(waiter)

    async def follow(self, after: int = 0) -> AsyncGenerator[tuple[int, Any], None]:
        """Yield ``(seq, event)`` from ``after`` until the run is done.

        The streaming face of `read`, for SSE. Starting at a non-zero ``after``
        replays the backlog first and then continues live, so a reconnecting
        stream and a fresh one are the same code path.
        """
        cursor = after
        while True:
            # Registered before the read, not after: an `append` landing in
            # between has to resolve a waiter that already exists. Register
            # second and that wake is lost — and for the last event of a run,
            # lost means this generator waits for an event that never comes.
            waiter = self._waiter()
            batch, cursor, done, gapped = self.read(cursor, limit=MAX_READ_LIMIT)
            if gapped:
                self._discard(waiter)
                return
            if not batch and not done:
                try:
                    await waiter
                finally:
                    self._discard(waiter)
                continue

            self._discard(waiter)
            for entry in batch:
                yield entry
            if not self.replayable:
                self._forget_through(cursor)
            if done:
                return

    def _waiter(self) -> asyncio.Future:
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        return waiter

    def _discard(self, waiter: asyncio.Future) -> None:
        if waiter in self._waiters:
            self._waiters.remove(waiter)

    def _forget_through(self, seq: int) -> None:
        """Drop the log up to ``seq``. Only sound with a single reader."""
        keep = seq - self.forgotten_through
        if keep > 0:
            self._nbytes -= sum(self._sizes[:keep])
            del self.events[:keep]
            del self._sizes[:keep]
            self.forgotten_through = seq

    def _over_cap(self, n: int, nbytes: int) -> bool:
        return (self._max_events > 0 and n > self._max_events) or (
            self._max_bytes > 0 and nbytes > self._max_bytes
        )

    def _trim(self) -> None:
        """Drop the oldest events until the ring fits. Never drops the tip."""
        if not self.replayable:
            return
        drop = 0
        n = len(self.events)
        nbytes = self._nbytes
        while n - drop > 1 and self._over_cap(n - drop, nbytes):
            nbytes -= self._sizes[drop]
            drop += 1
        if drop:
            del self.events[:drop]
            del self._sizes[:drop]
            self._nbytes = nbytes
            self.forgotten_through += drop


def _event_nbytes(event: Any) -> int:
    dump_json = getattr(event, "model_dump_json", None)
    if callable(dump_json):
        return len(dump_json())
    if isinstance(event, str):
        return len(event)
    if isinstance(event, (bytes, bytearray)):
        return len(event)
    return len(repr(event))


class JobStore:
    def __init__(
        self,
        retention_secs: float = DEFAULT_RETENTION_SECS,
        max_events: int | None = DEFAULT_MAX_EVENTS,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._retention_secs = retention_secs
        self._max_events = max_events
        self._max_bytes = max_bytes

    def create_job(
        self,
        runnable,
        params,
        job_id: str | None = None,
        replayable: bool = True,
    ) -> tuple[str, Job]:
        """Start ``runnable`` on its own task, tracked under ``job_id``.

        Raises `RunIdInUse` if a running job already holds that id. Pass
        ``replayable=False`` for a consumer that reads the run to completion in
        one pass and needs no reconnection (see `Job`).
        """
        _job_id: str = job_id if job_id is not None else uuid7(as_type="hex")  # type: ignore
        # `get_job` expires lazily, one id at a time; sweep here as well so the
        # dict itself stays bounded when finished jobs are never read back.
        self.reap()
        existing = self.get_job(_job_id)
        if existing is not None and not existing.done:
            raise RunIdInUse(_job_id)

        job = Job(
            replayable=replayable,
            max_events=self._max_events,
            max_bytes=self._max_bytes,
        )
        job.task = asyncio.create_task(self._run(runnable, params, job))
        # A job nobody can reconnect to has nothing to retain, so it goes as
        # soon as it ends instead of waiting out the window.
        if not replayable:
            job.task.add_done_callback(lambda _: self._forget(_job_id, job))
        self._jobs[_job_id] = job
        return _job_id, job

    def get_job(self, job_id: str, now: float | None = None) -> Job | None:
        """The job, or ``None`` once its window has lapsed (or it never existed here).

        Expiry is evaluated here rather than only in `reap`, so the retention
        window holds on an idle server too — otherwise a job's lifetime depends
        on whether more traffic happens to arrive.

        Callers must not read ``None`` as "finished cleanly": it also covers a
        run this process never had. Both mean the log is unavailable, which is
        what the `/events` route reports as ``expired``.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.expired(self._retention_secs, now):
            self._forget(job_id, job)
            return None
        return job

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
        """Forget every job whose retention window has passed."""
        now = time.monotonic() if now is None else now
        for job_id in [
            job_id for job_id, job in self._jobs.items() if job.expired(self._retention_secs, now)
        ]:
            del self._jobs[job_id]

    def _forget(self, job_id: str, job: Job) -> None:
        """Drop ``job_id``, but only while it still refers to ``job``."""
        if self._jobs.get(job_id) is job:
            del self._jobs[job_id]

    async def _run(self, runnable, params, job: Job):
        try:
            async for event in runnable(**params):
                job.append(event)
        finally:
            job.finish()
