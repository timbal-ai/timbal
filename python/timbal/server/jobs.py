import asyncio

from uuid_extensions import uuid7

JOB_DONE_SENTINEL = object()


class Job:
    def __init__(self, task: asyncio.Task, queue: asyncio.Queue):
        self.task = task
        self.queue = queue


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create_job(self, runnable, params, job_id: str | None = None) -> tuple[str, Job]:
        _job_id: str = job_id if job_id is not None else uuid7(as_type="str").replace("-", "")  # type: ignore
        queue = asyncio.Queue()
        task = asyncio.create_task(self._run(runnable, params, queue))
        task.add_done_callback(lambda _: self._jobs.pop(_job_id, None))
        job = Job(task, queue)
        self._jobs[_job_id] = job
        return _job_id, job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def _run(self, runnable, params, queue):
        try:
            async for event in runnable(**params):
                await queue.put(event)
        finally:
            await queue.put(JOB_DONE_SENTINEL)
