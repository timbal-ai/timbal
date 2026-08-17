import argparse
import asyncio
import contextlib
import json
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as e:
    raise ImportError(
        "fastapi and uvicorn are required to run the timbal server. Install them with: pip install 'timbal[server]'"
    ) from e
from dotenv import load_dotenv

from .. import __version__
from ..logs import setup_logging
from ..state import RunContext, set_run_context
from ..utils import ImportSpec, is_port_in_use
from .jobs import DEFAULT_READ_LIMIT, JobStore, RunIdInUse
from .voice import merge_voice_config

logger = structlog.get_logger("timbal.server.http")

# Ceiling on `/runs/{run_id}/events` long-polls, matching the platform's other
# `/events` endpoints. Long enough to be worth holding open, short enough to
# stay under the usual proxy idle timeouts.
MAX_WAIT_MS = 30_000


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    import_spec: ImportSpec,
) -> AsyncGenerator[None, None]:
    logger.info("loading_runnable", import_spec=import_spec)
    runnable = import_spec.load()
    app.state.runnable = runnable
    app.state.job_store = JobStore()
    app.state.voice_config = merge_voice_config(runnable)
    # Serverless voice boxes: serve one session, then exit (env read here at
    # server start — post-CRIU-restore — never at import time).
    from .single_session import init_single_session_guard

    app.state.single_session_guard = init_single_session_guard()
    from .livekit_session import maybe_start_livekit_session

    livekit_task = maybe_start_livekit_session(app)
    # Voice warmup off the boot path: pre-import the voice stack (and pre-load
    # the local turn-detection ONNX models) so the first voice session doesn't
    # pay those costs. Gated on actual voice intent — non-voice deployments
    # must not download/load ONNX models just because timbal[voice] is
    # installed. The playground launcher opts its child servers in via
    # TIMBAL_VOICE_WARMUP=1 (see voice.voice_warmup_intended).
    from ..core.agent import Agent
    from .voice import voice_warmup_intended, warmup_voice_stack

    warmup_task = (
        asyncio.create_task(warmup_voice_stack(app.state.voice_config))
        if isinstance(runnable, Agent) and voice_warmup_intended(runnable)
        else None
    )
    try:
        yield
    finally:
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
        if livekit_task is not None and not livekit_task.done():
            livekit_task.cancel()
            # Await the teardown: the driver's finally finalizes the recording
            # and disconnects the room. Cancelling without awaiting hands the
            # still-cleaning task to the loop's shutdown mass-cancel, whose
            # second CancelledError lands mid-finally and skips that cleanup.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await livekit_task
        if app.state.single_session_guard is not None:
            app.state.single_session_guard.shutdown()


def create_app() -> FastAPI:
    """Factory for the FastAPI app. Called by uvicorn in each worker process.

    Reads TIMBAL_RUNNABLE from the environment so that it works as a zero-arg
    factory with uvicorn's ``factory=True`` — required for multi-worker support
    since uvicorn spawns workers via multiprocessing and can't pickle app instances.
    """
    setup_logging()

    raw = os.environ.get("TIMBAL_RUNNABLE")
    if not raw:
        raise RuntimeError("TIMBAL_RUNNABLE environment variable is not set.")
    import_spec = ImportSpec.from_fqn(raw)

    app = FastAPI(lifespan=lambda app: lifespan(app, import_spec))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .rtc import router as rtc_router
    from .telephony import router as telephony_router
    from .voice import router as voice_router

    app.include_router(voice_router)
    app.include_router(rtc_router)
    app.include_router(telephony_router)

    @app.get("/healthcheck")
    async def healthcheck() -> Response:
        return Response(status_code=204)

    @app.get("/params_model_schema")
    async def params_model_schema() -> Response:
        params_model_schema = app.state.runnable.params_model_schema
        return JSONResponse(
            status_code=200,
            content=params_model_schema,
        )

    @app.get("/return_model_schema")
    async def return_model_schema() -> Response:
        return_model_schema = app.state.runnable.return_model_schema
        return JSONResponse(
            status_code=200,
            content=return_model_schema,
        )

    def _create_job(run_id: str | None, req_data: dict, replayable: bool = True):
        """Start a job, turning a colliding client-supplied run id into a 409.

        `context.id` is client-controlled, so two requests can name the same
        run. Letting the second overwrite the first would leave the first still
        executing with nothing pointing at it — unreadable, uncancellable, and
        never reaped.
        """
        try:
            return app.state.job_store.create_job(
                app.state.runnable, req_data, job_id=run_id, replayable=replayable
            )
        except RunIdInUse as e:
            raise HTTPException(
                status_code=409,
                detail=f"Run id '{e.job_id}' is already in use by a running run.",
            ) from e

    @app.post("/run")
    async def run(req: Request) -> Response:
        req_data = await req.json()
        run_context = req_data.pop("context", None)
        if run_context is None:
            run_context = req_data.pop("run_context", None)

        run_id = None
        if run_context is not None:
            run_context = RunContext.model_validate(run_context)
            set_run_context(run_context)
            run_id = run_context.id

        # Nothing can reconnect to a `/run`, so its events are dropped as they
        # are consumed instead of being kept for the retention window.
        _, job = _create_job(run_id, req_data, replayable=False)

        output_event = None
        async for _, event in job.follow():
            output_event = event

        return JSONResponse(
            status_code=200,
            content=output_event.model_dump() if output_event else None,
        )

    @app.post("/stream")
    async def stream(req: Request) -> Response:
        req_data = await req.json()
        run_context = req_data.pop("context", None)
        if run_context is None:
            run_context = req_data.pop("run_context", None)

        run_id = None
        if run_context is not None:
            run_context = RunContext.model_validate(run_context)
            set_run_context(run_context)
            run_id = run_context.id

        _, job = _create_job(run_id, req_data)

        async def event_streamer() -> AsyncGenerator[str, None]:
            # `id:` carries the seq so a client reading the stream always knows
            # the cursor to reconnect with, without having to look inside the
            # payload. Reconnection itself goes through
            # `GET /runs/{run_id}/events?after=`: this route is a POST, so it is
            # not reachable by `EventSource` and nothing here consumes
            # `Last-Event-ID`.
            async for seq, event in job.follow():
                yield f"id: {seq}\ndata: {json.dumps(event.model_dump(mode='json'))}\n\n"

        return StreamingResponse(event_streamer(), media_type="text/event-stream")

    @app.get("/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        after: int = 0,
        limit: int = DEFAULT_READ_LIMIT,
        wait_ms: int = 0,
    ) -> Response:
        """Replay a run's events after a cursor — the reconnect path for `/stream`.

        A dropped connection does not stop the run: the job keeps producing into
        its log. Poll here with the last `seq` seen to collect whatever was
        missed and to keep following (`wait_ms` long-polls instead of spinning).

        `expired` is the one field a client cannot skip. When the log is
        unavailable — reaped after retention, never held by this process, or
        belonging to a `/run` that kept no log — the honest answer looks exactly
        like a clean end-of-stream (`done: true`, no events, the cursor
        unmoved), so a client that only reads `done` stops early and silently
        loses the tail. `expired: true` means *terminal but possibly
        incomplete*: reconcile from wherever runs are persisted rather than
        trusting this response as the end.

        Note that "never held by this process" covers a run that is alive and
        well in a sibling worker: the log is process-local, so this route is
        only meaningful when runs are pinned to one process.
        """
        job = app.state.job_store.get_job(run_id)
        if job is None or not job.replayable:
            return JSONResponse(
                status_code=200,
                content={
                    "run_id": run_id,
                    "events": [],
                    "next_cursor": after,
                    "done": True,
                    "expired": True,
                },
            )

        events, next_cursor, done = job.read(after, limit)
        if not events and not done and wait_ms > 0:
            await job.wait(after, min(wait_ms, MAX_WAIT_MS) / 1000)
            events, next_cursor, done = job.read(after, limit)

        return JSONResponse(
            status_code=200,
            content={
                "run_id": run_id,
                "events": [
                    {"seq": seq, "data": event.model_dump(mode="json")} for seq, event in events
                ],
                "next_cursor": next_cursor,
                "done": done,
                "expired": False,
            },
        )

    @app.post("/cancel/{run_id}")
    async def cancel(run_id: str) -> Response:
        cancelled = app.state.job_store.cancel_job(run_id)
        if cancelled:
            return Response(status_code=204)
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found or already completed"},
        )

    return app


def run_server_cli(argv: list[str] | None = None) -> None:
    """CLI for the full HTTP server (also used by ``python -m timbal.server``)."""
    parser = argparse.ArgumentParser(description="Timbal HTTP server.")
    parser.add_argument("-v", "--version", action="store_true", help="Show version and exit.")
    parser.add_argument(
        "--import_spec",
        dest="import_spec",
        type=str,
        help="Path to a python module and optional object (format: path/to/file.py::object_name)",
    )
    parser.add_argument(
        "--host",
        dest="host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to.",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=4444,
        help="Port to bind to.",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=1,
        help="Number of worker processes. Defaults to number of CPUs, or 1 if using a GPU.",
    )
    args = parser.parse_args(argv)

    if args.version:
        print(f"timbal.server.http {__version__}")  # noqa: T201
        sys.exit(0)

    load_dotenv()

    # We can overwrite the env configuration with the --import_spec flag
    import_spec = args.import_spec
    if not import_spec:
        import_spec = os.getenv("TIMBAL_RUNNABLE")

    if not import_spec:
        print(  # noqa: T201
            "No import spec provided. Set TIMBAL_RUNNABLE env variable or use --import_spec to specify a module to load.",
            file=sys.stderr,
        )
        sys.exit(1)

    import_parts = import_spec.split("::")
    if len(import_parts) != 2:
        print("Invalid import spec format. Use 'path/to/file.py::object_name' or 'path/to/file.py'", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # Resolve to absolute path so workers can find it.
    import_path = str(Path(import_parts[0]).expanduser().resolve())
    import_spec = f"{import_path}::{import_parts[1]}"

    if is_port_in_use(args.port):
        print(f"Port {args.port} is already in use. Please use a different port.")  # noqa: T201
        sys.exit(1)

    if args.workers > 1:
        # The job store lives in one process's memory, and nothing routes a
        # request to the worker that owns a given run. So `/cancel/{run_id}`
        # 404s and `/runs/{run_id}/events` reports `expired` whenever the
        # request lands on a sibling — for a run that is alive and fine.
        logger.warning(
            "multi_worker_runs_are_not_addressable_across_workers",
            workers=args.workers,
            detail=(
                "Run cancellation and event replay are process-local. With more than one "
                "worker, /cancel/{run_id} and /runs/{run_id}/events only work when the "
                "request happens to reach the worker running that run. Use a single worker, "
                "or pin runs to a worker at the load balancer."
            ),
        )

    os.environ["TIMBAL_RUNNABLE"] = import_spec
    uvicorn.run(
        "timbal.server.http:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_config=None,
    )


if __name__ == "__main__":
    run_server_cli()
