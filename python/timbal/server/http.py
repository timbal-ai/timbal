import argparse
import json
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from ..logs import setup_logging
from ..state import RunContext, set_run_context
from ..utils import ImportSpec, is_port_in_use
from .jobs import JOB_DONE_SENTINEL, JobStore

logger = structlog.get_logger("timbal.server.http")


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    import_spec: ImportSpec,
) -> AsyncGenerator[None, None]:
    logger.info("loading_runnable", import_spec=import_spec)
    app.state.runnable = import_spec.load()
    app.state.job_store = JobStore()
    yield


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
    parts = raw.split("::")
    if len(parts) != 2:
        raise RuntimeError(f"Invalid TIMBAL_RUNNABLE format: {raw}")
    import_spec = ImportSpec(
        path=Path(parts[0]).expanduser().resolve(),
        target=parts[1],
    )

    app = FastAPI(lifespan=lambda app: lifespan(app, import_spec))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

        _, job = app.state.job_store.create_job(app.state.runnable, req_data, job_id=run_id)

        output_event = None
        while True:
            event = await job.queue.get()
            if event is JOB_DONE_SENTINEL:
                break
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

        _, job = app.state.job_store.create_job(app.state.runnable, req_data, job_id=run_id)

        async def event_streamer() -> AsyncGenerator[str, None]:
            while True:
                event = await job.queue.get()
                if event is JOB_DONE_SENTINEL:
                    break
                yield f"data: {json.dumps(event.model_dump())}\n\n"

        return StreamingResponse(event_streamer(), media_type="text/event-stream")

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


if __name__ == "__main__":
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
    args = parser.parse_args()

    if args.version:
        print(f"timbal.server.http {__version__}")  # noqa: T201
        sys.exit(0)

    load_dotenv()

    # We can overwrite the env configuration with the --import_spec flag
    import_spec = args.import_spec
    if not import_spec:
        import_spec = os.getenv("TIMBAL_RUNNABLE")
        if not import_spec:
            import_spec = os.getenv("TIMBAL_FLOW")  # Legacy
            if import_spec:
                print(  # noqa: T201
                    "TIMBAL_FLOW environment variable is deprecated. Please use TIMBAL_RUNNABLE instead.",
                    file=sys.stderr,
                )

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

    os.environ["TIMBAL_RUNNABLE"] = import_spec
    uvicorn.run(
        "timbal.server.http:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_config=None,
    )
