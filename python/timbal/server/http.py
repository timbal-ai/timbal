import argparse
import asyncio
import json
import os
import signal
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
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
    """Manages the lifecycle of the FastAPI application.

    This context manager handles the setup and teardown of the application,
    including loading the specified Python module and its runnable object.

    Args:
        app: The FastAPI application instance.
        import_spec: ImportSpec containing the path to Python module and target object name.
                     The loaded object will be set as app.state.runnable.

    Raises:
        ValueError: If the module or specified object cannot be loaded.
        ImportError: If the module cannot be imported.
        AttributeError: If the target object does not exist in the module.
    """
    # ? Any additional setup

    logger.info("loading_runnable", import_spec=import_spec)
    app.state.runnable = import_spec.load()
    app.state.job_store = JobStore()

    yield

    # ? Any additional cleanup


def create_app(
    import_spec: ImportSpec,
    shutdown_event: asyncio.Event,
) -> FastAPI:
    """Creates a FastAPI application for the Timbal HTTP server.

    This function creates a FastAPI application with endpoints for running Timbal
    runnables (tools, agents, workflows) over a REST API. It handles module loading,
    parameter validation, runnable execution, and streaming responses.

    Args:
        import_spec: ImportSpec containing the path to Python module and target object name.
        shutdown_event: Asyncio event to signal graceful shutdown.

    Returns:
        FastAPI: Configured FastAPI application with all endpoints.
    """
    app = FastAPI(lifespan=lambda app: lifespan(app, import_spec))

    @app.get("/healthcheck")
    async def healthcheck() -> Response:
        return Response(status_code=204)

    @app.post("/shutdown")
    async def shutdown() -> Response:
        shutdown_event.set()
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


async def main(
    host: str,
    port: int,
    workers: int,
    import_spec: ImportSpec,
) -> None:
    """Runs the HTTP server with the specified configuration.

    Sets up a FastAPI application with healthcheck, shutdown, run, and stream endpoints.
    Handles graceful shutdown on SIGTERM and SIGINT signals. Optionally enables
    ngrok tunneling for public access.

    Args:
        host: The hostname to bind the server to (e.g., '0.0.0.0', '127.0.0.1').
        port: The port number to listen on.
        workers: Number of worker processes to spawn.
        import_spec: ImportSpec containing the path to Python module and target object name.
                     The loaded object will be set as app.state.runnable.

    Raises:
        Exception: If server startup fails or runnable loading fails.
    """
    shutdown_event = asyncio.Event()

    app = create_app(import_spec, shutdown_event)

    if os.getenv("TIMBAL_ENABLE_NGROK", "false").lower() == "true":
        from pyngrok import ngrok

        public_url = ngrok.connect(str(port), "http")
        logger.info("ngrok_public_url", public_url=public_url)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level=None,
    )
    server = uvicorn.Server(config)

    def signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _signum, _frame: signal_handler())

    serve_task = asyncio.create_task(server.serve())
    await shutdown_event.wait()
    server.should_exit = True
    await serve_task


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
    setup_logging()

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
    import_path, import_target = import_parts
    import_spec = ImportSpec(
        path=Path(import_path).expanduser().resolve(),
        target=import_target,
    )

    if is_port_in_use(args.port):
        print(f"Port {args.port} is already in use. Please use a different port.")  # noqa: T201
        sys.exit(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            main(
                host=args.host,
                port=args.port,
                workers=args.workers,
                import_spec=import_spec,
            )
        )
    except Exception as e:
        logger.error("server_stopped", error=str(e))
    finally:
        pending = asyncio.all_tasks(loop)
        logger.info("loop_pending_tasks", count=len(pending))
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        logger.info("loop_closed")
