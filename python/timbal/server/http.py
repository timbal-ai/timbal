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
from ..state import RunContext
from ..types.file import File
from ..types.message import Message
from ..types.models import dump
from .utils import ModuleSpec, is_port_in_use, load_module

logger = structlog.get_logger("timbal.server.http")


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    module_spec: ModuleSpec,
) -> AsyncGenerator[None, None]:
    """Manages the lifecycle of the FastAPI application.

    This context manager handles the setup and teardown of the application,
    including loading the specified Python module and its object.

    Args:
        app: The FastAPI application instance.
        module_spec: A tuple containing (Path to Python module, Optional object name).
                     If object name is provided, it will be loaded from the module
                     and set as app.state.flow.

    Raises:
        ValueError: If the module or specified object cannot be loaded.
        NotImplementedError: If no object name is specified (currently unsupported).
    """
    # ? Any additional setup

    logger.info("loading_module", module_spec=module_spec)
    
    app.state.flow = load_module(module_spec)
    
    yield
    
    # ? Any additional cleanup


def create_app(
    module_spec: ModuleSpec, 
    shutdown_event: asyncio.Event,
) -> FastAPI:
    """HTTP server implementation for Timbal.

    This module provides the HTTP server implementation for running Timbal flows over a REST API.
    It handles module loading, parameter validation, and flow execution.

    Args:
        module_spec: Path to the flow module and object name.
        shutdown_event: Event to signal shutdown.
    """
    app = FastAPI(lifespan=lambda app: lifespan(app, module_spec))


    @app.get("/healthcheck")
    async def healthcheck() -> Response:
        return Response(status_code=204)


    @app.post("/shutdown")
    async def shutdown() -> Response:
        shutdown_event.set()
        return Response(status_code=204)


    @app.get("/params_model_schema")
    async def params_model_schema() -> Response:
        params_model_schema = app.state.flow.params_model_schema()
        return JSONResponse(
            status_code=200,
            content=params_model_schema,
        )


    @app.get("/return_model_schema")
    async def return_model_schema() -> Response:
        return_model_schema = app.state.flow.return_model_schema()
        return JSONResponse(
            status_code=200,
            content=return_model_schema,
        )


    @app.post("/run")
    async def run(req: Request) -> Response:
        req_data = await req.json()
        req_context = req_data.pop("context", None)
        if req_context is not None:
            req_context = RunContext.model_validate(req_context)
        else:
            req_context = RunContext()

        res_content = await app.state.flow.complete(context=req_context, **req_data)
        res_content = dump(res_content, req_context)

        return JSONResponse(
            status_code=200,
            content=res_content,
        )


    # ! WIP
    @app.post("/webhooks/twilio")
    async def twilio_webhook(req: Request) -> Response:
        form_data = await req.form()
        from_number = form_data.get("From", "") # e.g. whatsapp:+123456789
        from_number = from_number.replace("whatsapp:", "")
        message_body = form_data.get("Body", "")
        media_url = form_data.get("MediaUrl0", None)
        # content_type = form_data.get("MediaContentType0", "") if form_data.get("NumMedia", "0") != "0" else ""

        message_content = []
        # TODO We'll need to be able to pass specific authorization to perform the file download.
        if media_url is not None:
            message_content.append(File.validate(media_url))
        if message_body.strip():
            # ? We could add something here to allow the flow to distinguish between the user that's sending this.
            message_content.append(message_body)

        if not len(message_content):
            logger.warning("twilio_webhook_no_content", from_number=from_number)
            return Response(status_code=204)

        message_content.append(f"Message received at twilio webhook for {from_number}")

        message = Message.validate({
            "role": "user",
            "content": message_content,
        })

        req_context = RunContext()
        async for event in app.state.flow.run(context=req_context, prompt=message):
            if event.type == "STEP_OUTPUT":
                print(f"Agent: {event.output}")


    @app.post("/stream")
    async def stream(req: Request) -> Response:
        req_data = await req.json()
        req_context = req_data.pop("context", None)
        if req_context is not None:
            req_context = RunContext.model_validate(req_context)
        else:
            req_context = RunContext()

        # TODO Study if we need to filter these. Or if we need to add something to indicate chunks are for the response.
        async def event_streamer() -> AsyncGenerator[str, None]:
            async for event in app.state.flow.run(context=req_context, **req_data):
                event_content = dump(event, req_context) # Assumes dump returns serializable dict/list
                # Format as SSE message: data: <json_string>\n\n
                yield f"data: {json.dumps(event_content)}\n\n"

        return StreamingResponse(event_streamer(), media_type="text/event-stream")


    return app


async def main(
    host: str,
    port: int,
    workers: int,
    module_spec: ModuleSpec,
) -> None:
    """Runs the HTTP server with the specified configuration.

    Sets up a FastAPI application with healthcheck, shutdown, and run endpoints.
    Handles graceful shutdown on SIGTERM and SIGINT signals.

    Args:
        host: The hostname to bind the server to.
        port: The port number to listen on.
        workers: Number of worker processes to spawn.
        module_spec: A tuple containing (Path to Python module, Optional object name).
                     If object name is provided, it will be loaded from the module
                     and set as app.state.flow.
    """
    shutdown_event = asyncio.Event()

    app = create_app(module_spec, shutdown_event)

    if os.getenv("TIMBAL_ENABLE_NGROK", "false").lower() == "true":
        from pyngrok import ngrok
        public_url = ngrok.connect(port, "http")
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
    parser.add_argument(
        "-v", 
        "--version", 
        action="store_true", 
        help="Show version and exit."
    )
    parser.add_argument(
        "--module_spec",
        dest="module_spec",
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
        print(f"timbal.servers.http {__version__}") # noqa: T201
        sys.exit(0)

    # We can overwrite the env TIMBAL_FLOW variable with the --module_spec flag.
    module_spec = args.module_spec
    if not module_spec:
        module_spec = os.getenv("TIMBAL_FLOW")

    if not module_spec:
        print("No module spec provided. Set TIMBAL_FLOW env variable or use --module_spec to specify a module to load.") # noqa: T201
        sys.exit(1)

    module_parts = module_spec.split(":")
    if len(module_parts) > 2:
        print("Invalid module spec format. Use 'path/to/file.py:object_name' or 'path/to/file.py'") # noqa: T201
        sys.exit(1)
    elif len(module_parts) == 2:
        module_path, module_name = module_parts
        module_spec = ModuleSpec(
            path=Path(module_path).expanduser().resolve(), 
            object_name=module_name,
        )
    else:
        module_spec = ModuleSpec(
            path=Path(module_parts[0]).expanduser().resolve(), 
            object_name=None,
        )

    if is_port_in_use(args.port):
        print(f"Port {args.port} is already in use. Please use a different port.") # noqa: T201
        sys.exit(1)

    load_dotenv()
    setup_logging()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main(
            host=args.host, 
            port=args.port,
            workers=args.workers,
            module_spec=module_spec,
        ))
    except Exception as e:
        logger.error("server_stopped", error=str(e))
    finally:
        pending = asyncio.all_tasks(loop)
        logger.info("loop_pending_tasks", count=len(pending))
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        logger.info("loop_closed")
