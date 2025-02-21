import argparse
import asyncio
import importlib.util
import signal
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request

from .. import __version__
from ..logs import setup_logging
from .utils import is_port_in_use

logger = structlog.get_logger("timbal.servers.http")


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    module_spec: tuple[Path, str | None],
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
    
    # TODO Any additional setup

    logger.info("loading_module", module_spec=module_spec)
    spec = importlib.util.spec_from_file_location(module_spec[0].stem, module_spec[0].as_posix())
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if module_spec[1]:
            if hasattr(module, module_spec[1]):
                obj = getattr(module, module_spec[1])
                app.state.flow = obj
            else:
                raise ValueError(f"Module {module_spec[0]} has no object {module_spec[1]}")
        else:
            raise NotImplementedError("? support loading entire module")
    else:
        raise ValueError(f"Failed to load module {module_spec[0]}")
    
    yield
    
    # TODO Any additional cleanup
    

async def main(
    host: str,
    port: int,
    workers: int,
    module_spec: tuple[Path, str | None],
) -> None:
    """Runs the HTTP server with the specified configuration.

    Sets up a FastAPI application with healthcheck, shutdown, and run endpoints.
    Handles graceful shutdown on SIGTERM and SIGINT signals.

    Args:
        host: The hostname to bind the server to.
        port: The port number to listen on.
        workers: Number of worker processes to spawn.
        module_spec: A tuple containing (Path to Python module, Optional object name)
                    to be loaded by the lifespan manager.
    """
    shutdown_event = asyncio.Event()
    app = FastAPI(lifespan=lambda app: lifespan(app, module_spec))

    @app.get("/healthcheck")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/shutdown")
    async def shutdown() -> dict[str, str]:
        shutdown_event.set()
        return {"status": "ok"}

    @app.post("/run")
    async def run(request: Request) -> dict[str, Any]:
        data = await request.json()
        response = await app.state.flow.complete(**data)
        # TODO Serializers
        response = response["response"].content[0].text
        return {"status": "ok", "response": response}

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

    if not args.module_spec:
        print("No module spec provided. Use --module_spec to specify a module to load.") # noqa: T201
        sys.exit(1)

    module_parts = args.module_spec.split(":")
    if len(module_parts) > 2:
        print("Invalid module spec format. Use 'path/to/file.py:object_name' or 'path/to/file.py'") # noqa: T201
        sys.exit(1)
    elif len(module_parts) == 2:
        module_path, module_name = module_parts
        module_spec = (Path(module_path).expanduser().resolve(), module_name)
    else:
        module_spec = (Path(module_parts[0]).expanduser().resolve(), None)

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
