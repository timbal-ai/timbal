import argparse
import asyncio
import sys
from pathlib import Path

import structlog
import uvicorn
from dotenv import find_dotenv, load_dotenv

from ... import __version__
from ...logs import setup_logging
from ..utils import is_port_in_use
from .app import create_app

logger = structlog.get_logger("timbal.server.fs")


async def main(host: str, port: int, base_path: Path) -> None:
    app = create_app(base_path)
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=None,
    )
    
    server = uvicorn.Server(config)
    await server.serve()


parser = argparse.ArgumentParser(description="Timbal file system server.")
parser.add_argument(
    "-v", 
    "--version", 
    action="store_true", 
    help="Show version and exit."
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
    default=4488,
    help="Port to bind to.",
)
parser.add_argument(
    "--base-path",
    dest="base_path",
    type=str,
    help="Path to the directory to watch.",
)
args = parser.parse_args()

if args.version:
    print(f"timbal.servers.watch {__version__}") # noqa: T201
    sys.exit(0)

if is_port_in_use(args.port):
    print(f"Port {args.port} is already in use. Please use a different port.") # noqa: T201
    sys.exit(1)

if not args.base_path:
    print("Please provide a base path.") # noqa: T201
    sys.exit(1)
base_path = Path(args.base_path).expanduser().resolve()
if not base_path.is_dir():
    print(f"Path {base_path} is not a directory or does not exist.") # noqa: T201
    sys.exit(1)

logger.info("loading_dotenv", path=find_dotenv())
load_dotenv(override=True)
setup_logging()

logger.info("starting_fs_server", host=args.host, port=args.port, base_path=base_path)

try:
    asyncio.run(main(host=args.host, port=args.port, base_path=base_path))
except KeyboardInterrupt:
    logger.info("server_stopped_by_user")
except Exception as e:
    logger.error("server_error", error=str(e))
    sys.exit(1)
