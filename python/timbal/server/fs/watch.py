import argparse
import asyncio
import json
import signal
import sys
import io
import zipfile
from pathlib import Path
import os

import httpx
import structlog
import websockets
from dotenv import find_dotenv, load_dotenv
from watchfiles import awatch, Change

logger = structlog.get_logger("timbal.server.fs.watch")


class FileSystemWatcher:

    def __init__(self, base_path: Path, base_url: str):
        self.base_path = base_path
        self.base_url = base_url
        self.ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self.watcher_task = None
        self.shutdown_event = asyncio.Event()
        self._sync()


    # TODO Implement some logic for handling overwrites
    def _sync(self):
        with httpx.Client() as client:
            res = client.get(f"{self.base_url}/sync")
            res.raise_for_status()
            self.base_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(res.content)) as zip_file:
                zip_file.extractall(self.base_path)


    async def _watcher(self, ws: websockets.ClientConnection):
        # TODO Add the files in .dockerignore to the ignore_paths
        async for changes in awatch(self.base_path, debounce=300):
            if self.shutdown_event.is_set():
                break
                
            for change_type, file_path in changes:
                src_path = Path(file_path)
                
                if change_type == Change.added:
                    if src_path.is_dir():
                        # Handle directory creation - send mkdir + all files inside
                        await self._handle_dir_added(ws, src_path)
                    else:
                        operation = {
                            "type": "write",
                            "path": src_path.relative_to(self.base_path).as_posix(),
                            "content": src_path.read_text(encoding="utf-8"),
                        }
                        await self._send_operation(ws, operation)
                        
                elif change_type == Change.deleted:
                    operation = {
                        "type": "delete",
                        "path": src_path.relative_to(self.base_path).as_posix(),
                    }
                    await self._send_operation(ws, operation)
                    
                elif change_type == Change.modified:
                    if src_path.is_dir():
                        continue
                    operation = {
                        "type": "write",
                        "path": src_path.relative_to(self.base_path).as_posix(),
                        "content": src_path.read_text(encoding="utf-8"),
                    }
                    await self._send_operation(ws, operation)
                else:
                    logger.warning("unknown_change_type", change_type=change_type)
                    continue


    async def _send_operation(self, ws: websockets.ClientConnection, operation: dict):
        """Send operation via WebSocket with error handling"""
        try:
            logger.info("sending_operation", operation=operation)
            await ws.send(json.dumps(operation))
        except websockets.exceptions.ConnectionClosed:
            logger.info("connection_closed")
            raise


    async def _handle_dir_added(self, ws: websockets.ClientConnection, dir_path: Path):
        """Handle directory creation by sending mkdir + all files inside recursively"""
        # First send the directory creation
        mkdir_op = {
            "type": "mkdir",
            "path": dir_path.relative_to(self.base_path).as_posix(),
        }
        await self._send_operation(ws, mkdir_op)
        
        # Then send all files inside the directory
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    write_op = {
                        "type": "write",
                        "path": file_path.relative_to(self.base_path).as_posix(),
                        "content": file_path.read_text(encoding="utf-8"),
                    }
                    await self._send_operation(ws, write_op)
                except (OSError, UnicodeDecodeError) as e:
                    logger.warning("skipping_file", file_path=str(file_path), error=str(e))
            elif file_path.is_dir() and file_path != dir_path:
                # Send nested directories
                nested_mkdir_op = {
                    "type": "mkdir", 
                    "path": file_path.relative_to(self.base_path).as_posix(),
                }
                await self._send_operation(ws, nested_mkdir_op)


    def stop_watching(self):
        if self.watcher_task:
            self.watcher_task.cancel()
            logger.info("file_watcher_stopped")


    async def run(self):
        reconnect_delay = 1
        while not self.shutdown_event.is_set():
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("websocket_connection_established")
                    reconnect_delay = 1  # Reset delay on successful connection
                    self.watcher_task = asyncio.create_task(self._watcher(ws))
                    
                    # Wait for watcher to finish or shutdown signal
                    _, pending = await asyncio.wait(
                        [self.watcher_task, asyncio.create_task(self.shutdown_event.wait())],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in pending:
                        task.cancel()

                    if self.shutdown_event.is_set():
                        break

            except (websockets.exceptions.WebSocketException, ConnectionRefusedError) as e:
                logger.error("connection_failed", error=str(e))

            except Exception as e:
                logger.error("unexpected_error", error=str(e))

            if not self.shutdown_event.is_set():
                logger.info("reconnecting", delay=reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff

        self.stop_watching()
        logger.info("client_shutdown_gracefully")


    def shutdown(self):
        logger.info("shutdown_signal_received")
        self.shutdown_event.set()


async def main(client: FileSystemWatcher):
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, client.shutdown)
    
    await client.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timbal watch client.")
    parser.add_argument(
        "-v", 
        "--version", 
        action="store_true", 
        help="Show version and exit."
    )
    parser.add_argument(
        "--base-path",
        dest="base_path",
        type=str,
        help="Path to watch.",
        required=True
    )
    args = parser.parse_args()

    if args.version:
        print("dev")
        sys.exit(0)

    base_path = Path(args.base_path).expanduser().resolve()
    
    dotenv_path = find_dotenv()
    if dotenv_path:
        print(f"Loading .env from {dotenv_path}")
        load_dotenv(override=True)
    
    base_url = os.getenv("TIMBAL_BASE_URL", "http://localhost:4488")

    watcher = FileSystemWatcher(base_path=base_path, base_url=base_url)
    asyncio.run(main(watcher))
