import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from .operations import operation_adapter

logger = structlog.get_logger("timbal.server.fs.fastapi")


def create_app(base_path: Path) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Timbal File System Server")
    
    # Set base path for operations
    from .operations.base import BaseOperation
    BaseOperation._base_path = base_path
    
    # Connection tracking
    connections: set[WebSocket] = set()
    
    async def register_connection(websocket: WebSocket):
        """Register a new WebSocket connection."""
        connections.add(websocket)
        logger.info("client_connected", total_connections=len(connections))
        
        # Auto-sync notification on new connection
        try:
            sync_notification = {
                "type": "sync_available",
                "sync_url": "/sync",
                "message": "Use HTTP GET /sync to download all files as ZIP"
            }
            await websocket.send_text(json.dumps(sync_notification))
        except Exception as e:
            logger.error("auto_sync_notification_failed", error=str(e))
    
    async def unregister_connection(websocket: WebSocket):
        """Unregister a WebSocket connection."""
        connections.discard(websocket)
        logger.info("client_disconnected", total_connections=len(connections))
    
    async def handle_message(_websocket: WebSocket, message: str) -> dict[str, Any]:
        """Handle incoming WebSocket messages using operation classes."""
        try:
            data = json.loads(message)
            operation = operation_adapter.validate_python(data)
            return await operation(base_path)
                
        except json.JSONDecodeError:
            return {"error": "Invalid JSON message"}

        except Exception as e:
            logger.error("error_handling_message", error=str(e))
            return {"error": f"Invalid message format: {str(e)}"}
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}
    
    @app.get("/sync")
    async def sync_files():
        """HTTP endpoint to download all files as ZIP."""
        try:
            zip_buffer = BytesIO()
            file_count = 0
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in base_path.rglob("*"):
                    if file_path.is_file():
                        try:
                            relative_path = file_path.relative_to(base_path)
                            zip_file.write(file_path, relative_path)
                            file_count += 1
                        except Exception:
                            continue
            
            zip_data = zip_buffer.getvalue()
            
            return Response(
                content=zip_data,
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=files.zip",
                    "X-File-Count": str(file_count),
                    "X-Compressed-Size": str(len(zip_data))
                }
            )
            
        except Exception as e:
            logger.error("sync_endpoint_error", error=str(e))
            return Response(
                content=f"Sync failed: {str(e)}",
                status_code=500
            )
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time file operations."""
        await websocket.accept()
        await register_connection(websocket)
        
        try:
            while True:
                message = await websocket.receive_text()
                response = await handle_message(websocket, message)
                await websocket.send_text(json.dumps(response))
                
        except WebSocketDisconnect:
            logger.info("websocket_disconnected")
        except Exception as e:
            logger.error("websocket_error", error=str(e))
        finally:
            await unregister_connection(websocket)
    
    return app