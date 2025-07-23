from pathlib import Path
from typing import Any, Literal

import structlog

from .base import BaseOperation

logger = structlog.get_logger("timbal.server.fs.operations.ls")


class ListOperation(BaseOperation):
    """List directory contents."""
    type: Literal["list"] = "list"

    async def __call__(self, base_path: Path) -> dict[str, Any]:
        """Execute the ls operation."""
        try:
            full_path = base_path / self.path
            
            if not full_path.exists():
                return {"error": f"Directory does not exist: {self.path}"}
                
            if not full_path.is_dir():
                return {"error": f"Path is not a directory: {self.path}"}
                
            contents = []
            for item in full_path.iterdir():
                contents.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
                
            return {"success": True, "contents": contents}
            
        except Exception as e:
            logger.error("list_directory_failed", path=self.path, error=str(e))
            return {"error": f"Failed to list directory: {str(e)}"}