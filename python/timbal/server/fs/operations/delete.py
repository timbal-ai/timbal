import shutil
from pathlib import Path
from typing import Any, Literal

import structlog

from .base import BaseOperation

logger = structlog.get_logger("timbal.server.fs.operations.delete")


class DeleteOperation(BaseOperation):
    """Delete a file or directory."""
    type: Literal["delete"] = "delete"

    async def __call__(self, base_path: Path) -> dict[str, Any]:
        try:
            full_path = base_path / self.path
                
            if full_path.is_file():
                full_path.unlink()
                logger.info("file_deleted", path=self.path)
                return {"success": True, "message": f"File deleted: {self.path}"}
            
            if full_path.is_dir():
                shutil.rmtree(full_path)
                logger.info("directory_deleted", path=self.path)
                return {"success": True, "message": f"Directory deleted: {self.path}"}
                
            return {"error": f"Path is not an existing file or directory: {self.path}"}
            
        except Exception as e:
            logger.error("delete_failed", path=self.path, error=str(e))
            return {"error": f"Failed to delete file or directory: {str(e)}"}