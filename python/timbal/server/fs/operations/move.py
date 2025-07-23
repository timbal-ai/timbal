from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import field_validator

from .base import BaseOperation

logger = structlog.get_logger("timbal.server.fs.operations.mv")


class MoveOperation(BaseOperation):
    """Move/rename a file or directory."""
    type: Literal["move"] = "move"
    new_path: str

    @field_validator('new_path')
    @classmethod
    def validate_new_path(cls, v: str) -> str:
        """Validate that new_path is within allowed directory."""
        if cls._base_path is None:
            return v
            
        try:
            full_path = cls._base_path / v
            full_path.resolve().relative_to(cls._base_path.resolve())
            return v
        except ValueError as e:
            raise ValueError("New path outside allowed directory") from e

    async def __call__(self, base_path: Path) -> dict[str, Any]:
        """Execute the move operation."""
        try:
            old_path = base_path / self.path
            new_path = base_path / self.new_path
            
            if not old_path.exists():
                return {"error": f"Source path does not exist: {self.path}"}
                
            if new_path.exists():
                return {"error": f"Destination path already exists: {self.new_path}"}
                
            # Create parent directories if they don't exist
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            old_path.rename(new_path)
            logger.info("file_moved", old_path=self.path, new_path=self.new_path)
            return {"success": True, "message": f"Moved {self.path} to {self.new_path}"}
            
        except Exception as e:
            logger.error("move_failed", old_path=self.path, new_path=self.new_path, error=str(e))
            return {"error": f"Failed to move file: {str(e)}"}