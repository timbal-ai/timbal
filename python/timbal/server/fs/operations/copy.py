import shutil
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import field_validator

from .base import BaseOperation

logger = structlog.get_logger("timbal.server.fs.operations.cp")


class CopyOperation(BaseOperation):
    """Copy a file or directory."""
    type: Literal["copy"] = "copy"
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
        """Execute the copy operation."""
        try:
            old_path = base_path / self.path
            new_path = base_path / self.new_path
            
            if not old_path.exists():
                return {"error": f"Source path does not exist: {self.path}"}
                
            if new_path.exists():
                return {"error": f"Destination path already exists: {self.new_path}"}
                
            # Create parent directories if they don't exist
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            if old_path.is_file():
                shutil.copy2(old_path, new_path)
                logger.info("file_copied", old_path=self.path, new_path=self.new_path)
                return {"success": True, "message": f"Copied file {self.path} to {self.new_path}"}
            elif old_path.is_dir():
                shutil.copytree(old_path, new_path)
                logger.info("directory_copied", old_path=self.path, new_path=self.new_path)
                return {"success": True, "message": f"Copied directory {self.path} to {self.new_path}"}
            else:
                return {"error": f"Source path is neither file nor directory: {self.path}"}
                
        except Exception as e:
            logger.error("copy_failed", old_path=self.path, new_path=self.new_path, error=str(e))
            return {"error": f"Failed to copy: {str(e)}"}
