from pathlib import Path
from typing import Any, Literal

import structlog

from .base import BaseOperation

logger = structlog.get_logger("timbal.server.fs.operations.mkdir")


class MkdirOperation(BaseOperation):
    """Create a directory."""
    type: Literal["mkdir"] = "mkdir"

    async def __call__(self, base_path: Path) -> dict[str, Any]:
        """Execute the mkdir operation."""
        try:
            full_path = base_path / self.path
            
            if full_path.exists():
                return {"error": f"Directory already exists: {self.path}"}
                
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info("directory_created", path=self.path)
            return {"success": True, "message": f"Directory created: {self.path}"}
            
        except Exception as e:
            logger.error("create_directory_failed", path=self.path, error=str(e))
            return {"error": f"Failed to create directory: {str(e)}"}