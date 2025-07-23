from pathlib import Path
from typing import Any, Literal

import structlog

from .base import BaseOperation

logger = structlog.get_logger("timbal.server.fs.operations.read")


class ReadOperation(BaseOperation):
    """Read file content."""
    type: Literal["read"] = "read"

    async def __call__(self, base_path: Path) -> dict[str, Any]:
        """Execute the read operation."""
        try:
            full_path = base_path / self.path
            
            if not full_path.exists():
                return {"error": f"File does not exist: {self.path}"}
                
            if not full_path.is_file():
                return {"error": f"Path is not a file: {self.path}"}
                
            content = full_path.read_text(encoding="utf-8")
            return {"success": True, "content": content}
            
        except Exception as e:
            logger.error("read_file_failed", path=self.path, error=str(e))
            return {"error": f"Failed to read file: {str(e)}"}