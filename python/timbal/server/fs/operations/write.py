from pathlib import Path
from typing import Any, Literal

import structlog

from .base import BaseOperation

logger = structlog.get_logger("timbal.server.fs.operations.write")


class WriteOperation(BaseOperation):
    """Write content to file."""
    type: Literal["write"] = "write"
    content: str = ""

    async def __call__(self, base_path: Path) -> dict[str, Any]:
        """Execute the write operation."""
        try:
            full_path = base_path / self.path
            
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(self.content, encoding="utf-8")
            logger.info("file_written", path=self.path)
            return {"success": True, "message": f"File written: {self.path}"}
            
        except Exception as e:
            logger.error("write_file_failed", path=self.path, error=str(e))
            return {"error": f"Failed to write file: {str(e)}"}