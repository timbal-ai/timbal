"""
List tool for directory contents with path expansion support.

Supports ~ (home directory) and environment variables in paths.
"""
import os
from pathlib import Path

import structlog

from ..core.tool import Tool

logger = structlog.get_logger("timbal.tools.list")


class List(Tool):

    def __init__(self, **kwargs):
        
        async def _list(path: str) -> list[str]:
            path = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
            # If path does not exist, or is not a directory, the following line will raise an error
            return list(path.iterdir())
        
        super().__init__(
            name="list",
            description="List all files and subdirectories in the given directory path",
            handler=_list,
            **kwargs
        )
