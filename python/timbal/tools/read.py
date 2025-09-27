"""
Read tool for file access with path expansion support.

Returns a File object with content formatted for LLM consumption.
Supports ~ (home directory) and environment variables in paths.
"""
import os
from pathlib import Path

import structlog

from ..core.tool import Tool
from ..types.file import File

logger = structlog.get_logger("timbal.tools.read")


class Read(Tool):

    def __init__(self, **kwargs):
        
        # TODO Add the possibility to read specific line ranges or limit the number of lines read
        async def _read(path: str) -> File:
            path = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
            return File.validate(path)
            
        super().__init__(
            name="read",
            description="Read a file at the specified path",
            handler=_read,
            **kwargs
        )
