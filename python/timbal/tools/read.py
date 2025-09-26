"""
ReadCode tool

A tool for reading file contents with optional character limits.
"""

import os
from pathlib import Path

import structlog

from timbal import Tool

logger = structlog.get_logger("timbal.tools.read_code")


class Read(Tool):

    def __init__(self, **kwargs):
        
        async def read_code(
            path: str,
            max_chars: int | None = None,
        ) -> str:
            """
            Read a file and return its content with metadata.
            
            Args:
                file_path: Path to the file to read
                max_lines: Maximum number of lines to read (None for all)
            
            Returns:
                The file content
            """
            p = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
            
            if not p.exists():
                return f"File not found: {path}"
            
            if p.is_dir():
                return f"Path is a directory, not a file: {path}"
            
            try:
                with open(p, encoding="utf-8") as f:
                    text = f.read()
            except PermissionError:
                return f"Permission denied: cannot read file {path}"
            
            if max_chars is not None:
                text = text[:max_chars]

            return text
            
        super().__init__(
            name="read_code",
            handler=read_code,
            **kwargs
        )