"""
Read tool for file access with path expansion support.

Returns a File object with content formatted for LLM consumption.
Supports ~ (home directory) and environment variables in paths.
"""

import hashlib
import os
from itertools import islice
from pathlib import Path

import structlog

from ..core.tool import Tool
from ..state import get_run_context
from ..types.file import File

logger = structlog.get_logger("timbal.tools.read")


class Read(Tool):
    def __init__(self, **kwargs):
        async def _read(path: str, start_line: int | None = None, end_line: int | None = None) -> File | str:
            """
            Read a file at the specified path.

            Args:
                path: Path to the file to read
                start_line: Optional starting line number (1-indexed, inclusive)
                end_line: Optional ending line number (1-indexed, inclusive)

            Returns:
                File object with content (optionally sliced to line range)
            """
            run_context = get_run_context()
            # Resolve path with base_path security if run_context exists
            if run_context:
                path = run_context.resolve_cwd(path)
            else:
                # No run context - just expand and resolve normally
                path = Path(os.path.expandvars(os.path.expanduser(path))).resolve()

            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {path}")
            if path.is_dir():
                contents = "\n".join(item.name for item in path.iterdir())
                if not contents:
                    return "Empty directory"
                return contents

            # Update file state tracking with new hash
            if run_context:
                new_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                session = await run_context.get_session()
                if "fs_state" not in session:
                    session["fs_state"] = {}
                session["fs_state"][str(path)] = new_hash

            file = File.validate(path)

            # These are file types that are not text and handled specially by Timbal FileContent
            if file.__source_extension__ in [".xlsx", ".eml", ".docx"]:
                return file
            elif file.__content_type__.startswith("image/"):
                return file
            elif file.__content_type__.startswith("audio/"):
                return file
            elif file.__content_type__ == "application/pdf":
                return file

            # ? Enable multiple encodings
            if start_line is None and end_line is None:
                return file.read().decode("utf-8")

            # Read the specified line range efficiently
            with open(path, encoding="utf-8") as f:
                # Calculate slice parameters (convert from 1-indexed to 0-indexed)
                start_idx = (start_line - 1) if start_line is not None else 0
                # Calculate how many lines to read
                if end_line is not None:
                    if start_line is not None:
                        num_lines = end_line - start_line + 1
                    else:
                        num_lines = end_line
                else:
                    num_lines = None  # Read until end
                # Use islice for efficient line reading
                # Skip lines before start_idx, then take num_lines
                lines = list(islice(f, start_idx, start_idx + num_lines if num_lines else None))

            # Return empty string if no lines found (out of range)
            content = "".join(lines)
            return content if content else ""

        super().__init__(
            name="read",
            description="Read a file at the specified path. Optionally specify start_line and end_line to read only a specific line range.",
            handler=_read,
            **kwargs,
        )
