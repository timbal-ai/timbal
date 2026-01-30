"""
Write tool for file creation and editing with path expansion support.

Creates new files or modifies existing ones with diff output.
Supports ~ (home directory) and environment variables in paths.
"""

import difflib
import hashlib
import os
from pathlib import Path

import structlog

from ..core.tool import Tool
from ..state import get_run_context

logger = structlog.get_logger("timbal.tools.write")


class Write(Tool):
    # TODO Add parameter to limit permissions to a specific path
    def __init__(self, **kwargs):
        async def _write(path: str, content: str) -> str:
            """
            Write content to a file, creating it if it doesn't exist or overwriting if it does.

            Creates parent directories automatically if they don't exist.
            Returns a unified diff showing the changes made.

            Args:
                path: Path to the file to write (supports ~ and environment variables)
                content: The complete content to write to the file

            Returns:
                Unified diff showing the changes made (empty for new files)
            """
            # Resolve path with base_path security if run_context exists
            run_context = get_run_context()
            if run_context:
                path = run_context.resolve_cwd(path)
            else:
                # No run context - just expand and resolve normally
                path = Path(os.path.expandvars(os.path.expanduser(path))).resolve()

            if path.exists() and path.is_dir():
                raise ValueError(f"Path is a directory, not a file: {path}")

            # Read original content if file exists for diff generation
            original_content = ""
            if path.exists():
                original_bytes = path.read_bytes()
                original_content = original_bytes.decode("utf-8")

            # Generate clean, IDE-style diff with minimal context
            diff_lines = list(
                difflib.unified_diff(
                    original_content.splitlines(keepends=False),
                    content.splitlines(keepends=False),
                    fromfile=f"a/{path.name}",
                    tofile=f"b/{path.name}",
                    lineterm="",
                    n=3,  # 3 lines of context (standard)
                )
            )

            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            # Update file state tracking with new hash
            if run_context:
                new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                session = await run_context.get_session()
                if "fs_state" not in session:
                    session["fs_state"] = {}
                session["fs_state"][str(path)] = new_hash

            return "\n".join(diff_lines)

        super().__init__(
            name="write",
            description=(
                "Write content to a file, creating it if it doesn't exist or overwriting if it does. "
                "Automatically creates parent directories. Returns a diff showing changes."
            ),
            handler=_write,
            **kwargs,
        )
