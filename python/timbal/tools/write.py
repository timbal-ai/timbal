"""
Write tool for file creation and editing with path expansion support.

Creates new files or modifies existing ones with diff output.
Supports dry-run mode to preview changes without writing files.
Supports ~ (home directory) and environment variables in paths.
"""
import difflib
import os
from pathlib import Path

import structlog

from timbal import Tool

logger = structlog.get_logger("timbal.tools.write")


class Write(Tool):

    # TODO Add parameter to limit permissions to a specific path
    def __init__(self, **kwargs):

        async def _write(path: str, content: str, dry_run: bool = False):
            path = Path(os.path.expandvars(os.path.expanduser(path))).resolve()

            if path.is_dir():
                raise ValueError(f"Path is a directory, not a file: {path}")

            original_content = ""
            file_exists = path.exists()
            if file_exists:
                original_content = path.read_text(encoding="utf-8")

            # Check if content is the same
            if original_content == content:
                return "Content already matches - no changes needed"

            # Generate diff (always)
            diff = "\n".join(difflib.unified_diff(
                original_content.splitlines(),
                content.splitlines(),
                fromfile=str(path) + (" (existing)" if file_exists else " (new file)"),
                tofile=str(path) + " (modified)",
                lineterm=""
            ))

            if dry_run:
                # Return diff without writing
                action = "modify" if file_exists else "create"
                return f"Preview - would {action} {path}:\n\n{diff}"

            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write the actual file
            path.write_text(content, encoding="utf-8")
            action = "modified" if file_exists else "created"
            return f"File {action}: {path}\n\nChanges:\n{diff}"


        super().__init__(
            name="write",
            description="Create a new file or edit an existing one at the specified path",
            handler=_write,
            **kwargs
        )
