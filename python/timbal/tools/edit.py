"""
Edit tool for making targeted changes to existing files.

Performs exact string replacements in files with diff output.
Supports replacing all occurrences or just the first match.
Supports ~ (home directory) and environment variables in paths.
"""
import difflib
import hashlib
import os
from pathlib import Path

import structlog

from ..core.tool import Tool
from ..errors import FileModifiedError, FileNotReadError
from ..state import get_run_context

logger = structlog.get_logger("timbal.tools.edit")


class Edit(Tool):

    # TODO Add parameter to limit permissions to a specific path
    def __init__(self, **kwargs):

        async def _edit(
            path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ) -> str:
            """
            Edit a file by replacing old_string with new_string using EXACT string matching.

            CRITICAL: old_string must match the file content EXACTLY, including:
            - All whitespace (spaces, tabs, newlines)
            - Indentation (must match precisely)
            - Line endings
            - Every character exactly as it appears in the file
            
            If the match fails, read the file first to see the exact content.

            Args:
                path: Path to the file to edit
                old_string: The exact string to replace (must match character-for-character including all whitespace)
                new_string: The replacement string
                replace_all: If True, replace all occurrences. If False, replace only the first occurrence

            Returns:
                Unified diff showing the changes made
            """
            if old_string == new_string:
                raise ValueError("No changes made - old_string and new_string are identical")

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
                raise ValueError(f"Path is a directory, not a file: {path}")

            # Verify file state tracking BEFORE reading - fail fast
            # This is a security feature, we don't enforce the user to keep track of the fs state in the run context
            if run_context and hasattr(run_context, "_fs_state"):
                # Check if file has been read in this conversation
                if str(path) not in run_context._fs_state:
                    raise FileNotReadError(
                        f"Cannot edit {path} - file has not been read in this conversation. "
                        f"Please read the file first to understand its current state."
                    )

            original_bytes = path.read_bytes()
            # Verify file hasn't changed since last read
            # This is a security feature, we don't enforce the user to keep track of the fs state in the run context
            if run_context and hasattr(run_context, "_fs_state"):
                current_hash = hashlib.sha256(original_bytes).hexdigest()
                stored_hash = run_context._fs_state[str(path)]
                if stored_hash and current_hash != stored_hash:
                    raise FileModifiedError(
                        f"Cannot edit {path} - file has been modified since you last read it. "
                        f"Please read the file again to see the current state."
                    )

            original_content = original_bytes.decode("utf-8")
            if old_string not in original_content:
                raise ValueError(f"String not found in file: '{old_string}'")

            if replace_all:
                new_content = original_content.replace(old_string, new_string)
            else:
                new_content = original_content.replace(old_string, new_string, 1)

            # Generate clean, IDE-style diff with minimal context
            diff_lines = list(difflib.unified_diff(
                original_content.splitlines(keepends=False),
                new_content.splitlines(keepends=False),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
                lineterm="",
                n=3  # 3 lines of context (standard)
            ))

            path.write_text(new_content, encoding="utf-8")
            
            # Update file state tracking with new hash
            if run_context and hasattr(run_context, "_fs_state"):
                new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
                run_context._fs_state[str(path)] = new_hash

            return "\n".join(diff_lines)

        super().__init__(
            name="edit",
            description=(
                "Edit an existing file by replacing old_string with new_string. "
                "CRITICAL: Uses EXACT string matching - old_string must match the file content "
                "character-for-character including all whitespace, indentation, tabs, and line endings."
            ),
            handler=_edit,
            **kwargs
        )
