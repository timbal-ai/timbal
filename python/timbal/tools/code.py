"""
Tool handlers for agents to read and analyze code files.
"""

import os
import difflib
from pathlib import Path
import tempfile
from typing import Optional


async def read_file(
    file_path: str,
    max_chars: Optional[int] = None,
) -> str:
    """
    Read a file and return its content with metadata.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (None for all)
    
    Returns:
        The file content
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    
    if path.is_dir():
        return f"Path is a directory, not a file: {file_path}"
    
    try:
        with open(path, 'r', encoding="utf-8") as f:
            text = f.read()
    except PermissionError:
        return f"Permission denied: cannot read file {file_path}"
    
    if max_chars is not None:
        text = text[:max_chars]

    return text


async def edit_code(
    path: str,
    content: str,
    dry_run: bool = False,
):
    """ 
    Write content to a file

    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        dry_run: Whether to write the file or not. If not, return a unified diff.

    Returns:
        Whether the file was written successfully or not
    """
    
    p = Path(path)
    original_content = p.read_text() if p.exists() else ""
    if original_content == content:
        return "Content already in file"
    
    if dry_run:
        diff = "\n".join(difflib.unified_diff(
            original_content.splitlines(), 
            content.splitlines(),
            fromfile=str(p), 
            tofile=str(p) + " (new)", 
            lineterm=""
        ))
        diff_path = p.with_name(p.name + ".diff")
        fd, tmp = tempfile.mkstemp(dir=str(diff_path.parent))
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(diff)
            os.replace(tmp, str(diff_path))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return "Content written to diff file: " + str(diff_path)

    p.write_text(content, encoding="utf-8")
    return "Content written to the file"