"""
EditCode tool

A secure tool for editing code files.
Supports dry-run mode to preview changes and generates diff files.
"""

import difflib
import os
from pathlib import Path

import structlog

from timbal import Tool

logger = structlog.get_logger("timbal.tools.edit_code")

class Edit(Tool):

    def __init__(
        self, 
        **kwargs
    ):
        async def edit_code(
            path: str,
            content: str,
            dry_run: bool = False,
        ):           
            try:
                p = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
                
                # Read original content if file exists
                original_content = ""
                if p.exists():
                    if p.is_dir():
                        return f"Path is a directory, not a file: {path}"
                    try:
                        original_content = p.read_text(encoding="utf-8")
                    except PermissionError:
                        return f"Permission denied: cannot read file {path}"
                
                # Check if content is the same
                if original_content == content:
                    return "Content already in file"
                
                if dry_run:
                    # Generate diff
                    diff = "\n".join(difflib.unified_diff(
                        original_content.splitlines(), 
                        content.splitlines(),
                        fromfile=str(p), 
                        tofile=str(p) + " (new)", 
                        lineterm=""
                    ))
                    
                    # Create diff file in the same directory
                    diff_path = p.with_name(p.name + ".diff")
                    try:
                        diff_path.write_text(diff, encoding="utf-8")
                        return f"Diff written to: {diff_path}"
                    except PermissionError:
                        return f"Permission denied: cannot write diff file {diff_path}"
                        
                # Write the actual file
                try:
                    p.write_text(content, encoding="utf-8")
                    return f"Content written to: {p}"
                except PermissionError:
                    return f"Permission denied: cannot write to file {path}"
                    
            except Exception as e:
                return f"Unexpected error: {e}"


        super().__init__(
            name="edit_code",
            handler=edit_code,
            **kwargs
        )