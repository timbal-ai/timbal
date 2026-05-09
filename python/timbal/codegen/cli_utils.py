"""Shared argparse helpers for the codegen CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def arg_input(value: str) -> str:
    """argparse ``type`` callable supporting file and stdin redirection.

    Convention:

    - ``"-"`` reads from stdin (useful for piping).
    - ``"@path/to/file"`` reads the file contents as text. The leading ``@``
      is stripped before opening, so the path may be absolute or relative.
    - Any other value is returned unchanged.

    This sidesteps shell quoting problems for arguments that contain
    embedded newlines, parentheses, or backslashes (e.g. JSON config blobs
    or Python function definitions). Pass the payload via a file with
    ``--config @./payload.json`` and let the shell stay out of the way.
    """
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        path = Path(value[1:])
        try:
            return path.read_text()
        except OSError as e:
            raise argparse.ArgumentTypeError(f"cannot read {path}: {e}") from e
    return value
