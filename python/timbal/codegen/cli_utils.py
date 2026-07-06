"""Shared argparse helpers for the codegen CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_json_arg(raw: str, flag: str, *, hint: str = "") -> object:
    """Parse a CLI argument as JSON, raising a clear ``ValueError`` on failure.

    ``json.JSONDecodeError`` messages like ``Expecting value: line 1 column 1``
    give no clue that the fix is quoting; this wraps them with the flag name,
    the offending input, and an optional usage hint.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        message = f"{flag} must be a valid JSON literal (got {raw!r}): {e}."
        if hint:
            message = f"{message} {hint}"
        raise ValueError(message) from e


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
