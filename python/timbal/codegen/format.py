import importlib.util
import subprocess
import sys
from pathlib import Path

if importlib.util.find_spec("ruff") is None:
    raise ImportError(
        "ruff is required for codegen formatting. "
        "Install it with: pip install 'timbal[codegen]'"
    )

_RUFF_CMD = [sys.executable, "-m", "ruff"]

RUFF_FORMAT_ARGS = [
    *_RUFF_CMD,
    "format",
    "--isolated",
    "--line-length",
    "79",
    "--config",
    "indent-width=4",
    "--config",
    "format.indent-style='space'",
    "--config",
    "format.quote-style='double'",
    "--config",
    "format.skip-magic-trailing-comma=false",
]

RUFF_FIX_ARGS = [
    *_RUFF_CMD,
    "check",
    "--isolated",
    "--fix",
    "--select",
    "F401,F811,I",
]


def format_code(code: str, source_path: Path) -> str:
    # 1. Fix unused imports (F401), redefinitions (F811), and sort imports (I).
    result = subprocess.run(
        [*RUFF_FIX_ARGS, "--stdin-filename", str(source_path), "-"],
        input=code,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"ruff check --fix failed:\n{result.stderr}")
    code = result.stdout or code

    # 2. Format.
    result = subprocess.run(
        [*RUFF_FORMAT_ARGS, "--stdin-filename", str(source_path), "-"],
        input=code,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ruff format failed:\n{result.stderr}")
    return result.stdout
