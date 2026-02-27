import subprocess
from pathlib import Path

RUFF_ARGS = [
    "ruff",
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


def format_code(code: str, source_path: Path) -> str:
    result = subprocess.run(
        [*RUFF_ARGS, "--stdin-filename", str(source_path), "-"],
        input=code,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ruff format failed:\n{result.stderr}")
    return result.stdout
