import argparse
import os
import sys

# Set env vars before any timbal imports to silence deprecation warnings
os.environ.setdefault("TIMBAL_DELTA_EVENTS", "true")
os.environ.setdefault("TIMBAL_SUPPRESS_EVENTS", "tracing_setup")

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .. import __version__
from ..logs import setup_logging
from ..utils import ImportSpec
from .runner import run_evals
from .utils import discover_config, discover_eval_files, parse_eval_file

console = Console()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Timbal Evals CLI",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Path to eval file or directory containing evals. Defaults to current directory.",
    )
    parser.add_argument(
        "--runnable",
        type=str,
        default=None,
        help="Fully qualified name of the Runnable to evaluate (e.g., path/to/file.py::my_agent). "
        "If not provided, will look for 'runnable' in evalconf.yaml.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level. Use DEBUG or INFO to see runnable logs. Defaults to WARNING.",
    )
    parser.add_argument(
        "-s",
        "--no-capture",
        action="store_true",
        help="Disable output capture. Show stdout/stderr in real-time (like pytest -s).",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Show version and exit.",
    )
    args = parser.parse_args()

    if args.version:
        console.print(f"timbal.eval {__version__}")
        sys.exit(0)

    load_dotenv(override=True)

    # Set env var before setup_logging reads it
    os.environ["TIMBAL_LOG_LEVEL"] = args.log_level
    setup_logging()

    # Discover config file (evalconf.yaml)
    config = discover_config(args.path)

    # CLI --runnable overrides config file
    runnable_fqn = args.runnable or config.get("runnable")
    if not runnable_fqn:
        console.print("[red]Error:[/red] No runnable specified. Use --runnable or add 'runnable' to evalconf.yaml.")
        sys.exit(1)

    runnable_fqn_parts = runnable_fqn.split("::")
    if len(runnable_fqn_parts) != 2:
        console.print("[red]Error:[/red] Invalid import spec format. Use 'path/to/file.py::object_name'")
        sys.exit(1)

    runnable_path, runnable_target = runnable_fqn_parts
    runnable_spec = ImportSpec(
        path=Path(runnable_path).expanduser().resolve(),
        target=runnable_target,
    )

    try:
        runnable = runnable_spec.load()
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to load runnable: {e}")
        sys.exit(1)

    eval_files = discover_eval_files(args.path)
    if not eval_files:
        console.print(f"[yellow]Warning:[/yellow] No eval files found in {args.path}")
        sys.exit(0)

    evals = [eval for eval_file in eval_files for eval in parse_eval_file(eval_file, runnable)]
    if not evals:
        console.print(f"[yellow]Warning:[/yellow] No evals found in {len(eval_files)} file(s)")
        sys.exit(0)

    summary = asyncio.run(
        run_evals(
            evals,
            capture=not args.no_capture,
        )
    )

    # Exit with error code if any eval failed
    sys.exit(0 if summary.failed == 0 else 1)
