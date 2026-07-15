import argparse
import os
import sys

os.environ.setdefault("TIMBAL_SUPPRESS_EVENTS", "tracing_setup")

import asyncio
from pathlib import Path

from dotenv import load_dotenv
try:
    from rich.console import Console
except ImportError as e:
    raise ImportError(
        "rich is required to run the timbal evals CLI. "
        "Install it with: pip install 'timbal[evals]'"
    ) from e

from .. import __version__
from ..logs import setup_logging
from ..utils import ImportSpec
from .runner import run_evals
from .utils import collect_evals, discover_config, dump_summary

console = Console()


def parse_path_and_filter(path_arg: str) -> tuple[Path, str | None]:
    """Parse path argument, extracting optional ::eval_name filter.

    Args:
        path_arg: Path string, optionally with ::eval_name suffix

    Returns:
        Tuple of (path, eval_name_filter or None)
    """
    if "::" in path_arg:
        path_str, eval_name = path_arg.rsplit("::", 1)
        return Path(path_str), eval_name
    return Path(path_arg), None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Timbal Evals CLI",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default=str(Path.cwd()),
        help="Path to eval file or directory. Use ::eval_name to run a specific eval (e.g., evals/test.yaml::my_eval).",
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
        "-t",
        "--tags",
        type=str,
        default=None,
        help="Filter evals by tags. Comma-separated list (e.g., 'smoke,fast'). "
        "Evals matching ANY of the tags will run.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write results as JSON to the given file. Use '-' to print JSON to stdout.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["pretty", "json"],
        default="pretty",
        help="Output format: 'pretty' (default) is the rich terminal report; "
        "'json' streams one JSON event per line (JSONL) to stdout as each eval completes.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Run up to N evals concurrently (default 1). Results are reported as they complete.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Show version and exit.",
    )
    args = parser.parse_args()

    if args.format == "json" and args.output == "-":
        console.print("[red]Error:[/red] --format json already streams JSON to stdout; use -o <file> instead of '-'")
        sys.exit(1)

    if args.output == "-" or args.format == "json":
        # Reserve stdout for JSON; route all human-readable output (headers,
        # eval trees, failures, summary, warnings) to stderr so pipelines can
        # parse stdout reliably.
        from . import display

        console.file = sys.stderr
        display.console.file = sys.stderr

    if args.version:
        console.print(f"timbal.eval {__version__}")
        sys.exit(0)

    load_dotenv(override=True)

    # Set env var before setup_logging reads it
    os.environ["TIMBAL_LOG_LEVEL"] = args.log_level
    setup_logging()

    # Parse path and optional eval name filter
    path, eval_name_filter = parse_path_and_filter(args.path)

    # Discover config file (evalconf.yaml)
    config = discover_config(path)

    # CLI --runnable overrides config file
    runnable_fqn = args.runnable or config.get("runnable")
    runnable = None

    if runnable_fqn:
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

    tag_filters = {t.strip() for t in args.tags.split(",")} if args.tags else None

    try:
        evals = collect_evals(path, runnable, eval_name_filter, tag_filters)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not evals:
        # Warn but keep going: an empty run must still produce the JSON
        # events/document that -o and --format json consumers expect.
        suffix = f" matching tags: {', '.join(sorted(tag_filters))}" if tag_filters else ""
        console.print(f"[yellow]Warning:[/yellow] No evals found in {path}{suffix}")

    if args.format == "json":
        from .reporters import JsonReporter

        reporter = JsonReporter()
    else:
        from .reporters import PrettyReporter

        reporter = PrettyReporter()

    summary = asyncio.run(
        run_evals(
            evals,
            capture=not args.no_capture,
            reporter=reporter,
            max_concurrency=args.jobs,
        )
    )

    if args.output:
        import json

        summary_dict = asyncio.run(dump_summary(summary))
        summary_json = json.dumps(summary_dict, indent=2, default=str)
        if args.output == "-":
            sys.stdout.write(summary_json + "\n")
        else:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(summary_json)
            console.print(f"[dim]Results written to {output_path}[/dim]")

    # Exit with error code if any eval failed
    sys.exit(0 if summary.failed == 0 else 1)
