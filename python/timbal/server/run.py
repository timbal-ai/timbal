import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ..logs import setup_logging
from ..state import RunContext, set_run_context
from ..utils import ImportSpec


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a single run of a Timbal runnable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m timbal.server.run path/to/agent.py::agent
  python -m timbal.server.run path/to/agent.py::agent --input '{"prompt": "hello"}'
  python -m timbal.server.run path/to/agent.py::agent --input '{"prompt": "hello"}' --stream
  echo '{"prompt": "hello"}' | python -m timbal.server.run path/to/agent.py::agent
        """,
    )
    parser.add_argument(
        "import_spec",
        help="Path to a Python file and object name (format: path/to/file.py::object_name)",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="JSON string of input params. Reads from stdin if omitted.",
    )
    parser.add_argument(
        "--context",
        "-c",
        default=None,
        help='JSON string of RunContext fields (e.g. \'{"id": "my-run-id"}\')',
    )
    parser.add_argument(
        "--stream",
        "-s",
        action="store_true",
        help="Print every event as it arrives instead of only the final output event.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
    load_dotenv()
    setup_logging()

    parts = args.import_spec.split("::")
    if len(parts) != 2:
        print(  # noqa: T201
            f"Invalid import spec '{args.import_spec}'. Expected format: path/to/file.py::object_name",
            file=sys.stderr,
        )
        sys.exit(1)

    import_spec = ImportSpec(
        path=Path(parts[0]).expanduser().resolve(),
        target=parts[1],
    )
    runnable = import_spec.load()

    if args.input is not None:
        raw_input = args.input
    elif not sys.stdin.isatty():
        raw_input = sys.stdin.read()
    else:
        raw_input = "{}"

    try:
        params = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # Optionally wire up a RunContext
    if args.context is not None:
        try:
            run_context = RunContext.model_validate(json.loads(args.context))
        except (json.JSONDecodeError, Exception) as e:
            print(f"Invalid JSON context: {e}", file=sys.stderr)  # noqa: T201
            sys.exit(1)
        set_run_context(run_context)

    output_event = None
    async for event in runnable(**params):
        if args.stream:
            print(json.dumps(event.model_dump()))  # noqa: T201
        if event.type == "OUTPUT":
            output_event = event

    if not args.stream and output_event:
        print(json.dumps(output_event.model_dump()))  # noqa: T201

    if output_event is None or output_event.status.code != "success":
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
