import argparse
import sys
from pathlib import Path

import libcst as cst

from timbal.codegen.format import format_code
from timbal.codegen.transformers import SystemPromptSetter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m timbal.codegen",
        description="Modify timbal agent/workflow source files.",
    )
    parser.add_argument(
        "fqn",
        help="Fully Qualified Name of the timbal runnable. E.g. agent.py::agent",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the result without writing to disk.",
    )

    subparsers = parser.add_subparsers(dest="operation", required=True)

    sp = subparsers.add_parser(
        "set-system-prompt", help="Set the agent system prompt. Omit or pass empty string to remove it."
    )
    sp.add_argument("value", nargs="?", default="", help="The system prompt text.")

    args = parser.parse_args()

    parts = args.fqn.split("::")
    if len(parts) != 2:
        raise ValueError(f"Invalid FQN: {args.fqn!r}. Expected format: path/to/file.py::object_name")
    source_path = Path(parts[0])
    entry_point = parts[1]

    if not source_path.exists():
        print(f"error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    source = source_path.read_text()
    tree = cst.parse_module(source)

    if args.operation == "set-system-prompt":
        transformer = SystemPromptSetter(entry_point, args.value)
    else:
        print(f"error: unknown operation: {args.operation}", file=sys.stderr)
        sys.exit(1)

    new_tree = tree.visit(transformer)
    formatted = format_code(new_tree.code, source_path)

    if args.dry_run:
        print(formatted)
    else:
        source_path.write_text(formatted)


if __name__ == "__main__":
    main()
