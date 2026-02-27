import argparse
import sys
from pathlib import Path

import libcst as cst

from timbal.codegen.format import format_code
from timbal.codegen.transformers import load_modules


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

    transformer_modules = load_modules()
    for mod in transformer_modules.values():
        mod.register(subparsers)

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

    # Map CLI operation name (e.g. "set-model") to module name (e.g. "set_model").
    module_name = args.operation.replace("-", "_")
    mod = transformer_modules.get(module_name)
    if mod is None:
        print(f"error: unknown operation: {args.operation}", file=sys.stderr)
        sys.exit(1)

    transformer = mod.run(entry_point, args, tree=tree)
    new_tree = tree.visit(transformer)
    formatted = format_code(new_tree.code, source_path, entry_point=entry_point)

    if args.dry_run:
        print(formatted)
    else:
        source_path.write_text(formatted)


if __name__ == "__main__":
    main()
