import argparse
import json
import sys
from pathlib import Path

from timbal.codegen.pipeline import apply_operation, get_flow, parse_fqn
from timbal.codegen.transformers import load_modules


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m timbal.codegen",
        description="Modify timbal agent/workflow source files.",
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Path to a workspace member directory containing timbal.yaml. Defaults to the current directory.",
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

    # Read-only operations (not CST transformers)
    subparsers.add_parser("get-flow", help="Print the graph for the workspace entry point.")

    args = parser.parse_args()

    workspace_path = Path(args.path)
    operation = args.operation

    if operation == "get-flow":
        try:
            flow = get_flow(workspace_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(flow, indent=2))
        return

    module_name = operation.replace("-", "_")

    # Collect operation-specific kwargs from the parsed args.
    skip = {"path", "dry_run", "operation"}
    kwargs = {k: v for k, v in vars(args).items() if k not in skip}

    try:
        formatted = apply_operation(workspace_path, module_name, **kwargs)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(formatted)
    else:
        source_path, _ = parse_fqn(workspace_path)
        source_path.write_text(formatted)


if __name__ == "__main__":
    main()
