import argparse
import json
import sys
from pathlib import Path


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

    # Read-only operations (not CST transformers)
    subparsers.add_parser("get-flow", help="Print the graph for the workspace entry point.")
    subparsers.add_parser("list-tools", help="List available framework tool types.")

    # Test run operation
    test_parser = subparsers.add_parser("test", help="Execute a single test run of the workspace entry point.")
    test_parser.add_argument("--input", "-i", default=None, help="JSON string of input params.")
    test_parser.add_argument(
        "--context", "-c", default=None, help='JSON string of RunContext fields (e.g. \'{"id": "my-run-id"}\').'
    )
    test_parser.add_argument(
        "--stream", "-s", action="store_true", help="Print every event instead of only the final output event."
    )

    # Defer transformer module loading (pulls in libcst + timbal.codegen which
    # are expensive) — only needed for transformer operations, not for
    # list-tools, get-flow, or test.
    _lightweight_ops = {"list-tools", "get-flow", "test"}
    if not (_lightweight_ops & set(sys.argv[1:])):
        from timbal.codegen.transformers import load_modules

        transformer_modules = load_modules()
        for mod in transformer_modules.values():
            mod.register(subparsers)

    args = parser.parse_args()

    workspace_path = Path(args.path)
    operation = args.operation

    if operation == "list-tools":
        from timbal.codegen.tool_discovery import get_framework_tools

        tools = [
            {
                "type": cls,
                "module": ft.module,
                "name": ft.name,
                "description": ft.description,
                "provider": ft.provider,
                "provider_logo": ft.provider_logo,
            }
            for cls, ft in sorted(get_framework_tools().items())
        ]
        tools = {"tools": tools}
        print(json.dumps(tools, indent=2))
        return

    if operation == "get-flow":
        from timbal.codegen.flow import get_flow

        try:
            flow = get_flow(workspace_path)
            print("flow", flow)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(flow, indent=2))
        return

    if operation == "test":
        import asyncio
        import os

        os.environ["TIMBAL_LOG_LEVEL"] = "CRITICAL"
        os.environ["TIMBAL_DELTA_EVENTS"] = "true"

        from ..logs import setup_logging

        setup_logging()

        from ..state import RunContext
        from . import parse_fqn
        from .test import run_test

        try:
            import_spec = parse_fqn(workspace_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            params = json.loads(args.input) if args.input else {}
        except json.JSONDecodeError as e:
            print(f"error: invalid JSON input: {e}", file=sys.stderr)
            sys.exit(1)

        run_context = None
        if args.context is not None:
            try:
                run_context = RunContext.model_validate(json.loads(args.context))
            except (json.JSONDecodeError, Exception) as e:
                print(f"error: invalid JSON context: {e}", file=sys.stderr)
                sys.exit(1)

        asyncio.run(run_test(import_spec, params, run_context=run_context, stream=args.stream))
        return

    # Transformer operations — heavy imports already loaded above.
    from timbal.codegen import parse_fqn
    from timbal.codegen.transformers import apply_operation

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
        parse_fqn(workspace_path).path.write_text(formatted)


if __name__ == "__main__":
    main()
