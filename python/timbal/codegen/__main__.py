import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="timbal-codegen",
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
    get_flow_parser = subparsers.add_parser("get-flow", help="Print the graph for the workspace entry point.")
    get_flow_parser.add_argument(
        "--format",
        choices=["compact", "json"],
        default="json",
        help="Output format: 'json' (default) is the full ReactFlow-compatible JSON; 'compact' is token-efficient and LLM-readable.",
    )
    get_tools_parser = subparsers.add_parser(
        "get-tools",
        help="Browse tools by provider, or search/filter with pagination.",
    )
    get_tools_parser.add_argument(
        "--provider", default=None, help="Filter tools by provider name (use 'system' for tools with no provider)."
    )
    get_tools_parser.add_argument(
        "--search", default=None, help="Case-insensitive substring search on tool name, type, and description."
    )
    get_tools_parser.add_argument("--limit", type=int, default=50, help="Max tools to return (default 50).")
    get_tools_parser.add_argument("--offset", type=int, default=0, help="Number of tools to skip (default 0).")
    get_tools_parser.add_argument(
        "--no-cache", action="store_true", help="Skip the disk cache and force a full rediscovery."
    )
    get_models_parser = subparsers.add_parser(
        "get-models",
        help="Browse models by provider, or search/filter with pagination.",
    )
    get_models_parser.add_argument(
        "--provider", default=None, help="Filter models by provider name."
    )
    get_models_parser.add_argument(
        "--search", default=None, help="Case-insensitive substring search on model id, display_name, and description."
    )
    get_models_parser.add_argument("--limit", type=int, default=50, help="Max models to return (default 50).")
    get_models_parser.add_argument("--offset", type=int, default=0, help="Number of models to skip (default 0).")

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
    _lightweight_ops = {"get-models", "get-tools", "get-flow", "test"}
    if not (_lightweight_ops & set(sys.argv[1:])):
        from timbal.codegen.transformers import load_modules

        transformer_modules = load_modules()
        for mod in transformer_modules.values():
            mod.register(subparsers)

    args = parser.parse_args()

    workspace_path = Path(args.path)
    operation = args.operation

    if operation == "get-models":
        from timbal.codegen.model_discovery import get_models, get_provider_summaries

        provider_filter = args.provider
        search_filter = args.search

        # No filters → return provider summaries.
        if provider_filter is None and search_filter is None:
            print(json.dumps({"providers": get_provider_summaries()}, indent=2))
            return

        # Build filtered model list.
        models = get_models()

        if provider_filter is not None:
            models = [m for m in models if m.get("provider") == provider_filter]

        if search_filter is not None:
            q = search_filter.lower()
            models = [
                m
                for m in models
                if q in (m.get("id") or "").lower()
                or q in (m.get("display_name") or "").lower()
                or q in (m.get("description") or "").lower()
            ]

        total = len(models)
        models = models[args.offset : args.offset + args.limit]

        print(json.dumps({"models": models, "total": total, "limit": args.limit, "offset": args.offset}, indent=2))
        return

    if operation == "get-tools":
        from timbal.codegen.tool_discovery import get_framework_tools, get_provider_summaries

        no_cache = args.no_cache
        provider_filter = args.provider
        search_filter = args.search

        # No filters → return provider summaries.
        if provider_filter is None and search_filter is None:
            print(json.dumps({"providers": get_provider_summaries(no_cache=no_cache)}, indent=2))
            return

        # Build filtered tool list.
        tools = [
            {
                "type": cls,
                "module": ft.module,
                "name": ft.name,
                "description": ft.description,
                "provider": ft.provider,
                "provider_logo": ft.provider_logo,
            }
            for cls, ft in sorted(get_framework_tools(no_cache=no_cache).items())
        ]

        if provider_filter is not None:
            if provider_filter == "system":
                tools = [t for t in tools if t["provider"] is None]
            else:
                tools = [t for t in tools if t["provider"] == provider_filter]

        if search_filter is not None:
            q = search_filter.lower()
            tools = [
                t
                for t in tools
                if q in (t["name"] or "").lower()
                or q in (t["type"] or "").lower()
                or q in (t["description"] or "").lower()
            ]

        total = len(tools)
        tools = tools[args.offset : args.offset + args.limit]

        print(json.dumps({"tools": tools, "total": total, "limit": args.limit, "offset": args.offset}, indent=2))
        return

    if operation == "get-flow":
        from timbal.codegen.flow import format_compact, get_flow

        try:
            flow = get_flow(workspace_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        if args.format == "json":
            print(json.dumps(flow, indent=2))
        else:
            print(format_compact(flow))
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
