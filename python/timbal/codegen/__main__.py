import argparse
import sys
from pathlib import Path

import libcst as cst
import yaml

from timbal.codegen.format import format_code
from timbal.codegen.transformers import load_modules

TIMBAL_YAML = "timbal.yaml"


def _parse_fqn(workspace_path: Path) -> tuple[Path, str]:
    """Read timbal.yaml from *workspace_path* and return (source_path, entry_point)."""
    yaml_path = workspace_path / TIMBAL_YAML
    if not yaml_path.exists():
        print(f"error: {TIMBAL_YAML} not found in {workspace_path}", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    fqn = (data or {}).get("fqn")
    if not fqn:
        print(f"error: 'fqn' field missing in {yaml_path}", file=sys.stderr)
        sys.exit(1)

    parts = fqn.split("::")
    if len(parts) != 2:
        print(
            f"error: invalid fqn in {yaml_path}: {fqn!r}. Expected format: file.py::object_name",
            file=sys.stderr,
        )
        sys.exit(1)

    source_path = workspace_path / parts[0]
    entry_point = parts[1]
    return source_path, entry_point


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

    args = parser.parse_args()

    workspace_path = Path(args.path)
    source_path, entry_point = _parse_fqn(workspace_path)

    if not source_path.exists():
        print(f"error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    source = source_path.read_text()
    tree = cst.parse_module(source)

    # Map CLI operation name (e.g. "set-config") to module name (e.g. "set_config").
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
