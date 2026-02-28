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


class _NameCounter(cst.CSTTransformer):
    """Count every occurrence of each Name node in the module."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def visit_Name(self, node: cst.Name) -> None:
        self.counts[node.value] = self.counts.get(node.value, 0) + 1


def _count_name_in_subtree(node: cst.CSTNode, name: str) -> int:
    """Count occurrences of a Name node with the given value in a CST subtree."""
    count = 0
    if isinstance(node, cst.Name) and node.value == name:
        count += 1
    for child in node.children:
        count += _count_name_in_subtree(child, name)
    return count


def _remove_unused_code(code: str, protected: set[str]) -> str:
    """Iteratively remove unused top-level variables and functions.

    A definition is considered unused when its name has no references outside
    its own definition statements. The loop repeats until no more removals
    are possible, so cascading dead code is handled.
    """
    while True:
        tree = cst.parse_module(code)

        # Collect top-level definition names.
        var_defs: set[str] = set()
        func_defs: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, cst.FunctionDef):
                func_defs.add(stmt.name.value)
            elif isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Assign):
                        for target in item.targets:
                            if isinstance(target.target, cst.Name):
                                var_defs.add(target.target.value)

        candidates = (var_defs | func_defs) - protected
        if not candidates:
            return code

        # Count every Name occurrence in the tree.
        counter = _NameCounter()
        tree.visit(counter)

        # Compute "definition weight" — how many times each candidate name
        # appears within its own definition statements. A name is unused if
        # all its occurrences are within its own definitions (no external refs).
        def_weight: dict[str, int] = {}
        for stmt in tree.body:
            if isinstance(stmt, cst.FunctionDef) and stmt.name.value in candidates:
                name = stmt.name.value
                def_weight[name] = def_weight.get(name, 0) + _count_name_in_subtree(stmt, name)
            elif isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Assign):
                        for target in item.targets:
                            if isinstance(target.target, cst.Name) and target.target.value in candidates:
                                name = target.target.value
                                def_weight[name] = def_weight.get(name, 0) + _count_name_in_subtree(stmt, name)

        unused = {
            name for name in candidates
            if counter.counts.get(name, 0) <= def_weight.get(name, 1)
        }
        if not unused:
            return code

        unused_vars = unused & var_defs
        unused_funcs = unused & func_defs

        class _Remover(cst.CSTTransformer):
            def leave_FunctionDef(
                self,
                original_node: cst.FunctionDef,
                updated_node: cst.FunctionDef,
            ) -> cst.FunctionDef | cst.RemovalSentinel:
                if updated_node.name.value in unused_funcs:
                    return cst.RemovalSentinel.REMOVE
                return updated_node

            def leave_SimpleStatementLine(
                self,
                original_node: cst.SimpleStatementLine,
                updated_node: cst.SimpleStatementLine,
            ) -> cst.SimpleStatementLine | cst.RemovalSentinel:
                for item in updated_node.body:
                    if isinstance(item, cst.Assign):
                        for target in item.targets:
                            if isinstance(target.target, cst.Name) and target.target.value in unused_vars:
                                return cst.RemovalSentinel.REMOVE
                return updated_node

        code = tree.visit(_Remover()).code


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

    # Clean up unused code, then format.
    code = _remove_unused_code(new_tree.code, protected={entry_point})
    formatted = format_code(code, source_path)

    if args.dry_run:
        print(formatted)
    else:
        source_path.write_text(formatted)


if __name__ == "__main__":
    main()
