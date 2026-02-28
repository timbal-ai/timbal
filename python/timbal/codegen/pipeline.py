from pathlib import Path
from types import SimpleNamespace

import libcst as cst
import yaml

from timbal.codegen.format import format_code
from timbal.codegen.transformers import load_modules

TIMBAL_YAML = "timbal.yaml"

_transformer_modules = None


def _get_transformer_modules() -> dict:
    global _transformer_modules
    if _transformer_modules is None:
        _transformer_modules = load_modules()
    return _transformer_modules


def parse_fqn(workspace_path: str | Path) -> tuple[Path, str]:
    """Read timbal.yaml from *workspace_path* and return (source_path, entry_point)."""
    workspace_path = Path(workspace_path)
    yaml_path = workspace_path / TIMBAL_YAML
    if not yaml_path.exists():
        raise FileNotFoundError(f"{TIMBAL_YAML} not found in {workspace_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    fqn = (data or {}).get("fqn")
    if not fqn:
        raise ValueError(f"'fqn' field missing in {yaml_path}")

    parts = fqn.split("::")
    if len(parts) != 2:
        raise ValueError(f"invalid fqn in {yaml_path}: {fqn!r}. Expected format: file.py::object_name")

    return workspace_path / parts[0], parts[1]


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


def remove_unused_code(code: str, protected: set[str]) -> str:
    """Iteratively remove unused top-level variables and functions.

    A definition is considered unused when its name has no references outside
    its own definition statements. The loop repeats until no more removals
    are possible, so cascading dead code is handled.
    """
    while True:
        tree = cst.parse_module(code)

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

        counter = _NameCounter()
        tree.visit(counter)

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

        unused = {name for name in candidates if counter.counts.get(name, 0) <= def_weight.get(name, 1)}
        if not unused:
            return code

        unused_vars = unused & var_defs
        unused_funcs = unused & func_defs

        class _Remover(cst.CSTTransformer):
            def leave_FunctionDef(self, original_node, updated_node):
                if updated_node.name.value in unused_funcs:
                    return cst.RemovalSentinel.REMOVE
                return updated_node

            def leave_SimpleStatementLine(self, original_node, updated_node):
                for item in updated_node.body:
                    if isinstance(item, cst.Assign):
                        for target in item.targets:
                            if isinstance(target.target, cst.Name) and target.target.value in unused_vars:
                                return cst.RemovalSentinel.REMOVE
                return updated_node

        code = tree.visit(_Remover()).code


def apply_operation(workspace_path: str | Path, operation: str, **kwargs) -> str:
    """Run a codegen operation and return the formatted result.

    Does NOT write to disk — the caller decides what to do with the output.

    Args:
        workspace_path: Path to directory containing timbal.yaml.
        operation: One of "add_tool", "remove_tool", "set_config".
        **kwargs: Operation-specific parameters (passed as argparse.Namespace attrs).

    Returns:
        The formatted source code after applying the operation.
    """
    workspace_path = Path(workspace_path)
    source_path, entry_point = parse_fqn(workspace_path)

    if not source_path.exists():
        raise FileNotFoundError(f"source file not found: {source_path}")

    source = source_path.read_text()
    tree = cst.parse_module(source)

    modules = _get_transformer_modules()
    mod = modules.get(operation)
    if mod is None:
        raise ValueError(f"unknown operation: {operation}")

    args = SimpleNamespace(**kwargs)
    transformer = mod.run(entry_point, args, tree=tree)
    new_tree = tree.visit(transformer)

    code = remove_unused_code(new_tree.code, protected={entry_point})
    return format_code(code, source_path)
