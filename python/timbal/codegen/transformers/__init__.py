import importlib
import pkgutil
from pathlib import Path
from types import SimpleNamespace

import libcst as cst

from timbal.codegen import parse_fqn
from timbal.codegen.format import format_code

_transformer_modules = None


def load_modules() -> dict:
    modules = {}
    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        mod = importlib.import_module(f"timbal.codegen.transformers.{info.name}")
        modules[info.name] = mod
    return modules


def _get_transformer_modules() -> dict:
    global _transformer_modules
    if _transformer_modules is None:
        _transformer_modules = load_modules()
    return _transformer_modules


class _UsageAnalyzer(cst.CSTVisitor):
    """Single-pass visitor that counts global name usage and per-definition self-references."""

    def __init__(self, candidates: set[str]) -> None:
        self._candidates = candidates
        self._current_def: str | None = None
        self.counts: dict[str, int] = {}
        self.self_counts: dict[str, int] = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        name = node.name.value
        if name in self._candidates and self._current_def is None:
            self._current_def = name
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self._current_def == node.name.value:
            self._current_def = None

    def visit_Name(self, node: cst.Name) -> None:
        val = node.value
        self.counts[val] = self.counts.get(val, 0) + 1
        if self._current_def is not None and val == self._current_def:
            self.self_counts[val] = self.self_counts.get(val, 0) + 1


class _Remover(cst.CSTTransformer):
    """Remove top-level definitions whose names are in the unused sets."""

    def __init__(self, unused_funcs: set[str], unused_vars: set[str]) -> None:
        self._unused_funcs = unused_funcs
        self._unused_vars = unused_vars

    def leave_FunctionDef(self, original_node, updated_node):
        if updated_node.name.value in self._unused_funcs:
            return cst.RemovalSentinel.REMOVE
        return updated_node

    def leave_SimpleStatementLine(self, original_node, updated_node):
        for item in updated_node.body:
            if isinstance(item, cst.Assign):
                for target in item.targets:
                    if isinstance(target.target, cst.Name) and target.target.value in self._unused_vars:
                        return cst.RemovalSentinel.REMOVE
        return updated_node


def remove_unused_code(code: str, protected: set[str]) -> str:
    """Iteratively remove unused top-level variables and functions.

    A definition is considered unused when its name has no references outside
    its own definition statements. The loop repeats until no more removals
    are possible, so cascading dead code is handled.
    """
    tree = cst.parse_module(code)

    while True:
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
            return tree.code

        analyzer = _UsageAnalyzer(candidates)
        tree.visit(analyzer)

        # For variable assignments, count self-references by scanning the
        # assignment statement's subtree (these are flat, so cheap).
        var_self_counts: dict[str, int] = {}
        for stmt in tree.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Assign):
                        for target in item.targets:
                            if isinstance(target.target, cst.Name) and target.target.value in candidates:
                                name = target.target.value
                                # Count Name nodes in the assignment value + target itself.
                                count = 0
                                stack = list(stmt.children)
                                while stack:
                                    child = stack.pop()
                                    if isinstance(child, cst.Name) and child.value == name:
                                        count += 1
                                    stack.extend(child.children)
                                var_self_counts[name] = var_self_counts.get(name, 0) + count

        def_weight: dict[str, int] = {}
        for name in candidates:
            if name in func_defs:
                def_weight[name] = analyzer.self_counts.get(name, 0)
            else:
                def_weight[name] = var_self_counts.get(name, 0)

        unused = {name for name in candidates if analyzer.counts.get(name, 0) <= def_weight.get(name, 1)}
        if not unused:
            return tree.code

        unused_vars = unused & var_defs
        unused_funcs = unused & func_defs

        tree = tree.visit(_Remover(unused_funcs, unused_vars))


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
    spec = parse_fqn(workspace_path)

    if not spec.path.exists():
        raise FileNotFoundError(f"source file not found: {spec.path}")

    source = spec.path.read_text()
    tree = cst.parse_module(source)

    modules = _get_transformer_modules()
    mod = modules.get(operation)
    if mod is None:
        raise ValueError(f"unknown operation: {operation}")

    args = SimpleNamespace(**kwargs)
    transformer = mod.run(spec.target, args, tree=tree)
    new_tree = tree.visit(transformer)

    code = remove_unused_code(new_tree.code, protected={spec.target})
    return format_code(code, spec.path)
