import importlib
import pkgutil
import re
from pathlib import Path
from types import SimpleNamespace

import libcst as cst

from timbal.codegen import parse_fqn
from timbal.codegen.cst_utils import collect_assignments, resolve_runnable_name
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


def reorder_step_calls(code: str, entry_point: str) -> str:
    """Topologically sort workflow .step() calls so dependencies come first.

    Extracts the dependency graph from depends_on lists and step_span()
    references in kwargs, then reorders the .step() statements accordingly.
    Steps with no ordering constraint preserve their relative source order.
    """
    tree = cst.parse_module(code)
    assignments = collect_assignments(tree)
    body = list(tree.body)

    # Collect step call indices and their metadata.
    step_calls: list[tuple[int, str, set[str]]] = []  # (index, step_name, deps)

    for i, stmt in enumerate(body):
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not (isinstance(item, cst.Expr) and isinstance(item.value, cst.Call)):
                continue
            call = item.value
            if not (
                isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name)
                and call.func.value.value == entry_point
                and call.func.attr.value == "step"
            ):
                continue

            # Resolve step name from first arg.
            if not call.args:
                continue
            first_arg = call.args[0].value
            step_name = None
            if isinstance(first_arg, cst.Name):
                var_name = first_arg.value
                if var_name in assignments:
                    step_name = resolve_runnable_name(assignments[var_name])
                if step_name is None:
                    step_name = var_name
            if step_name is None:
                continue

            # Collect dependencies from depends_on and step_span references.
            deps: set[str] = set()
            call_code = cst.parse_module("").code_for_node(call)

            # depends_on list.
            for arg in call.args[1:]:
                if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "depends_on":
                    if isinstance(arg.value, cst.List):
                        for el in arg.value.elements:
                            if isinstance(el.value, (cst.SimpleString, cst.ConcatenatedString)):
                                deps.add(el.value.evaluated_value)

            # step_span("...") references in any kwarg value.
            for match in re.finditer(r'step_span\("([^"]+)"\)', call_code):
                deps.add(match.group(1))

            step_calls.append((i, step_name, deps))

    if len(step_calls) <= 1:
        return code

    # Check if already in valid order — skip rewrite if so.
    name_to_pos = {name: order for order, (_, name, _) in enumerate(step_calls)}
    needs_sort = False
    for _, name, deps in step_calls:
        for dep in deps:
            if dep in name_to_pos and name_to_pos[dep] > name_to_pos[name]:
                needs_sort = True
                break
        if needs_sort:
            break

    if not needs_sort:
        return code

    # Kahn's algorithm, using original index as tiebreaker for stable order.
    original_order = {name: idx for idx, (_, name, _) in enumerate(step_calls)}
    graph: dict[str, set[str]] = {name: set() for _, name, _ in step_calls}
    in_degree: dict[str, int] = {name: 0 for _, name, _ in step_calls}

    for _, name, deps in step_calls:
        for dep in deps:
            if dep in graph:
                graph[dep].add(name)
                in_degree[name] += 1

    queue = sorted(
        [n for n, d in in_degree.items() if d == 0],
        key=lambda n: original_order[n],
    )
    sorted_names: list[str] = []
    while queue:
        node = queue.pop(0)
        sorted_names.append(node)
        for neighbor in sorted(graph[node], key=lambda n: original_order[n]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort(key=lambda n: original_order[n])

    # Build mapping: new_body_index -> old_body_index for step statements.
    idx_by_name = {name: body_idx for body_idx, name, _ in step_calls}
    step_body_indices = [body_idx for body_idx, _, _ in step_calls]
    sorted_body_indices = [idx_by_name[name] for name in sorted_names]

    # Swap step statements into their new positions.
    new_body = list(body)
    for target_idx, source_idx in zip(step_body_indices, sorted_body_indices):
        new_body[target_idx] = body[source_idx]

    return tree.with_changes(body=new_body).code


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

    if getattr(transformer, "needs_reorder", False):
        code = reorder_step_calls(code, spec.target)

    return format_code(code, spec.path)
