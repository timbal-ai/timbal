import argparse

import libcst as cst

from ..cst_utils import (
    collect_assignments,
    collect_step_names,
    resolve_entry_point_type,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "remove-edge",
        help="Remove an edge between two workflow steps.",
    )
    sp.add_argument(
        "--source",
        required=True,
        help="Source step name.",
    )
    sp.add_argument(
        "--target",
        required=True,
        help="Target step name.",
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    ep_type = resolve_entry_point_type(tree, entry_point) if tree else None

    if ep_type != "Workflow":
        raise ValueError("remove-edge requires a Workflow entry point.")

    assignments = collect_assignments(tree) if tree else {}
    step_names = collect_step_names(tree, entry_point, assignments) if tree else {}
    return EdgeRemover(entry_point, args.source, args.target, assignments, step_names)


class EdgeRemover(cst.CSTTransformer):
    """Remove an edge between two workflow steps by modifying the target's .step() call."""

    def __init__(
        self,
        entry_point: str,
        source: str,
        target: str,
        assignments: dict[str, cst.Call],
        step_names: dict[str, str] | None = None,
    ):
        self.entry_point = entry_point
        self.source = source
        self.target = target
        self.assignments = assignments
        self.step_names = step_names or {}
        # Resolve source to runtime name for matching against depends_on and step_span.
        self._resolved_source = self.step_names.get(source, source)

    def _is_step_call(self, call: cst.Call) -> bool:
        return (
            isinstance(call.func, cst.Attribute)
            and isinstance(call.func.value, cst.Name)
            and call.func.value.value == self.entry_point
            and call.func.attr.value == "step"
        )

    def _matches_target(self, call: cst.Call) -> bool:
        """Match by variable name or runtime name."""
        if not call.args:
            return False
        first_arg = call.args[0].value
        if isinstance(first_arg, cst.Name):
            var_name = first_arg.value
            if var_name == self.target:
                return True
            runtime_name = self.step_names.get(var_name)
            if runtime_name is not None and runtime_name == self.target:
                return True
        return False

    def _references_source(self, arg: cst.Arg) -> bool:
        """Check if an argument's value references the source step via step_span."""
        code = cst.parse_module("").code_for_node(arg.value)
        return f'step_span("{self._resolved_source}")' in code

    def leave_Expr(self, original_node: cst.Expr, updated_node: cst.Expr) -> cst.Expr:
        """Remove edge references from the target's .step() call."""
        call = updated_node.value
        if not isinstance(call, cst.Call):
            return updated_node
        if not self._is_step_call(call) or not self._matches_target(call):
            return updated_node

        step_call_code = self._build_step_call_code(call)
        parsed = cst.parse_module(step_call_code + "\n")
        for stmt in parsed.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Expr) and isinstance(item.value, cst.Call):
                        return updated_node.with_changes(value=item.value)
        return updated_node

    def _build_step_call_code(self, existing_call: cst.Call) -> str:
        """Rebuild the .step() call with source references removed."""
        first_arg = existing_call.args[0]
        step_ref = cst.parse_module("").code_for_node(first_arg.value)

        parts = [step_ref]

        for arg in existing_call.args[1:]:
            if not isinstance(arg.keyword, cst.Name):
                value_code = cst.parse_module("").code_for_node(arg.value).strip()
                parts.append(value_code)
                continue

            key = arg.keyword.value

            # Remove depends_on entry for source, or the whole kwarg if it becomes empty.
            if key == "depends_on" and isinstance(arg.value, cst.List):
                remaining = []
                for el in arg.value.elements:
                    if isinstance(el.value, (cst.SimpleString, cst.ConcatenatedString)):
                        if el.value.evaluated_value != self._resolved_source:
                            remaining.append(f'"{el.value.evaluated_value}"')
                    else:
                        remaining.append(cst.parse_module("").code_for_node(el.value).strip())
                if remaining:
                    parts.append(f"depends_on=[{', '.join(remaining)}]")
                continue

            # Remove when kwarg if it references source.
            if key == "when" and self._references_source(arg):
                continue

            # Remove param kwargs that reference source.
            if self._references_source(arg):
                continue

            # Keep everything else.
            value_code = cst.parse_module("").code_for_node(arg.value).strip()
            parts.append(f"{key}={value_code}")

        return f"{self.entry_point}.step({', '.join(parts)})"
