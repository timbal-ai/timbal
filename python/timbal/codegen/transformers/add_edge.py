import argparse

import libcst as cst

from ..cst_utils import (
    collect_assignments,
    collect_step_names,
    has_import,
    resolve_entry_point_type,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "add-edge",
        help="Add an ordering or conditional edge between two workflow steps.",
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
    sp.add_argument(
        "--when",
        default=None,
        help=(
            "Python expression for a conditional edge. "
            'E.g. \'lambda: get_run_context().step_span("agent_a").output.content != ""\''
        ),
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    ep_type = resolve_entry_point_type(tree, entry_point) if tree else None

    if ep_type != "Workflow":
        raise ValueError("add-edge requires a Workflow entry point.")

    when_expr = args.when if args.when else None
    assignments = collect_assignments(tree) if tree else {}
    step_names = collect_step_names(tree, entry_point, assignments) if tree else {}

    return EdgeAdder(entry_point, args.source, args.target, when_expr, assignments, step_names)


class EdgeAdder(cst.CSTTransformer):
    """Add an ordering or conditional edge between two workflow steps."""

    needs_reorder = True

    def __init__(
        self,
        entry_point: str,
        source: str,
        target: str,
        when_expr: str | None,
        assignments: dict[str, cst.Call],
        step_names: dict[str, str] | None = None,
    ):
        self.entry_point = entry_point
        self.source = source
        self.target = target
        self.when_expr = when_expr
        self.assignments = assignments
        self.step_names = step_names or {}

    def _is_step_call(self, call: cst.Call) -> bool:
        return (
            isinstance(call.func, cst.Attribute)
            and isinstance(call.func.value, cst.Name)
            and call.func.value.value == self.entry_point
            and call.func.attr.value == "step"
        )

    def _matches_target(self, call: cst.Call) -> bool:
        """Check if a .step() call is the target step.

        Matches when self.target equals either the variable name or the
        runtime name (the ``name=`` kwarg) of the step.
        """
        if not call.args:
            return False
        first_arg = call.args[0].value
        if isinstance(first_arg, cst.Name):
            var_name = first_arg.value
            # Match by variable name.
            if var_name == self.target:
                return True
            # Match by runtime name (name= kwarg).
            runtime_name = self.step_names.get(var_name)
            if runtime_name is not None and runtime_name == self.target:
                return True
        return False

    def _get_existing_depends_on(self, call: cst.Call) -> list[str]:
        """Extract existing depends_on list from a .step() call."""
        for arg in call.args[1:]:
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "depends_on":
                if isinstance(arg.value, cst.List):
                    deps = []
                    for el in arg.value.elements:
                        if isinstance(el.value, (cst.SimpleString, cst.ConcatenatedString)):
                            deps.append(el.value.evaluated_value)
                    return deps
        return []

    def leave_Expr(self, original_node: cst.Expr, updated_node: cst.Expr) -> cst.Expr:
        """Update the target's .step() call to add the edge."""
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

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if self.when_expr and not has_import(original_node, "timbal.state", "get_run_context"):
            body = list(updated_node.body)
            import_stmt = cst.parse_statement("from timbal.state import get_run_context\n")
            import_insert_idx = 0
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, (cst.Import, cst.ImportFrom)):
                            import_insert_idx = i + 1
            body.insert(import_insert_idx, import_stmt)
            return updated_node.with_changes(body=body)
        return updated_node

    def _resolve_source_name(self) -> str:
        """Resolve self.source to the runtime step name.

        If self.source is a step variable name whose runtime ``name=`` kwarg
        differs, return that runtime name. Only considers variables that are
        actually used as workflow steps (not the workflow itself).
        """
        if self.source in self.step_names:
            return self.step_names[self.source]
        return self.source

    def _build_step_call_code(self, existing_call: cst.Call) -> str:
        """Build the updated .step() call source code."""
        first_arg = existing_call.args[0]
        step_ref = cst.parse_module("").code_for_node(first_arg.value)

        parts = [step_ref]

        # Keys we'll be overriding.
        overridden_keys = {"depends_on"}
        if self.when_expr:
            overridden_keys.add("when")

        # Collect existing kwargs that we're NOT overriding.
        for arg in existing_call.args[1:]:
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value not in overridden_keys:
                value_code = cst.parse_module("").code_for_node(arg.value).strip()
                parts.append(f"{arg.keyword.value}={value_code}")

        # Merge source into depends_on, resolving variable name to runtime name.
        resolved_source = self._resolve_source_name()
        existing_deps = self._get_existing_depends_on(existing_call)
        all_deps = list(dict.fromkeys(existing_deps + [resolved_source]))  # dedupe, preserve order
        deps = ", ".join(f'"{d}"' for d in all_deps)
        parts.append(f"depends_on=[{deps}]")

        # when kwarg
        if self.when_expr:
            parts.append(f"when={self.when_expr}")

        return f"{self.entry_point}.step({', '.join(parts)})"
