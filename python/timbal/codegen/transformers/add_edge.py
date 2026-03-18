import argparse
import json

import libcst as cst

from ..cst_utils import (
    collect_assignments,
    has_import,
    resolve_entry_point_type,
    resolve_runnable_name,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "add-edge",
        help="Add an edge between two workflow steps (data flow, ordering, or conditional).",
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
        "--params",
        default=None,
        help=(
            "JSON mapping target input params to source step outputs. "
            'E.g. \'{"prompt": {"step": "agent_a"}, "context": {"step": "agent_b", "key": "valor"}}\'.'
        ),
    )
    sp.add_argument(
        "--when",
        default=None,
        help=(
            "Python expression for a conditional edge. "
            'E.g. \'lambda: get_run_context().step_span("agent_a").output.content != ""\''
        ),
    )


def _parse_params(raw_params: str) -> dict[str, dict]:
    """Parse and validate --params JSON."""
    param_map: dict[str, dict] = {}
    raw = json.loads(raw_params)
    for param_name, spec in raw.items():
        if not isinstance(spec, dict) or "step" not in spec:
            raise ValueError(
                f"Invalid --params entry for '{param_name}'. "
                'Each value must be a dict with at least a "step" key.'
            )
        param_map[param_name] = spec
    return param_map


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    ep_type = resolve_entry_point_type(tree, entry_point) if tree else None

    if ep_type != "Workflow":
        raise ValueError("add-edge requires a Workflow entry point.")

    param_map = _parse_params(args.params) if args.params else {}
    when_expr = args.when if args.when else None
    assignments = collect_assignments(tree) if tree else {}

    return EdgeAdder(entry_point, args.source, args.target, param_map, when_expr, assignments)


class EdgeAdder(cst.CSTTransformer):
    """Add an edge between two workflow steps by modifying the target's .step() call."""

    def __init__(
        self,
        entry_point: str,
        source: str,
        target: str,
        param_map: dict[str, dict],
        when_expr: str | None,
        assignments: dict[str, cst.Call],
    ):
        self.entry_point = entry_point
        self.source = source
        self.target = target
        self.param_map = param_map
        self.when_expr = when_expr
        self.assignments = assignments

    def _is_step_call(self, call: cst.Call) -> bool:
        return (
            isinstance(call.func, cst.Attribute)
            and isinstance(call.func.value, cst.Name)
            and call.func.value.value == self.entry_point
            and call.func.attr.value == "step"
        )

    def _matches_target(self, call: cst.Call) -> bool:
        if not call.args:
            return False
        first_arg = call.args[0].value
        if isinstance(first_arg, cst.Name):
            var_name = first_arg.value
            if var_name in self.assignments:
                resolved = resolve_runnable_name(self.assignments[var_name])
                if resolved is not None:
                    return resolved == self.target
            return var_name == self.target
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
        needs_import = self.param_map or self.when_expr
        if needs_import and not has_import(original_node, "timbal.state", "get_run_context"):
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

    def _build_step_call_code(self, existing_call: cst.Call) -> str:
        """Build the updated .step() call source code."""
        first_arg = existing_call.args[0]
        step_ref = cst.parse_module("").code_for_node(first_arg.value)

        parts = [step_ref]

        # Keys we'll be overriding.
        overridden_keys = set(self.param_map.keys())
        if not self.param_map:
            # Pure ordering or when-only edge: merge into depends_on.
            overridden_keys.add("depends_on")
        if self.when_expr:
            overridden_keys.add("when")

        # Collect existing kwargs that we're NOT overriding.
        for arg in existing_call.args[1:]:
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value not in overridden_keys:
                value_code = cst.parse_module("").code_for_node(arg.value).strip()
                parts.append(f"{arg.keyword.value}={value_code}")

        # For pure ordering (no params), merge source into depends_on.
        if not self.param_map:
            existing_deps = self._get_existing_depends_on(existing_call)
            all_deps = list(dict.fromkeys(existing_deps + [self.source]))  # dedupe, preserve order
            deps = ", ".join(f'"{d}"' for d in all_deps)
            parts.append(f"depends_on=[{deps}]")

        # when kwarg
        if self.when_expr:
            parts.append(f"when={self.when_expr}")

        # Param kwargs (data flow)
        for param_name, spec in self.param_map.items():
            source_step = spec["step"]
            key = spec.get("key")
            if key:
                parts.append(
                    f'{param_name}=lambda: get_run_context().step_span("{source_step}").output["{key}"]'
                )
            else:
                parts.append(
                    f'{param_name}=lambda: get_run_context().step_span("{source_step}").output'
                )

        return f"{self.entry_point}.step({', '.join(parts)})"
