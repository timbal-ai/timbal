import argparse

import libcst as cst

from ..cli_utils import arg_input
from ..cst_utils import (
    StepCallRewriter,
    collect_assignments,
    collect_chained_step_names,
    collect_step_names,
    has_import,
    insert_imports,
    require_step,
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
        type=arg_input,
        help=(
            "Python expression for a conditional edge. "
            'E.g. \'lambda: get_run_context().step_span("agent_a").output.content != ""\'. '
            "Use '@path' to read from file or '-' to read from stdin."
        ),
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    ep_type = resolve_entry_point_type(tree, entry_point) if tree else None

    if ep_type != "Workflow":
        raise ValueError("add-edge requires a Workflow entry point.")

    when_expr = args.when if args.when else None
    assignments = collect_assignments(tree)
    step_names = collect_step_names(tree, entry_point, assignments)
    chained_step_names = collect_chained_step_names(tree, entry_point, assignments)

    require_step(args.target, step_names, chained_step_names, kind="Target", operation="add-edge")
    # Sources only need to exist in the graph — they are referenced by runtime
    # name inside the target's depends_on list.
    source = require_step(args.source, {**step_names, **chained_step_names}, kind="Source", operation="add-edge")

    return EdgeAdder(entry_point, source, args.target, when_expr, assignments, step_names)


class EdgeAdder(StepCallRewriter):
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

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if self.when_expr and not has_import(original_node, "timbal.state", "get_run_context"):
            body = list(updated_node.body)
            insert_imports(body, [cst.parse_statement("from timbal.state import get_run_context\n")])
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
