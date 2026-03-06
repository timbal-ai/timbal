import argparse

import libcst as cst

from ..cst_utils import collect_assignments, resolve_entry_point_type, resolve_runnable_name


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "remove-step",
        help="Remove a step from the workflow by name.",
    )
    sp.add_argument("--name", required=True, help="The step name to remove. E.g. agent_b.")


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    if tree is not None:
        ep_type = resolve_entry_point_type(tree, entry_point)
        if ep_type is not None and ep_type != "Workflow":
            raise ValueError(f"remove-step requires a Workflow entry point, but '{entry_point}' is a {ep_type}.")

    assignments = collect_assignments(tree) if tree else {}
    return StepRemover(entry_point, args.name, assignments)


class StepRemover(cst.CSTTransformer):
    def __init__(self, entry_point: str, step_name: str, assignments: dict[str, cst.Call]):
        self.entry_point = entry_point
        self.step_name = step_name
        self.assignments = assignments

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine | cst.RemovalSentinel:
        """Remove expression statements that are workflow.step() calls for the target step."""
        for item in updated_node.body:
            if isinstance(item, cst.Expr) and isinstance(item.value, cst.Call):
                call = item.value
                if self._is_step_call(call) and self._matches_step_name(call):
                    return cst.RemovalSentinel.REMOVE
        return updated_node

    def _is_step_call(self, call: cst.Call) -> bool:
        """Check if a Call node is entry_point.step(...)."""
        return (
            isinstance(call.func, cst.Attribute)
            and isinstance(call.func.value, cst.Name)
            and call.func.value.value == self.entry_point
            and call.func.attr.value == "step"
        )

    def _matches_step_name(self, call: cst.Call) -> bool:
        """Check if the first argument of .step() resolves to the target step name."""
        if not call.args:
            return False
        first_arg = call.args[0].value
        if isinstance(first_arg, cst.Name):
            var_name = first_arg.value
            if var_name in self.assignments:
                resolved = resolve_runnable_name(self.assignments[var_name])
                if resolved is not None:
                    return resolved == self.step_name
            return var_name == self.step_name
        return False
