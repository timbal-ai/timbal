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

    # ------------------------------------------------------------------
    # Standalone: workflow.step(X)  as an expression statement
    # ------------------------------------------------------------------

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine | cst.RemovalSentinel:
        """Remove the target step's standalone .step() call, and clean up references in other steps."""
        for item in updated_node.body:
            if isinstance(item, cst.Expr) and isinstance(item.value, cst.Call):
                call = item.value
                if not self._is_standalone_step_call(call):
                    continue
                # Remove the target step entirely.
                if self._matches_step_name(call):
                    return cst.RemovalSentinel.REMOVE
                # For other steps, clean up references to the removed step.
                cleaned = self._clean_references(call)
                if cleaned is not None:
                    new_item = item.with_changes(value=cleaned)
                    return updated_node.with_changes(body=[new_item])
        return updated_node

    # ------------------------------------------------------------------
    # Chained: Workflow(...).step(A).step(B).step(X)
    # ------------------------------------------------------------------

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call,
    ) -> cst.BaseExpression:
        """Handle .step() calls in chained syntax."""
        if not self._is_dot_step(updated_node):
            return updated_node

        # Only handle chained calls (receiver is a Call, not a bare Name).
        receiver = updated_node.func.value
        if isinstance(receiver, cst.Name):
            # Standalone call like workflow.step(X) — handled in leave_SimpleStatementLine.
            return updated_node

        # This is a chained .step() call.
        if self._matches_step_name(updated_node):
            # Splice out: return the receiver (previous link in the chain).
            # Transfer parentheses from the removed node so multiline formatting stays valid.
            if updated_node.lpar and hasattr(receiver, "lpar"):
                receiver = receiver.with_changes(lpar=updated_node.lpar, rpar=updated_node.rpar)
            return receiver

        # Clean up edge references to the removed step in this chained call.
        cleaned = self._clean_references(updated_node)
        if cleaned is not None:
            return cleaned

        return updated_node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_dot_step(self, call: cst.Call) -> bool:
        """Check if a Call node is <something>.step(...)."""
        return isinstance(call.func, cst.Attribute) and call.func.attr.value == "step"

    def _is_standalone_step_call(self, call: cst.Call) -> bool:
        """Check if a Call node is entry_point.step(...) as a standalone expression."""
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

    def _references_step(self, arg: cst.Arg) -> bool:
        """Check if an argument's value references the removed step via step_span."""
        code = cst.parse_module("").code_for_node(arg.value)
        return f'step_span("{self.step_name}")' in code

    def _clean_references(self, call: cst.Call) -> cst.Call | None:
        """Remove references to the removed step from a .step() call.

        Returns None if no changes are needed, or the updated Call node.
        """
        first_arg = call.args[0]
        changed = False
        new_args = [first_arg]

        for arg in call.args[1:]:
            if not isinstance(arg.keyword, cst.Name):
                new_args.append(arg)
                continue

            key = arg.keyword.value

            # Clean depends_on: remove the step name from the list.
            if key == "depends_on" and isinstance(arg.value, cst.List):
                remaining = []
                for el in arg.value.elements:
                    if isinstance(el.value, (cst.SimpleString, cst.ConcatenatedString)):
                        if el.value.evaluated_value == self.step_name:
                            changed = True
                            continue
                    remaining.append(el)
                if remaining:
                    # Fix comma separators on remaining elements.
                    fixed = []
                    for i, el in enumerate(remaining):
                        if i < len(remaining) - 1:
                            fixed.append(el.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))
                        else:
                            fixed.append(el.with_changes(comma=cst.MaybeSentinel.DEFAULT))
                    new_args.append(arg.with_changes(value=arg.value.with_changes(elements=fixed)))
                else:
                    changed = True
                continue

            # Remove when= if it references the removed step.
            if key == "when" and self._references_step(arg):
                changed = True
                continue

            # Remove param kwargs that reference the removed step.
            if self._references_step(arg):
                changed = True
                continue

            new_args.append(arg)

        if not changed:
            return None

        # Fix comma separators on args.
        fixed_args = []
        for i, arg in enumerate(new_args):
            if i < len(new_args) - 1:
                fixed_args.append(arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))
            else:
                fixed_args.append(arg.with_changes(comma=cst.MaybeSentinel.DEFAULT))

        return call.with_changes(args=fixed_args)
