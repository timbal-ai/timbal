import argparse

import libcst as cst

from ..utils import collect_assignments, resolve_runnable_name


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "rename",
        help="Rename a step or tool.",
    )
    sp.add_argument("old_name", help="Current name of the step or tool to rename.")
    sp.add_argument("--to", required=True, dest="new_name", help="New name for the step or tool.")


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    old_name = args.old_name
    new_name = args.new_name

    assignments = collect_assignments(tree) if tree else {}

    # Find the variable whose runtime name matches old_name.
    old_var = None
    for var_name, call in assignments.items():
        resolved = resolve_runnable_name(call)
        if resolved == old_name:
            old_var = var_name
            break

    if old_var is None:
        raise ValueError(f"No step or tool found with name '{old_name}'.")

    if old_var == entry_point:
        raise ValueError(
            f"Cannot rename the entry point variable '{entry_point}'. "
            "The entry point name is referenced by timbal.yaml."
        )

    return Renamer(old_var, new_name, old_name, new_name)


class Renamer(cst.CSTTransformer):
    def __init__(self, old_var: str, new_var: str, old_runtime_name: str, new_runtime_name: str):
        self.old_var = old_var
        self.new_var = new_var
        self.old_runtime_name = old_runtime_name
        self.new_runtime_name = new_runtime_name
        # Context tracking for string replacement.
        self._in_depends_on = False
        self._in_step_span = False

    # -- Global Name rename ---------------------------------------------------

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if updated_node.value == self.old_var:
            return updated_node.with_changes(value=self.new_var)
        return updated_node

    # -- Update name= kwarg in the constructor --------------------------------

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        # After leave_Name, the target is already renamed. Check for new_var.
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.new_var:
                if isinstance(updated_node.value, cst.Call):
                    new_call = self._update_name_kwarg(updated_node.value)
                    return updated_node.with_changes(value=new_call)
        return updated_node

    # -- Context tracking for string replacement ------------------------------

    def visit_Arg(self, node: cst.Arg) -> bool:
        if isinstance(node.keyword, cst.Name) and node.keyword.value == "depends_on":
            self._in_depends_on = True
        return True

    def leave_Arg(self, original_node: cst.Arg, updated_node: cst.Arg) -> cst.Arg:
        self._in_depends_on = False
        return updated_node

    def visit_Call(self, node: cst.Call) -> bool:
        if isinstance(node.func, cst.Attribute) and node.func.attr.value == "step_span":
            self._in_step_span = True
        return True

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if isinstance(original_node.func, cst.Attribute) and original_node.func.attr.value == "step_span":
            self._in_step_span = False
        return updated_node

    # -- String replacement in depends_on and step_span -----------------------

    def leave_SimpleString(
        self, original_node: cst.SimpleString, updated_node: cst.SimpleString
    ) -> cst.SimpleString:
        if not (self._in_depends_on or self._in_step_span):
            return updated_node
        evaluated = updated_node.evaluated_value
        if evaluated == self.old_runtime_name:
            return updated_node.with_changes(value=f'"{self.new_runtime_name}"')
        return updated_node

    # -- Helpers --------------------------------------------------------------

    def _update_name_kwarg(self, call: cst.Call) -> cst.Call:
        """Update the name= kwarg in a constructor Call to the new runtime name."""
        for i, arg in enumerate(call.args):
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "name":
                new_arg = arg.with_changes(value=cst.SimpleString(f'"{self.new_runtime_name}"'))
                new_args = [*call.args[:i], new_arg, *call.args[i + 1:]]
                return call.with_changes(args=new_args)
        return call
