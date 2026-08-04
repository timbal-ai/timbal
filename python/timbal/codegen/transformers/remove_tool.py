import argparse

import libcst as cst

from ..cst_utils import (
    collect_assignments,
    collect_step_names,
    is_bare_function_step,
    resolve_entry_point_type,
    resolve_runnable_name,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "remove-tool",
        help="Remove a tool from the agent's tools list by name.",
    )
    sp.add_argument("--name", required=True, help="The tool name to remove. E.g. my_search_tool.")
    sp.add_argument(
        "--step",
        default=None,
        help="Target step name within a Workflow. When provided, the tool is removed from the step's tools list.",
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    step = getattr(args, "step", None)
    if tree is not None:
        ep_type = resolve_entry_point_type(tree, entry_point)
        if step:
            if ep_type is not None and ep_type != "Workflow":
                raise ValueError(f"--step requires a Workflow entry point, but '{entry_point}' is a {ep_type}.")
        else:
            if ep_type is not None and ep_type != "Agent":
                raise ValueError(f"remove-tool requires an Agent entry point, but '{entry_point}' is a {ep_type}.")

    target = step if step else entry_point
    assignments = collect_assignments(tree) if tree else {}

    # Validate the target resolves to a constructor assignment (mirrors
    # add-tool). Without this, an aliased entry point like ``agent = _agent``
    # would ride ``allow_noop`` to a phantom success: the transformer never
    # matches anything, the file is unchanged, and the tool stays in place.
    if tree is not None:
        if step:
            step_names = collect_step_names(tree, entry_point, assignments)
            if (
                step not in step_names
                and step not in assignments
                and not is_bare_function_step(tree, entry_point, step, assignments)
            ):
                raise ValueError(
                    f"Workflow step '{step}' not found. "
                    "Use the step variable name from .step(...), not the runtime name."
                )
        elif entry_point not in assignments:
            raise ValueError(
                f"Entry point variable '{entry_point}' not found in source. "
                "Ensure timbal.yaml fqn matches the Agent/Workflow variable name."
            )

    return ToolRemover(target, args.name, assignments)


class ToolRemover(cst.CSTTransformer):
    # Removing a tool that is already absent is an idempotent success.
    allow_noop = True

    def __init__(self, entry_point: str, tool_name: str, assignments: dict[str, cst.Call]):
        self.entry_point = entry_point
        self.tool_name = tool_name
        self.assignments = assignments

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call):
                    return updated_node.with_changes(
                        value=self._remove_from_tools(updated_node.value),
                    )
        return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:  # noqa: ARG002
        """Handle annotated assignments, e.g. ``agent: Agent = Agent(...)``."""
        if (
            isinstance(updated_node.target, cst.Name)
            and updated_node.target.value == self.entry_point
            and isinstance(updated_node.value, cst.Call)
        ):
            return updated_node.with_changes(
                value=self._remove_from_tools(updated_node.value),
            )
        return updated_node

    def _remove_from_tools(self, call: cst.Call) -> cst.Call:
        for i, arg in enumerate(call.args):
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "tools":
                if isinstance(arg.value, cst.List):
                    elements = [
                        el
                        for el in arg.value.elements
                        if resolve_runnable_name(el.value, self.assignments) != self.tool_name
                    ]
                    new_list = arg.value.with_changes(elements=elements)
                    new_arg = arg.with_changes(value=new_list)
                    new_args = [*call.args[:i], new_arg, *call.args[i + 1 :]]
                    return call.with_changes(args=new_args)
        return call
