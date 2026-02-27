import argparse

import libcst as cst

from ..utils import collect_assignments, resolve_runnable_name


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "remove-tool",
        help="Remove a tool from the agent's tools list by name.",
    )
    sp.add_argument("value", help="The tool name to remove. E.g. my_search_tool.")


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    assignments = collect_assignments(tree) if tree else {}
    return ToolRemover(entry_point, args.value, assignments)


class ToolRemover(cst.CSTTransformer):
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
