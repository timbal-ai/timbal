import argparse

import libcst as cst

from ..utils import FRAMEWORK_TOOL_NAMES, resolve_runnable_name

# Framework tools: class name -> module path
FRAMEWORK_TOOLS = {
    "Bash": "timbal.tools",
    "CalaSearch": "timbal.tools",
    "Edit": "timbal.tools",
    "Read": "timbal.tools",
    "WebSearch": "timbal.tools",
    "Write": "timbal.tools",
}

TOOL_TYPES = [*FRAMEWORK_TOOLS.keys(), "Custom"]


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "add-tool",
        help="Add a tool to the agent's tools list.",
    )
    sp.add_argument("--type", choices=TOOL_TYPES, required=True, dest="tool_type", help="Type of tool to add.")
    sp.add_argument(
        "--definition",
        default=None,
        help="Full function definition for custom tools. E.g. 'def my_tool(query: str) -> str:\\n    return query'",
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    assignments = _collect_assignments(tree) if tree else {}
    tool_type = args.tool_type

    if tool_type == "Custom":
        if not args.definition:
            raise ValueError("--definition is required for Custom tools.")
        # Parse the definition to extract the function name.
        func_tree = cst.parse_module(args.definition)
        func_def = None
        for stmt in func_tree.body:
            if isinstance(stmt, cst.FunctionDef):
                func_def = stmt
                break
        if func_def is None:
            raise ValueError("--definition must contain a function definition.")
        tool_name = func_def.name.value
        return ToolAdder(entry_point, tool_name, assignments, tool_type="Custom", definition=args.definition)

    return ToolAdder(entry_point, tool_type, assignments, tool_type=tool_type)


def _collect_assignments(tree: cst.Module) -> dict[str, cst.Call]:
    """Build a map of variable_name -> Call node for all top-level assignments."""
    result = {}
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign) and isinstance(item.value, cst.Call):
                    for target in item.targets:
                        if isinstance(target.target, cst.Name):
                            result[target.target.value] = item.value
    return result


def _has_import(tree: cst.Module, module: str, name: str) -> bool:
    """Check if `from <module> import <name>` already exists."""
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.ImportFrom) and not isinstance(item.names, cst.ImportStar):
                    # Reconstruct dotted module name.
                    parts = []
                    node = item.module
                    while isinstance(node, cst.Attribute):
                        parts.append(node.attr.value)
                        node = node.value
                    if isinstance(node, cst.Name):
                        parts.append(node.value)
                    mod = ".".join(reversed(parts))

                    if mod == module:
                        for alias in item.names:
                            if isinstance(alias, cst.ImportAlias):
                                imported = alias.name.value if isinstance(alias.name, cst.Name) else ""
                                if imported == name:
                                    return True
    return False


class ToolAdder(cst.CSTTransformer):
    def __init__(
        self,
        entry_point: str,
        tool_name: str,
        assignments: dict[str, cst.Call],
        *,
        tool_type: str,
        definition: str | None = None,
    ):
        self.entry_point = entry_point
        self.tool_name = tool_name
        self.runtime_name = FRAMEWORK_TOOL_NAMES.get(tool_name, tool_name)
        self.assignments = assignments
        self.tool_type = tool_type
        self.definition = definition

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef | cst.RemovalSentinel:
        if self.tool_type != "Custom" or not self.definition:
            return updated_node
        # Replace existing function with the new definition.
        if updated_node.name.value == self.tool_name:
            return cst.parse_statement(self.definition + "\n")
        return updated_node

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call):
                    return updated_node.with_changes(
                        value=self._add_to_tools(updated_node.value),
                    )
            # Update handler= in Tool wrapper assignments whose name matches.
            if self.tool_type == "Custom" and self.definition and isinstance(updated_node.value, cst.Call):
                name = resolve_runnable_name(updated_node.value)
                if name == self.runtime_name:
                    new_args = []
                    for arg in updated_node.value.args:
                        if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "handler":
                            new_args.append(arg.with_changes(value=cst.Name(self.tool_name)))
                        else:
                            new_args.append(arg)
                    return updated_node.with_changes(
                        value=updated_node.value.with_changes(args=new_args),
                    )
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        imports_to_add: list[cst.BaseStatement] = []
        funcs_to_add: list[cst.BaseStatement] = []

        if self.tool_type != "Custom":
            # Add import if missing.
            module = FRAMEWORK_TOOLS[self.tool_name]
            if not _has_import(original_node, module, self.tool_name):
                imports_to_add.append(cst.parse_statement(f"from {module} import {self.tool_name}\n"))
        else:
            # Add function definition if not already defined (replacement is handled in leave_FunctionDef).
            already_defined = (
                any(
                    isinstance(stmt, cst.FunctionDef) and stmt.name.value == self.tool_name
                    for stmt in original_node.body
                )
                or self.tool_name in self.assignments
            )
            if not already_defined:
                funcs_to_add.append(cst.parse_statement(self.definition + "\n"))

        if not imports_to_add and not funcs_to_add:
            return updated_node

        body = list(updated_node.body)

        # Insert imports after the last existing import.
        if imports_to_add:
            import_insert_idx = 0
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, (cst.Import, cst.ImportFrom)):
                            import_insert_idx = i + 1
            for stmt in reversed(imports_to_add):
                body.insert(import_insert_idx, stmt)

        # Insert function definition before the first assignment that references
        # the tool name (e.g. a Tool wrapper), or before the entry point.
        if funcs_to_add:
            insert_idx = len(body)
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, cst.Assign) and isinstance(item.value, cst.Call):
                            resolved = resolve_runnable_name(item.value)
                            if resolved == self.runtime_name:
                                insert_idx = min(insert_idx, i)
                            for target in item.targets:
                                if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                                    insert_idx = min(insert_idx, i)
            for stmt in reversed(funcs_to_add):
                body.insert(insert_idx, stmt)

        return updated_node.with_changes(body=body)

    def _add_to_tools(self, call: cst.Call) -> cst.Call:
        if self.tool_type != "Custom":
            new_value = cst.Call(func=cst.Name(self.tool_name))
        else:
            new_value = cst.Name(self.tool_name)

        for i, arg in enumerate(call.args):
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "tools":
                if isinstance(arg.value, cst.List):
                    # Check if already present.
                    for el in arg.value.elements:
                        name = resolve_runnable_name(el.value, self.assignments)
                        if name == self.runtime_name:
                            return call

                    new_element = cst.Element(value=new_value)
                    new_list = arg.value.with_changes(
                        elements=[*arg.value.elements, new_element],
                    )
                    new_arg = arg.with_changes(value=new_list)
                    new_args = [*call.args[:i], new_arg, *call.args[i + 1 :]]
                    return call.with_changes(args=new_args)

        # No tools kwarg yet — add one.
        new_arg = cst.Arg(
            keyword=cst.Name("tools"),
            value=cst.List(elements=[cst.Element(value=new_value)]),
        )
        return call.with_changes(args=[*call.args, new_arg])
