import argparse

import libcst as cst

from ..utils import (
    FRAMEWORK_TOOL_NAMES,
    collect_assignments,
    has_import,
    resolve_entry_point_type,
    resolve_runnable_name,
)

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
    sp.add_argument(
        "--name",
        default=None,
        dest="tool_name",
        help="Explicit name for the tool. When omitted, the tool uses its default name.",
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    if tree is not None:
        ep_type = resolve_entry_point_type(tree, entry_point)
        if ep_type is not None and ep_type != "Agent":
            raise ValueError(f"add-tool requires an Agent entry point, but '{entry_point}' is a {ep_type}.")

    assignments = collect_assignments(tree) if tree else {}
    tool_type = args.tool_type

    if tool_type == "Custom":
        if not args.definition:
            raise ValueError("--definition is required for Custom tools.")
        func_tree = cst.parse_module(args.definition)
        func_def = None
        for stmt in func_tree.body:
            if isinstance(stmt, cst.FunctionDef):
                func_def = stmt
                break
        if func_def is None:
            raise ValueError("--definition must contain a function definition.")
        func_name = func_def.name.value
        var_name = args.tool_name if args.tool_name else f"{func_name}_tool"
        runtime_name = args.tool_name if args.tool_name else func_name
        return ToolAdder(
            entry_point, assignments,
            tool_type="Custom",
            class_name=None,
            func_name=func_name,
            var_name=var_name,
            runtime_name=runtime_name,
            definition=args.definition,
            tool_name=args.tool_name,
        )

    # Framework tool.
    var_name = args.tool_name if args.tool_name else FRAMEWORK_TOOL_NAMES[tool_type]
    runtime_name = args.tool_name if args.tool_name else FRAMEWORK_TOOL_NAMES[tool_type]
    return ToolAdder(
        entry_point, assignments,
        tool_type=tool_type,
        class_name=tool_type,
        func_name=None,
        var_name=var_name,
        runtime_name=runtime_name,
        tool_name=args.tool_name,
    )


class ToolAdder(cst.CSTTransformer):
    def __init__(
        self,
        entry_point: str,
        assignments: dict[str, cst.Call],
        *,
        tool_type: str,
        class_name: str | None,
        func_name: str | None,
        var_name: str,
        runtime_name: str,
        definition: str | None = None,
        tool_name: str | None = None,
    ):
        self.entry_point = entry_point
        self.assignments = assignments
        self.tool_type = tool_type
        self.class_name = class_name  # e.g. "WebSearch" (None for custom)
        self.func_name = func_name  # e.g. "my_search" (None for framework)
        self.var_name = var_name  # variable name for the assignment
        self.runtime_name = runtime_name  # name used by remove-tool
        self.definition = definition
        self.tool_name = tool_name  # explicit --name override (None = use default)
        # Track whether an existing variable assignment was found and updated.
        self._assignment_updated = False

    # -- FunctionDef: replace existing custom function body -----------------

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef | cst.RemovalSentinel:
        if self.tool_type != "Custom" or not self.definition:
            return updated_node
        if updated_node.name.value == self.func_name:
            return cst.parse_statement(self.definition + "\n")
        return updated_node

    # -- Assign: update entry point tools list + existing variable assignments

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if not isinstance(target.target, cst.Name):
                continue
            target_name = target.target.value

            # Entry point — update the tools list.
            if target_name == self.entry_point and isinstance(updated_node.value, cst.Call):
                return updated_node.with_changes(
                    value=self._add_to_tools(updated_node.value),
                )

            # Existing variable assignment for this tool — update it.
            if target_name != self.entry_point and isinstance(updated_node.value, cst.Call):
                resolved = resolve_runnable_name(updated_node.value)
                if resolved == self.runtime_name:
                    self._assignment_updated = True
                    return updated_node.with_changes(
                        value=self._build_assignment_call(),
                    )
        return updated_node

    # -- Module: add imports, function defs, and variable assignments -------

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        imports_to_add: list[cst.BaseStatement] = []
        stmts_to_add: list[cst.BaseStatement] = []

        # --- Imports ---
        if self.tool_type != "Custom":
            module = FRAMEWORK_TOOLS[self.class_name]
            if not has_import(original_node, module, self.class_name):
                imports_to_add.append(cst.parse_statement(f"from {module} import {self.class_name}\n"))
        else:
            # Custom tools need `from timbal.core import Tool` for the wrapper.
            if not has_import(original_node, "timbal.core", "Tool"):
                imports_to_add.append(cst.parse_statement("from timbal.core import Tool\n"))

        # --- Function definition (custom tools) ---
        if self.tool_type == "Custom" and self.definition:
            already_defined = any(
                isinstance(stmt, cst.FunctionDef) and stmt.name.value == self.func_name
                for stmt in original_node.body
            )
            if not already_defined:
                stmts_to_add.append(cst.parse_statement(self.definition + "\n"))

        # --- Variable assignment (if not already updated in leave_Assign) ---
        if not self._assignment_updated:
            assignment_code = f"{self.var_name} = {self._build_assignment_code()}\n"
            stmts_to_add.append(cst.parse_statement(assignment_code))

        if not imports_to_add and not stmts_to_add:
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

        # Insert statements before the earliest relevant assignment:
        # either a tool wrapper assignment or the entry point.
        if stmts_to_add:
            insert_idx = len(body)
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, cst.Assign) and isinstance(item.value, cst.Call):
                            resolved = resolve_runnable_name(item.value)
                            if resolved == self.runtime_name:
                                insert_idx = min(insert_idx, i)
                            for t in item.targets:
                                if isinstance(t.target, cst.Name) and t.target.value == self.entry_point:
                                    insert_idx = min(insert_idx, i)
            for stmt in reversed(stmts_to_add):
                body.insert(insert_idx, stmt)

        return updated_node.with_changes(body=body)

    # -- Helpers ------------------------------------------------------------

    def _build_assignment_call(self) -> cst.Call:
        """Build the CST Call node for the variable assignment RHS."""
        args: list[cst.Arg] = []
        if self.tool_name:
            args.append(cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString(f'"{self.tool_name}"')))
        if self.tool_type != "Custom":
            return cst.Call(func=cst.Name(self.class_name), args=args)
        else:
            args.append(cst.Arg(keyword=cst.Name("handler"), value=cst.Name(self.func_name)))
            return cst.Call(func=cst.Name("Tool"), args=args)

    def _build_assignment_code(self) -> str:
        """Build the source code string for the variable assignment RHS."""
        name_part = f'name="{self.tool_name}", ' if self.tool_name else ""
        if self.tool_type != "Custom":
            return f"{self.class_name}({name_part.rstrip(', ')})"
        else:
            return f"Tool({name_part}handler={self.func_name})"

    def _add_to_tools(self, call: cst.Call) -> cst.Call:
        """Add or update the tool reference in the tools=[...] list."""
        new_ref = cst.Name(self.var_name)

        for i, arg in enumerate(call.args):
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "tools":
                if isinstance(arg.value, cst.List):
                    # Check if already present (as Name or inline Call).
                    for j, el in enumerate(arg.value.elements):
                        name = resolve_runnable_name(el.value, self.assignments)
                        if name == self.runtime_name:
                            if isinstance(el.value, cst.Call):
                                # Migrate inline Call → Name reference.
                                updated_el = el.with_changes(value=new_ref)
                                new_elements = [*arg.value.elements[:j], updated_el, *arg.value.elements[j + 1 :]]
                                new_list = arg.value.with_changes(elements=new_elements)
                                new_arg = arg.with_changes(value=new_list)
                                new_args = [*call.args[:i], new_arg, *call.args[i + 1 :]]
                                return call.with_changes(args=new_args)
                            # Already a Name reference — no change needed.
                            return call

                    # Not present — append.
                    new_element = cst.Element(value=new_ref)
                    new_list = arg.value.with_changes(
                        elements=[*arg.value.elements, new_element],
                    )
                    new_arg = arg.with_changes(value=new_list)
                    new_args = [*call.args[:i], new_arg, *call.args[i + 1 :]]
                    return call.with_changes(args=new_args)

        # No tools kwarg yet — add one.
        new_arg = cst.Arg(
            keyword=cst.Name("tools"),
            value=cst.List(elements=[cst.Element(value=new_ref)]),
        )
        return call.with_changes(args=[*call.args, new_arg])
