import argparse
import json

import libcst as cst

from ..cli_utils import arg_input
from ..cst_utils import (
    build_cst_value,
    collect_assignments,
    has_import,
    resolve_entry_point_type,
    resolve_runnable_name,
)
from ..tool_discovery import get_framework_tool_names, get_framework_tools, validate_tool_config


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "add-tool",
        help="Add a tool to the agent's tools list.",
    )
    sp.add_argument("--type", required=True, dest="tool_type", help="Type of tool to add.")
    sp.add_argument(
        "--definition",
        default=None,
        type=arg_input,
        help=(
            "Full function definition for custom tools. "
            "E.g. 'def my_tool(query: str) -> str:\\n    return query'. "
            "Use '@path' to read from file or '-' to read from stdin."
        ),
    )
    sp.add_argument(
        "--name",
        default=None,
        dest="tool_name",
        help="Explicit name for the tool. When omitted, the tool uses its default name.",
    )
    sp.add_argument(
        "--config",
        default=None,
        type=arg_input,
        help=(
            'Tool constructor params as JSON. E.g. \'{"limit": 5}\'. '
            "Validated against the tool's schema. "
            "Use '@path' to read from file or '-' to read from stdin."
        ),
    )
    sp.add_argument(
        "--step",
        default=None,
        help="Target step name within a Workflow. When provided, the tool is added to the step's tools list.",
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
                raise ValueError(f"add-tool requires an Agent entry point, but '{entry_point}' is a {ep_type}.")

    target = step if step else entry_point
    assignments = collect_assignments(tree) if tree else {}
    tool_type = args.tool_type
    valid_types = [*get_framework_tools().keys(), "Custom"]

    if tool_type not in valid_types:
        raise ValueError(f"--type must be one of {valid_types}, got '{tool_type}'.")

    # Reject `--name ""` early so downstream code can treat tool_name as a
    # tri-state (None / non-empty string) and the build/merge helpers don't
    # disagree about whether empty means "absent" or "explicit empty value".
    if args.tool_name is not None and args.tool_name == "":
        raise ValueError("--name cannot be empty. Omit --name to use the default tool name.")

    config = json.loads(args.config) if getattr(args, "config", None) else {}
    if config:
        # Reject keys that would collide with explicit flags / derived values.
        # Without this, the generated source has duplicate kwargs and ruff
        # rejects it with a SyntaxError ("Duplicate keyword argument").
        if "name" in config and args.tool_name is not None:
            raise ValueError(
                "--name and --config both set 'name'; pass it via --name only "
                "(or omit --name and set 'name' inside --config)."
            )
        if tool_type == "Custom" and "handler" in config:
            raise ValueError(
                "Custom tools derive 'handler' from --definition; remove 'handler' from --config."
            )
        # Validate against the tool's schema. Custom tools are wrapped in `Tool`
        # so they share its schema; framework tools validate against their own.
        validate_tool_config("Tool" if tool_type == "Custom" else tool_type, config)

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
        var_name = args.tool_name if args.tool_name is not None else f"{func_name}_tool"
        runtime_name = args.tool_name if args.tool_name is not None else func_name
        return ToolAdder(
            entry_point, assignments,
            target=target,
            tool_type="Custom",
            class_name=None,
            func_name=func_name,
            var_name=var_name,
            runtime_name=runtime_name,
            definition=args.definition,
            tool_name=args.tool_name,
            config=config,
        )

    # Framework tool.
    framework_names = get_framework_tool_names()
    var_name = args.tool_name if args.tool_name is not None else framework_names[tool_type]
    runtime_name = args.tool_name if args.tool_name is not None else framework_names[tool_type]
    return ToolAdder(
        entry_point, assignments,
        target=target,
        tool_type=tool_type,
        class_name=tool_type,
        func_name=None,
        var_name=var_name,
        runtime_name=runtime_name,
        tool_name=args.tool_name,
        config=config,
    )


class ToolAdder(cst.CSTTransformer):
    def __init__(
        self,
        entry_point: str,
        assignments: dict[str, cst.Call],
        *,
        target: str,
        tool_type: str,
        class_name: str | None,
        func_name: str | None,
        var_name: str,
        runtime_name: str,
        definition: str | None = None,
        tool_name: str | None = None,
        config: dict | None = None,
    ):
        self.entry_point = entry_point
        self.target = target  # variable to add tools to (entry_point or step var)
        self.assignments = assignments
        self.tool_type = tool_type
        self.class_name = class_name  # e.g. "WebSearch" (None for custom)
        self.func_name = func_name  # e.g. "my_search" (None for framework)
        self.var_name = var_name  # variable name for the assignment
        self.runtime_name = runtime_name  # name used by remove-tool
        self.definition = definition
        self.tool_name = tool_name  # explicit --name override (None = use default)
        self.config = config or {}  # extra constructor kwargs to merge into the tool call
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

    # -- Assign: update target tools list + existing variable assignments

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if not isinstance(target.target, cst.Name):
                continue
            target_name = target.target.value

            # Target (entry point or step) — update the tools list.
            if target_name == self.target and isinstance(updated_node.value, cst.Call):
                return updated_node.with_changes(
                    value=self._add_to_tools(updated_node.value),
                )

            # Existing variable assignment for this tool — update it.
            if target_name != self.target and isinstance(updated_node.value, cst.Call):
                resolved = resolve_runnable_name(updated_node.value)
                if resolved == self.runtime_name:
                    self._assignment_updated = True
                    return updated_node.with_changes(
                        value=self._merge_config_into_call(updated_node.value),
                    )
        return updated_node

    # -- Module: add imports, function defs, and variable assignments -------

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        imports_to_add: list[cst.BaseStatement] = []
        stmts_to_add: list[cst.BaseStatement] = []

        # --- Imports ---
        if self.tool_type != "Custom":
            module = get_framework_tools()[self.class_name].module
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
        # either a tool wrapper assignment or the target (entry point / step).
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
                                if isinstance(t.target, cst.Name) and t.target.value == self.target:
                                    insert_idx = min(insert_idx, i)
            for stmt in reversed(stmts_to_add):
                body.insert(insert_idx, stmt)

        return updated_node.with_changes(body=body)

    # -- Helpers ------------------------------------------------------------

    def _build_assignment_call(self) -> cst.Call:
        """Build the CST Call node for the variable assignment RHS."""
        args: list[cst.Arg] = []
        if self.tool_name is not None:
            args.append(cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString(f'"{self.tool_name}"')))
        if self.tool_type == "Custom":
            args.append(cst.Arg(keyword=cst.Name("handler"), value=cst.Name(self.func_name)))
        # Skip keys that are already emitted above; otherwise the output would
        # contain duplicate keyword arguments. `run()` rejects these cases with
        # a clear error, so this is just defense in depth.
        skip_keys: set[str] = set()
        if self.tool_name is not None:
            skip_keys.add("name")
        if self.tool_type == "Custom":
            skip_keys.add("handler")
        for key, value in self.config.items():
            if value is None or key in skip_keys:
                continue
            args.append(cst.Arg(keyword=cst.Name(key), value=build_cst_value(value)))
        if self.tool_type != "Custom":
            return cst.Call(func=cst.Name(self.class_name), args=args)
        return cst.Call(func=cst.Name("Tool"), args=args)

    def _build_assignment_code(self) -> str:
        """Build the source code string for the variable assignment RHS."""
        # Generate code via CST so kwarg ordering and quoting stay consistent.
        return cst.parse_module("").code_for_node(self._build_assignment_call())

    def _merge_config_into_call(self, existing: cst.Call) -> cst.Call:
        """Merge new config kwargs into an existing tool Call, preserving the rest.

        - Drops kwargs whose name is in self.config (they are being overridden or removed).
        - Re-emits the `name=` kwarg from --name when provided (overriding any prior value).
        - Appends new config kwargs (skipping those whose new value is None — that means remove).
        """
        override_keys = set(self.config.keys())
        if self.tool_name is not None:
            override_keys.add("name")

        args: list[cst.Arg] = [
            a for a in existing.args
            if not (isinstance(a.keyword, cst.Name) and a.keyword.value in override_keys)
        ]
        # Re-add name= when --name was provided. We then skip "name" in the
        # config loop below to avoid emitting two `name=` kwargs (which would
        # be a SyntaxError). `run()` already rejects --name + config['name']
        # together, so this is defense in depth.
        if self.tool_name is not None:
            args.append(cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString(f'"{self.tool_name}"')))
        # Append new config kwargs.
        for key, value in self.config.items():
            if value is None:
                continue
            if key == "name" and self.tool_name is not None:
                continue
            args.append(cst.Arg(keyword=cst.Name(key), value=build_cst_value(value)))
        return existing.with_changes(args=args)

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
