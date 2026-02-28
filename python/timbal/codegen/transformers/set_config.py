import argparse
import json

import libcst as cst

from ..utils import (
    FRAMEWORK_TOOL_CONFIGS,
    FRAMEWORK_TOOL_NAMES,
    build_cst_value,
    collect_assignments,
    has_import,
    resolve_runnable_name,
    validate_tool_config,
)

AGENT_FIELDS = {
    "name",
    "description",
    "model",
    "system_prompt",
    "max_iter",
    "model_params",
    "skills_path",
}


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "set-config",
        help="Set configuration on the agent or on a specific tool.",
    )
    sp.add_argument(
        "tool_name",
        nargs="?",
        default=None,
        help="Tool name to configure. Omit to configure the agent itself.",
    )
    sp.add_argument(
        "--config",
        required=True,
        help='Configuration as a JSON object. E.g. \'{"model": "openai/gpt-4o-mini"}\'.',
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    config = json.loads(args.config)

    if args.tool_name:
        assignments = collect_assignments(tree) if tree else {}
        tool_class = _resolve_tool_class(tree, entry_point, args.tool_name, assignments)
        if tool_class is None:
            raise ValueError(f"Tool '{args.tool_name}' not found in agent tools list.")
        validate_tool_config(tool_class, config)
        var_name = FRAMEWORK_TOOL_NAMES.get(tool_class, args.tool_name)
        return ToolConfigSetter(entry_point, args.tool_name, config, assignments, tool_class, var_name)

    unknown = set(config.keys()) - AGENT_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown agent config field(s): {', '.join(sorted(unknown))}. "
            f"Valid fields: {', '.join(sorted(AGENT_FIELDS))}."
        )
    return AgentConfigSetter(entry_point, config)


def _resolve_tool_class(
    tree: cst.Module, entry_point: str, tool_name: str, assignments: dict[str, cst.Call]
) -> str | None:
    """Find the framework class name for a tool in the agent's tools list."""
    # Find the tools list in the entry point.
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign):
                    for target in item.targets:
                        if isinstance(target.target, cst.Name) and target.target.value == entry_point:
                            if isinstance(item.value, cst.Call):
                                return _find_tool_class(item.value, tool_name, assignments)
    return None


def _find_tool_class(call: cst.Call, tool_name: str, assignments: dict[str, cst.Call]) -> str | None:
    """Find the framework class name for a tool in a Call's tools=[...] kwarg."""
    for arg in call.args:
        if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "tools":
            if isinstance(arg.value, cst.List):
                for el in arg.value.elements:
                    resolved = resolve_runnable_name(el.value, assignments)
                    if resolved == tool_name:
                        return _class_name_from_element(el.value, assignments)
    return None


def _class_name_from_element(element: cst.BaseExpression, assignments: dict[str, cst.Call]) -> str | None:
    """Extract the class name (e.g. 'WebSearch') from a tool element."""
    if isinstance(element, cst.Call) and isinstance(element.func, cst.Name):
        return element.func.value
    if isinstance(element, cst.Name):
        var_name = element.value
        if var_name in assignments:
            call = assignments[var_name]
            if isinstance(call.func, cst.Name):
                return call.func.value
    return None


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


class AgentConfigSetter(cst.CSTTransformer):
    def __init__(self, entry_point: str, config: dict):
        self.entry_point = entry_point
        self.config = config

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call):
                    new_call = self._update_agent_kwargs(updated_node.value)
                    return updated_node.with_changes(value=new_call)
        return updated_node

    def _update_agent_kwargs(self, call: cst.Call) -> cst.Call:
        args = list(call.args)

        for field, value in self.config.items():
            # Drop existing kwarg for this field.
            args = [a for a in args if not (isinstance(a.keyword, cst.Name) and a.keyword.value == field)]

            # Add new kwarg (unless null = remove).
            if value is not None:
                args.append(cst.Arg(
                    keyword=cst.Name(field),
                    value=build_cst_value(value),
                ))

        return call.with_changes(args=args)


# ---------------------------------------------------------------------------
# Tool config
# ---------------------------------------------------------------------------


class ToolConfigSetter(cst.CSTTransformer):
    def __init__(
        self,
        entry_point: str,
        tool_name: str,
        config: dict,
        assignments: dict[str, cst.Call],
        tool_class: str,
        var_name: str,
    ):
        self.entry_point = entry_point
        self.tool_name = tool_name
        self.config = config
        self.assignments = assignments
        self.tool_class = tool_class  # e.g. "WebSearch"
        self.var_name = var_name  # e.g. "web_search"
        self._var_updated = False
        self._needs_migration = False
        self._inline_call: cst.Call | None = None  # original inline call args for migration

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if not isinstance(target.target, cst.Name):
                continue
            target_name = target.target.value

            # Entry point — check for inline tools to migrate.
            if target_name == self.entry_point and isinstance(updated_node.value, cst.Call):
                new_call = self._migrate_inline(updated_node.value)
                if new_call is not updated_node.value:
                    return updated_node.with_changes(value=new_call)

            # Variable assignment — update config kwargs.
            if target_name != self.entry_point and isinstance(updated_node.value, cst.Call):
                resolved = resolve_runnable_name(updated_node.value)
                if resolved == self.tool_name:
                    self._var_updated = True
                    return updated_node.with_changes(
                        value=self._build_configured_call(updated_node.value),
                    )
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self._needs_migration:
            return updated_node

        stmts_to_add: list[cst.BaseStatement] = []

        # Build the variable assignment merging existing inline args with new config.
        new_call = self._build_configured_call(self._inline_call)
        assignment_code = f"{self.var_name} = {cst.parse_module('').code_for_node(new_call)}\n"
        stmts_to_add.append(cst.parse_statement(assignment_code))

        if not stmts_to_add:
            return updated_node

        body = list(updated_node.body)

        # Insert before the entry point assignment.
        insert_idx = len(body)
        for i, stmt in enumerate(body):
            if isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Assign):
                        for t in item.targets:
                            if isinstance(t.target, cst.Name) and t.target.value == self.entry_point:
                                insert_idx = min(insert_idx, i)

        for stmt in reversed(stmts_to_add):
            body.insert(insert_idx, stmt)

        return updated_node.with_changes(body=body)

    def _migrate_inline(self, call: cst.Call) -> cst.Call:
        """Replace an inline tool Call with a Name reference in the tools list."""
        for i, arg in enumerate(call.args):
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "tools":
                if isinstance(arg.value, cst.List):
                    for j, el in enumerate(arg.value.elements):
                        resolved = resolve_runnable_name(el.value, self.assignments)
                        if resolved == self.tool_name and isinstance(el.value, cst.Call):
                            self._needs_migration = True
                            self._inline_call = el.value
                            new_ref = cst.Name(self.var_name)
                            updated_el = el.with_changes(value=new_ref)
                            new_elements = [*arg.value.elements[:j], updated_el, *arg.value.elements[j + 1:]]
                            new_list = arg.value.with_changes(elements=new_elements)
                            new_arg = arg.with_changes(value=new_list)
                            new_args = [*call.args[:i], new_arg, *call.args[i + 1:]]
                            return call.with_changes(args=new_args)
        return call

    def _build_configured_call(self, existing_call: cst.Call) -> cst.Call:
        """Merge new config kwargs into an existing Call, preserving all other args."""
        # Keep existing args except those being overridden or removed.
        args = [
            a for a in existing_call.args
            if not (isinstance(a.keyword, cst.Name) and a.keyword.value in self.config)
        ]
        # Append new/updated config kwargs (None = remove, already dropped above).
        for key, value in self.config.items():
            if value is not None:
                args.append(cst.Arg(keyword=cst.Name(key), value=build_cst_value(value)))
        return existing_call.with_changes(args=args)
