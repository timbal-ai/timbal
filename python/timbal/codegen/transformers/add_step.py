import argparse
import json

import libcst as cst

from ..utils import (
    collect_assignments,
    get_framework_tool_names,
    get_framework_tools,
    has_import,
    resolve_entry_point_type,
    resolve_runnable_name,
)
from .set_config import AGENT_FIELDS


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "add-step",
        help="Add a step to the workflow.",
    )
    sp.add_argument("--type", required=True, dest="step_type", help="Type of step to add.")
    sp.add_argument(
        "--definition",
        default=None,
        help="Full function definition for custom steps. E.g. 'def process(x: str) -> str:\\n    return x.upper()'",
    )
    sp.add_argument(
        "--name",
        default=None,
        dest="step_name",
        help="Explicit name for the step. When omitted, the step uses its default name.",
    )
    sp.add_argument(
        "--config",
        default=None,
        help='Agent constructor params as JSON. E.g. \'{"name": "agent_a", "model": "openai/gpt-4o-mini"}\'.',
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    if tree is not None:
        ep_type = resolve_entry_point_type(tree, entry_point)
        if ep_type is not None and ep_type != "Workflow":
            raise ValueError(f"add-step requires a Workflow entry point, but '{entry_point}' is a {ep_type}.")

    assignments = collect_assignments(tree) if tree else {}
    step_type = args.step_type
    config = json.loads(args.config) if args.config else {}
    valid_types = [*get_framework_tools().keys(), "Agent", "Custom"]

    if step_type not in valid_types:
        raise ValueError(f"--type must be one of {valid_types}, got '{step_type}'.")

    if step_type == "Custom":
        if not args.definition:
            raise ValueError("--definition is required for Custom steps.")
        func_tree = cst.parse_module(args.definition)
        func_def = None
        for stmt in func_tree.body:
            if isinstance(stmt, cst.FunctionDef):
                func_def = stmt
                break
        if func_def is None:
            raise ValueError("--definition must contain a function definition.")
        func_name = func_def.name.value
        runtime_name = args.step_name if args.step_name else func_name
        return StepAdder(
            entry_point, assignments,
            step_type="Custom",
            class_name=None,
            func_name=func_name,
            var_name=None,
            runtime_name=runtime_name,
            definition=args.definition,
            step_name=args.step_name,
            config=config,
        )

    if step_type == "Agent":
        if not config.get("name") and not args.step_name:
            raise ValueError("Agent steps require a name (via --config '{\"name\": ...}' or --name).")
        # Validate config against AGENT_FIELDS
        unknown = set(config.keys()) - AGENT_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown agent config field(s): {', '.join(sorted(unknown))}. "
                f"Valid fields: {', '.join(sorted(AGENT_FIELDS))}."
            )
        agent_name = config.get("name") or args.step_name
        var_name = args.step_name if args.step_name else agent_name
        runtime_name = agent_name
        return StepAdder(
            entry_point, assignments,
            step_type="Agent",
            class_name="Agent",
            func_name=None,
            var_name=var_name,
            runtime_name=runtime_name,
            config=config,
        )

    # Framework tool step.
    framework_names = get_framework_tool_names()
    var_name = args.step_name if args.step_name else framework_names[step_type]
    runtime_name = args.step_name if args.step_name else framework_names[step_type]
    return StepAdder(
        entry_point, assignments,
        step_type=step_type,
        class_name=step_type,
        func_name=None,
        var_name=var_name,
        runtime_name=runtime_name,
        step_name=args.step_name,
    )


class StepAdder(cst.CSTTransformer):
    def __init__(
        self,
        entry_point: str,
        assignments: dict[str, cst.Call],
        *,
        step_type: str,
        class_name: str | None,
        func_name: str | None,
        var_name: str | None,
        runtime_name: str,
        definition: str | None = None,
        step_name: str | None = None,
        config: dict | None = None,
    ):
        self.entry_point = entry_point
        self.assignments = assignments
        self.step_type = step_type
        self.class_name = class_name  # e.g. "WebSearch", "Agent" (None for Custom)
        self.func_name = func_name  # e.g. "process" (None for framework/Agent)
        self.var_name = var_name  # variable name for assignment (None for Custom)
        self.runtime_name = runtime_name  # name used for identification
        self.definition = definition
        self.step_name = step_name  # explicit --name override
        self.config = config or {}
        self._assignment_updated = False
        self._step_call_updated = False

    # -- FunctionDef: replace existing custom function body -----------------

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef | cst.RemovalSentinel:
        if self.step_type != "Custom" or not self.definition:
            return updated_node
        if updated_node.name.value == self.func_name:
            return cst.parse_statement(self.definition + "\n")
        return updated_node

    # -- Assign: update existing variable assignments for idempotency ------

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        if self.step_type == "Custom":
            return updated_node

        for target in updated_node.targets:
            if not isinstance(target.target, cst.Name):
                continue
            target_name = target.target.value

            # Skip the entry point itself.
            if target_name == self.entry_point:
                continue

            # Existing variable assignment for this step — update it.
            if isinstance(updated_node.value, cst.Call):
                resolved = resolve_runnable_name(updated_node.value)
                if resolved == self.runtime_name:
                    self._assignment_updated = True
                    return updated_node.with_changes(
                        value=self._build_assignment_call(),
                    )
        return updated_node

    # -- ExpressionStatement: update existing .step() call -----------------

    def leave_Expr(self, original_node: cst.Expr, updated_node: cst.Expr) -> cst.Expr:
        """Update existing .step() call for idempotency."""
        call = updated_node.value
        if not isinstance(call, cst.Call):
            return updated_node
        if not self._is_step_call(call):
            return updated_node

        # Check if first arg resolves to our step name.
        if call.args and self._resolve_step_name(call.args[0].value) == self.runtime_name:
            self._step_call_updated = True
            # Rebuild with just the positional arg (preserving any existing kwargs from set-config).
            return updated_node.with_changes(value=self._build_step_call())
        return updated_node

    # -- Module: add imports, definitions, assignments, and .step() calls --

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        imports_to_add: list[cst.BaseStatement] = []
        stmts_to_add: list[cst.BaseStatement] = []

        # --- Imports ---
        if self.step_type == "Agent":
            if not has_import(original_node, "timbal", "Agent"):
                imports_to_add.append(cst.parse_statement("from timbal import Agent\n"))
        elif self.step_type != "Custom":
            module = get_framework_tools()[self.class_name].module
            if not has_import(original_node, module, self.class_name):
                imports_to_add.append(cst.parse_statement(f"from {module} import {self.class_name}\n"))

        # --- Function definition (Custom type) ---
        if self.step_type == "Custom" and self.definition:
            already_defined = any(
                isinstance(stmt, cst.FunctionDef) and stmt.name.value == self.func_name
                for stmt in original_node.body
            )
            if not already_defined:
                stmts_to_add.append(cst.parse_statement(self.definition + "\n"))

        # --- Variable assignment (Agent / framework tool types) ---
        if self.step_type != "Custom" and not self._assignment_updated:
            assignment_code = f"{self.var_name} = {self._build_assignment_code()}\n"
            stmts_to_add.append(cst.parse_statement(assignment_code))

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

        # Insert definitions/assignments before the entry point.
        if stmts_to_add:
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

        # --- Append .step() call ---
        if not self._step_call_updated:
            step_call_code = self._build_step_call_code()
            step_stmt = cst.parse_statement(step_call_code + "\n")

            # Find insertion point: after last existing .step() call,
            # or after the entry point assignment.
            step_insert_idx = None
            entry_point_idx = None
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, cst.Expr) and isinstance(item.value, cst.Call):
                            if self._is_step_call(item.value):
                                step_insert_idx = i + 1
                        if isinstance(item, cst.Assign):
                            for t in item.targets:
                                if isinstance(t.target, cst.Name) and t.target.value == self.entry_point:
                                    entry_point_idx = i + 1

            insert_at = step_insert_idx if step_insert_idx is not None else entry_point_idx
            if insert_at is not None:
                body.insert(insert_at, step_stmt)
            else:
                body.append(step_stmt)

        return updated_node.with_changes(body=body)

    # -- Helpers ------------------------------------------------------------

    def _is_step_call(self, call: cst.Call) -> bool:
        """Check if a Call node is entry_point.step(...)."""
        return (
            isinstance(call.func, cst.Attribute)
            and isinstance(call.func.value, cst.Name)
            and call.func.value.value == self.entry_point
            and call.func.attr.value == "step"
        )

    def _resolve_step_name(self, element: cst.BaseExpression) -> str | None:
        """Resolve the name of a step from the first arg of .step()."""
        if isinstance(element, cst.Name):
            var_name = element.value
            if self.assignments and var_name in self.assignments:
                call = self.assignments[var_name]
                resolved = resolve_runnable_name(call)
                if resolved is not None:
                    return resolved
            return var_name
        return None

    def _build_assignment_call(self) -> cst.Call:
        """Build the CST Call node for the variable assignment RHS."""
        if self.step_type == "Agent":
            from ..utils import build_cst_value

            args: list[cst.Arg] = []
            for key, value in self.config.items():
                args.append(cst.Arg(keyword=cst.Name(key), value=build_cst_value(value)))
            return cst.Call(func=cst.Name("Agent"), args=args)

        # Framework tool.
        args = []
        if self.step_name:
            args.append(cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString(f'"{self.step_name}"')))
        return cst.Call(func=cst.Name(self.class_name), args=args)

    def _build_assignment_code(self) -> str:
        """Build the source code string for the variable assignment RHS."""
        if self.step_type == "Agent":
            config_parts = ", ".join(
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                for k, v in self.config.items()
            )
            return f"Agent({config_parts})"

        # Framework tool.
        name_part = f'name="{self.step_name}"' if self.step_name else ""
        return f"{self.class_name}({name_part})"

    def _build_step_call(self) -> cst.Call:
        """Build the CST Call node for workflow.step(...)."""
        step_call_code = self._build_step_call_code()
        parsed = cst.parse_module(step_call_code + "\n")
        for stmt in parsed.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Expr) and isinstance(item.value, cst.Call):
                        return item.value
        raise RuntimeError("Failed to parse step call")

    def _build_step_call_code(self) -> str:
        """Build the source code string for workflow.step(...)."""
        step_ref = self.func_name if self.step_type == "Custom" else self.var_name
        return f"{self.entry_point}.step({step_ref})"
