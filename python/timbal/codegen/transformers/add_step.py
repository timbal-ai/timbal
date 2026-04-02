import argparse
import json

import libcst as cst

from ..cst_utils import (
    build_cst_value,
    collect_assignments,
    has_import,
    resolve_entry_point_type,
    resolve_runnable_name,
)
from ..tool_discovery import get_framework_tool_names, get_framework_tools
from .set_config import AGENT_FIELDS

START_X = 100
START_Y = 100
NODE_SPACING_X = 360
FALLBACK_STACK_DY = 140


def _eval_number(node: cst.BaseExpression) -> float | None:
    """Extract a numeric value from a CST node."""
    if isinstance(node, cst.Integer):
        return float(node.value)
    if isinstance(node, cst.Float):
        return float(node.value)
    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
        inner = _eval_number(node.expression)
        return -inner if inner is not None else None
    return None


def _extract_position_from_call(call: cst.Call) -> dict[str, float] | None:
    """Extract {"x": float, "y": float} from a Call node's metadata kwarg."""
    for arg in call.args:
        if not (isinstance(arg.keyword, cst.Name) and arg.keyword.value == "metadata"):
            continue
        if not isinstance(arg.value, cst.Dict):
            continue
        for el in arg.value.elements:
            if not isinstance(el, cst.DictElement):
                continue
            if not (isinstance(el.key, cst.SimpleString) and el.key.evaluated_value == "position"):
                continue
            if not isinstance(el.value, cst.Dict):
                continue
            pos: dict[str, float] = {}
            for pos_el in el.value.elements:
                if not isinstance(pos_el, cst.DictElement):
                    continue
                if isinstance(pos_el.key, cst.SimpleString):
                    key = pos_el.key.evaluated_value
                    if key in ("x", "y"):
                        val = _eval_number(pos_el.value)
                        if val is not None:
                            pos[key] = val
            if "x" in pos and "y" in pos:
                return pos
    return None


def _count_workflow_steps(tree: cst.Module, entry_point: str) -> int:
    """Count total .step() calls — both chained on the constructor and standalone."""
    count = 0
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            # Standalone: entry_point.step(...)
            if isinstance(item, cst.Expr) and isinstance(item.value, cst.Call):
                call = item.value
                if (
                    isinstance(call.func, cst.Attribute)
                    and isinstance(call.func.value, cst.Name)
                    and call.func.value.value == entry_point
                    and call.func.attr.value == "step"
                ):
                    count += 1
            # Chained: workflow = Workflow().step().step()
            if isinstance(item, cst.Assign):
                for t in item.targets:
                    if isinstance(t.target, cst.Name) and t.target.value == entry_point:
                        node = item.value
                        while isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute):
                            if node.func.attr.value == "step":
                                count += 1
                            node = node.func.value
    return count


def _compute_default_position(
    assignments: dict[str, cst.Call],
    entry_point: str,
    unpositioned_count: int = 0,
) -> dict[str, float]:
    """Compute a default position for a new step.

    Collects saved positions from step assignments, then generates virtual
    positions for unpositioned steps (bare functions) at the default column
    (START_X) using the stacked fallback (FALLBACK_STACK_DY).  The new node
    is placed one column (NODE_SPACING_X) to the right of the rightmost
    column, vertically centred on that column's nodes.
    """
    positions = []
    for var_name, call in assignments.items():
        if var_name == entry_point:
            continue
        pos = _extract_position_from_call(call)
        if pos is not None:
            positions.append(pos)

    # Virtual positions for unpositioned steps (bare functions, etc.)
    for i in range(unpositioned_count):
        positions.append({"x": float(START_X), "y": float(START_Y + i * FALLBACK_STACK_DY)})

    if not positions:
        return {"x": START_X, "y": START_Y}

    max_x = max(p["x"] for p in positions)
    rightmost_nodes = [p for p in positions if p["x"] == max_x]
    avg_y = sum(p["y"] for p in rightmost_nodes) / len(rightmost_nodes)
    return {"x": max_x + NODE_SPACING_X, "y": avg_y}


def _clean_position(position: dict[str, float]) -> dict[str, int | float]:
    """Convert float values to int when they are whole numbers."""
    return {
        k: int(v) if isinstance(v, float) and v == int(v) else v
        for k, v in position.items()
    }


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
    sp.add_argument("--x", default=None, type=float, help="X canvas position. Auto-computed if omitted.")
    sp.add_argument("--y", default=None, type=float, help="Y canvas position. Auto-computed if omitted.")


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    if tree is not None:
        ep_type = resolve_entry_point_type(tree, entry_point)
        if ep_type is not None and ep_type != "Workflow":
            raise ValueError(f"add-step requires a Workflow entry point, but '{entry_point}' is a {ep_type}.")

    assignments = collect_assignments(tree) if tree else {}
    step_type = args.step_type
    config = json.loads(args.config) if args.config else {}

    if args.x is not None and args.y is not None:
        position = _clean_position({"x": args.x, "y": args.y})
    else:
        total_steps = _count_workflow_steps(tree, entry_point) if tree else 0
        positioned_count = sum(
            1 for var, call in assignments.items()
            if var != entry_point and _extract_position_from_call(call) is not None
        )
        unpositioned_count = max(0, total_steps - positioned_count)
        position = _clean_position(_compute_default_position(assignments, entry_point, unpositioned_count))

    valid_types = [*get_framework_tools().keys(), "Agent", "Custom"]

    if step_type not in valid_types:
        raise ValueError(
            f"--type '{step_type}' is not valid. "
            f"Use 'Agent' for agent steps, 'Custom' for custom function steps, "
            f"or a framework tool class name. "
            f"Run `timbal-codegen get-tools` to browse available tool types."
        )

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
        var_name = runtime_name
        # When the function name collides with the Tool variable name, add
        # a _fn suffix to the function to avoid shadowing.
        if func_name == var_name:
            func_name = f"{func_name}_fn"
            # Rewrite the definition with the renamed function.
            definition = args.definition.replace(f"def {func_def.name.value}(", f"def {func_name}(", 1)
        else:
            definition = args.definition
        return StepAdder(
            entry_point, assignments,
            step_type="Custom",
            class_name="Tool",
            func_name=func_name,
            var_name=var_name,
            runtime_name=runtime_name,
            definition=definition,
            step_name=args.step_name,
            config=config,
            position=position,
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
            position=position,
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
        position=position,
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
        position: dict[str, int | float] | None = None,
    ):
        self.entry_point = entry_point
        self.assignments = assignments
        self.step_type = step_type
        self.class_name = class_name  # e.g. "WebSearch", "Agent" (None for Custom)
        self.func_name = func_name  # e.g. "process_fn" (None for framework/Agent)
        self.var_name = var_name  # variable name for Tool wrapper assignment
        self.runtime_name = runtime_name  # name used for identification
        self.definition = definition
        self.step_name = step_name  # explicit --name override
        self.config = config or {}
        self.position = position or {"x": 0, "y": 0}
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
        elif self.step_type == "Custom":
            if not has_import(original_node, "timbal.core", "Tool"):
                imports_to_add.append(cst.parse_statement("from timbal.core import Tool\n"))
        else:
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

        # --- Variable assignment (Agent / framework tool / Custom Tool wrapper) ---
        if not self._assignment_updated:
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

    def _build_metadata_arg(self) -> cst.Arg:
        """Build a ``metadata={"position": {...}}`` CST Arg."""
        return cst.Arg(
            keyword=cst.Name("metadata"),
            value=build_cst_value({"position": self.position}),
        )

    def _build_assignment_call(self) -> cst.Call:
        """Build the CST Call node for the variable assignment RHS."""
        metadata_arg = self._build_metadata_arg()

        if self.step_type == "Agent":
            args: list[cst.Arg] = []
            for key, value in self.config.items():
                args.append(cst.Arg(keyword=cst.Name(key), value=build_cst_value(value)))
            args.append(metadata_arg)
            return cst.Call(func=cst.Name("Agent"), args=args)

        if self.step_type == "Custom":
            args: list[cst.Arg] = [
                cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString(f'"{self.runtime_name}"')),
                cst.Arg(keyword=cst.Name("handler"), value=cst.Name(self.func_name)),
                metadata_arg,
            ]
            return cst.Call(func=cst.Name("Tool"), args=args)

        # Framework tool.
        args = []
        if self.step_name:
            args.append(cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString(f'"{self.step_name}"')))
        args.append(metadata_arg)
        return cst.Call(func=cst.Name(self.class_name), args=args)

    def _metadata_code(self) -> str:
        """Build the metadata keyword argument as a source code fragment."""
        x, y = self.position["x"], self.position["y"]
        return f'metadata={{"position": {{"x": {x}, "y": {y}}}}}'

    def _build_assignment_code(self) -> str:
        """Build the source code string for the variable assignment RHS."""
        meta = self._metadata_code()

        if self.step_type == "Agent":
            config_parts = ", ".join(
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                for k, v in self.config.items()
            )
            sep = ", " if config_parts else ""
            return f"Agent({config_parts}{sep}{meta})"

        if self.step_type == "Custom":
            return f'Tool(name="{self.runtime_name}", handler={self.func_name}, {meta})'

        # Framework tool.
        name_part = f'name="{self.step_name}", ' if self.step_name else ""
        return f"{self.class_name}({name_part}{meta})"

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
        return f"{self.entry_point}.step({self.var_name})"
