import argparse
import json
import re

import libcst as cst

from ..cst_utils import (
    collect_assignments,
    has_import,
    resolve_entry_point_type,
    resolve_runnable_name,
)


def key_to_accessor(key: str) -> str:
    """Convert a dot-notation key path to Python accessor syntax.

    Examples:
        "output"                              → ".output"
        "output.cleaned"                      → ".output.cleaned"
        "output.0.something.something_else.0" → ".output[0].something.something_else[0]"
    """
    parts = key.split(".")
    result = ""
    for part in parts:
        if part.isdigit():
            result += f"[{part}]"
        else:
            result += f".{part}"
    return result


def accessor_to_key(accessor: str) -> str:
    """Convert Python accessor syntax back to dot-notation key path.

    Examples:
        ".output"                                    → "output"
        ".output.cleaned"                            → "output.cleaned"
        ".output[0].something.something_else[0]"     → "output.0.something.something_else.0"
        '["cleaned"]'                                → "cleaned"
        '.output["cleaned"]'                         → "output.cleaned"
    """
    # Normalize bracket access: ["key"] → .key, [0] → .0
    s = re.sub(r'\["([^"]+)"\]', r".\1", accessor)
    s = re.sub(r"\[(\d+)\]", r".\1", s)
    # Strip leading dot
    return s.lstrip(".")


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "set-param",
        help="Set a parameter on a workflow step (static value or mapped from another step's output).",
    )
    sp.add_argument(
        "--target",
        required=True,
        help="Target step name.",
    )
    sp.add_argument(
        "--name",
        required=True,
        help="Parameter name to set on the target step.",
    )
    sp.add_argument(
        "--type",
        required=True,
        choices=["map", "value"],
        dest="param_type",
        help="Parameter type: 'map' to wire from another step's output, 'value' for a static value.",
    )
    sp.add_argument(
        "--source",
        default=None,
        help="Source step name (required for type=map).",
    )
    sp.add_argument(
        "--key",
        default=None,
        help=(
            "Dot-notation path into the source step's output (optional, for type=map). "
            'E.g. "output.cleaned", "output.0.items", "output.result.name".'
        ),
    )
    sp.add_argument(
        "--value",
        default=None,
        help="Static value as a JSON literal (required for type=value). Use 'null' to remove the param.",
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    ep_type = resolve_entry_point_type(tree, entry_point) if tree else None

    if ep_type != "Workflow":
        raise ValueError("set-param requires a Workflow entry point.")

    param_type = args.param_type

    if param_type == "map":
        if not args.source:
            raise ValueError("--source is required for type=map.")
        return ParamSetter(
            entry_point=entry_point,
            target=args.target,
            param_name=args.name,
            param_type="map",
            source=args.source,
            key=args.key,
            assignments=collect_assignments(tree) if tree else {},
        )

    if param_type == "value":
        if args.value is None:
            raise ValueError("--value is required for type=value.")
        value = json.loads(args.value)
        return ParamSetter(
            entry_point=entry_point,
            target=args.target,
            param_name=args.name,
            param_type="value",
            value=value,
            assignments=collect_assignments(tree) if tree else {},
        )

    raise ValueError(f"Unknown param type: {param_type}")


class ParamSetter(cst.CSTTransformer):
    """Set a parameter on a workflow step's .step() call."""

    def __init__(
        self,
        entry_point: str,
        target: str,
        param_name: str,
        param_type: str,
        source: str | None = None,
        key: str | None = None,
        value: object = None,
        assignments: dict[str, cst.Call] | None = None,
    ):
        self.entry_point = entry_point
        self.target = target
        self.param_name = param_name
        self.param_type = param_type
        self.source = source
        self.key = key
        self.value = value
        self.assignments = assignments or {}
        self.needs_reorder = param_type == "map"

    def _is_step_call(self, call: cst.Call) -> bool:
        return (
            isinstance(call.func, cst.Attribute)
            and isinstance(call.func.value, cst.Name)
            and call.func.value.value == self.entry_point
            and call.func.attr.value == "step"
        )

    def _matches_target(self, call: cst.Call) -> bool:
        if not call.args:
            return False
        first_arg = call.args[0].value
        if isinstance(first_arg, cst.Name):
            var_name = first_arg.value
            if var_name in self.assignments:
                resolved = resolve_runnable_name(self.assignments[var_name])
                if resolved is not None:
                    return resolved == self.target
            return var_name == self.target
        return False

    def leave_Expr(self, original_node: cst.Expr, updated_node: cst.Expr) -> cst.Expr:
        call = updated_node.value
        if not isinstance(call, cst.Call):
            return updated_node
        if not self._is_step_call(call) or not self._matches_target(call):
            return updated_node

        step_call_code = self._build_step_call_code(call)
        parsed = cst.parse_module(step_call_code + "\n")
        for stmt in parsed.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Expr) and isinstance(item.value, cst.Call):
                        return updated_node.with_changes(value=item.value)
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if self.param_type == "map" and not has_import(original_node, "timbal.state", "get_run_context"):
            body = list(updated_node.body)
            import_stmt = cst.parse_statement("from timbal.state import get_run_context\n")
            import_insert_idx = 0
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, (cst.Import, cst.ImportFrom)):
                            import_insert_idx = i + 1
            body.insert(import_insert_idx, import_stmt)
            return updated_node.with_changes(body=body)
        return updated_node

    def _build_step_call_code(self, existing_call: cst.Call) -> str:
        first_arg = existing_call.args[0]
        step_ref = cst.parse_module("").code_for_node(first_arg.value)

        parts = [step_ref]

        # Keep existing kwargs except the one we're setting.
        for arg in existing_call.args[1:]:
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == self.param_name:
                continue
            if isinstance(arg.keyword, cst.Name):
                value_code = cst.parse_module("").code_for_node(arg.value).strip()
                parts.append(f"{arg.keyword.value}={value_code}")
            else:
                value_code = cst.parse_module("").code_for_node(arg.value).strip()
                parts.append(value_code)

        # Add the new param (unless value type with None = remove).
        if self.param_type == "map":
            base = f'get_run_context().step_span("{self.source}")'
            if self.key:
                accessor = key_to_accessor(self.key)
                parts.append(f"{self.param_name}=lambda: {base}{accessor}")
            else:
                parts.append(f"{self.param_name}=lambda: {base}.output")
        elif self.param_type == "value" and self.value is not None:
            parts.append(f"{self.param_name}={json.dumps(self.value)}")

        return f"{self.entry_point}.step({', '.join(parts)})"
