"""Shared CST helpers for the add-guardrail / remove-guardrail transformers.

Lives outside ``transformers/`` on purpose. That package is scanned by ``pkgutil`` and
every module in it is imported as an operation, so a helper module placed there would be
registered as a bogus operation — and, worse, having one transformer import another's
privates couples unrelated operations together at import time.
"""

import libcst as cst

from .cst_utils import (
    collect_assignments,
    collect_step_names,
    is_bare_function_step,
    resolve_entry_point_type,
)

__all__ = [
    "guardrails_kwarg_index",
    "literal_shorthands",
    "rail_name",
    "string_element",
    "string_value",
    "validate_guardrail_target",
]


def validate_guardrail_target(
    tree: cst.Module | None,
    entry_point: str,
    step: str | None,
    operation: str,
) -> tuple[str, dict]:
    """Resolve and validate the Agent the operation targets. Mirrors remove-tool."""
    if tree is not None:
        ep_type = resolve_entry_point_type(tree, entry_point)
        if step:
            if ep_type is not None and ep_type != "Workflow":
                raise ValueError(f"--step requires a Workflow entry point, but '{entry_point}' is a {ep_type}.")
        else:
            if ep_type is not None and ep_type != "Agent":
                raise ValueError(f"{operation} requires an Agent entry point, but '{entry_point}' is a {ep_type}.")

    target = step if step else entry_point
    assignments = collect_assignments(tree) if tree else {}

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
    return target, assignments


def string_value(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.SimpleString):
        return node.evaluated_value if isinstance(node.evaluated_value, str) else None
    return None


def string_element(value: str) -> cst.Element:
    return cst.Element(value=cst.SimpleString(f'"{value}"'))


def rail_name(spec: str) -> str:
    """The rail identity of a shorthand — ``"pii:redact"`` and ``"pii"`` are the same rail."""
    return spec.partition(":")[0].strip().lower()


def guardrails_kwarg_index(call: cst.Call) -> int | None:
    for i, arg in enumerate(call.args):
        if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "guardrails":
            return i
    return None


def literal_shorthands(value: cst.BaseExpression) -> list[str] | None:
    """The kwarg's current shorthand list, or None when it isn't literal strings.

    A ``"default"`` string expands to its preset shorthands so list edits compose.
    """
    if (s := string_value(value)) is not None:
        if s.strip().lower() == "default":
            from timbal.guardrails.presets import DEFAULT_SHORTHANDS

            return list(DEFAULT_SHORTHANDS)
        return [s]
    if isinstance(value, cst.List):
        out: list[str] = []
        for el in value.elements:
            s = string_value(el.value)
            if s is None:
                return None
            out.append(s)
        return out
    return None
