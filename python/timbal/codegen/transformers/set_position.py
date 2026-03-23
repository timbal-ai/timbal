import argparse

import libcst as cst

from ..cst_utils import (
    build_cst_value,
    collect_assignments,
    resolve_entry_point_type,
    resolve_runnable_name,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "set-position",
        help="Set the (x, y) canvas position for a node, stored in metadata.",
    )
    sp.add_argument(
        "--name",
        default=None,
        help="Step name to position. Omit to position the entry point itself.",
    )
    sp.add_argument("--x", required=True, type=float, help="X coordinate.")
    sp.add_argument("--y", required=True, type=float, help="Y coordinate.")


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    ep_type = resolve_entry_point_type(tree, entry_point) if tree else None

    position = {"x": args.x, "y": args.y}

    if ep_type == "Workflow":
        if not args.name:
            raise ValueError("set-position on a Workflow entry point requires a step --name.")
        assignments = collect_assignments(tree) if tree else {}
        return StepPositionSetter(entry_point, args.name, position, assignments)

    # Agent (or unknown) entry point — set on the constructor directly.
    if args.name:
        raise ValueError("--name is only supported for Workflow entry points.")

    return ConstructorPositionSetter(entry_point, position)


def _merge_position_into_metadata(existing_args: list[cst.Arg], position: dict) -> list[cst.Arg]:
    """Replace or insert the ``metadata`` kwarg with the position merged in.

    If a ``metadata`` kwarg already exists and is a literal Dict, the
    ``"position"`` key is upserted while preserving other keys.  Otherwise a
    fresh ``metadata={"position": {...}}`` kwarg is appended.
    """
    new_args: list[cst.Arg] = []
    found = False

    for arg in existing_args:
        if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "metadata":
            found = True
            if isinstance(arg.value, cst.Dict):
                # Keep every element except an existing "position" key.
                elements = [
                    el
                    for el in arg.value.elements
                    if not (
                        isinstance(el, cst.DictElement)
                        and isinstance(el.key, cst.SimpleString)
                        and el.key.evaluated_value == "position"
                    )
                ]
                elements.append(
                    cst.DictElement(
                        key=build_cst_value("position"),
                        value=build_cst_value(position),
                    )
                )
                new_args.append(arg.with_changes(value=cst.Dict(elements=elements)))
            else:
                # Non-literal metadata — wrap in a {**existing, "position": ...} isn't
                # expressible via build_cst_value, so just overwrite with a fresh dict.
                new_args.append(
                    cst.Arg(
                        keyword=cst.Name("metadata"),
                        value=build_cst_value({"position": position}),
                    )
                )
        else:
            new_args.append(arg)

    if not found:
        new_args.append(
            cst.Arg(
                keyword=cst.Name("metadata"),
                value=build_cst_value({"position": position}),
            )
        )

    return new_args


class ConstructorPositionSetter(cst.CSTTransformer):
    """Set position metadata on a top-level Agent (or other Runnable) constructor."""

    def __init__(self, entry_point: str, position: dict):
        self.entry_point = entry_point
        self.position = position

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call):
                    new_args = _merge_position_into_metadata(list(updated_node.value.args), self.position)
                    return updated_node.with_changes(value=updated_node.value.with_changes(args=new_args))
        return updated_node


class StepPositionSetter(cst.CSTTransformer):
    """Set position metadata on a workflow step's constructor variable."""

    def __init__(self, entry_point: str, step_name: str, position: dict, assignments: dict[str, cst.Call]):
        self.entry_point = entry_point
        self.step_name = step_name
        self.position = position
        self.assignments = assignments

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if not isinstance(target.target, cst.Name):
                continue
            target_name = target.target.value
            if target_name == self.entry_point:
                continue
            if isinstance(updated_node.value, cst.Call):
                resolved = resolve_runnable_name(updated_node.value)
                if resolved == self.step_name:
                    new_args = _merge_position_into_metadata(list(updated_node.value.args), self.position)
                    return updated_node.with_changes(
                        value=updated_node.value.with_changes(args=new_args),
                    )
        return updated_node
