"""remove-guardrail: remove a guardrail from the Agent's guardrails list by rail name.

```bash
python -m timbal.codegen remove-guardrail --name pii
python -m timbal.codegen remove-guardrail --name injection --step agent_a
```

Matches list entries by rail *name* (the part before ``:``), so ``--name pii`` removes
``"pii"``, ``"pii:redact"``, or ``"pii:block"``. A ``guardrails="default"`` string is
expanded to its shorthand list first. Removing the last rail drops the kwarg entirely.
Removing an already-absent rail is an idempotent success.
"""

import argparse

import libcst as cst

from ..guardrail_specs import (
    guardrails_kwarg_index,
    literal_shorthands,
    rail_name,
    string_element,
    validate_guardrail_target,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "remove-guardrail",
        help="Remove a guardrail from the agent's guardrails list by rail name.",
    )
    sp.add_argument("--name", required=True, help='The rail name to remove (e.g. "pii", "injection").')
    sp.add_argument(
        "--step",
        default=None,
        help="Target step name within a Workflow. When provided, the guardrail is removed from that step's Agent.",
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    target, _assignments = validate_guardrail_target(
        tree, entry_point, getattr(args, "step", None), "remove-guardrail"
    )
    return GuardrailRemover(target, args.name.strip().lower())


class GuardrailRemover(cst.CSTTransformer):
    # Removing an absent rail is an idempotent success.
    allow_noop = True

    def __init__(self, target: str, name: str) -> None:
        self.target = target
        self.name = name

    def _edit_call(self, call: cst.Call) -> cst.Call:
        idx = guardrails_kwarg_index(call)
        if idx is None:
            return call  # nothing configured — idempotent
        current = literal_shorthands(call.args[idx].value)
        if current is None:
            raise ValueError(
                "The existing guardrails= value is not a literal string/list of shorthands "
                "(it may hold rail instances or a variable). Edit the source directly."
            )
        remaining = [s for s in current if rail_name(s) != self.name]
        if remaining == current:
            return call  # not present — idempotent (still normalizes "default" only on a hit)
        if not remaining:
            return call.with_changes(args=[*call.args[:idx], *call.args[idx + 1 :]])
        new_value = cst.List(elements=[string_element(s) for s in remaining])
        new_arg = cst.Arg(keyword=cst.Name("guardrails"), value=new_value)
        return call.with_changes(args=[*call.args[:idx], new_arg, *call.args[idx + 1 :]])

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:  # noqa: ARG002
        for assign_target in updated_node.targets:
            if (
                isinstance(assign_target.target, cst.Name)
                and assign_target.target.value == self.target
                and isinstance(updated_node.value, cst.Call)
            ):
                return updated_node.with_changes(value=self._edit_call(updated_node.value))
        return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:  # noqa: ARG002
        if (
            isinstance(updated_node.target, cst.Name)
            and updated_node.target.value == self.target
            and isinstance(updated_node.value, cst.Call)
        ):
            return updated_node.with_changes(value=self._edit_call(updated_node.value))
        return updated_node
