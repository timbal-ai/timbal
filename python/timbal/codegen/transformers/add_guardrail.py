"""add-guardrail: wire a guardrail shorthand into the Agent's guardrails list.

```bash
python -m timbal.codegen add-guardrail --spec "pii:redact"
python -m timbal.codegen add-guardrail --spec default          # set guardrails="default"
python -m timbal.codegen add-guardrail --spec "moderation:warn" --step agent_a
```

Semantics on the existing ``guardrails=`` kwarg:

- absent → ``guardrails=["<spec>"]`` (or ``guardrails="default"`` for the default preset)
- string ``"default"`` → expanded to its shorthand list, then the spec is merged in
- string shorthand → converted to a two-element list
- list of string literals → the spec is appended; an entry with the same rail *name*
  (the part before ``:``) is replaced, since duplicate rail names are invalid at runtime
- anything non-literal (a variable, rail instances) → loud error; edit the code directly
"""

import argparse

import libcst as cst

from ..guardrail_specs import (
    guardrails_kwarg_index,
    literal_shorthands,
    rail_name,
    string_element,
    validate_guardrail_target,
    validate_shorthand,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "add-guardrail",
        help="Add a guardrail shorthand to the agent's guardrails list.",
    )
    sp.add_argument(
        "--spec",
        required=True,
        help='Guardrail shorthand: "default", or "<name>[:action]" (e.g. "pii:redact", "injection:block").',
    )
    sp.add_argument(
        "--step",
        default=None,
        help="Target step name within a Workflow. When provided, the guardrail is added to that step's Agent.",
    )


def _validate_spec(spec: str) -> None:
    """Reject unknown shorthands loudly at the CLI boundary."""
    if spec.strip().lower() == "default":
        return
    validate_shorthand(spec)


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    _validate_spec(args.spec)
    target, _assignments = validate_guardrail_target(
        tree, entry_point, getattr(args, "step", None), "add-guardrail"
    )
    return GuardrailAdder(target, args.spec.strip())


class GuardrailAdder(cst.CSTTransformer):
    def __init__(self, target: str, spec: str) -> None:
        self.target = target
        self.spec = spec
        self.matched = False

    def _edit_call(self, call: cst.Call) -> cst.Call:
        self.matched = True
        idx = guardrails_kwarg_index(call)

        if self.spec.lower() == "default":
            new_value: cst.BaseExpression = cst.SimpleString('"default"')
        else:
            current = [] if idx is None else literal_shorthands(call.args[idx].value)
            if current is None:
                raise ValueError(
                    "The existing guardrails= value is not a literal string/list of shorthands "
                    "(it may hold rail instances or a variable). Edit the source directly."
                )
            name = rail_name(self.spec)
            merged = [s for s in current if rail_name(s) != name]
            merged.append(self.spec)
            if merged == current:
                return call  # already present — idempotent
            new_value = cst.List(elements=[string_element(s) for s in merged])

        new_arg = cst.Arg(keyword=cst.Name("guardrails"), value=new_value)
        if idx is None:
            return call.with_changes(args=[*call.args, new_arg])
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
