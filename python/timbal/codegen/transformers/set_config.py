import argparse

import libcst as cst

from ..cli_utils import arg_input, parse_json_arg
from ..cst_utils import (
    assignment_resolves_to,
    build_cst_value,
    collect_assignments,
    insert_before_assignments,
    is_bare_function_step,
    merge_config_kwargs,
    resolve_entry_point_type,
    resolve_runnable_name,
    wrap_bare_function_step,
)
from ..tool_discovery import get_framework_tool_names, validate_tool_config

AGENT_FIELDS = {
    "name",
    "description",
    "model",
    "system_prompt",
    "max_iter",
    "max_tokens",
    "temperature",
    "base_url",
    "api_key",
    "model_params",  # Deprecated, kept for backward compatibility
    "skills_path",
    "voice_config",  # dict consumed by the voice server (merge_voice_config)
    "guardrails",  # literal shorthand list; incremental edits via add-/remove-guardrail
    "guardrail_mode",
    "max_guardrail_retries",
}


def _validate_agent_config(config: dict) -> None:
    """Field-specific validation beyond the AGENT_FIELDS allowlist — catches typos at
    the CLI instead of at agent construction."""
    unknown = set(config.keys()) - AGENT_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown agent config field(s): {', '.join(sorted(unknown))}. "
            f"Valid fields: {', '.join(sorted(AGENT_FIELDS))}."
        )
    mode = config.get("guardrail_mode")
    if mode is not None and mode not in ("enforce", "shadow"):
        raise ValueError(f"Invalid guardrail_mode {mode!r}. Must be 'enforce' or 'shadow'.")
    rails = config.get("guardrails")
    if rails is not None:
        from timbal.guardrails.presets import coerce_rail

        if isinstance(rails, str):
            rails = [rails]
        if not isinstance(rails, list) or not all(isinstance(s, str) for s in rails):
            raise ValueError(
                'guardrails must be a JSON list of shorthand strings (e.g. ["pii:redact", "injection:block"]) '
                'or the string "default".'
            )
        for spec in rails:
            if spec.strip().lower() != "default":
                coerce_rail(spec)  # raises with the valid names/actions


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "set-config",
        help="Set configuration on the agent/step or on a specific tool.",
    )
    sp.add_argument(
        "--name",
        default=None,
        help="Tool or step name to configure. Omit to configure the entry point itself.",
    )
    sp.add_argument(
        "--config",
        default=None,
        type=arg_input,
        help=(
            'Configuration as a JSON object. E.g. \'{"model": "openai/gpt-4o-mini"}\'. '
            "Use '@path' to read from file or '-' to read from stdin."
        ),
    )


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    ep_type = resolve_entry_point_type(tree, entry_point) if tree else None

    config = parse_json_arg(args.config, "--config") if args.config else {}

    # --- Workflow entry point ---
    if ep_type == "Workflow":
        if not args.name:
            raise ValueError("set-config on a Workflow entry point requires a step name.")
        if not config:
            raise ValueError("--config is required for set-config.")

        assignments = collect_assignments(tree) if tree else {}

        # Bare function used directly as a step → wrap in Tool first.
        wrapped_tree = None
        if tree and is_bare_function_step(tree, entry_point, args.name, assignments):
            tree = wrap_bare_function_step(tree, entry_point, args.name)
            wrapped_tree = tree
            assignments = collect_assignments(tree)

        step_class = _resolve_step_class(args.name, assignments)
        if step_class == "Agent":
            _validate_agent_config(config)
        transformer = StepConstructorConfigSetter(entry_point, args.name, config, assignments)
        if wrapped_tree is not None:
            return transformer, wrapped_tree
        return transformer

    # --- Agent entry point ---
    if ep_type is not None and ep_type != "Agent":
        raise ValueError(f"set-config requires an Agent or Workflow entry point, but '{entry_point}' is a {ep_type}.")

    if args.name and args.name != entry_point:
        assignments = collect_assignments(tree) if tree else {}
        tool_class = _resolve_tool_class(entry_point, args.name, assignments)
        if tool_class is None:
            raise ValueError(f"Tool '{args.name}' not found in agent tools list.")
        validate_tool_config(tool_class, config)
        var_name = get_framework_tool_names().get(tool_class, args.name)
        return ToolConfigSetter(entry_point, args.name, config, assignments, tool_class, var_name)

    if not config:
        raise ValueError("--config is required for set-config.")

    _validate_agent_config(config)

    # ``ep_type is None`` with a call on the RHS: the entry point may be a
    # factory call like ``agent = build()`` (local or imported) — appending
    # config kwargs to that call would corrupt the module, so fail loudly.
    # Constructors can't be told apart from factories statically for imports,
    # so use PEP 8 naming: CapWords callees (``Tool``, ``WebSearch``, custom
    # runnables) are treated as constructors, everything else as a factory.
    # (Aliased imports like ``Agent as A`` and module-qualified ``timbal.Agent``
    # are canonicalized by resolve_entry_point_type and never land here.)
    if tree is not None and ep_type is None:
        ep_call = collect_assignments(tree).get(entry_point)
        if ep_call is not None:
            if isinstance(ep_call.func, cst.Name):
                func_name = ep_call.func.value
            elif isinstance(ep_call.func, cst.Attribute):
                func_name = ep_call.func.attr.value
            else:
                func_name = None
            is_local_func = func_name is not None and any(
                isinstance(stmt, cst.FunctionDef) and stmt.name.value == func_name for stmt in tree.body
            )
            looks_like_function = func_name is None or not func_name[:1].isupper()
            if is_local_func or looks_like_function:
                raise ValueError(
                    f"Entry point '{entry_point}' is built by calling '{func_name or '<expression>'}()', "
                    f"which looks like a factory function, not an Agent(...) constructor. set-config "
                    f"cannot modify a factory-built agent; construct the Agent directly in the "
                    f"'{entry_point}' assignment."
                )

    return AgentConfigSetter(entry_point, config)


def _resolve_tool_class(
    entry_point: str, tool_name: str, assignments: dict[str, cst.Call]
) -> str | None:
    """Find the framework class name for a tool in the agent's tools list.

    ``assignments`` covers both plain and annotated entry point assignments
    (see ``collect_assignments``).
    """
    call = assignments.get(entry_point)
    if call is not None:
        return _find_tool_class(call, tool_name, assignments)
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


def _resolve_step_class(step_name: str, assignments: dict[str, cst.Call]) -> str | None:
    """Find the class name for a step variable (e.g. 'Agent', 'WebSearch')."""
    # Look up in assignments by step_name directly.
    if step_name in assignments:
        call = assignments[step_name]
        if isinstance(call.func, cst.Name):
            return call.func.value
    # Also check by resolved name.
    for _var_name, call in assignments.items():
        resolved = resolve_runnable_name(call)
        if resolved == step_name and isinstance(call.func, cst.Name):
            return call.func.value
    return None


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


class AgentConfigSetter(cst.CSTTransformer):
    def __init__(self, entry_point: str, config: dict):
        self.entry_point = entry_point
        self.config = config
        # True once the entry point constructor was found and updated. An
        # unchanged file with matched=True is an idempotent save (success);
        # matched=False means the source shape was never recognized (failure).
        self.matched = False

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call):
                    self.matched = True
                    new_call = self._update_agent_kwargs(updated_node.value)
                    return updated_node.with_changes(value=new_call)
        return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:  # noqa: ARG002
        """Handle annotated assignments, e.g. ``agent: Agent = Agent(...)``."""
        if (
            isinstance(updated_node.target, cst.Name)
            and updated_node.target.value == self.entry_point
            and isinstance(updated_node.value, cst.Call)
        ):
            self.matched = True
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
        self.matched = False  # target tool found — unchanged file means idempotent save
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
                    self.matched = True
                    return updated_node.with_changes(value=new_call)

            # Variable assignment — update config kwargs.
            if target_name != self.entry_point and isinstance(updated_node.value, cst.Call):
                resolved = resolve_runnable_name(updated_node.value)
                if resolved == self.tool_name:
                    self.matched = True
                    self._var_updated = True
                    return updated_node.with_changes(
                        value=self._build_configured_call(updated_node.value),
                    )
        return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:  # noqa: ARG002
        """Handle annotated assignments, e.g. ``agent: Agent = Agent(..., tools=[...])``."""
        if not isinstance(updated_node.target, cst.Name) or not isinstance(updated_node.value, cst.Call):
            return updated_node
        target_name = updated_node.target.value

        # Entry point — check for inline tools to migrate.
        if target_name == self.entry_point:
            new_call = self._migrate_inline(updated_node.value)
            if new_call is not updated_node.value:
                self.matched = True
                return updated_node.with_changes(value=new_call)
            return updated_node

        # Variable assignment — update config kwargs.
        resolved = resolve_runnable_name(updated_node.value)
        if resolved == self.tool_name:
            self.matched = True
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
        # Insert before the entry point assignment (plain or annotated).
        insert_before_assignments(body, stmts_to_add, target_names={self.entry_point})
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
        return merge_config_kwargs(existing_call, self.config)


class StepConstructorConfigSetter(cst.CSTTransformer):
    """Update a step's constructor kwargs (e.g. model, system_prompt) without touching .step() call.

    When the config includes a ``name`` change, references in ``depends_on``
    lists and ``step_span()`` calls are updated to match the new runtime name.
    """

    def __init__(
        self,
        entry_point: str,
        step_name: str,
        config: dict,
        assignments: dict[str, cst.Call],
    ):
        self.entry_point = entry_point
        self.step_name = step_name
        self.config = config
        self.assignments = assignments
        self.matched = False  # target step found — unchanged file means idempotent save
        # When renaming, track the old→new runtime name for reference updates.
        self._new_runtime_name: str | None = config.get("name")
        # Context tracking for string replacement (mirrors Renamer).
        self._in_depends_on = False
        self._in_step_span = False

    # -- Constructor kwargs update ---------------------------------------------

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        if assignment_resolves_to(updated_node, self.entry_point, self.step_name):
            self.matched = True
            return updated_node.with_changes(
                value=merge_config_kwargs(updated_node.value, self.config),
            )
        return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:  # noqa: ARG002
        """Handle annotated step variables, e.g. ``agent_a: Agent = Agent(...)``."""
        if assignment_resolves_to(updated_node, self.entry_point, self.step_name):
            self.matched = True
            return updated_node.with_changes(
                value=merge_config_kwargs(updated_node.value, self.config),
            )
        return updated_node

    # -- Context tracking for depends_on / step_span string replacement --------

    def visit_Arg(self, node: cst.Arg) -> bool:
        if isinstance(node.keyword, cst.Name) and node.keyword.value == "depends_on":
            self._in_depends_on = True
        return True

    def leave_Arg(self, original_node: cst.Arg, updated_node: cst.Arg) -> cst.Arg:
        self._in_depends_on = False
        return updated_node

    def visit_Call(self, node: cst.Call) -> bool:
        if isinstance(node.func, cst.Attribute) and node.func.attr.value == "step_span":
            self._in_step_span = True
        return True

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if isinstance(original_node.func, cst.Attribute) and original_node.func.attr.value == "step_span":
            self._in_step_span = False
        return updated_node

    def leave_SimpleString(
        self, original_node: cst.SimpleString, updated_node: cst.SimpleString
    ) -> cst.SimpleString:
        if self._new_runtime_name is None:
            return updated_node
        if not (self._in_depends_on or self._in_step_span):
            return updated_node
        evaluated = updated_node.evaluated_value
        if evaluated == self.step_name:
            return updated_node.with_changes(value=f'"{self._new_runtime_name}"')
        return updated_node
