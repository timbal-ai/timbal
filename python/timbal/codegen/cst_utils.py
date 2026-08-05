from __future__ import annotations

from typing import TYPE_CHECKING

import libcst as cst

from .tool_discovery import get_framework_tool_names

if TYPE_CHECKING:
    pass

ENTRY_POINT_TYPES = {"Agent", "Workflow"}


def _root_constructor_name(value: cst.BaseExpression) -> str | None:
    """Return the root constructor's class name for an assignment value.

    Walks down chained method calls (e.g. ``Workflow(...).step(...).step(...)``)
    to find the root constructor. Module-qualified constructors
    (e.g. ``timbal.Agent(...)``) resolve to the attribute name.
    """
    call = value
    # Chained method calls have a Call as the attribute base; module access
    # (``timbal.Agent``) has a Name/Attribute base and must not be walked.
    while (
        isinstance(call, cst.Call)
        and isinstance(call.func, cst.Attribute)
        and isinstance(call.func.value, cst.Call)
    ):
        call = call.func.value
    if isinstance(call, cst.Call):
        if isinstance(call.func, cst.Name):
            return call.func.value
        if isinstance(call.func, cst.Attribute):
            return call.func.attr.value
    return None


def _collect_entry_point_aliases(tree: cst.Module) -> dict[str, str]:
    """Map local import aliases to canonical entry point class names.

    Covers ``from timbal import Agent as A`` so aliased constructors are
    recognized like the canonical names.
    """
    aliases: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.ImportFrom) and not isinstance(item.names, cst.ImportStar):
                    for alias in item.names:
                        if (
                            isinstance(alias, cst.ImportAlias)
                            and isinstance(alias.name, cst.Name)
                            and alias.name.value in ENTRY_POINT_TYPES
                            and alias.asname is not None
                            and isinstance(alias.asname.name, cst.Name)
                        ):
                            aliases[alias.asname.name.value] = alias.name.value
    return aliases


def resolve_entry_point_type(tree: cst.Module, entry_point: str) -> str | None:
    """Return the constructor class name ('Agent' or 'Workflow') for the entry point variable.

    Inspects top-level assignments (plain and annotated, e.g.
    ``agent: Agent = Agent(...)``) to find `entry_point = ClassName(...)` and
    returns the class name if it's a known entry point type. Import aliases
    (``Agent as A``) and module-qualified constructors (``timbal.Agent(...)``)
    are canonicalized. Returns None if not found.
    """
    aliases = _collect_entry_point_aliases(tree)

    def _canonical(value: cst.BaseExpression) -> str | None:
        cls_name = _root_constructor_name(value)
        cls_name = aliases.get(cls_name, cls_name)
        return cls_name if cls_name in ENTRY_POINT_TYPES else None

    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign):
                    for target in item.targets:
                        if isinstance(target.target, cst.Name) and target.target.value == entry_point:
                            cls_name = _canonical(item.value)
                            if cls_name is not None:
                                return cls_name
                elif isinstance(item, cst.AnnAssign):
                    if (
                        isinstance(item.target, cst.Name)
                        and item.target.value == entry_point
                        and item.value is not None
                    ):
                        cls_name = _canonical(item.value)
                        if cls_name is not None:
                            return cls_name
    return None


def _get_string_value(node: cst.BaseExpression) -> str | None:
    """Extract the string value from a CST string node."""
    if isinstance(node, cst.SimpleString):
        return node.evaluated_value
    if isinstance(node, cst.ConcatenatedString):
        return node.evaluated_value
    return None


def _get_kwarg(call: cst.Call, name: str) -> cst.BaseExpression | None:
    """Get the value of a keyword argument from a Call node."""
    for arg in call.args:
        if isinstance(arg.keyword, cst.Name) and arg.keyword.value == name:
            return arg.value
    return None


def _name_from_call(call: cst.Call) -> str | None:
    """Extract the runnable name from a Call node (constructor invocation).

    Mirrors the runtime resolution order:
    1. Explicit name= kwarg  →  that string
    2. handler= kwarg that is a Name  →  the function name (= __name__ at runtime)
    3. Callable name itself (e.g. WebSearch() → "WebSearch")
    """
    name_val = _get_kwarg(call, "name")
    if name_val is not None:
        return _get_string_value(name_val)

    handler_val = _get_kwarg(call, "handler")
    if isinstance(handler_val, cst.Name):
        return handler_val.value

    # Fall back to the callable name, mapping to runtime name for framework tools.
    if isinstance(call.func, cst.Name):
        return get_framework_tool_names().get(call.func.value, call.func.value)

    return None


def resolve_runnable_name(
    element: cst.BaseExpression,
    assignments: dict[str, cst.Call] | None = None,
) -> str | None:
    """Resolve the runtime name of a runnable from a CST element.

    Given a CST node from inside a list (e.g. tools=[...], steps=[...]),
    determine what name the runnable will have at runtime.

    Cases:
    - Bare Name (e.g. `my_func`):
        Look up the variable in assignments. If it's assigned to a Call,
        try to extract name from that Call. Otherwise fall back to the
        variable name itself (bare function → __name__ = variable name).
    - Inline Call (e.g. `CalaSearch(name="x")`):
        Extract name from the Call's kwargs.
    """
    if isinstance(element, cst.Name):
        var_name = element.value
        if assignments and var_name in assignments:
            call = assignments[var_name]
            resolved = _name_from_call(call)
            if resolved is not None:
                return resolved
        # Bare function reference — name = variable name.
        return var_name

    if isinstance(element, cst.Call):
        return _name_from_call(element)

    return None


def build_cst_value(value: object) -> cst.BaseExpression:
    """Recursively convert a Python value into a CST expression."""
    if isinstance(value, bool):
        return cst.Name("True" if value else "False")
    if isinstance(value, int):
        if value < 0:
            return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer(str(-value)))
        return cst.Integer(str(value))
    if isinstance(value, float):
        # Ensure the string representation has a decimal point (cst.Float rejects "0").
        s = str(value)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        if value < 0:
            return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Float(s[1:]))
        return cst.Float(s)
    if isinstance(value, str):
        # Use repr() to properly escape special characters (newlines, quotes, etc.)
        # then wrap as a SimpleString CST node.
        return cst.SimpleString(repr(value))
    if value is None:
        return cst.Name("None")
    if isinstance(value, list):
        elements = [cst.Element(value=build_cst_value(v)) for v in value]
        return cst.List(elements=elements)
    if isinstance(value, dict):
        elements = [cst.DictElement(key=build_cst_value(k), value=build_cst_value(v)) for k, v in value.items()]
        return cst.Dict(elements=elements)
    raise TypeError(f"Unsupported type for CST conversion: {type(value)}")


def collect_assignments(tree: cst.Module) -> dict[str, cst.Call]:
    """Build a map of variable_name -> Call node for all top-level assignments.

    Covers plain assignments (``x = Call(...)``) and annotated assignments
    (``x: T = Call(...)``).
    """
    result = {}
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign) and isinstance(item.value, cst.Call):
                    for target in item.targets:
                        if isinstance(target.target, cst.Name):
                            result[target.target.value] = item.value
                elif (
                    isinstance(item, cst.AnnAssign)
                    and isinstance(item.target, cst.Name)
                    and isinstance(item.value, cst.Call)
                ):
                    result[item.target.value] = item.value
    return result


def collect_step_names(
    tree: cst.Module,
    entry_point: str,
    assignments: dict[str, cst.Call],
) -> dict[str, str]:
    """Build a mapping of step variable name → runtime name.

    Scans the tree for ``<entry_point>.step(<var>, ...)`` calls and resolves
    each step's runtime name from its assignment.  Returns a dict like
    ``{"agent_b": "Agent 2", "translator": "translator"}``.
    """
    var_to_name: dict[str, str] = {}
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not isinstance(item, cst.Expr) or not isinstance(item.value, cst.Call):
                continue
            call = item.value
            if not (
                isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name)
                and call.func.value.value == entry_point
                and call.func.attr.value == "step"
                and call.args
            ):
                continue
            first_arg = call.args[0].value
            if isinstance(first_arg, cst.Name):
                var_name = first_arg.value
                resolved = resolve_runnable_name(
                    assignments[var_name],
                ) if var_name in assignments else None
                var_to_name[var_name] = resolved if resolved is not None else var_name
    return var_to_name


def has_step_expr(tree: cst.Module, entry_point: str) -> bool:
    """True when a standalone ``<entry_point>.step(...)`` statement exists.

    Used to validate workflow entry points that ``collect_assignments`` cannot
    see (e.g. an alias like ``workflow = _wf`` whose steps are registered via
    ``workflow.step(...)`` statements).
    """
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if (
                isinstance(item, cst.Expr)
                and isinstance(item.value, cst.Call)
                and is_step_call(item.value, entry_point)
            ):
                return True
    return False


def collect_chained_step_names(
    tree: cst.Module,
    entry_point: str,
    assignments: dict[str, cst.Call],
) -> dict[str, str]:
    """Build a step variable → runtime name mapping for chained ``.step()`` calls.

    Covers steps added in the entry point assignment itself, e.g.
    ``workflow = Workflow(...).step(a).step(b)``. These steps exist in the
    graph but cannot be modified by the standalone-statement transformers
    (set-param, add-edge, remove-edge).
    """
    var_to_name: dict[str, str] = {}
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if isinstance(item, cst.Assign):
                if not any(
                    isinstance(t.target, cst.Name) and t.target.value == entry_point
                    for t in item.targets
                ):
                    continue
                node = item.value
            elif isinstance(item, cst.AnnAssign):
                if not (isinstance(item.target, cst.Name) and item.target.value == entry_point):
                    continue
                node = item.value
            else:
                continue
            while isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute):
                if node.func.attr.value == "step" and node.args:
                    first_arg = node.args[0].value
                    resolved = resolve_runnable_name(first_arg, assignments)
                    if isinstance(first_arg, cst.Name):
                        var_to_name[first_arg.value] = resolved if resolved is not None else first_arg.value
                    elif resolved is not None:
                        var_to_name[resolved] = resolved
                node = node.func.value
    return var_to_name


def require_step(
    ref: str,
    step_names: dict[str, str],
    chained_step_names: dict[str, str] | None = None,
    *,
    kind: str = "Target",
    operation: str,
) -> str:
    """Resolve *ref* (a step variable name or runtime name) to the runtime name.

    ``step_names`` are the steps the operation can address. When *ref* only
    matches ``chained_step_names`` (steps the operation cannot modify), or
    matches nothing, a ``ValueError`` with an actionable message is raised.
    """
    if ref in step_names.values():
        return ref
    if ref in step_names:
        return step_names[ref]

    chained = chained_step_names or {}
    if ref in chained or ref in chained.values():
        raise ValueError(
            f"{kind} step '{ref}' is added via a chained .step() call, which {operation} "
            f"cannot modify. Rewrite it as a standalone '<workflow>.step(...)' statement first."
        )

    available = sorted(set(step_names.values()) | set(chained.values()))
    if available:
        raise ValueError(
            f"{kind} step '{ref}' not found in workflow. Available steps: {', '.join(available)}."
        )
    raise ValueError(f"{kind} step '{ref}' not found in workflow. The workflow has no steps.")


def is_bare_function_step(
    tree: cst.Module,
    entry_point: str,
    step_name: str,
    assignments: dict[str, cst.Call],
) -> bool:
    """Check if *step_name* is a bare ``def`` used directly in a ``.step()`` call.

    Returns ``True`` when:
    1. No top-level assignment resolves to *step_name* (i.e. it's not already
       wrapped in ``Tool`` / ``Agent`` / etc.).
    2. A ``FunctionDef`` with that name exists at the module level.
    3. That name appears as the first positional arg in an
       ``<entry_point>.step(step_name, ...)`` call.
    """
    # Already wrapped?
    for _var, call in assignments.items():
        if resolve_runnable_name(call) == step_name:
            return False

    # Has a matching FunctionDef?
    has_func_def = any(
        isinstance(stmt, cst.FunctionDef) and stmt.name.value == step_name
        for stmt in tree.body
    )
    if not has_func_def:
        return False

    # Used as a workflow step?
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not (isinstance(item, cst.Expr) and isinstance(item.value, cst.Call)):
                continue
            call = item.value
            if (
                isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name)
                and call.func.value.value == entry_point
                and call.func.attr.value == "step"
                and call.args
                and isinstance(call.args[0].value, cst.Name)
                and call.args[0].value.value == step_name
            ):
                return True

    return False


class _BareFunctionWrapper(cst.CSTTransformer):
    """Rename ``def step_name(...)`` → ``def step_name_fn(...)`` and insert a
    ``Tool(name="step_name", handler=step_name_fn)`` assignment."""

    def __init__(self, entry_point: str, step_name: str) -> None:
        self.entry_point = entry_point
        self.step_name = step_name
        self._func_renamed = False

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef  # noqa: ARG002
    ) -> cst.FunctionDef:
        if updated_node.name.value == self.step_name:
            self._func_renamed = True
            return updated_node.with_changes(name=cst.Name(f"{self.step_name}_fn"))
        return updated_node

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if not self._func_renamed:
            return updated_node

        body = list(updated_node.body)

        # Add ``from timbal.core import Tool`` if missing.
        if not has_import(original_node, "timbal.core", "Tool"):
            insert_imports(body, [cst.parse_statement("from timbal.core import Tool\n")])

        # Build ``step_name = Tool(name="step_name", handler=step_name_fn)``
        # and insert it before the entry-point assignment (plain or annotated).
        # Also anchor at the first standalone ``.step()`` statement so aliased
        # entry points (not visible as assignments) still get the Tool defined
        # before its first reference.
        assignment_code = (
            f'{self.step_name} = Tool(name="{self.step_name}", handler={self.step_name}_fn)\n'
        )
        insert_before_assignments(
            body, [cst.parse_statement(assignment_code)],
            target_names={self.entry_point}, step_calls_of=self.entry_point,
        )

        return updated_node.with_changes(body=body)


def wrap_bare_function_step(tree: cst.Module, entry_point: str, step_name: str) -> cst.Module:
    """Wrap a bare function step in a ``Tool`` and return the modified tree."""
    return tree.visit(_BareFunctionWrapper(entry_point, step_name))


def is_step_call(call: cst.Call, entry_point: str) -> bool:
    """Check if a Call node is ``<entry_point>.step(...)``."""
    return (
        isinstance(call.func, cst.Attribute)
        and isinstance(call.func.value, cst.Name)
        and call.func.value.value == entry_point
        and call.func.attr.value == "step"
    )


def step_matches_target(
    call: cst.Call,
    target: str,
    step_names: dict[str, str],
    assignments: dict[str, cst.Call] | None = None,
) -> bool:
    """Check if a ``.step()`` call's first argument refers to *target*.

    Matches when *target* equals either the step variable name or its runtime
    name (from ``step_names``, or resolved from *assignments* when provided).
    """
    if not call.args:
        return False
    first_arg = call.args[0].value
    if isinstance(first_arg, cst.Name):
        var_name = first_arg.value
        if var_name == target:
            return True
        runtime_name = step_names.get(var_name)
        if runtime_name is None and assignments and var_name in assignments:
            runtime_name = resolve_runnable_name(assignments[var_name])
        if runtime_name is not None and runtime_name == target:
            return True
    return False


class StepCallRewriter(cst.CSTTransformer):
    """Base for transformers that rewrite the target step's ``.step()`` call.

    Subclasses set ``entry_point``, ``target``, ``step_names`` and
    ``assignments`` attributes and implement ``_build_step_call_code(call)``.
    Sets ``matched = True`` when the target ``.step()`` call was found, so an
    unchanged file counts as an idempotent save (see ``apply_operation``).
    """

    def leave_Expr(self, original_node: cst.Expr, updated_node: cst.Expr) -> cst.Expr:  # noqa: ARG002
        call = updated_node.value
        if not isinstance(call, cst.Call):
            return updated_node
        if not is_step_call(call, self.entry_point) or not step_matches_target(
            call, self.target, self.step_names, self.assignments,
        ):
            return updated_node

        self.matched = True
        new_call = parse_call_statement(self._build_step_call_code(call))
        if new_call is not None:
            return updated_node.with_changes(value=new_call)
        return updated_node


def assignment_resolves_to(assign: cst.Assign | cst.AnnAssign, entry_point: str, name: str) -> bool:
    """True when a (non-entry-point) assignment's Call value resolves to *name*.

    Covers plain assignments and annotated assignments (``x: T = Call(...)``).
    """
    if not isinstance(assign.value, cst.Call):
        return False
    if isinstance(assign, cst.AnnAssign):
        targets = [assign.target]
    else:
        targets = [t.target for t in assign.targets]
    for target in targets:
        if not isinstance(target, cst.Name) or target.value == entry_point:
            continue
        if resolve_runnable_name(assign.value) == name:
            return True
    return False


def parse_call_statement(code: str) -> cst.Call | None:
    """Parse *code* (a single expression statement) and return its Call node."""
    parsed = cst.parse_module(code + "\n")
    for stmt in parsed.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Expr) and isinstance(item.value, cst.Call):
                    return item.value
    return None


def parse_function_def(definition: str) -> cst.FunctionDef:
    """Parse a ``--definition`` source string and return its FunctionDef.

    Raises ``ValueError`` when the source contains no function definition.
    """
    func_tree = cst.parse_module(definition)
    for stmt in func_tree.body:
        if isinstance(stmt, cst.FunctionDef):
            return stmt
    raise ValueError("--definition must contain a function definition.")


def insert_imports(body: list[cst.BaseStatement], imports: list[cst.BaseStatement]) -> None:
    """Insert import statements after the last existing import (in place)."""
    import_insert_idx = 0
    for i, stmt in enumerate(body):
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, (cst.Import, cst.ImportFrom)):
                    import_insert_idx = i + 1
    for stmt in reversed(imports):
        body.insert(import_insert_idx, stmt)


def insert_before_assignments(
    body: list[cst.BaseStatement],
    stmts: list[cst.BaseStatement],
    *,
    target_names: set[str],
    runnable_names: set[str] = frozenset(),
    step_calls_of: str | None = None,
) -> None:
    """Insert statements before the earliest anchoring statement (in place).

    Anchors are top-level assignments (plain or annotated) whose target is in
    *target_names*, or whose Call value resolves (via ``resolve_runnable_name``)
    to a name in *runnable_names*. When *step_calls_of* is set, standalone
    ``<step_calls_of>.step(...)`` statements also anchor — this covers aliased
    entry points that never appear as assignments. Appends at the end when no
    anchor is found.
    """
    insert_idx = len(body)
    for i, stmt in enumerate(body):
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign):
                    if runnable_names and isinstance(item.value, cst.Call):
                        if resolve_runnable_name(item.value) in runnable_names:
                            insert_idx = min(insert_idx, i)
                    for t in item.targets:
                        if isinstance(t.target, cst.Name) and t.target.value in target_names:
                            insert_idx = min(insert_idx, i)
                elif isinstance(item, cst.AnnAssign):
                    if isinstance(item.target, cst.Name):
                        if item.target.value in target_names:
                            insert_idx = min(insert_idx, i)
                        elif (
                            runnable_names
                            and isinstance(item.value, cst.Call)
                            and resolve_runnable_name(item.value) in runnable_names
                        ):
                            insert_idx = min(insert_idx, i)
                elif (
                    step_calls_of is not None
                    and isinstance(item, cst.Expr)
                    and isinstance(item.value, cst.Call)
                    and is_step_call(item.value, step_calls_of)
                ):
                    insert_idx = min(insert_idx, i)
    for stmt in reversed(stmts):
        body.insert(insert_idx, stmt)


def validate_tools_target(
    tree: cst.Module | None,
    entry_point: str,
    step: str | None,
    *,
    operation: str,
) -> tuple[str, dict[str, cst.Call]]:
    """Validate the add-tool/add-mcp target and return (target, assignments).

    With ``--step`` the entry point must be a Workflow and the step must exist;
    without it the entry point must be an Agent assigned in the source.
    """
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


def merge_config_kwargs(call: cst.Call, config: dict) -> cst.Call:
    """Merge *config* kwargs into an existing Call, preserving all other args.

    Existing kwargs named in *config* are dropped (overridden or removed);
    new kwargs are appended, skipping ``None`` values (None = remove).
    """
    args = [
        a for a in call.args
        if not (isinstance(a.keyword, cst.Name) and a.keyword.value in config)
    ]
    for key, value in config.items():
        if value is not None:
            args.append(cst.Arg(keyword=cst.Name(key), value=build_cst_value(value)))
    return call.with_changes(args=args)


def has_import(tree: cst.Module, module: str, name: str) -> bool:
    """Check if `from <module> import <name>` already exists."""
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.ImportFrom) and not isinstance(item.names, cst.ImportStar):
                    parts = []
                    node = item.module
                    while isinstance(node, cst.Attribute):
                        parts.append(node.attr.value)
                        node = node.value
                    if isinstance(node, cst.Name):
                        parts.append(node.value)
                    mod = ".".join(reversed(parts))

                    if mod == module:
                        for alias in item.names:
                            if isinstance(alias, cst.ImportAlias):
                                imported = alias.name.value if isinstance(alias.name, cst.Name) else ""
                                if imported == name:
                                    return True
    return False
