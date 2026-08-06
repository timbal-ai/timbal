"""Org tool library operations.

Two operations live here (dispatched from ``__main__`` like the other
read-only/special ops, not via the transformer table):

- ``extract-tool``: pull a custom tool out of a workspace member's source as a
  self-contained, portable module + manifest (JSON to stdout). The dependency
  closure is computed with libcst scope analysis: the handler function, every
  top-level helper/constant/class it transitively references, and the imports
  they use.
- ``add-library-tool``: vendor a library tool module into a workspace member
  (``tools/<module>.py`` with a provenance header) and wire it into the
  entry point's ``tools=[...]`` list.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import libcst as cst
from libcst.metadata import Assignment, GlobalScope, MetadataWrapper, ScopeProvider

from timbal.codegen import parse_fqn

from .cst_utils import (
    insert_imports,
    resolve_runnable_name,
    validate_tools_target,
)
from .format import format_code

PROVENANCE_PREFIX = "# timbal-tool:"
PROVENANCE_RE = re.compile(r"^# timbal-tool: (?P<name>\S+)@(?P<rev>\S+)\s*$")

VENDOR_DIR = "tools"

# Import name -> PyPI distribution name, for the common mismatches.
_IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
}


# ---------------------------------------------------------------------------
# Small CST helpers
# ---------------------------------------------------------------------------


def _walk(node: cst.CSTNode):
    """Yield every descendant of *node* (including itself)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.children)


def _get_kwarg(call: cst.Call, name: str) -> cst.BaseExpression | None:
    for arg in call.args:
        if isinstance(arg.keyword, cst.Name) and arg.keyword.value == name:
            return arg.value
    return None


def _string_value(node: cst.BaseExpression | None) -> str | None:
    if isinstance(node, (cst.SimpleString, cst.ConcatenatedString)):
        return node.evaluated_value
    return None


def _constructor_name(call: cst.Call) -> str | None:
    if isinstance(call.func, cst.Name):
        return call.func.value
    if isinstance(call.func, cst.Attribute):
        return call.func.attr.value
    return None


def _bound_names(stmt: cst.BaseStatement) -> set[str]:
    """Top-level names bound by a module-level statement."""
    names: set[str] = set()
    if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
        names.add(stmt.name.value)
        return names
    if not isinstance(stmt, cst.SimpleStatementLine):
        return names
    for item in stmt.body:
        if isinstance(item, cst.Assign):
            for target in item.targets:
                if isinstance(target.target, cst.Name):
                    names.add(target.target.value)
                elif isinstance(target.target, (cst.Tuple, cst.List)):
                    for el in target.target.elements:
                        if isinstance(el.value, cst.Name):
                            names.add(el.value.value)
        elif isinstance(item, cst.AnnAssign):
            if isinstance(item.target, cst.Name):
                names.add(item.target.value)
        elif isinstance(item, cst.Import):
            for alias in item.names:
                if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
                    names.add(alias.asname.name.value)
                else:
                    # ``import a.b`` binds ``a``.
                    node = alias.name
                    while isinstance(node, cst.Attribute):
                        node = node.value
                    if isinstance(node, cst.Name):
                        names.add(node.value)
        elif isinstance(item, cst.ImportFrom) and not isinstance(item.names, cst.ImportStar):
            for alias in item.names:
                if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
                    names.add(alias.asname.name.value)
                elif isinstance(alias.name, cst.Name):
                    names.add(alias.name.value)
    return names


def _is_import_stmt(stmt: cst.BaseStatement) -> bool:
    return isinstance(stmt, cst.SimpleStatementLine) and all(
        isinstance(item, (cst.Import, cst.ImportFrom)) for item in stmt.body
    )


def _import_top_level_modules(stmt: cst.BaseStatement) -> list[tuple[str, bool]]:
    """Return ``(top_level_module, is_relative)`` pairs for an import statement."""
    result: list[tuple[str, bool]] = []
    if not isinstance(stmt, cst.SimpleStatementLine):
        return result
    for item in stmt.body:
        if isinstance(item, cst.Import):
            for alias in item.names:
                node = alias.name
                while isinstance(node, cst.Attribute):
                    node = node.value
                if isinstance(node, cst.Name):
                    result.append((node.value, False))
        elif isinstance(item, cst.ImportFrom):
            if item.relative:
                result.append(("", True))
                continue
            node = item.module
            while isinstance(node, cst.Attribute):
                node = node.value
            if isinstance(node, cst.Name):
                result.append((node.value, False))
    return result


# ---------------------------------------------------------------------------
# extract-tool
# ---------------------------------------------------------------------------


def _stdlib_modules() -> frozenset[str]:
    return frozenset(sys.stdlib_module_names)


def _requirements_for_imports(workspace_path: Path, modules: set[str]) -> list[str]:
    """Best-effort mapping of imported top-level modules to dependency specs.

    Looks each module up in the member's pyproject dependencies; falls back to
    the (dist-mapped) import name when no spec matches.
    """
    import tomllib

    deps: list[str] = []
    pyproject = workspace_path / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text())
            deps = data.get("project", {}).get("dependencies", []) or []
        except tomllib.TOMLDecodeError:
            deps = []

    def _norm(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    spec_by_name: dict[str, str] = {}
    for spec in deps:
        m = re.match(r"\s*([A-Za-z0-9._-]+)", spec)
        if m:
            spec_by_name[_norm(m.group(1))] = spec.strip()

    requirements: list[str] = []
    for module in sorted(modules):
        dist = _IMPORT_TO_DIST.get(module, module)
        spec = spec_by_name.get(_norm(dist))
        requirements.append(spec if spec is not None else dist)
    return requirements


def _scan_integrations(stmts: list[cst.BaseStatement]) -> list[str]:
    found: set[str] = set()
    for stmt in stmts:
        for node in _walk(stmt):
            if (
                isinstance(node, cst.Call)
                and isinstance(node.func, cst.Name)
                and node.func.value == "Integration"
                and node.args
            ):
                value = _string_value(node.args[0].value)
                if value is not None:
                    found.add(value)
    return sorted(found)


def _scan_env_vars(stmts: list[cst.BaseStatement]) -> list[str]:
    found: set[str] = set()

    def _is_os_environ(node: cst.BaseExpression) -> bool:
        return (
            isinstance(node, cst.Attribute)
            and isinstance(node.value, cst.Name)
            and node.value.value == "os"
            and node.attr.value == "environ"
        )

    for stmt in stmts:
        for node in _walk(stmt):
            if isinstance(node, cst.Call):
                func = node.func
                # os.getenv("X") / os.environ.get("X")
                if isinstance(func, cst.Attribute) and node.args:
                    if (
                        isinstance(func.value, cst.Name)
                        and func.value.value == "os"
                        and func.attr.value == "getenv"
                    ) or (_is_os_environ(func.value) and func.attr.value == "get"):
                        value = _string_value(node.args[0].value)
                        if value is not None:
                            found.add(value)
            elif isinstance(node, cst.Subscript) and _is_os_environ(node.value):
                # os.environ["X"]
                for el in node.slice:
                    if isinstance(el.slice, cst.Index):
                        value = _string_value(el.slice.value)
                        if value is not None:
                            found.add(value)
    return sorted(found)


def _extract_params(func: cst.FunctionDef, module: cst.Module) -> list[dict]:
    params: list[dict] = []
    all_params = [
        *func.params.posonly_params,
        *func.params.params,
        *func.params.kwonly_params,
    ]
    for p in all_params:
        entry: dict = {"name": p.name.value}
        entry["annotation"] = (
            module.code_for_node(p.annotation.annotation).strip() if p.annotation is not None else None
        )
        entry["default"] = module.code_for_node(p.default).strip() if p.default is not None else None
        params.append(entry)
    return params


def _docstring_summary(func: cst.FunctionDef) -> str | None:
    doc = func.get_docstring()
    if not doc:
        return None
    # First paragraph only.
    return doc.strip().split("\n\n")[0].strip()


def extract_tool(workspace_path: str | Path, tool_ref: str, step: str | None = None) -> dict:
    """Extract *tool_ref* from the workspace entry point as a portable module.

    Returns a manifest dict with the self-contained ``source`` plus inferred
    metadata (params, requirements, integrations, env vars).

    Raises ``ValueError`` for framework tools and non-portable closures
    (references to the entry point, relative/local-module imports, or names
    bound by unsupported statement shapes).
    """
    workspace_path = Path(workspace_path)
    spec = parse_fqn(workspace_path)
    if not spec.path.exists():
        raise FileNotFoundError(f"source file not found: {spec.path}")

    source = spec.path.read_text()
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError as e:
        raise ValueError(f"Cannot parse {spec.path}: {e}") from e

    entry_point = spec.target
    target, assignments = validate_tools_target(tree, entry_point, step, operation="extract-tool")

    target_call = assignments.get(target)
    if target_call is None:
        raise ValueError(f"Could not find the constructor call for '{target}'.")
    tools_val = _get_kwarg(target_call, "tools")
    if not isinstance(tools_val, cst.List) or not tools_val.elements:
        raise ValueError(f"'{target}' has no tools list to extract from.")

    # -- Locate the tool element ------------------------------------------
    element: cst.BaseExpression | None = None
    runtime_name: str | None = None
    for el in tools_val.elements:
        resolved = resolve_runnable_name(el.value, assignments)
        if resolved == tool_ref or (isinstance(el.value, cst.Name) and el.value.value == tool_ref):
            element = el.value
            runtime_name = resolved if resolved is not None else tool_ref
            break
    if element is None:
        available = sorted(
            {n for el in tools_val.elements if (n := resolve_runnable_name(el.value, assignments)) is not None}
        )
        raise ValueError(
            f"Tool '{tool_ref}' not found in '{target}' tools list. "
            f"Available tools: {', '.join(available) if available else '(none)'}."
        )

    # -- Classify the element and pick the binding + seed names ------------
    func_defs = {stmt.name.value: stmt for stmt in tree.body if isinstance(stmt, cst.FunctionDef)}

    binding: str
    tool_call: cst.Call | None = None  # the Tool(...) wrapper call, when present
    synthesized_binding_code: str | None = None
    seed_names: set[str] = set()
    inline_element: cst.Call | None = None

    if isinstance(element, cst.Name):
        var_name = element.value
        if var_name in assignments:
            ctor_call = assignments[var_name]
            ctor = _constructor_name(ctor_call)
            if ctor != "Tool":
                raise ValueError(
                    f"Tool '{tool_ref}' is a {ctor or 'non-Tool'} instance. "
                    "Only custom tools (Tool(...) wrappers or bare functions) can be extracted."
                )
            binding = var_name
            tool_call = ctor_call
            seed_names = {var_name}
        elif var_name in func_defs:
            binding = var_name
            seed_names = {var_name}
        else:
            raise ValueError(
                f"Tool '{tool_ref}' resolves to '{var_name}', which is neither a top-level "
                "assignment nor a top-level function definition."
            )
    elif isinstance(element, cst.Call):
        ctor = _constructor_name(element)
        if ctor != "Tool":
            raise ValueError(
                f"Tool '{tool_ref}' is an inline {ctor or 'non-Tool'} call. "
                "Only custom tools (Tool(...) wrappers or bare functions) can be extracted."
            )
        inline_element = element
        safe = re.sub(r"\W+", "_", runtime_name or tool_ref).strip("_").lower() or "tool"
        binding = f"{safe}_tool"
        tool_call = element
        synthesized_binding_code = f"{binding} = {tree.code_for_node(element)}\n"
    else:
        raise ValueError(f"Unsupported tools list element for '{tool_ref}'.")

    handler_name: str | None = None
    if tool_call is not None:
        handler_val = _get_kwarg(tool_call, "handler")
        if not isinstance(handler_val, cst.Name):
            raise ValueError(
                f"Tool '{tool_ref}' has a non-portable handler (expected a plain function "
                "reference, e.g. handler=my_func)."
            )
        handler_name = handler_val.value
    else:
        handler_name = binding

    # -- Scope analysis: statement-level dependency graph ------------------
    wrapper = MetadataWrapper(tree, unsafe_skip_copy=True)
    scopes = wrapper.resolve(ScopeProvider)
    all_scopes = {s for s in scopes.values() if s is not None}
    global_scope = next((s for s in all_scopes if isinstance(s, GlobalScope)), None)
    if global_scope is None:
        raise ValueError("Could not resolve the module's global scope.")

    body = list(tree.body)
    stmt_of_node: dict[int, int] = {}
    for i, stmt in enumerate(body):
        for node in _walk(stmt):
            stmt_of_node[id(node)] = i

    name_def_stmts: dict[str, set[int]] = {}
    for i, stmt in enumerate(body):
        for name in _bound_names(stmt):
            name_def_stmts.setdefault(name, set()).add(i)

    # stmt index -> global names it references.
    stmt_deps: dict[int, set[str]] = {}
    inline_ids: set[int] = {id(n) for n in _walk(inline_element)} if inline_element is not None else set()
    inline_deps: set[str] = set()
    for scope in all_scopes:
        for access in scope.accesses:
            for referent in access.referents:
                if not isinstance(referent, Assignment):
                    continue  # builtins etc.
                if referent.scope is not global_scope:
                    continue
                stmt_idx = stmt_of_node.get(id(access.node))
                if stmt_idx is not None:
                    stmt_deps.setdefault(stmt_idx, set()).add(referent.name)
                if id(access.node) in inline_ids:
                    inline_deps.add(referent.name)

    if inline_element is not None:
        seed_names = inline_deps

    # -- BFS the closure ----------------------------------------------------
    needed_names: set[str] = set()
    included_stmts: set[int] = set()
    queue = list(seed_names)
    while queue:
        name = queue.pop()
        if name in needed_names:
            continue
        needed_names.add(name)
        for stmt_idx in name_def_stmts.get(name, set()):
            if stmt_idx in included_stmts:
                continue
            included_stmts.add(stmt_idx)
            for dep in stmt_deps.get(stmt_idx, set()):
                if dep not in needed_names:
                    queue.append(dep)

    # -- Portability validation --------------------------------------------
    if entry_point in needed_names or target in needed_names:
        raise ValueError(
            f"Tool '{tool_ref}' references '{entry_point if entry_point in needed_names else target}' "
            "(the entry point) and cannot be extracted as a standalone module."
        )

    unresolved = {n for n in needed_names if n not in name_def_stmts}
    # Names without a top-level binding are fine only when they resolve to
    # builtins or scoped locals; anything the BFS queued came from a global
    # referent, so leftovers here indicate an unsupported binding shape.
    if unresolved - set(dir(__import__("builtins"))):
        raise ValueError(
            f"Tool '{tool_ref}' references names bound by unsupported statements "
            f"(e.g. loops, with-blocks, or conditionals): {', '.join(sorted(unresolved))}."
        )

    ordered = sorted(included_stmts)
    local_stems = {p.stem for p in workspace_path.glob("*.py")} | {
        p.name for p in workspace_path.iterdir() if p.is_dir() and not p.name.startswith(".")
    }
    for i in ordered:
        stmt = body[i]
        if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
            continue
        if not isinstance(stmt, cst.SimpleStatementLine) or not all(
            isinstance(item, (cst.Assign, cst.AnnAssign, cst.Import, cst.ImportFrom)) for item in stmt.body
        ):
            snippet = tree.code_for_node(stmt).strip().split("\n")[0]
            raise ValueError(
                f"Tool '{tool_ref}' depends on an unsupported top-level statement: '{snippet}'. "
                "Only imports, assignments, functions, and classes can be extracted."
            )
        for module_name, is_relative in _import_top_level_modules(stmt):
            if is_relative:
                raise ValueError(
                    f"Tool '{tool_ref}' uses a relative import, which cannot be extracted."
                )
            if module_name in local_stems:
                raise ValueError(
                    f"Tool '{tool_ref}' imports the local module '{module_name}'. "
                    "Extraction only supports single-module closures; inline the helper first."
                )

    # -- Emit the portable module -------------------------------------------
    import_stmts = [i for i in ordered if _is_import_stmt(body[i])]
    other_stmts = [i for i in ordered if i not in set(import_stmts)]
    parts: list[str] = []
    for i in import_stmts:
        parts.append(tree.code_for_node(body[i]).strip("\n"))
    if import_stmts:
        parts.append("")
    for i in other_stmts:
        parts.append(tree.code_for_node(body[i]).strip("\n"))
        parts.append("")
    if synthesized_binding_code is not None:
        parts.append(synthesized_binding_code.strip("\n"))
        parts.append("")
    module_code = "\n".join(parts).strip("\n") + "\n"
    formatted = format_code(module_code, spec.path.with_name("tool.py"))

    # -- Manifest metadata ----------------------------------------------------
    emitted_stmts = [body[i] for i in ordered]
    handler_def = func_defs.get(handler_name) if handler_name else None

    description: str | None = None
    if tool_call is not None:
        description = _string_value(_get_kwarg(tool_call, "description"))
    if description is None and handler_def is not None:
        description = _docstring_summary(handler_def)

    stdlib = _stdlib_modules()
    external_modules: set[str] = set()
    for i in import_stmts:
        for module_name, is_relative in _import_top_level_modules(body[i]):
            if not is_relative and module_name and module_name not in stdlib and module_name != "timbal":
                external_modules.add(module_name)

    return {
        "name": runtime_name,
        "binding": binding,
        "description": description,
        "source": formatted,
        "params": _extract_params(handler_def, tree) if handler_def is not None else [],
        "requirements": _requirements_for_imports(workspace_path, external_modules),
        "integrations": _scan_integrations(emitted_stmts),
        "env_vars": _scan_env_vars(emitted_stmts),
    }


# ---------------------------------------------------------------------------
# add-library-tool
# ---------------------------------------------------------------------------


def _infer_binding(source_tree: cst.Module) -> tuple[str, str]:
    """Infer ``(binding, runtime_name)`` from a library tool module.

    Prefers the last top-level ``x = Tool(...)`` assignment; falls back to the
    last top-level function definition (bare-function tools). The runtime name
    is what the tool answers to inside an agent: the Tool's ``name=`` kwarg,
    else the handler function's name, else the binding itself.
    """
    binding: str | None = None
    tool_call: cst.Call | None = None
    last_func: str | None = None
    for stmt in source_tree.body:
        if isinstance(stmt, cst.FunctionDef):
            last_func = stmt.name.value
        elif isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign) and isinstance(item.value, cst.Call):
                    if _constructor_name(item.value) == "Tool":
                        for t in item.targets:
                            if isinstance(t.target, cst.Name):
                                binding = t.target.value
                                tool_call = item.value
    if binding is not None:
        runtime_name = _string_value(_get_kwarg(tool_call, "name"))
        if runtime_name is None:
            handler_val = _get_kwarg(tool_call, "handler")
            runtime_name = handler_val.value if isinstance(handler_val, cst.Name) else binding
        return binding, runtime_name
    if last_func is not None:
        return last_func, last_func
    raise ValueError("Could not infer the tool binding from the module (no Tool assignment or function found).")


def _find_import_alias(tree: cst.Module, module: str) -> str | None:
    """Return the local name bound by an existing ``from <module> import ...``."""
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not isinstance(item, cst.ImportFrom) or isinstance(item.names, cst.ImportStar):
                continue
            parts: list[str] = []
            node = item.module
            while isinstance(node, cst.Attribute):
                parts.append(node.attr.value)
                node = node.value
            if isinstance(node, cst.Name):
                parts.append(node.value)
            if ".".join(reversed(parts)) != module:
                continue
            for alias in item.names:
                if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
                    return alias.asname.name.value
                if isinstance(alias.name, cst.Name):
                    return alias.name.value
    return None


class LibraryToolAdder(cst.CSTTransformer):
    """Add the library import and append the local name to tools=[...]."""

    # Re-adding is an idempotent success.
    allow_noop = True

    def __init__(
        self,
        target: str,
        assignments: dict[str, cst.Call],
        *,
        local_name: str,
        binding: str,
        import_module: str,
    ) -> None:
        self.target = target
        self.assignments = assignments
        self.local_name = local_name
        self.binding = binding
        self.import_module = import_module

    def _add_to_tools(self, call: cst.Call) -> cst.Call:
        new_ref = cst.Name(self.local_name)
        for i, arg in enumerate(call.args):
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "tools":
                if isinstance(arg.value, cst.List):
                    for el in arg.value.elements:
                        if isinstance(el.value, cst.Name) and el.value.value == self.local_name:
                            return call  # already present
                    new_list = arg.value.with_changes(
                        elements=[*arg.value.elements, cst.Element(value=new_ref)],
                    )
                    new_args = [*call.args[:i], arg.with_changes(value=new_list), *call.args[i + 1 :]]
                    return call.with_changes(args=new_args)
        new_arg = cst.Arg(keyword=cst.Name("tools"), value=cst.List(elements=[cst.Element(value=new_ref)]))
        return call.with_changes(args=[*call.args, new_arg])

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:  # noqa: ARG002
        for target in updated_node.targets:
            if (
                isinstance(target.target, cst.Name)
                and target.target.value == self.target
                and isinstance(updated_node.value, cst.Call)
            ):
                return updated_node.with_changes(value=self._add_to_tools(updated_node.value))
        return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:  # noqa: ARG002
        if (
            isinstance(updated_node.target, cst.Name)
            and updated_node.target.value == self.target
            and isinstance(updated_node.value, cst.Call)
        ):
            return updated_node.with_changes(value=self._add_to_tools(updated_node.value))
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if _find_import_alias(original_node, self.import_module) is not None:
            return updated_node
        if self.local_name != self.binding:
            import_stmt = f"from {self.import_module} import {self.binding} as {self.local_name}\n"
        else:
            import_stmt = f"from {self.import_module} import {self.binding}\n"
        body = list(updated_node.body)
        insert_imports(body, [cst.parse_statement(import_stmt)])
        return updated_node.with_changes(body=body)


def vendor_module_name(tool_name: str) -> str:
    """Sanitize a library tool name into a valid python module name."""
    module_name = re.sub(r"\W+", "_", tool_name).strip("_").lower() or "tool"
    if not module_name.isidentifier():
        module_name = f"_{module_name}"
    return module_name


def run_add_library_tool(workspace_path: str | Path, args: argparse.Namespace) -> None:
    """Vendor a library tool module and wire it into the entry point.

    Writes ``tools/<module>.py`` (with a provenance header when provided) and
    updates the entry point source. With ``--dry-run``, prints the updated
    entry point source and writes nothing.
    """
    workspace_path = Path(workspace_path)
    spec = parse_fqn(workspace_path)
    if not spec.path.exists():
        raise FileNotFoundError(f"source file not found: {spec.path}")

    module_source: str = args.source
    try:
        source_tree = cst.parse_module(module_source)
    except cst.ParserSyntaxError as e:
        raise ValueError(f"--source is not valid python: {e}") from e

    inferred_binding, runtime_name = _infer_binding(source_tree)
    binding = args.binding or inferred_binding
    tool_name = args.tool
    module_name = vendor_module_name(tool_name)
    import_module = f"{VENDOR_DIR}.{module_name}"

    source = spec.path.read_text()
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError as e:
        raise ValueError(f"Cannot parse {spec.path}: {e}") from e

    step = getattr(args, "step", None)
    target, assignments = validate_tools_target(tree, spec.target, step, operation="add-library-tool")

    # Resolve the local name for the imported binding. An existing import of
    # the vendor module wins (idempotent re-add); otherwise a collision with
    # any top-level name gets an aliased import.
    existing_alias = _find_import_alias(tree, import_module)
    if existing_alias is not None:
        local_name = existing_alias
    else:
        bound_names: set[str] = set()
        for stmt in tree.body:
            bound_names |= _bound_names(stmt)
        if binding in bound_names:
            local_name = module_name if module_name != binding else f"{binding}_lib"
            if local_name in bound_names:
                raise ValueError(
                    f"Cannot import library tool '{tool_name}': both '{binding}' and the "
                    f"fallback alias '{local_name}' already exist in {spec.path.name}. "
                    "Rename the local symbol first."
                )
        else:
            local_name = binding

    # Runtime tool names must be unique within the agent. A different tool
    # already answering to this name is a hard conflict, not something to
    # silently shadow.
    target_call = assignments.get(target)
    if target_call is not None:
        tools_val = _get_kwarg(target_call, "tools")
        if isinstance(tools_val, cst.List):
            for el in tools_val.elements:
                if isinstance(el.value, cst.Name) and el.value.value == local_name:
                    continue  # our own reference — idempotent
                if resolve_runnable_name(el.value, assignments) == runtime_name:
                    raise ValueError(
                        f"A different tool named '{runtime_name}' already exists in '{target}'. "
                        "Remove or rename it before adding this library tool."
                    )

    transformer = LibraryToolAdder(
        target,
        assignments,
        local_name=local_name,
        binding=binding,
        import_module=import_module,
    )
    new_tree = tree.visit(transformer)
    formatted = format_code(new_tree.code, spec.path)

    header = ""
    if getattr(args, "provenance", None):
        header = (
            f"{PROVENANCE_PREFIX} {args.provenance}\n"
            "# Vendored from the org tool library. Edits make this a fork and stop updates.\n\n"
        )
    vendor_content = header + module_source.strip("\n") + "\n"

    if getattr(args, "dry_run", False):
        print(formatted)
        return

    vendor_path = workspace_path / VENDOR_DIR / f"{module_name}.py"
    vendor_path.parent.mkdir(parents=True, exist_ok=True)
    vendor_path.write_text(vendor_content)
    spec.path.write_text(formatted)

    print(
        json.dumps(
            {
                "vendored": str(vendor_path.relative_to(workspace_path)),
                "binding": binding,
                "local_name": local_name,
                "name": runtime_name,
                "module": import_module,
            }
        )
    )
