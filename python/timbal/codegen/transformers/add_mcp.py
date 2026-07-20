"""add-mcp — add an MCPServer to an Agent's tools list.

Generates a declarative ``MCPServer(...)`` variable assignment and references it
in ``tools=[...]``. Secrets are never hardcoded: ``$VAR`` / ``${VAR}``
placeholders inside string values are emitted as ``os.environ[...]`` lookups so
the generated source reads the value at runtime.
"""

import argparse
import keyword
import re

import libcst as cst

from ..cli_utils import arg_input, parse_json_arg
from ..cst_utils import (
    collect_assignments,
    collect_step_names,
    has_import,
    is_bare_function_step,
    resolve_entry_point_type,
    resolve_runnable_name,
)

_ENV_PLACEHOLDER = re.compile(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)")

# Ordered constructor kwargs per transport, mirroring MCPServer's field order.
_STDIO_KEYS = ("command", "args", "env")
_HTTP_KEYS = ("url", "headers")


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "add-mcp",
        help="Add an MCP server to the agent's tools list.",
    )
    sp.add_argument(
        "--name",
        default=None,
        help="Identifier for the server. Used as the variable name and for remove-tool/set-config targeting.",
    )
    sp.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=None,
        help="Transport type. Inferred from --command (stdio) or --url (http) when omitted.",
    )
    sp.add_argument("--command", default=None, help="Executable for stdio transport. E.g. 'npx'.")
    sp.add_argument(
        "--args",
        default=None,
        dest="server_args",
        type=arg_input,
        help='Command arguments as a JSON array. E.g. \'["-y", "@modelcontextprotocol/server-filesystem", "."]\'.',
    )
    sp.add_argument(
        "--env",
        default=None,
        type=arg_input,
        help='Environment variables as a JSON object. Values may use $VAR / ${VAR} placeholders, '
        "emitted as os.environ lookups instead of hardcoded secrets.",
    )
    sp.add_argument("--url", default=None, help="Server URL for http transport.")
    sp.add_argument(
        "--headers",
        default=None,
        type=arg_input,
        help='HTTP headers as a JSON object. E.g. \'{"Authorization": "Bearer $TIMBAL_API_KEY"}\'. '
        "Values may use $VAR / ${VAR} placeholders, emitted as os.environ lookups instead of hardcoded secrets.",
    )
    sp.add_argument(
        "--from-json",
        default=None,
        dest="from_json",
        type=arg_input,
        help="Standard 'mcpServers' JSON config (claude-desktop/cursor style); adds every server in it. "
        "Mutually exclusive with the per-server flags. Use '@path' to read from file or '-' to read from stdin.",
    )
    sp.add_argument(
        "--step",
        default=None,
        help="Target step name within a Workflow. When provided, the server is added to the step's tools list.",
    )


def _var_name_for(server_name: str) -> str:
    """Derive a valid Python identifier from the server name."""
    var = re.sub(r"\W", "_", server_name)
    if not var or var[0].isdigit():
        var = f"mcp_{var}"
    if keyword.iskeyword(var):
        var = f"{var}_mcp"
    return var


def _infer_transport(spec: dict) -> str:
    explicit = spec.get("transport") or spec.get("type")
    if explicit is not None:
        # Accept the common aliases used by mcpServers configs.
        aliases = {"stdio": "stdio", "http": "http", "streamable-http": "http", "streamable_http": "http", "sse": "http"}
        transport = aliases.get(explicit)
        if transport is None:
            raise ValueError(f"Unknown MCP transport '{explicit}'. Use 'stdio' or 'http'.")
        return transport
    if spec.get("command") and spec.get("url"):
        raise ValueError("Both --command and --url given; pass --transport to disambiguate.")
    if spec.get("command"):
        return "stdio"
    if spec.get("url"):
        return "http"
    raise ValueError("Cannot infer transport: provide --command (stdio) or --url (http).")


def _build_server_kwargs(name: str, spec: dict) -> dict:
    """Normalize a raw spec (CLI flags or an mcpServers entry) into MCPServer kwargs."""
    transport = _infer_transport(spec)
    kwargs: dict = {"name": name, "transport": transport}
    keys = _STDIO_KEYS if transport == "stdio" else _HTTP_KEYS
    for key in keys:
        value = spec.get(key)
        if value is not None:
            kwargs[key] = value

    # Validate the full spec against the runtime model so errors surface at
    # codegen time, not when the generated file first runs. $VAR placeholders
    # are plain strings, so they pass through validation unchanged.
    from timbal.core.mcp import MCPServer

    try:
        MCPServer(**kwargs)
    except Exception as e:
        raise ValueError(f"Invalid MCP server config for '{name}': {e}") from e

    return kwargs


def _servers_from_args(args: argparse.Namespace) -> list[tuple[str, dict]]:
    """Return a list of (server_name, constructor_kwargs)."""
    if args.from_json is not None:
        per_server_flags = [args.name, args.transport, args.command, args.server_args, args.env, args.url, args.headers]
        if any(f is not None for f in per_server_flags):
            raise ValueError("--from-json is mutually exclusive with the per-server flags.")
        config = parse_json_arg(args.from_json, "--from-json")
        servers = config.get("mcpServers") if isinstance(config, dict) else None
        if not isinstance(servers, dict) or not servers:
            raise ValueError('--from-json must be a JSON object with a non-empty "mcpServers" key.')
        return [(name, _build_server_kwargs(name, spec)) for name, spec in servers.items()]

    if not args.name:
        raise ValueError("--name is required (or use --from-json).")

    spec: dict = {"transport": args.transport, "command": args.command, "url": args.url}
    if args.server_args is not None:
        server_args = parse_json_arg(args.server_args, "--args")
        if not isinstance(server_args, list):
            raise ValueError("--args must be a JSON array.")
        spec["args"] = server_args
    for flag, key in ((args.env, "env"), (args.headers, "headers")):
        if flag is not None:
            value = parse_json_arg(flag, f"--{key}")
            if not isinstance(value, dict):
                raise ValueError(f"--{key} must be a JSON object.")
            spec[key] = value
    return [(args.name, _build_server_kwargs(args.name, spec))]


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    step = getattr(args, "step", None)
    if tree is not None:
        ep_type = resolve_entry_point_type(tree, entry_point)
        if step:
            if ep_type is not None and ep_type != "Workflow":
                raise ValueError(f"--step requires a Workflow entry point, but '{entry_point}' is a {ep_type}.")
        else:
            if ep_type is not None and ep_type != "Agent":
                raise ValueError(f"add-mcp requires an Agent entry point, but '{entry_point}' is a {ep_type}.")

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

    servers = _servers_from_args(args)
    return MCPAdder(assignments, target=target, servers=servers)


# -- Code emission ------------------------------------------------------------


def _string_expr(value: str) -> tuple[str, bool]:
    """Return (python_expression, uses_os_environ) for a string value.

    ``$VAR`` / ``${VAR}`` placeholders become ``os.environ`` lookups; a string
    that is exactly one placeholder becomes a bare lookup, otherwise an f-string.
    """
    matches = list(_ENV_PLACEHOLDER.finditer(value))
    if not matches:
        return repr(value), False

    def env_var(m: re.Match) -> str:
        return m.group("braced") or m.group("plain")

    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        return f'os.environ["{env_var(matches[0])}"]', True

    parts: list[str] = []
    last = 0
    for m in matches:
        literal = value[last : m.start()]
        parts.append(literal.replace("\\", "\\\\").replace('"', '\\"').replace("{", "{{").replace("}", "}}"))
        parts.append(f"{{os.environ['{env_var(m)}']}}")
        last = m.end()
    parts.append(value[last:].replace("\\", "\\\\").replace('"', '\\"').replace("{", "{{").replace("}", "}}"))
    return 'f"' + "".join(parts) + '"', True


def _value_expr(value: object) -> tuple[str, bool]:
    """Return (python_expression, uses_os_environ) for any JSON value."""
    if isinstance(value, str):
        return _string_expr(value)
    if isinstance(value, list):
        exprs = [_value_expr(v) for v in value]
        return "[" + ", ".join(e for e, _ in exprs) + "]", any(env for _, env in exprs)
    if isinstance(value, dict):
        items = [(repr(k), _value_expr(v)) for k, v in value.items()]
        return (
            "{" + ", ".join(f"{k}: {e}" for k, (e, _) in items) + "}",
            any(env for _, (_, env) in items),
        )
    return repr(value), False


def _server_call_code(kwargs: dict) -> tuple[str, bool]:
    """Build the ``MCPServer(...)`` call source and whether it needs ``import os``."""
    parts: list[str] = []
    uses_env = False
    for key, value in kwargs.items():
        expr, env = _value_expr(value)
        uses_env = uses_env or env
        parts.append(f"{key}={expr}")
    return f"MCPServer({', '.join(parts)})", uses_env


class MCPAdder(cst.CSTTransformer):
    def __init__(self, assignments: dict[str, cst.Call], *, target: str, servers: list[tuple[str, dict]]):
        self.assignments = assignments
        self.target = target
        # (var_name, runtime_name, constructor_kwargs)
        self.servers = [(_var_name_for(name), name, kwargs) for name, kwargs in servers]
        self._updated: set[str] = set()  # runtime names whose assignment was updated in place
        self._uses_env = any(_server_call_code(kwargs)[1] for _, _, kwargs in self.servers)

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:  # noqa: ARG002
        for target in updated_node.targets:
            if not isinstance(target.target, cst.Name):
                continue
            target_name = target.target.value

            if target_name == self.target and isinstance(updated_node.value, cst.Call):
                return updated_node.with_changes(value=self._add_to_tools(updated_node.value))

            # Existing assignment for one of the servers — replace it wholesale
            # (add-mcp takes the full spec each time, so no merge semantics).
            if target_name != self.target and isinstance(updated_node.value, cst.Call):
                resolved = resolve_runnable_name(updated_node.value)
                for _var, runtime_name, kwargs in self.servers:
                    if resolved == runtime_name:
                        self._updated.add(runtime_name)
                        call_code, _ = _server_call_code(kwargs)
                        return updated_node.with_changes(value=cst.parse_expression(call_code))
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        imports_to_add: list[cst.BaseStatement] = []
        stmts_to_add: list[cst.BaseStatement] = []

        if not has_import(original_node, "timbal.core", "MCPServer"):
            imports_to_add.append(cst.parse_statement("from timbal.core import MCPServer\n"))
        if self._uses_env and not _has_os_import(original_node):
            imports_to_add.append(cst.parse_statement("import os\n"))

        for var_name, runtime_name, kwargs in self.servers:
            if runtime_name in self._updated:
                continue
            call_code, _ = _server_call_code(kwargs)
            stmts_to_add.append(cst.parse_statement(f"{var_name} = {call_code}\n"))

        if not imports_to_add and not stmts_to_add:
            return updated_node

        body = list(updated_node.body)

        if imports_to_add:
            import_insert_idx = 0
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, (cst.Import, cst.ImportFrom)):
                            import_insert_idx = i + 1
            for stmt in reversed(imports_to_add):
                body.insert(import_insert_idx, stmt)

        if stmts_to_add:
            insert_idx = len(body)
            runtime_names = {rn for _, rn, _ in self.servers}
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, cst.Assign) and isinstance(item.value, cst.Call):
                            if resolve_runnable_name(item.value) in runtime_names:
                                insert_idx = min(insert_idx, i)
                            for t in item.targets:
                                if isinstance(t.target, cst.Name) and t.target.value == self.target:
                                    insert_idx = min(insert_idx, i)
            for stmt in reversed(stmts_to_add):
                body.insert(insert_idx, stmt)

        return updated_node.with_changes(body=body)

    def _add_to_tools(self, call: cst.Call) -> cst.Call:
        """Add or update all server references in the tools=[...] list."""
        for var_name, runtime_name, _kwargs in self.servers:
            call = self._add_one_to_tools(call, var_name, runtime_name)
        return call

    def _add_one_to_tools(self, call: cst.Call, var_name: str, runtime_name: str) -> cst.Call:
        new_ref = cst.Name(var_name)

        for i, arg in enumerate(call.args):
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "tools":
                if isinstance(arg.value, cst.List):
                    for j, el in enumerate(arg.value.elements):
                        name = resolve_runnable_name(el.value, self.assignments)
                        if name == runtime_name:
                            if isinstance(el.value, cst.Call):
                                # Migrate inline Call → Name reference.
                                updated_el = el.with_changes(value=new_ref)
                                new_elements = [*arg.value.elements[:j], updated_el, *arg.value.elements[j + 1 :]]
                                new_list = arg.value.with_changes(elements=new_elements)
                                new_arg = arg.with_changes(value=new_list)
                                return call.with_changes(args=[*call.args[:i], new_arg, *call.args[i + 1 :]])
                            return call

                    new_element = cst.Element(value=new_ref)
                    new_list = arg.value.with_changes(elements=[*arg.value.elements, new_element])
                    new_arg = arg.with_changes(value=new_list)
                    return call.with_changes(args=[*call.args[:i], new_arg, *call.args[i + 1 :]])

        new_arg = cst.Arg(
            keyword=cst.Name("tools"),
            value=cst.List(elements=[cst.Element(value=new_ref)]),
        )
        return call.with_changes(args=[*call.args, new_arg])


def _has_os_import(tree: cst.Module) -> bool:
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Import):
                    for alias in item.names:
                        if isinstance(alias.name, cst.Name) and alias.name.value == "os":
                            return True
    return False
