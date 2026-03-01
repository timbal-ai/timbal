import json

from mcp.server.fastmcp import FastMCP

from timbal.codegen.pipeline import apply_operation, get_flow, parse_fqn
from timbal.codegen.transformers.add_tool import FRAMEWORK_TOOLS

mcp = FastMCP(
    "timbal-codegen",
    instructions=(
        "Use these tools to modify Timbal agents and workflows. "
        "Always use these tools instead of editing the agent source file directly — "
        "they handle imports, formatting, and dead code cleanup automatically. "
        "Each tool requires a `path` parameter: the ABSOLUTE path to the directory containing "
        "`timbal.yaml` (the workforce member directory). Always use full absolute paths, never relative."
    ),
)

TOOL_TYPES = [*FRAMEWORK_TOOLS.keys(), "Custom"]


@mcp.tool()
def add_tool(
    path: str,
    tool_type: str,
    definition: str | None = None,
    name: str | None = None,
) -> str:
    """Add a tool to the agent's tools list.

    Args:
        path: Absolute path to the directory containing timbal.yaml (the workforce member directory).
        tool_type: Type of tool to add. One of: Bash, CalaSearch, Edit, Read, WebSearch, Write, Custom.
        definition: Full function definition as a Python code string (required for Custom tools). E.g. 'def my_tool(query: str) -> str:\\n    return query.upper()'.
        name: Explicit name for the tool. When omitted, the tool uses its default name (class default for framework tools, function name for custom tools).
    """
    if tool_type not in TOOL_TYPES:
        return f"Error: tool_type must be one of {TOOL_TYPES}"
    result = apply_operation(path, "add_tool", tool_type=tool_type, definition=definition, tool_name=name)
    source_path, _ = parse_fqn(path)
    source_path.write_text(result)
    return f"Tool '{name or tool_type}' added successfully."


@mcp.tool()
def remove_tool(path: str, tool_name: str) -> str:
    """Remove a tool from the agent's tools list by its runtime name. Also cleans up the tool's variable assignment and function definition if they become unused.

    Args:
        path: Absolute path to the directory containing timbal.yaml (the workforce member directory).
        tool_name: The runtime name of the tool to remove (e.g. 'web_search', 'my_custom_tool').
    """
    result = apply_operation(path, "remove_tool", value=tool_name)
    source_path, _ = parse_fqn(path)
    source_path.write_text(result)
    return f"Tool '{tool_name}' removed successfully."


@mcp.tool()
def set_config(path: str, config: str, tool_name: str | None = None) -> str:
    """Set configuration on the agent or on a specific tool. Uses partial update semantics — only the fields specified in config are changed, all other fields are preserved.

    Args:
        path: Absolute path to the directory containing timbal.yaml (the workforce member directory).
        config: Configuration as a JSON object string. E.g. '{"model": "openai/gpt-4o"}' for agent config, or '{"allowed_domains": ["example.com"]}' for tool config. Set a field to null to remove it.
        tool_name: The runtime name of the tool to configure. Omit to configure the agent itself.
    """
    try:
        json.loads(config)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON in config: {e}"
    result = apply_operation(path, "set_config", tool_name=tool_name, config=config)
    source_path, _ = parse_fqn(path)
    source_path.write_text(result)
    target = f"tool '{tool_name}'" if tool_name else "agent"
    return f"Configuration updated on {target}."


@mcp.tool()
def get_flow_tool(path: str) -> str:
    """Get the ReactFlow-compatible graph for a workspace entry point. Returns a JSON object with _version, nodes, and edges.

    Args:
        path: Absolute path to the directory containing timbal.yaml (the workforce member directory).
    """
    flow = get_flow(path)
    return json.dumps(flow, indent=2)


def main() -> None:
    import os
    import sys

    print(f"[timbal-codegen] cwd: {os.getcwd()}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
