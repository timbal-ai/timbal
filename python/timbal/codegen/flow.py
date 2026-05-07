import contextlib
import inspect
import io
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from timbal.codegen import parse_fqn


def _get_callable_source(fn: Any) -> str | None:
    """Extract a readable source representation of a callable.

    For inline lambdas, returns the lambda expression. For named functions,
    returns ``<fn_name>``. Returns None if the source can't be retrieved
    and the callable has no name.
    """
    try:
        source = inspect.getsource(fn)
        source = textwrap.dedent(source).strip()
        # For inline lambdas (e.g. `when=lambda: ...`), extract just the lambda.
        if "lambda" in source:
            idx = source.find("lambda")
            if idx >= 0:
                source = source[idx:]
                # Strip trailing comma or paren from inline usage.
                source = source.rstrip(" ,)")
        return source
    except (OSError, TypeError):
        return f"<{fn.__name__}>" if hasattr(fn, "__name__") else None


def _get_when_source(step: Any) -> str | None:
    """Extract the source code of a step's when callable."""
    if not step.when or "callable" not in step.when:
        return None
    return _get_callable_source(step.when["callable"])


def _extract_key_from_lambda(fn: Any, source_step: str) -> str | None:
    """Extract the dot-notation key path from a runtime lambda's source.

    Given a lambda like:
        lambda: get_run_context().step_span("agent_a").output[0].something
    Returns:
        "output.0.something"

    Returns None if the lambda just accesses `.output` with no further path.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None

    # Find everything after step_span("source_step")
    pattern = rf'step_span\("{re.escape(source_step)}"\)(\..*?)(?:\s*[,)]|$)'
    match = re.search(pattern, source)
    if not match:
        return None

    accessor = match.group(1)  # e.g. ".output[0].something" or ".output"
    if accessor == ".output":
        return None

    # Convert Python accessor syntax to dot notation.
    from timbal.codegen.transformers.set_param import accessor_to_key

    return accessor_to_key(accessor)


def _is_model_enum(values: list) -> bool:
    """Return True if a list of enum values looks like LLM model IDs (provider/name)."""
    return len(values) > 5 and all(isinstance(v, str) and "/" in v for v in values[:5])


def _compact_field(field: dict) -> dict:
    """Replace a large model-ID enum in a single field schema with an x-timbal-ref marker.

    Handles both direct enums and enums nested inside anyOf (e.g. Literal | str).
    Preserves all other keys (value, description, default, …) on the field.
    """
    field = dict(field)
    if _is_model_enum(field.get("enum", [])):
        field = {k: v for k, v in field.items() if k != "enum"}
        field["x-timbal-ref"] = "models"
    elif "anyOf" in field:
        for item in field["anyOf"]:
            if _is_model_enum(item.get("enum", [])):
                field = {k: v for k, v in field.items() if k not in ("anyOf", "$ref")}
                field["type"] = "string"
                field["x-timbal-ref"] = "models"
                break
    return field


def _compact_params_schema(schema: dict) -> dict:
    """Apply _compact_field to every property in a params JSON schema."""
    schema = dict(schema)
    if "properties" in schema:
        schema["properties"] = {name: _compact_field(prop) for name, prop in schema["properties"].items()}

    # Drop $defs entries that are no longer referenced after compaction.
    if "$defs" in schema:
        import json

        schema_without_defs = {k: v for k, v in schema.items() if k != "$defs"}
        body = json.dumps(schema_without_defs)
        schema["$defs"] = {k: v for k, v in schema["$defs"].items() if f'"#/$defs/{k}"' in body}
        if not schema["$defs"]:
            del schema["$defs"]

    return schema


def _compact_config(config: dict) -> dict:
    """Apply _compact_field to every field in an _annotate_config result.

    Config is a flat dict of {field_name: field_schema} where each value
    is a schema dict (with anyOf / enum / type / …) plus a 'value' key.
    """
    return {name: _compact_field(field) for name, field in config.items()}


def _enrich_params_schema(runnable: Any) -> dict[str, Any]:
    """Enrich the params JSON schema with default info from fixed and runtime params."""
    schema = _compact_params_schema(runnable.params_model_schema)
    properties = schema.get("properties", {})

    for param_name, prop in properties.items():
        if param_name in runnable._default_runtime_params:
            info = runnable._default_runtime_params[param_name]
            deps = info.get("dependencies", [])
            source = deps[0] if deps else None
            if source:
                entry: dict[str, Any] = {
                    "type": "map",
                    "source": source,
                }
                key = _extract_key_from_lambda(info["callable"], source)
                if key:
                    entry["key"] = key
            else:
                # No statically-resolvable upstream step (helper-fn lambda,
                # partial, lambda defined elsewhere, etc.). Don't emit a
                # phantom map edge — record the callable as opaque.
                expr = _get_callable_source(info["callable"]) or "<callable>"
                entry = {"type": "callable", "expr": expr}
            prop["value"] = entry
        elif param_name in runnable._default_fixed_params:
            value = runnable._default_fixed_params[param_name]
            prop["value"] = {
                "type": "value",
                "value": value,
            }

    return schema


def _build_node(runnable: Any, *, include_tools: bool = True) -> dict[str, Any]:
    """Build a ReactFlow-compatible node dict from a live Runnable instance."""
    from timbal.core.agent import Agent
    from timbal.core.runnable import Runnable
    from timbal.core.workflow import Workflow

    if isinstance(runnable, Agent):
        node_type = "agent"
    elif isinstance(runnable, Workflow):
        node_type = "workflow"
    else:
        node_type = "tool"

    config = _compact_config(runnable.get_config())

    if node_type == "agent" and include_tools:
        config["tools"] = [_build_node(t, include_tools=False) for t in runnable.tools if isinstance(t, Runnable)]

    node: dict[str, Any] = {
        "id": runnable._path,
        "type": node_type,
        "position": runnable.metadata.get("position", {"x": 0, "y": 0}),
        "data": {
            "config": config,
            "params": _enrich_params_schema(runnable),
            "return": runnable.return_model_schema,
            "metadata": runnable.metadata,
        },
    }

    return node


def get_flow(workspace_path: str | Path) -> dict[str, Any]:
    """Return a ReactFlow-compatible graph for the workspace entry point.

    Imports the runnable defined by timbal.yaml's FQN and builds a JSON-
    serialisable dict with ``_version``, ``nodes``, and ``edges``.

    Args:
        workspace_path: Path to directory containing timbal.yaml.

    Returns:
        ``{"_version": ..., "nodes": [...], "edges": [...]}``
    """
    workspace_path = Path(workspace_path)
    spec = parse_fqn(workspace_path)

    if not spec.path.exists():
        raise FileNotFoundError(f"source file not found: {spec.path}")

    # Suppress all stdout during import — structlog's default PrintLogger
    # writes to stdout and would pollute the JSON output.
    with contextlib.redirect_stdout(io.StringIO()):
        from timbal import __version__
        from timbal.core.agent import Agent
        from timbal.core.workflow import Workflow

        runnable = spec.load()

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    if isinstance(runnable, Workflow):
        for step in runnable._steps.values():
            nodes.append(_build_node(step, include_tools=isinstance(step, Agent)))

        for _step_name, step in runnable._steps.items():
            when_source = _get_when_source(step) if step.when else None
            for prev_name in step.previous_steps:
                prev_step = runnable._steps[prev_name]
                edge: dict[str, Any] = {
                    "id": f"{prev_step._path}->{step._path}",
                    "source": prev_step._path,
                    "target": step._path,
                }
                if when_source is not None:
                    edge["when"] = when_source
                edges.append(edge)
    else:
        nodes.append(_build_node(runnable, include_tools=isinstance(runnable, Agent)))

    return {
        "_version": __version__,
        "nodes": nodes,
        "edges": edges,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Compact formatter — token-efficient, LLM-readable summary
# ──────────────────────────────────────────────────────────────────────────────

def _short(node_id: str | None) -> str:
    """'workflow.step_name' → 'step_name'. Tolerates missing/None ids."""
    if not node_id:
        return "?"
    return node_id.split(".")[-1]


def _fmt_type(prop: dict) -> str:
    if prop.get("x-timbal-ref") == "models":
        return "model"
    t = prop.get("type", "")
    if t == "object":
        return "dict"
    if t == "array":
        item_type = _fmt_type(prop.get("items", {})) if prop.get("items") else "any"
        return f"list[{item_type}]"
    if "anyOf" in prop:
        types = [_fmt_type(x) for x in prop["anyOf"] if x.get("type") != "null"]
        return "|".join(types) if types else "any"
    return t or "any"


def _config_val(field: dict) -> Any:
    """Extract plain value from a config field dict produced by _compact_config."""
    v = field.get("value")
    if isinstance(v, dict) and v.get("type") == "value":
        return v["value"]
    return v


def _fmt_fallback_entry(entry: Any) -> str:
    """Render one fallback entry compactly. Shows the model name and any
    non-default ModelEntry fields (max_retries, retry_delay, base_url).

    api_key is intentionally not surfaced — it's already masked upstream.
    """
    if isinstance(entry, str):
        return entry
    if not isinstance(entry, dict):
        return str(entry)
    model = entry.get("model") or "?"
    extras: list[str] = []
    if entry.get("max_retries") not in (None, 2):
        extras.append(f"retries={entry['max_retries']}")
    if entry.get("retry_delay") not in (None, 1.0):
        extras.append(f"delay={entry['retry_delay']}")
    if entry.get("base_url"):
        extras.append(f"base_url={entry['base_url']}")
    if extras:
        return f"{model}({', '.join(extras)})"
    return model


def _fmt_node_lines(node: dict, indent: str) -> list[str]:
    lines: list[str] = []
    ntype = node["type"]
    name = _short(node["id"])
    config = node["data"].get("config", {})
    params = node["data"].get("params", {})
    props = params.get("properties", {})
    required = set(params.get("required", []))

    if ntype == "agent":
        model = _config_val(config.get("model", {})) or "?"
        extras: list[str] = [model]
        for key in ("max_iter", "max_tokens", "temperature"):
            val = _config_val(config.get(key, {}))
            if val is not None:
                extras.append(f"{key}={val}")
        lines.append(f"{indent}agent  {name}  [{', '.join(extras)}]")

        fallbacks = _config_val(config.get("fallbacks", {})) or []
        if fallbacks:
            names = [_fmt_fallback_entry(f) for f in fallbacks]
            lines.append(f"{indent}  fallbacks: {', '.join(names)}")

        sp = _config_val(config.get("system_prompt", {}))
        if sp:
            sp_short = sp[:120].replace("\n", " ")
            if len(sp) > 120:
                sp_short += "…"
            lines.append(f'{indent}  system_prompt: "{sp_short}"')

        tools: list[dict] = config.get("tools", [])
        tool_names = [_short(t["id"]) for t in tools] if tools else []
        lines.append(f"{indent}  tools: {', '.join(tool_names) if tool_names else 'none'}")

        for pname, prop in props.items():
            val = prop.get("value", {})
            if isinstance(val, dict) and val.get("type") == "map":
                source = _short(val.get("source") or "?")
                key = val.get("key")
                src = f"{source}.{key}" if key else source
                lines.append(f"{indent}  {pname} ← {src}")
            elif isinstance(val, dict) and val.get("type") == "callable":
                expr = val.get("expr") or "<callable>"
                lines.append(f"{indent}  {pname} ← {expr}")

    elif ntype == "tool":
        sig_parts: list[str] = []
        for pname, prop in props.items():
            val = prop.get("value", {})
            if isinstance(val, dict) and val.get("type") == "map":
                source = _short(val.get("source") or "?")
                key = val.get("key")
                src = f"{source}.{key}" if key else source
                sig_parts.append(f"{pname}:←{src}")
            elif isinstance(val, dict) and val.get("type") == "callable":
                expr = val.get("expr") or "<callable>"
                sig_parts.append(f"{pname}:←{expr}")
            elif isinstance(val, dict) and val.get("type") == "value":
                sig_parts.append(f"{pname}={repr(val['value'])[:30]}")
            else:
                opt = "" if pname in required else "?"
                sig_parts.append(f"{pname}{opt}:{_fmt_type(prop)}")

        ret = node["data"].get("return", {})
        ret_t = _fmt_type(ret) if ret else ""
        ret_str = f" → {ret_t}" if ret_t and ret_t not in ("any", "") else ""
        lines.append(f"{indent}tool   {name}({', '.join(sig_parts)}){ret_str}")

    elif ntype == "workflow":
        lines.append(f"{indent}workflow  {name}")

    return lines


def _build_edge_lines(edges: list[dict]) -> list[str]:
    """Render edges grouped by source, one line per source node."""
    targets_of: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    all_sources: list[str] = []
    seen_sources: set[str] = set()
    for e in edges:
        src, tgt = _short(e["source"]), _short(e["target"])
        targets_of[src].append((tgt, e.get("when")))
        if src not in seen_sources:
            all_sources.append(src)
            seen_sources.add(src)

    lines: list[str] = []
    for src in all_sources:
        nexts = targets_of[src]
        if len(nexts) == 1:
            tgt, when = nexts[0]
            arrow = f"→[{when}] " if when else "→ "
            lines.append(f"{src} {arrow}{tgt}")
        else:
            parts = []
            for tgt, when in nexts:
                parts.append(f"[{when}] {tgt}" if when else tgt)
            lines.append(f"{src} → {', '.join(parts)}")
    return lines


def format_compact(flow: dict) -> str:
    """Render a get-flow result as a compact, token-efficient text summary."""
    nodes = flow.get("nodes", [])
    edges = flow.get("edges", [])

    if not nodes:
        return "(empty flow)"

    lines: list[str] = []

    # A standalone runnable (Agent/Tool as entry point) has no workflow prefix
    # in its ID (e.g. "my_agent"). Workflow step nodes always have "wf.step" form.
    is_standalone = len(nodes) == 1 and "." not in nodes[0]["id"]

    if is_standalone:
        node = nodes[0]
        ntype = node["type"].upper()
        name = _short(node["id"])
        config = node["data"].get("config", {})

        extras: list[str] = []
        if ntype == "AGENT":
            model = _config_val(config.get("model", {})) or "?"
            extras.append(model)
            for key in ("max_iter", "max_tokens", "temperature"):
                val = _config_val(config.get(key, {}))
                if val is not None:
                    extras.append(f"{key}={val}")
        header = f"{ntype} {name}"
        if extras:
            header += f"  [{', '.join(extras)}]"
        lines.append(header)

        if ntype == "AGENT":
            fallbacks = _config_val(config.get("fallbacks", {})) or []
            if fallbacks:
                names = [_fmt_fallback_entry(f) for f in fallbacks]
                lines.append(f"fallbacks: {', '.join(names)}")
            sp = _config_val(config.get("system_prompt", {}))
            if sp:
                sp_short = sp[:120].replace("\n", " ") + ("…" if len(sp) > 120 else "")
                lines.append(f'system_prompt: "{sp_short}"')
            tools: list[dict] = config.get("tools", [])
            tool_names = [_short(t["id"]) for t in tools]
            lines.append(f"tools: {', '.join(tool_names) if tool_names else 'none'}")
    else:
        # Workflow entry point — steps have IDs like "wf_name.step_name"
        workflow_name = nodes[0]["id"].split(".")[0]
        lines.append(f"WORKFLOW {workflow_name}")
        lines.append("")
        lines.append("STEPS")
        for node in nodes:
            for line in _fmt_node_lines(node, indent="  "):
                lines.append(line)

    if edges:
        lines.append("")
        lines.append("EDGES")
        for line in _build_edge_lines(edges):
            lines.append(f"  {line}")

    return "\n".join(lines)
