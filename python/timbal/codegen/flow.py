import contextlib
import inspect
import io
import re
import textwrap
from pathlib import Path
from typing import Any

from timbal.codegen import parse_fqn


def _get_when_source(step: Any) -> str | None:
    """Extract the source code of a step's when callable."""
    if not step.when or "callable" not in step.when:
        return None
    fn = step.when["callable"]
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


def _enrich_params_schema(runnable: Any) -> dict[str, Any]:
    """Enrich the params JSON schema with default info from fixed and runtime params."""
    schema = runnable.params_model_schema
    properties = schema.get("properties", {})

    for param_name, prop in properties.items():
        if param_name in runnable._default_runtime_params:
            info = runnable._default_runtime_params[param_name]
            deps = info.get("dependencies", [])
            source = deps[0] if deps else None
            entry: dict[str, Any] = {
                "type": "map",
                "source": source,
            }
            if source:
                key = _extract_key_from_lambda(info["callable"], source)
                if key:
                    entry["key"] = key
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

    config = runnable.get_config()

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
    # writes to stdout, and module-level warnings (e.g. ChunkEvents
    # deprecation) would pollute the JSON output.
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
