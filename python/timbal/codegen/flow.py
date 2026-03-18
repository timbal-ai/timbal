import contextlib
import inspect
import io
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

    return {
        "id": runnable._path,
        "type": node_type,
        "data": {
            "config": config,
            "params": runnable.params_model_schema,
            "return": runnable.return_model_schema,
            "metadata": runnable.metadata,
        },
    }


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
