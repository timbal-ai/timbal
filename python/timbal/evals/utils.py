from pathlib import Path
from typing import Any

import yaml

from ..core.runnable import Runnable
from ..state.tracing.span import Span
from ..state.tracing.trace import Trace
from .models import Eval

CONFIG_FILENAME = "evalconf.yaml"


def resolve_target(trace: Trace, target: str) -> tuple[Span | None, Any]:
    """Resolve a target path to a span and value.

    Target format: "span.path.property.nested.path"

    Examples:
        - "agent.validate_user.input.age" -> (Span, age_value)
        - "agent.llm.output" -> (Span, output_value)
        - "agent.search.input.query.text" -> (Span, nested text value)

    Args:
        trace: The trace to search in.
        target: Dot-separated path like "agent.validate_user.input.age".

    Returns:
        Tuple of (span, value). Span is None if not found.
        Value is None if span found but property path doesn't exist.
    """
    if not target:
        return None, None

    parts = target.split(".")

    # Try progressively longer paths to find the span
    # e.g., for "agent.validate_user.input.age", try:
    #   - "agent.validate_user.input.age" (no match)
    #   - "agent.validate_user.input" (no match)
    #   - "agent.validate_user" (match!)
    span: Span | None = None
    prop_parts: list[str] = []

    for i in range(len(parts), 0, -1):
        span_path = ".".join(parts[:i])
        spans = trace.get_path(span_path)
        if spans:
            # TODO Multiple matching spans (e.g. multiple iteration same tool uses)
            span = spans[0]
            prop_parts = parts[i:]
            break

    if span is None:
        return None, None

    if not prop_parts:
        return span, None

    # Resolve property path on the span
    value: Any = span
    for part in prop_parts:
        if value is None:
            return span, None

        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list):
            if part.isdigit():
                idx = int(part)
                value = value[idx] if idx < len(value) else None
            else:
                value = None
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            value = None

    return span, value


def discover_config(path: Path) -> dict[str, Any]:
    """Discover evalconf.yaml by walking up the directory tree.

    Searches from the given path upward to find evalconf.yaml.
    If path is a file, starts from its parent directory.
    """
    current = path if path.is_dir() else path.parent

    while current != current.parent:
        config_file = current / CONFIG_FILENAME
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        current = current.parent

    # Check root
    config_file = current / CONFIG_FILENAME
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f) or {}

    return {}


def discover_eval_files(path: Path) -> list[Path]:
    """Discover all eval files given a path.
    If the path is a directory, it will search recursively for files matching the pattern "eval*.yaml".
    If the path is a file, it'll simply check the file is .yaml and return it.
    """
    eval_files = []

    # If path doesn't exist, return empty list
    if not path.exists():
        return eval_files

    if path.is_dir():
        eval_file_set = set(path.rglob("eval*.yaml")) | set(path.rglob("*eval.yaml"))
        eval_files = list(eval_file_set)
    else:
        if not path.name.endswith(".yaml"):
            raise ValueError(f"Invalid evals path: {path}")
        eval_files.append(path)

    return eval_files


def parse_eval_file(path: Path, runnable: Runnable) -> list[Eval]:
    """Parse an eval file and return a list of Eval objects."""
    with open(path) as f:
        evals = yaml.safe_load(f)

    if not isinstance(evals, list):
        raise ValueError(f"Invalid eval file: {path}")

    evals = [Eval.model_validate({"path": path, "runnable": runnable, **eval}) for eval in evals]
    return evals
