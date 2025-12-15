from pathlib import Path
from typing import Any

import structlog
import yaml

from ..core.runnable import Runnable
from ..state.tracing.span import Span
from ..state.tracing.trace import Trace
from ..utils import ImportSpec
from .models import Eval

logger = structlog.get_logger("timbal.evals.utils")

CONFIG_FILENAME = "evalconf.yaml"


def resolve_target(trace: Trace, target: str) -> tuple[Span | None, Any]:
    """Resolve a target path to a span and value.

    Target format: "span.path.property.nested.path"

    Examples:
        - "agent.validate_user.input.age" -> (Span, age_value)
        - "agent.llm.output" -> (Span, output_value)
        - "agent.search.input.query.text" -> (Span, nested text value)
        - "agent.llm.usage.input_tokens" -> (Span, sum of input_tokens across all models)
        - "agent.llm.usage.claude-haiku-4-5-20251001:input_tokens" -> (Span, specific model's input_tokens)

    Smart usage resolution:
        When accessing span.usage with a key like "input_tokens", the resolver will:
        1. First try exact match in usage dict (e.g., if there's a "input_tokens" key)
        2. If not found and there's only one model in usage, return that model's metric
        3. If multiple models exist, sum the metric across all models
        4. You can still use full keys like "claude-haiku-4-5-20251001:input_tokens" for specific models

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
    for idx, part in enumerate(prop_parts):
        if value is None:
            return span, None

        if isinstance(value, dict):
            # Check if this dict is a usage dict (previous part was "usage")
            is_usage_dict = idx > 0 and prop_parts[idx - 1] == "usage"

            # Smart usage resolution: handle model-prefixed keys
            if part not in value and is_usage_dict:
                resolved = _resolve_usage_key(value, part)
                if resolved is not None:
                    value = resolved
                else:
                    value = None
            else:
                value = value.get(part)
        elif isinstance(value, list):
            if part.isdigit():
                list_idx = int(part)
                value = value[list_idx] if list_idx < len(value) else None
            else:
                value = None
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            value = None

    return span, value


def _resolve_usage_key(usage: dict[str, int], key: str) -> int | None:
    """Resolve a usage key smartly.

    Examples:
        usage = {"claude-haiku-4-5-20251001:input_tokens": 100, "claude-haiku-4-5-20251001:output_tokens": 50}
        _resolve_usage_key(usage, "input_tokens") -> 100 (single model, returns its value)

        usage = {"model1:input_tokens": 100, "model2:input_tokens": 200}
        _resolve_usage_key(usage, "input_tokens") -> 300 (multiple models, returns sum)

        usage = {"claude-haiku-4-5-20251001:input_tokens": 100, "claude-haiku-4-5-20241120:input_tokens": 50}
        _resolve_usage_key(usage, "claude-haiku-4-5:input_tokens") -> 150 (partial model match, returns sum)

    Args:
        usage: The usage dictionary with model:metric keys
        key: The metric key to resolve. Can be:
             - Just metric: "input_tokens" (matches all models)
             - Partial model: "claude-haiku-4-5:input_tokens" (matches all claude-haiku-4-5* models)
             - Full model: "claude-haiku-4-5-20251001:input_tokens" (exact match)

    Returns:
        The resolved value, or None if not found
    """
    # If exact key exists, return it
    if key in usage:
        return usage[key]

    # Check if the key contains a model prefix (model:metric or just metric)
    if ":" in key:
        # User specified a model prefix (e.g., "claude-haiku-4-5:input_tokens")
        model_prefix, metric = key.split(":", 1)
        matching_values = []
        for usage_key, usage_value in usage.items():
            if ":" in usage_key:
                usage_model, usage_metric = usage_key.split(":", 1)
                # Match if model starts with prefix and metric matches
                if usage_model.startswith(model_prefix) and usage_metric == metric:
                    matching_values.append(usage_value)
    else:
        # No model specified, just a metric (e.g., "input_tokens")
        # Find all matching model:metric keys
        matching_values = []
        for usage_key, usage_value in usage.items():
            if ":" in usage_key:
                _, metric = usage_key.split(":", 1)
                if metric == key:
                    matching_values.append(usage_value)

    # If we found matches, sum them
    if matching_values:
        return sum(matching_values)

    return None


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


def _load_runnable(runnable_fqn: str, eval_file_path: Path) -> Runnable:
    """Load a runnable from a fully qualified name.

    Args:
        runnable_fqn: Fully qualified name like "path/to/file.py::object_name"
        eval_file_path: Path to the eval file (used for resolving relative paths)

    Returns:
        The loaded Runnable instance.
    """
    parts = runnable_fqn.split("::")
    if len(parts) != 2:
        raise ValueError(f"Invalid runnable format: '{runnable_fqn}'. Use 'path/to/file.py::object_name'")

    runnable_path, runnable_target = parts

    # Resolve relative paths from the eval file's directory
    runnable_path_obj = Path(runnable_path)
    if not runnable_path_obj.is_absolute():
        runnable_path_obj = (eval_file_path.parent / runnable_path_obj).resolve()
    else:
        runnable_path_obj = runnable_path_obj.expanduser().resolve()

    runnable_spec = ImportSpec(
        path=runnable_path_obj,
        target=runnable_target,
    )
    return runnable_spec.load()


def parse_eval_file(path: Path, runnable: Runnable | None = None) -> list[Eval]:
    """Parse an eval file and return a list of Eval objects.

    Each eval can optionally specify its own 'runnable' key to override the default.
    The runnable should be a fully qualified name like "path/to/file.py::object_name".
    Relative paths are resolved from the eval file's directory.

    If no default runnable is provided, each eval must specify its own 'runnable' key.
    """
    with open(path) as f:
        evals = yaml.safe_load(f)

    if not isinstance(evals, list):
        raise ValueError(f"Invalid eval file: {path}")

    parsed_evals = []
    for eval_dict in evals:
        # Check if this eval has its own runnable override
        eval_runnable = runnable
        if "runnable" in eval_dict:
            runnable_fqn = eval_dict.pop("runnable")
            try:
                eval_runnable = _load_runnable(runnable_fqn, path)
                logger.debug("Loaded per-eval runnable", eval_name=eval_dict.get("name"), runnable=runnable_fqn)
            except Exception as e:
                logger.error(
                    "Failed to load per-eval runnable",
                    eval_name=eval_dict.get("name"),
                    runnable=runnable_fqn,
                    error=str(e),
                )
                raise ValueError(
                    f"Failed to load runnable '{runnable_fqn}' for eval '{eval_dict.get('name')}': {e}"
                ) from e

        if eval_runnable is None:
            raise ValueError(
                f"No runnable specified for eval '{eval_dict.get('name')}' in {path}. "
                "Either add 'runnable' key to the eval or use --runnable flag."
            )

        parsed_eval = Eval.model_validate({"path": path, "runnable": eval_runnable, **eval_dict})
        parsed_evals.append(parsed_eval)

    return parsed_evals
