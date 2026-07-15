import os
from pathlib import Path
from typing import Any

import structlog
import yaml

from ..core.runnable import Runnable
from ..state.tracing.span import Span
from ..state.tracing.trace import Trace
from ..utils import ImportSpec
from ..utils.serialization import dump
from .models import Eval, EvalResult, EvalSummary

logger = structlog.get_logger("timbal.evals.utils")

CONFIG_FILENAME = "evalconf.yaml"


def _parse_path_key_indices(path_key: str, target: str) -> dict[str, int]:
    """Parse a path_key to extract span occurrence indices, filtering to only spans in target.

    The path_key contains flow validators (seq#0, parallel#0) that don't exist as spans.
    This function extracts only the indices for span names that appear in the target path.

    Args:
        path_key: Path key with occurrence indices, e.g., "agent.seq#0.parallel#0.get_datetime#1.input.timezone"
        target: The actual span path, e.g., "agent.get_datetime.input.timezone"

    Returns:
        Dictionary mapping actual span paths to their occurrence indices.
        e.g., {"agent.get_datetime": 1} (skipping seq and parallel which aren't real spans)
    """
    import re

    # Extract all indexed parts from path_key: [(name, index), ...]
    indexed_parts: list[tuple[str, int]] = []
    for part in path_key.split("."):
        match = re.match(r"^(.+)#(\d+)$", part)
        if match:
            name, idx = match.groups()
            indexed_parts.append((name, int(idx)))

    # Extract the span path from target (parts before property paths like input, output)
    # We need to find which parts of target are span names vs property paths
    target_parts = target.split(".")

    # Build indices dict by matching indexed parts to target path segments
    indices: dict[str, int] = {}
    target_path_parts: list[str] = []
    indexed_idx = 0

    for target_part in target_parts:
        target_path_parts.append(target_part)
        current_path = ".".join(target_path_parts)

        # Check if this target part matches the next indexed part
        if indexed_idx < len(indexed_parts):
            indexed_name, indexed_occurrence = indexed_parts[indexed_idx]
            if target_part == indexed_name:
                indices[current_path] = indexed_occurrence
                indexed_idx += 1

    return indices


def resolve_target(trace: Trace, target: str, path_key: str = "") -> tuple[Span | None, Any]:
    """Resolve a target path to a span and value.

    Target format: "span.path.property.nested.path"

    Examples:
        - "agent.validate_user.input.age" -> (Span, age_value)
        - "agent.llm.output" -> (Span, output_value)
        - "agent.search.input.query.text" -> (Span, nested text value)
        - "agent.llm.usage.input_tokens" -> (Span, sum of input_tokens across all models)
        - "agent.llm.usage.anthropic/claude-haiku-4-5:input_tokens" -> (Span, that model's input_tokens)

    Smart usage resolution:
        When accessing span.usage with a key like "input_tokens", the resolver will:
        1. First try exact match in usage dict (e.g., if there's a "input_tokens" key)
        2. If not found and there's only one model in usage, return that model's metric
        3. If multiple models exist, sum the metric across all models
        4. You can still use full keys like "anthropic/claude-haiku-4-5:input_tokens" for specific models

    Args:
        trace: The trace to search in.
        target: Dot-separated path like "agent.validate_user.input.age".
        path_key: Optional path key with occurrence indices, e.g., "agent.seq#0.get_datetime#1.input.timezone".
                  When provided, uses the indices to select the correct span occurrence.

    Returns:
        Tuple of (span, value). Span is None if not found.
        Value is None if span found but property path doesn't exist.
    """
    if not target:
        return None, None

    # Parse path_key to get occurrence indices for each span
    path_indices = _parse_path_key_indices(path_key, target) if path_key else {}

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
            # Use path_key index if available, otherwise default to first span
            occurrence_idx = path_indices.get(span_path, 0)
            if occurrence_idx < len(spans):
                span = spans[occurrence_idx]
            else:
                # Fall back to first span if index is out of range
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
        usage = {"anthropic/claude-haiku-4-5:input_tokens": 100, "anthropic/claude-haiku-4-5:output_tokens": 50}
        _resolve_usage_key(usage, "input_tokens") -> 100 (single model, returns its value)

        usage = {"openai/gpt-4o:input_text_tokens": 100, "anthropic/claude-haiku-4-5:input_tokens": 200}
        _resolve_usage_key(usage, "input_text_tokens") -> 100
        _resolve_usage_key(usage, "input_tokens") -> 200

        usage = {"anthropic/claude-haiku-4-5-20251001:input_tokens": 100, "anthropic/claude-haiku-4-5-20241120:input_tokens": 50}
        _resolve_usage_key(usage, "anthropic/claude-haiku-4-5:input_tokens") -> 150 (partial model match, returns sum)

    Args:
        usage: The usage dictionary with `provider/model:metric` keys (split on the last ":").
        key: The metric key to resolve. Can be:
             - Just metric: "input_tokens" (matches all models)
             - Partial model: "anthropic/claude-haiku-4-5:input_tokens" (matches usage models starting with prefix)
             - Full model: "anthropic/claude-haiku-4-5-20251001:input_tokens" (exact match)

    Returns:
        The resolved value, or None if not found
    """
    # If exact key exists, return it
    if key in usage:
        return usage[key]

    # Check if the key contains a model prefix (model:metric or just metric)
    if ":" in key:
        # User specified a model prefix (e.g. "anthropic/claude-haiku-4-5:input_tokens").
        # Billing keys use "provider/model:metric" — split on the last ":" only.
        model_prefix, metric = key.rsplit(":", 1)
        matching_values = []
        for usage_key, usage_value in usage.items():
            if ":" in usage_key:
                usage_model, usage_metric = usage_key.rsplit(":", 1)
                if usage_model.startswith(model_prefix) and usage_metric == metric:
                    matching_values.append(usage_value)
    else:
        # No model specified, just a metric (e.g., "input_tokens")
        matching_values = []
        for usage_key, usage_value in usage.items():
            if ":" in usage_key:
                _, metric = usage_key.rsplit(":", 1)
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


IGNORED_DIR_NAMES = {"node_modules", "__pycache__", "site-packages"}


def _is_ignored_dir(path: Path) -> bool:
    """Check whether a directory should be skipped during eval file discovery."""
    if path.name.startswith("."):
        return True
    if path.name in IGNORED_DIR_NAMES:
        return True
    # Virtualenvs not named with a leading dot (e.g. "venv", "env")
    if (path / "pyvenv.cfg").exists():
        return True
    return False


def discover_eval_files(path: Path) -> list[Path]:
    """Discover all eval files given a path.

    If the path is a directory, it searches recursively for files matching "eval*.yaml"
    or "*eval.yaml", skipping hidden directories, virtualenvs, node_modules,
    site-packages and __pycache__ (so e.g. fixtures shipped inside a project's .venv
    are never picked up).
    If the path is a file, it'll simply check the file is .yaml and return it.
    """
    if not path.exists():
        return []

    if not path.is_dir():
        if not path.name.endswith(".yaml"):
            raise ValueError(f"Invalid evals path: {path}")
        return [path]

    eval_files: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(path):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not _is_ignored_dir(current / d)]
        for filename in filenames:
            if not filename.endswith(".yaml"):
                continue
            stem = filename[: -len(".yaml")]
            if filename.startswith("eval") or stem.endswith("eval"):
                eval_files.add(current / filename)

    return sorted(eval_files)


async def dump_result(result: EvalResult) -> dict[str, Any]:
    """Serialize a single EvalResult to a JSON-safe dict.

    Purpose-built instead of EvalResult.model_dump() because results hold
    non-serializable objects (the Runnable instance, raw OutputEvents, etc.).
    """
    output_event = result.agent_output
    return {
        "name": result.eval.name,
        "path": str(result.eval.path),
        "description": result.eval.description,
        "tags": result.eval.tags,
        "passed": result.passed,
        "duration": result.duration,
        "error": result.error.model_dump() if result.error else None,
        "params": await dump(result.eval.params),
        "output": await dump(output_event.output) if output_event is not None else None,
        "usage": output_event.usage if output_event is not None else {},
        "validators": [
            {
                "target": vr.target,
                "name": vr.name,
                "value": await dump(vr.value),
                "passed": vr.passed,
                "evaluated": vr.evaluated,
                "error": vr.error,
                "actual_value": await dump(vr.actual_value),
            }
            for vr in result.validator_results
        ],
        "captured_stdout": result.captured_stdout,
        "captured_stderr": result.captured_stderr,
    }


async def dump_summary(summary: EvalSummary) -> dict[str, Any]:
    """Serialize an EvalSummary to a JSON-safe dict."""
    return {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "total_duration": summary.total_duration,
        "results": [await dump_result(r) for r in summary.results],
    }


def collect_evals(
    path: Path,
    runnable: Runnable | None = None,
    eval_name: str | None = None,
    tags: set[str] | None = None,
) -> list[Eval]:
    """Discover, parse, and filter evals under a path.

    Raises ValueError on duplicate eval names or when eval_name matches nothing.
    Returns an empty list when no eval files, evals, or tag matches are found.
    """
    eval_files = discover_eval_files(path)
    evals = [eval for eval_file in eval_files for eval in parse_eval_file(eval_file, runnable)]

    seen_names: dict[str, Path] = {}
    for e in evals:
        if e.name in seen_names:
            raise ValueError(f"Duplicate eval name '{e.name}' found in:\n  - {seen_names[e.name]}\n  - {e.path}")
        seen_names[e.name] = e.path

    if eval_name is not None:
        evals = [e for e in evals if e.name == eval_name]
        if not evals:
            raise ValueError(f"No eval found with name '{eval_name}'")

    if tags:
        evals = [e for e in evals if tags & set(e.tags)]

    return evals


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

    # Resolve relative paths from the eval file's directory, falling back to the
    # current working directory (so eval files in e.g. an "evals/" subdirectory can
    # reference project-root modules like "agent.py::agent").
    runnable_path_obj = Path(runnable_path)
    if not runnable_path_obj.is_absolute():
        candidate = (eval_file_path.parent / runnable_path_obj).resolve()
        if not candidate.exists():
            cwd_candidate = (Path.cwd() / runnable_path_obj).resolve()
            if cwd_candidate.exists():
                candidate = cwd_candidate
        runnable_path_obj = candidate
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
