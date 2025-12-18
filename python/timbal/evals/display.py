from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from .models import Eval, EvalResult, EvalSummary, ValidatorResult
from .validators.base import BaseValidator

console = Console()


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]WARNING:[/bold yellow] {message}")


class OutputCapture:
    """Context manager to capture stdout and stderr."""

    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()
        self._stdout_redirector = None
        self._stderr_redirector = None

    def __enter__(self):
        self._stdout_redirector = redirect_stdout(self.stdout)
        self._stderr_redirector = redirect_stderr(self.stderr)
        self._stdout_redirector.__enter__()
        self._stderr_redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stdout_redirector.__exit__(exc_type, exc_val, exc_tb)  # type: ignore
        self._stderr_redirector.__exit__(exc_type, exc_val, exc_tb)  # type: ignore
        # Don't suppress exceptions
        return False

    def get_stdout(self) -> str:
        return self.stdout.getvalue()

    def get_stderr(self) -> str:
        return self.stderr.getvalue()


def print_header(evals: list[Eval]) -> None:
    """Print pytest-style header."""
    console.print()
    console.rule("[bold]Timbal Evals[/bold]", style="blue")
    console.print()

    # Collect unique files
    files = set(str(e.path) for e in evals)
    file_count = len(files)
    file_label = "file" if file_count == 1 else "files"
    console.print(f"[dim]collected {len(evals)} evals from {file_count} {file_label}[/dim]")
    console.print()


# Tag color palette - expanded for variety
TAG_COLORS = [
    "cyan",
    "magenta",
    "yellow",
    "blue",
    "green",
    "red",
    "bright_cyan",
    "bright_magenta",
    "bright_yellow",
    "bright_blue",
    "bright_green",
    "bright_red",
    "dark_cyan",
    "dark_magenta",
    "dark_orange",
    "purple",
    "orange1",
    "deep_pink3",
    "spring_green3",
    "dodger_blue2",
    "gold3",
    "medium_purple3",
    "dark_sea_green",
    "indian_red",
]


def _get_tag_color(tag: str) -> str:
    """Get a consistent color for a tag based on its name hash."""
    # Use sum of character codes * position for better distribution
    tag_hash = sum((i + 1) * ord(c) for i, c in enumerate(tag))
    return TAG_COLORS[tag_hash % len(TAG_COLORS)]


def print_eval_result(result: EvalResult) -> None:
    """Print eval result with validator tree."""
    eval = result.eval
    path = eval.path
    name = eval.name
    tags = eval.tags
    duration_str = f"{result.duration:.2f}s"

    # Build status badge
    if result.passed:
        status = Text(" PASSED ", style="bold white on green")
    else:
        status = Text(" FAILED ", style="bold white on red")

    # Build header line
    header = Text()
    header.append_text(status)
    header.append(" ")
    header.append(str(path), style="dim")
    header.append("::", style="dim")
    header.append(name, style="bold")

    # Add tags
    for tag in tags:
        color = _get_tag_color(tag)
        header.append(" ")
        header.append("[", style="dim")
        header.append(tag, style=color)
        header.append("]", style="dim")

    header.append(f" [{duration_str}]", style="dim")

    # Build tree with header as root
    tree = Tree(header)

    # Build a lookup map for validator results by (target, name)
    validator_results_map: dict[tuple[str, str], ValidatorResult] = {}
    for vr in result.validator_results:
        validator_results_map[(vr.target, vr.name)] = vr

    # Add validators
    validators = eval._validators
    if validators:
        # Add root path as first branch
        root_path = eval.runnable._path
        root_branch = tree.add(f"[yellow]{root_path}[/yellow]")
        _build_validator_tree(root_branch, validators, root_path, validator_results_map)

    console.print(tree)
    console.print()


def print_eval_line(result: EvalResult) -> None:
    """Print a single eval result line (pytest-style). Deprecated - use print_eval_result."""
    print_eval_result(result)


def print_failure_details(result: EvalResult) -> None:
    """Print detailed failure information."""
    path = result.eval.path
    name = result.eval.name

    console.print()
    console.rule(f"[red]FAILED[/red] {path}::{name}", style="red")

    if result.captured_stdout:
        console.print(
            Panel(
                result.captured_stdout.rstrip(),
                title="[dim]Captured stdout[/dim]",
                title_align="left",
                border_style="dim",
                padding=(0, 1),
            )
        )

    if result.captured_stderr:
        console.print(
            Panel(
                result.captured_stderr.rstrip(),
                title="[yellow]Captured stderr[/yellow]",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    # Show failed validators
    failed_validators = [vr for vr in result.validator_results if not vr.passed]
    if failed_validators:
        for vr in failed_validators:
            header = f"[red bold]✗[/red bold] [yellow]{vr.target}[/yellow].[green]{vr.name}[/green]"
            if vr.traceback:
                content = vr.traceback.rstrip()
            elif vr.error:
                content = vr.error
            else:
                content = "Validation failed"
            console.print(
                Panel(
                    content,
                    title=header,
                    title_align="left",
                    border_style="red",
                    padding=(0, 1),
                )
            )

    if result.error:
        if result.error.traceback:
            error_text = result.error.traceback.rstrip()
        else:
            error_text = f"[red bold]{result.error.type}[/red bold]: {result.error.message}"
        console.print(
            Panel(
                error_text,
                title="[red]Error[/red]",
                title_align="left",
                border_style="red",
                padding=(0, 1),
            )
        )


def print_failures(results: list[EvalResult]) -> None:
    """Print all failure details."""
    failed = [r for r in results if not r.passed]
    if failed:
        console.print()
        console.rule("[bold red]FAILURES[/bold red]", style="red")
        for result in failed:
            print_failure_details(result)


def print_summary(summary: EvalSummary) -> None:
    """Print pytest-style summary."""
    console.print()

    # Build summary line
    parts = []
    if summary.failed > 0:
        parts.append(f"[bold red]{summary.failed} failed[/bold red]")
    if summary.passed > 0:
        parts.append(f"[bold green]{summary.passed} passed[/bold green]")

    duration_str = f"{summary.total_duration:.2f}s"
    summary_text = ", ".join(parts) if parts else "[dim]no evals run[/dim]"

    # Determine overall status color
    if summary.failed > 0:
        status_style = "red"
        status_char = "!"
    else:
        status_style = "green"
        status_char = "="

    console.print()
    console.rule(
        f"{summary_text} [dim]in {duration_str}[/dim]",
        style=status_style,
        characters=status_char,
    )
    console.print()


def _format_validator_value(value: Any) -> str:
    """Format a validator value for display."""
    if isinstance(value, str):
        # Truncate long strings
        if len(value) > 30:
            return f'"{value[:27]}..."'
        return f'"{value}"'
    elif isinstance(value, list):
        if len(value) > 3:
            return f"[{len(value)} items]"
        return str(value)
    elif isinstance(value, dict):
        return "{...}"
    else:
        return str(value)


def _normalize_validator(v: Any) -> tuple[str, str, Any, bool, list[str]]:
    """Normalize a validator to (target, name, value, negate, transform) tuple."""
    if isinstance(v, tuple):
        # Legacy tuple format - assume no negate/transform
        if len(v) == 3:
            return (*v, False, [])
        return v
    elif isinstance(v, BaseValidator):
        return (v.target or "", v.name, v.value, v.negate, v.transform)
    else:
        return ("", "unknown!", v, False, [])


FLOW_VALIDATORS_SET = {"seq!", "parallel!", "any!"}

# Mapping from base validator to negated display name
NEGATED_VALIDATOR_NAMES: dict[str, str] = {
    "eq!": "ne!",
    "contains!": "not_contains!",
    "contains_all!": "not_contains_all!",
    "contains_any!": "not_contains_any!",
    "starts_with!": "not_starts_with!",
    "ends_with!": "not_ends_with!",
    "pattern!": "not_pattern!",
    "type!": "not_type!",
    "semantic!": "not_semantic!",
    "language!": "not_language!",
}


def _format_validator_status(
    target: str,
    validator: str,
    value: Any,
    results_map: dict[tuple[str, str], ValidatorResult] | None,
    rel_path: str = "",
    negate: bool = False,
    transform: list[str] | None = None,
) -> str:
    """Format a validator line with pass/fail status indicator."""
    # Look up result if available
    result = results_map.get((target, validator)) if results_map else None

    if result is not None:
        if result.passed:
            status = "[bold green]✓[/bold green]"
        else:
            status = "[bold red]✗[/bold red]"
    else:
        # No result yet (e.g., during planning/preview)
        status = "[dim]○[/dim]"

    # Use cyan for flow validators, green for others
    is_flow = validator in FLOW_VALIDATORS_SET
    color = "bold cyan" if is_flow else "green"

    # Get display name - use negated name if negate=True
    display_name = NEGATED_VALIDATOR_NAMES.get(validator, validator) if negate else validator

    # Format transform suffix
    transform_suffix = ""
    if transform:
        transform_str = " → ".join(transform)
        transform_suffix = f" [magenta]⤳ {transform_str}[/magenta]"

    # Don't show value for flow validators (too verbose)
    if is_flow:
        if rel_path:
            return f"{status} [dim]{rel_path}.[/dim][{color}]{display_name}[/{color}]{transform_suffix}"
        else:
            return f"{status} [{color}]{display_name}[/{color}]{transform_suffix}"
    else:
        formatted_val = _format_validator_value(value)
        if rel_path:
            return f"{status} [dim]{rel_path}.[/dim][{color}]{display_name}[/{color}] [dim]({formatted_val})[/dim]{transform_suffix}"
        else:
            return f"{status} [{color}]{display_name}[/{color}] [dim]({formatted_val})[/dim]{transform_suffix}"


def _build_validator_tree(
    parent: Tree,
    validators: list,
    base_path: str = "",
    results_map: dict[tuple[str, str], ValidatorResult] | None = None,
) -> int:
    """Build a tree of validators, returning the count.

    Recursively builds a tree where flow validators (seq!, parallel!, any!)
    contain their steps as children, and each step contains its nested validators.
    """
    count = 0

    # Normalize all validators to tuples
    normalized = [_normalize_validator(v) for v in validators]

    # Filter to only validators that belong to this subtree
    def belongs_to_path(target: str, path: str) -> bool:
        if not path:
            return True
        return target == path or target.startswith(path + ".")

    subtree = [(t, v, val, neg, trans) for t, v, val, neg, trans in normalized if belongs_to_path(t, base_path)]

    # Find the root flow validator for this level (matching base_path exactly)
    root_flow = None
    for target, validator, value, neg, trans in subtree:
        if target == base_path and validator in ("seq!", "parallel!", "any!"):
            root_flow = (target, validator, value, neg, trans)
            break

    if root_flow:
        target, validator, steps, neg, trans = root_flow
        flow_line = _format_validator_status(target, validator, steps, results_map, "", neg, trans)
        branch = parent.add(flow_line)

        # Track which validators we've handled
        handled = set()
        handled.add((target, validator))

        # Extract step names for path building (handles both strings and dicts)
        step_names = []
        for step in steps:
            if isinstance(step, str):
                step_names.append(step)
            elif isinstance(step, dict) and len(step) == 1:
                step_key = next(iter(step.keys()))
                if step_key in ("seq!", "parallel!", "any!"):
                    # Nested flow validator - don't add as path segment
                    pass
                else:
                    step_names.append(step_key)

        # Get parent flow validator result to propagate to nested flow validators
        parent_result = results_map.get((target, validator)) if results_map else None

        for step in steps:
            # Handle nested flow validators (e.g., parallel! inside seq!)
            if isinstance(step, dict) and len(step) == 1:
                step_key, step_value = next(iter(step.items()))
                if step_key in ("seq!", "parallel!", "any!"):
                    # This is a nested flow validator
                    # Derive status from parent - if parent failed, nested likely contributed
                    if parent_result is not None:
                        if parent_result.passed:
                            status = "[bold green]✓[/bold green]"
                        else:
                            status = "[bold red]✗[/bold red]"
                    else:
                        status = "[dim]○[/dim]"
                    nested_flow_line = f"{status} [bold cyan]{step_key}[/bold cyan]"
                    nested_branch = branch.add(nested_flow_line)

                    # Add the nested flow's steps
                    if isinstance(step_value, list):
                        for nested_step in step_value:
                            if isinstance(nested_step, str):
                                nested_branch.add(f"[yellow]{nested_step}[/yellow]")
                            elif isinstance(nested_step, dict) and len(nested_step) == 1:
                                nested_step_name = next(iter(nested_step.keys()))
                                nested_step_spec = nested_step[nested_step_name]
                                nested_step_branch = nested_branch.add(f"[yellow]{nested_step_name}[/yellow]")

                                # Add validators for this nested step
                                nested_step_path = f"{base_path}.{nested_step_name}"
                                nested_step_subtree = [
                                    (t, v, val, neg, trans)
                                    for t, v, val, neg, trans in subtree
                                    if belongs_to_path(t, nested_step_path)
                                ]

                                # Check if this nested step has a flow validator
                                has_flow = any(
                                    t == nested_step_path and v in ("seq!", "parallel!", "any!")
                                    for t, v, val, neg, trans in nested_step_subtree
                                )

                                if has_flow:
                                    # Recursively build subtree for nested flow
                                    count += _build_validator_tree(
                                        nested_step_branch, validators, nested_step_path, results_map
                                    )
                                    for t, v, val, neg, trans in nested_step_subtree:
                                        handled.add((t, v))
                                else:
                                    # Add non-flow validators only
                                    for t, v, val, neg, trans in nested_step_subtree:
                                        if t != nested_step_path and v not in ("seq!", "parallel!", "any!"):
                                            prop_path = t[len(nested_step_path) :].lstrip(".")
                                            line = _format_validator_status(
                                                t, v, val, results_map, prop_path, neg, trans
                                            )
                                            nested_step_branch.add(line)
                                            handled.add((t, v))
                                            count += 1
                    continue

                # Regular span with nested validators
                step_name = step_key
                has_nested_spec = True  # This step is a dict, so it has nested spec
            else:
                step_name = step
                has_nested_spec = False  # This step is just a string, no nested spec

            step_path = f"{base_path}.{step_name}" if base_path else step_name
            step_branch = branch.add(f"[yellow]{step_name}[/yellow]")

            # Only show nested validators if this step was defined with nested spec
            if has_nested_spec:
                # Get validators that belong to this step only
                step_subtree = [
                    (t, v, val, neg, trans) for t, v, val, neg, trans in subtree if belongs_to_path(t, step_path)
                ]

                # Check if this step has a nested flow validator
                nested_flow = None
                for t, v, val, neg, trans in step_subtree:
                    if t == step_path and v in ("seq!", "parallel!", "any!"):
                        nested_flow = (t, v, val, neg, trans)
                        break

                if nested_flow:
                    # Recursively build subtree for nested flow
                    count += _build_validator_tree(step_branch, validators, step_path, results_map)
                    for t, v, val, neg, trans in step_subtree:
                        handled.add((t, v))
                else:
                    # Add value validators for this step
                    for t, v, val, neg, trans in step_subtree:
                        if v in ("seq!", "parallel!", "any!"):
                            continue
                        if t == step_path:
                            continue  # Skip the step itself
                        prop_path = t[len(step_path) :].lstrip(".")
                        line = _format_validator_status(t, v, val, results_map, prop_path, neg, trans)
                        step_branch.add(line)
                        handled.add((t, v))
                        count += 1

        # Add validators at base level that aren't under any step
        for t, v, val, neg, trans in subtree:
            if (t, v) in handled:
                continue
            if v in ("seq!", "parallel!", "any!"):
                continue
            if t == base_path:
                continue
            # Check if under any step
            rel_path = t[len(base_path) :].lstrip(".")
            first_segment = rel_path.split(".")[0] if "." in rel_path else rel_path
            if first_segment not in step_names:
                line = _format_validator_status(t, v, val, results_map, rel_path, neg, trans)
                parent.add(line)
                count += 1

    else:
        # No flow validator at this level, just add value validators
        for target, validator, value, neg, trans in subtree:
            if validator in ("seq!", "parallel!", "any!"):
                continue
            if target == base_path:
                continue
            rel_path = target[len(base_path) :].lstrip(".") if base_path else target
            line = _format_validator_status(target, validator, value, results_map, rel_path, neg, trans)
            parent.add(line)
            count += 1

    return count


def print_eval_tree(evals: list[Eval]) -> None:
    """Print a tree view of all evals and their validators."""
    console.print()

    total_validators = 0

    for eval in evals:
        # Build eval header with tags
        header = Text()
        header.append(eval.name, style="bold")

        for tag in eval.tags:
            color = _get_tag_color(tag)
            header.append(" ")
            header.append("[", style="dim")
            header.append(tag, style=color)
            header.append("]", style="dim")

        tree = Tree(header)

        # Add validators
        validators = eval._validators
        if validators:
            count = _build_validator_tree(tree, validators, eval.runnable._path)
            total_validators += count
        else:
            tree.add("[dim]no validators[/dim]")

        console.print(tree)
        console.print()

    # Summary
    eval_label = "eval" if len(evals) == 1 else "evals"
    validator_label = "validator" if total_validators == 1 else "validators"
    console.print(f"[dim]{len(evals)} {eval_label}, {total_validators} {validator_label}[/dim]")
    console.print()
