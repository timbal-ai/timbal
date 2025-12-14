from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from .models import Eval, EvalResult, EvalSummary
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

    # Add validators
    validators = eval._validators
    if validators:
        _build_validator_tree(tree, validators, eval.runnable._path)

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

    if result.error:
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


def _normalize_validator(v: Any) -> tuple[str, str, Any]:
    """Normalize a validator to (target, name, value) tuple."""
    if isinstance(v, tuple):
        return v
    elif isinstance(v, BaseValidator):
        return (v.target or "", v.name, v.value)
    else:
        return ("", "unknown!", v)


def _build_validator_tree(
    parent: Tree,
    validators: list,
    base_path: str = "",
) -> int:
    """Build a tree of validators, returning the count."""
    count = 0

    # Normalize all validators to tuples
    normalized = [_normalize_validator(v) for v in validators]

    # Group validators by their immediate path segment
    flow_validators = {}  # seq!, parallel!, any!
    value_validators = []  # eq!, contains!, etc.

    for target, validator, value in normalized:
        # Get path relative to base
        rel_path = target[len(base_path) :].lstrip(".") if target.startswith(base_path) else target

        if validator in ("seq!", "parallel!", "any!"):
            flow_validators[target] = (validator, value)
        else:
            value_validators.append((rel_path, validator, value))
            count += 1

    # Add flow validators as branches
    for target, (validator, steps) in flow_validators.items():
        rel_path = target[len(base_path) :].lstrip(".") if target.startswith(base_path) else target
        branch_label = f"[bold cyan]{validator}[/bold cyan]"
        if rel_path:
            branch_label = f"[dim]{rel_path}[/dim] {branch_label}"
        branch = parent.add(branch_label)

        # Add steps as sub-branches
        for step in steps:
            step_branch = branch.add(f"[yellow]{step}[/yellow]")

            # Find validators for this step
            step_path = f"{target}.{step}"
            step_validators = [
                (t, v, val)
                for t, v, val in normalized
                if t.startswith(step_path) and v not in ("seq!", "parallel!", "any!")
            ]

            for t, v, val in step_validators:
                # Get path relative to step
                prop_path = t[len(step_path) :].lstrip(".")
                formatted_val = _format_validator_value(val)
                if prop_path:
                    step_branch.add(f"[dim]{prop_path}.[/dim][green]{v}[/green] [dim]({formatted_val})[/dim]")
                else:
                    step_branch.add(f"[green]{v}[/green] [dim]({formatted_val})[/dim]")
                count += 1

    # Add standalone value validators (not under a flow validator)
    for rel_path, validator, value in value_validators:
        # Check if this validator is already under a flow validator
        is_nested = any(
            target in rel_path.split(".")[0]
            for target in [step for _, steps in flow_validators.values() for step in steps]
        )
        if not is_nested and rel_path:
            formatted_val = _format_validator_value(value)
            parent.add(f"[dim]{rel_path}.[/dim][green]{validator}[/green] [dim]({formatted_val})[/dim]")

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
