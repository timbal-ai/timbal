from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .models import Eval, EvalResult, EvalSummary

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
    console.print(f"[dim]collected {len(evals)} evals from {len(files)} file(s)[/dim]")
    console.print()


def print_eval_line(result: EvalResult) -> None:
    """Print a single eval result line (pytest-style)."""
    # Format: path/to/file.yaml::eval_name PASSED/FAILED [duration]
    path = result.eval.path
    name = result.eval.name
    duration_str = f"{result.duration:.2f}s"

    location = Text()
    location.append(str(path), style="dim")
    location.append("::", style="dim")
    location.append(name, style="bold")

    if result.passed:
        status = Text(" PASSED ", style="bold white on green")
    else:
        status = Text(" FAILED ", style="bold white on red")

    # Build the line
    line = Text()
    line.append_text(status)
    line.append(" ")
    line.append_text(location)
    line.append(f" [{duration_str}]", style="dim")

    console.print(line)


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
