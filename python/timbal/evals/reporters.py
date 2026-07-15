import json
import sys
from typing import Any, TextIO

from .models import Eval, EvalResult, EvalSummary
from .utils import dump_result


class Reporter:
    """Receives eval lifecycle callbacks.

    `result()` is called as each eval completes — in file order for sequential
    runs, in completion order for parallel runs.
    """

    def start(self, evals: list[Eval]) -> None:
        pass

    async def result(self, result: EvalResult) -> None:
        pass

    def finish(self, summary: EvalSummary) -> None:
        pass


class PrettyReporter(Reporter):
    """Human-readable Rich output (the default CLI experience)."""

    def __init__(self) -> None:
        from . import display  # deferred: pulls in rich

        self._display = display

    def start(self, evals: list[Eval]) -> None:
        self._display.print_header(evals)

    async def result(self, result: EvalResult) -> None:
        self._display.print_eval_result(result)

    def finish(self, summary: EvalSummary) -> None:
        self._display.print_failures(summary.results)
        self._display.print_summary(summary)


class JsonReporter(Reporter):
    """Streams one JSON object per line (JSONL) as evals complete.

    Events:
        {"event": "start", "total": N, "evals": [{"name", "path"}, ...]}
        {"event": "result", ...}   # one per eval, in completion order
        {"event": "summary", "total", "passed", "failed", "total_duration"}
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        # Grab the stream reference eagerly: output capture swaps sys.stdout
        # globally while evals run, and events must reach the real stream.
        self._stream = stream if stream is not None else sys.stdout

    def _emit(self, obj: dict[str, Any]) -> None:
        self._stream.write(json.dumps(obj, default=str) + "\n")
        self._stream.flush()

    def start(self, evals: list[Eval]) -> None:
        self._emit({
            "event": "start",
            "total": len(evals),
            "evals": [{"name": e.name, "path": str(e.path)} for e in evals],
        })

    async def result(self, result: EvalResult) -> None:
        self._emit({"event": "result", **await dump_result(result)})

    def finish(self, summary: EvalSummary) -> None:
        self._emit({
            "event": "summary",
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "total_duration": summary.total_duration,
        })
