import asyncio
import contextlib
import contextvars
import io
import sys
import time
import traceback
from io import StringIO
from typing import Any

from timbal.types.message import Message

from ..state import RunContext, set_call_id, set_parent_call_id, set_run_context
from .display import OutputCapture
from .models import Eval, EvalError, EvalResult, EvalSummary, ValidatorResult
from .reporters import Reporter
from .utils import resolve_target
from .validators.base import BaseValidator
from .validators.context import ValidationContext
from .validators.llm_base import LLMValidator

# ---------------------------------------------------------------------------
# Task-safe output capture for parallel runs
#
# contextlib.redirect_stdout swaps sys.stdout globally, which breaks when
# multiple evals run concurrently (overlapping enter/exit from different tasks
# restores the wrong stream). Instead, we swap sys.stdout/stderr ONCE for the
# whole batch with streams that route writes to a per-task buffer held in a
# contextvar. Each eval task sets its own buffer; asyncio tasks copy the
# context at creation, so buffers never leak across evals.
# ---------------------------------------------------------------------------

_stdout_buf: contextvars.ContextVar[StringIO | None] = contextvars.ContextVar("timbal_evals_stdout_buf", default=None)
_stderr_buf: contextvars.ContextVar[StringIO | None] = contextvars.ContextVar("timbal_evals_stderr_buf", default=None)

_context_capture_installed = False


class _ContextStream(io.TextIOBase):
    """Writes to the current task's capture buffer, falling back to the real stream."""

    def __init__(self, var: contextvars.ContextVar, real: Any) -> None:
        self._var = var
        self._real = real

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        buf = self._var.get()
        target = buf if buf is not None else self._real
        return target.write(s)

    def flush(self) -> None:
        buf = self._var.get()
        (buf if buf is not None else self._real).flush()


@contextlib.contextmanager
def _install_context_capture():
    """Swap sys.stdout/stderr for context-routing streams for the whole batch."""
    global _context_capture_installed
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _ContextStream(_stdout_buf, real_stdout)
    sys.stderr = _ContextStream(_stderr_buf, real_stderr)
    _context_capture_installed = True
    try:
        yield
    finally:
        _context_capture_installed = False
        sys.stdout, sys.stderr = real_stdout, real_stderr


async def run_eval(eval: Eval, capture: bool = True, ace: Any | None = None) -> EvalResult:
    """Run a single eval and return the result."""

    error: EvalError | None = None
    validator_results: list[ValidatorResult] = []
    captured_stdout = ""
    captured_stderr = ""
    cap = None
    ctx_bufs: tuple[StringIO, StringIO] | None = None

    agent_output: Any | None = None
    ace_context = []
    ace_active_policies = []
    ace_policies = []

    start_time = time.perf_counter()
    try:
        if capture:
            if _context_capture_installed:
                # Batch-level capture is active: just bind fresh buffers for this task.
                ctx_bufs = (StringIO(), StringIO())
                _stdout_buf.set(ctx_bufs[0])
                _stderr_buf.set(ctx_bufs[1])
            else:
                cap = OutputCapture()
                cap.__enter__()

        if ace:
            if "messages" in eval.params:
                raw = eval.params.pop("messages")
                if not isinstance(raw, list):
                    raw = [raw]
                messages = [Message.validate(m) for m in raw]
                input_key = "messages"
            elif "prompt" in eval.params:
                messages = [Message.validate(eval.params.pop("prompt"))]
                input_key = "prompt"

            ace_run_context = RunContext()
            set_run_context(ace_run_context)
            session = await ace_run_context.get_session()
            messages = await ace.handler(messages=messages)
            # messages = ace_messages
            ace_context = session.get("ace_context", [])
            ace_active_policies = session.get("ace_active_policies", [])
            ace_policies = session.get("ace_policies", [])
            # Reset for agent execution
            set_call_id(None)
            set_parent_call_id(None)

            input_kwargs = {input_key: messages[0] if input_key == "prompt" else messages}
        else:
            input_kwargs = {}

        run_context = RunContext()
        set_run_context(run_context)

        # TODO We should handle the cases where output_event.error is not None
        agent_output = await eval.runnable(**input_kwargs, **eval.params).collect() # type: ignore

        trace = run_context._trace
        validation_context = ValidationContext(trace=trace)

        # Run validators and collect results
        for validator in eval._validators:
            if isinstance(validator, BaseValidator):
                # Capture actual_value for LLM validators before validation
                actual_value = None
                if isinstance(validator, LLMValidator):
                    _, resolved_value = resolve_target(trace, validator.target, validator.path_key)
                    if isinstance(resolved_value, Message):
                        resolved_value = resolved_value.collect_text()
                    if isinstance(resolved_value, str):
                        resolved_value = validator.apply_transform(resolved_value)
                    actual_value = resolved_value

                try:
                    await validator(validation_context)
                    validator_results.append(
                        ValidatorResult(
                            target=validator.target,
                            name=validator.name,
                            value=validator.value,
                            path_key=validator.path_key,
                            passed=True,
                            actual_value=actual_value,
                        )
                    )
                except Exception as e:
                    validator_results.append(
                        ValidatorResult(
                            target=validator.target,
                            name=validator.name,
                            value=validator.value,
                            path_key=validator.path_key,
                            passed=False,
                            error=f"{type(e).__name__}: {e}",
                            traceback=traceback.format_exc(),
                            actual_value=actual_value,
                        )
                    )

    except Exception as e:
        error = EvalError(
            type=type(e).__name__,
            message=str(e),
            traceback=traceback.format_exc(),
        )
    finally:
        if cap:
            cap.__exit__(None, None, None)
            captured_stdout = cap.get_stdout()
            captured_stderr = cap.get_stderr()
        elif ctx_bufs is not None:
            captured_stdout = ctx_bufs[0].getvalue()
            captured_stderr = ctx_bufs[1].getvalue()
            _stdout_buf.set(None)
            _stderr_buf.set(None)

    duration = time.perf_counter() - start_time

    # Add results for invalid validators that were not executed
    for invalid_validator in eval._invalid_validators:
        validator_results.append(
            ValidatorResult(
                target=invalid_validator["target"],
                name=invalid_validator["name"],
                value=invalid_validator["value"],
                path_key=invalid_validator["path_key"],
                passed=False,
                error="Validator is invalid and was not executed",
                evaluated=False,
            )
        )

    # Eval passes if no error and all validators passed
    all_validators_passed = all(vr.passed for vr in validator_results)
    passed = error is None and all_validators_passed

    return EvalResult(
        eval=eval,
        passed=passed,
        agent_output=agent_output,
        duration=duration,
        error=error,
        validator_results=validator_results,
        captured_stdout=captured_stdout,
        captured_stderr=captured_stderr,
        ace_context=ace_context,
        ace_active_policies=ace_active_policies,
        ace_policies=ace_policies,
    )


async def run_evals(
    evals: list[Eval],
    capture: bool = True,
    reporter: Reporter | None = None,
    max_concurrency: int = 1,
) -> EvalSummary:
    """Run all evals and return summary.

    Args:
        evals: The evals to run.
        capture: Capture stdout/stderr produced by each eval.
        reporter: Receives lifecycle callbacks. Defaults to the pretty
            (Rich terminal) reporter. `reporter.result()` fires as each eval
            completes — in completion order when max_concurrency > 1.
        max_concurrency: Run up to this many evals concurrently.
    """
    if reporter is None:
        from .reporters import PrettyReporter

        reporter = PrettyReporter()

    summary = EvalSummary()

    reporter.start(evals)

    if max_concurrency <= 1:
        for eval in evals:
            # TODO Add the optional env variables to eval
            result = await run_eval(eval, capture=capture)
            summary.results.append(result)
            await reporter.result(result)
    else:
        capture_cm = _install_context_capture() if capture else contextlib.nullcontext()
        with capture_cm:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _run_one(e: Eval) -> EvalResult:
                async with semaphore:
                    return await run_eval(e, capture=capture)

            tasks = [asyncio.create_task(_run_one(e)) for e in evals]
            try:
                # Report results in completion order (streaming), not file order.
                for future in asyncio.as_completed(tasks):
                    result = await future
                    summary.results.append(result)
                    await reporter.result(result)
            finally:
                for task in tasks:
                    task.cancel()

    reporter.finish(summary)

    return summary
