import time
import traceback

from ..core.runnable import Runnable
from ..state import get_or_create_run_context, set_run_context
from .display import (
    OutputCapture,
    print_eval_line,
    print_failures,
    print_header,
    print_summary,
)
from .models import Eval, EvalError, EvalResult, EvalSummary


async def run_eval(eval: Eval, capture: bool = True) -> EvalResult:
    """Run a single eval and return the result."""
    run_context = get_or_create_run_context()
    set_run_context(run_context)

    error: EvalError | None = None
    captured_stdout = ""
    captured_stderr = ""
    cap = None

    start_time = time.perf_counter()
    try:
        if capture:
            cap = OutputCapture()
            cap.__enter__()

        # output_event = await runnable(**eval.params).collect()  # type: ignore

        # # TODO Run the validators in here
        # for validator in eval.validators:
        #     print(validator)
        #     await validator.run(run_context._trace)

        # # Check if the output event contains an error
        # if output_event is not None and getattr(output_event, "error", None) is not None:
        #     err = output_event.error
        #     error = EvalError(
        #         type=err.get("type", "UnknownError"),
        #         message=err.get("message", str(err)),
        #         traceback=err.get("traceback"),
        #     )

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

    duration = time.perf_counter() - start_time

    passed = error is None

    return EvalResult(
        eval=eval,
        passed=passed,
        duration=duration,
        error=error,
        captured_stdout=captured_stdout,
        captured_stderr=captured_stderr,
    )


async def run_evals(
    evals: list[Eval],
    capture: bool = True,
) -> EvalSummary:
    """Run all evals and return summary."""
    summary = EvalSummary()

    print_header(evals)

    for eval in evals:
        # TODO Add the optional env variables to eval
        result = await run_eval(eval, capture=capture)
        summary.results.append(result)

        print_eval_line(result)

    # Print failure details at the end (pytest-style)
    print_failures(summary.results)

    print_summary(summary)

    return summary
