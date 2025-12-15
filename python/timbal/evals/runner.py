import time
import traceback

from ..state import RunContext, set_run_context
from .display import (
    OutputCapture,
    print_eval_result,
    print_failures,
    print_header,
    print_summary,
)
from .models import Eval, EvalError, EvalResult, EvalSummary, ValidatorResult
from .validators.base import BaseValidator
from .validators.context import ValidationContext


async def run_eval(eval: Eval, capture: bool = True) -> EvalResult:
    """Run a single eval and return the result."""
    run_context = RunContext()  # type: ignore[call-arg]
    set_run_context(run_context)

    error: EvalError | None = None
    validator_results: list[ValidatorResult] = []
    captured_stdout = ""
    captured_stderr = ""
    cap = None

    start_time = time.perf_counter()
    try:
        if capture:
            cap = OutputCapture()
            cap.__enter__()

        # TODO We should handle the cases where output_event.error is not None
        _ = await eval.runnable(**eval.params).collect()  # type: ignore

        trace = run_context._trace
        validation_context = ValidationContext(trace=trace)

        # Run validators and collect results
        for validator in eval._validators:
            if isinstance(validator, BaseValidator):
                try:
                    await validator(validation_context)
                    validator_results.append(
                        ValidatorResult(
                            target=validator.target,
                            name=validator.name,
                            value=validator.value,
                            passed=True,
                        )
                    )
                except Exception as e:
                    validator_results.append(
                        ValidatorResult(
                            target=validator.target,
                            name=validator.name,
                            value=validator.value,
                            passed=False,
                            error=f"{type(e).__name__}: {e}",
                            traceback=traceback.format_exc(),
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

    duration = time.perf_counter() - start_time

    # Eval passes if no error and all validators passed
    all_validators_passed = all(vr.passed for vr in validator_results)
    passed = error is None and all_validators_passed

    return EvalResult(
        eval=eval,
        passed=passed,
        duration=duration,
        error=error,
        validator_results=validator_results,
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

        print_eval_result(result)

    # Print failure details at the end (pytest-style)
    print_failures(summary.results)

    print_summary(summary)

    return summary
