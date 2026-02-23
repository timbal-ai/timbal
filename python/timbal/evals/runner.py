import time
import traceback
from typing import Any

from timbal.types.message import Message

from ..state import RunContext, set_call_id, set_parent_call_id, set_run_context
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
from .validators.llm_base import LLMValidator
from .utils import resolve_target


async def run_eval(eval: Eval, capture: bool = True, ace: Any | None = None) -> EvalResult:
    """Run a single eval and return the result."""
    
    error: EvalError | None = None
    validator_results: list[ValidatorResult] = []
    captured_stdout = ""
    captured_stderr = ""
    cap = None

    ace_context = []
    ace_active_policies = []
    ace_policies = []

    start_time = time.perf_counter()
    try:
        if capture:
            cap = OutputCapture()
            cap.__enter__()

        raw = eval.params.get("messages", eval.params.get("prompt", []))
        if not isinstance(raw, list):
            raw = [raw]
        messages = [Message.validate(m) for m in raw]
        
        if ace:
            ace_run_context = RunContext()
            set_run_context(ace_run_context)
            session = await ace_run_context.get_session()
            ace_messages = await ace.handler(messages=messages)
            messages = ace_messages
            ace_context = session.get("ace_context", [])
            ace_active_policies = session.get("ace_active_policies", [])
            ace_policies = session.get("ace_policies", [])
            # Reset for agent execution
            set_call_id(None)
            set_parent_call_id(None)
        

        run_context = RunContext()
        set_run_context(run_context)

        # TODO We should handle the cases where output_event.error is not None
        _ = await eval.runnable(messages=messages).collect()  # type: ignore

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
        ace_context=ace_context,
        ace_active_policies=ace_active_policies,
        ace_policies=ace_policies,
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
