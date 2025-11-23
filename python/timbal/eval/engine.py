import inspect
import time
import traceback
from pathlib import Path
from typing import Any

import structlog
import yaml

from ..core.agent import Agent
from ..errors import EvalError
from ..state import get_run_context
from ..types.content import TextContent
from ..types.message import Message
from .types.result import EvalResult, EvalTestSuiteResult
from .types.test import Test
from .types.test_suite import TestSuite
from .types.turn import Turn

logger = structlog.get_logger("timbal.eval.engine")


async def eval_steps(turn: Turn, test_results: EvalTestSuiteResult | None = None) -> tuple[bool, list[str], list[dict]]:
    """Evaluate the steps taken by the agent during a turn.
    
    Extracts actual steps from the run context trace and validates them against
    the expected steps defined in turn.steps.validators.
    
    Args:
        turn: The turn object containing expected steps and validators
        test_results: Optional test results object to track validation counts
        
    Returns:
        Tuple of (success, explanations, actual_steps) where:
        - success: True if all validators pass, False otherwise
        - explanations: List of error messages if validators fail
        - actual_steps: List of actual steps extracted from trace
    """
    try:
        if turn.steps is None:
            return True, [], []
            
        run_context = get_run_context()
        if not run_context or not run_context._trace:
            return False, ["No trace data available"], []
            
        actual_steps = []
        
        for call_id, span in run_context._trace.items():
            if call_id is None or not span.path or "." not in span.path:
                continue
                
            path_parts = span.path.split(".")
            if len(path_parts) >= 2:
                tool_name = path_parts[1]
                if tool_name == "llm":
                    continue
                actual_steps.append({
                    "tool": tool_name,
                    "input": span.input or {}
                })
        
        if hasattr(run_context, '_last_usage') and run_context._last_usage:
            for key, value in run_context._last_usage.items():
                if ('_requests' in key or '_calls' in key) and value > 0:
                    parts = key.split(':')
                    if len(parts) == 2:
                        tool_key = parts[1]
                        tool_name = tool_key.replace('_requests', '').replace('_calls', '')
                        if not any(step.get("tool") == tool_name for step in actual_steps):
                            actual_steps.append({
                                "tool": tool_name,
                                "usage_count": value
                            })
        
        steps_total_usage = {}
        for call_id, span in run_context._trace.items():
            if call_id is None or not span.path or "." not in span.path:
                continue
            path_parts = span.path.split(".")
            if len(path_parts) >= 2 and path_parts[1] != "llm" and span.usage:
                for key, value in span.usage.items():
                    steps_total_usage[key] = steps_total_usage.get(key, 0) + value
        
        if hasattr(run_context, '_last_usage') and run_context._last_usage:
            for key, value in run_context._last_usage.items():
                if '_requests' in key or '_calls' in key:
                    steps_total_usage[key] = steps_total_usage.get(key, 0) + value
        
        if actual_steps:
            run_context._last_steps_usage = steps_total_usage
        
        if test_results is not None and run_context:
            run_context._eval_test_results = test_results
        
        explanations = []
        results = []
        if turn.steps and turn.steps.validators:
            for i, validator in enumerate(turn.steps.validators):
                if test_results is not None:
                    test_results.total_validations += 1
                
                is_async = hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func)
                try:
                    if is_async:
                        await validator(actual_steps)
                    else:
                        validator(actual_steps)
                    results.append(True)
                except EvalError as e:
                    explanations.append(str(e))
                    results.append(False)
                    logger.info("eval_steps_validator_result", validator_index=i, result="failed", error=str(e))
        
        final_result = all(results) if results else True
        logger.info("eval_steps_final_result", results=results, final_result=final_result, explanations=explanations)
        return final_result, explanations, actual_steps
    except Exception as e:
        error_msg = f"Error evaluating steps: {str(e)}"
        logger.error("eval_steps_exception", error=error_msg, exception=str(e))
        return False, [error_msg], []


async def eval_input(
    turn: Turn,
    actual_input: dict,
    test_results: EvalTestSuiteResult | None = None
) -> tuple[bool | None, list[str]]:
    """Validate the input provided to the agent during a turn.
    
    Validates top-level validators (like usage) and per-key validators
    (like content validation for specific input fields).
    
    Args:
        turn: The turn object containing input validators
        actual_input: The actual input dictionary to validate
        test_results: Optional test results object to track validation counts
        
    Returns:
        Tuple of (success, explanations) where:
        - success: True if all validators pass, False if any fail, None if no validators
        - explanations: List of error messages if validators fail
    """
    if not turn.input or not turn.input.validators:
        return None, []
    
    input_validators = turn.input.validators
    if not isinstance(input_validators, dict):
        return None, []
    
    top_level_validators = {}
    per_key_validators = {}
    
    for key, validators in input_validators.items():
        if key == "usage":
            top_level_validators[key] = validators
        else:
            per_key_validators[key] = validators
    
    if test_results is not None:
        run_context = get_run_context()
        if run_context:
            run_context._eval_test_results = test_results
    
    explanations = []
    results = []
    
    for validator_name, validators in top_level_validators.items():
        if not isinstance(validators, list):
            validators = [validators]
        
        for validator in validators:
            if test_results is not None:
                test_results.total_validations += 1
            
            is_async = hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func)
            try:
                if is_async:
                    await validator(None)
                else:
                    validator(None)
                results.append(True)
            except EvalError as e:
                explanations.append(f"Input {validator_name}: {str(e)}")
                results.append(False)
    
    if per_key_validators:
        all_actual_values = {}
        
        if turn.input:
            all_actual_values.update(turn.input.model_dump(exclude={"validators"}))
        
        run_context = get_run_context()
        if run_context:
            agent_span = run_context.current_span()
            if agent_span and hasattr(agent_span, 'input'):
                context_input = agent_span.input
                if isinstance(context_input, dict):
                    all_actual_values.update(context_input)
                elif context_input is not None:
                    all_actual_values["_raw_input"] = context_input
        
        for key, value in actual_input.items():
            if not (isinstance(value, dict) and "validators" in value and len(value) == 1):
                all_actual_values[key] = value
        
        all_actual_values = {k: v for k, v in all_actual_values.items() if v is not None}
        
        for key, validators in per_key_validators.items():
            if not isinstance(validators, list):
                validators = [validators]
            
            actual_value = all_actual_values.get(key) or ""
            
            if key == "prompt" and isinstance(actual_value, list):
                text_parts = []
                for item in actual_value:
                    text_parts.append(str(getattr(item, 'path', item)))
                actual_value = " ".join(text_parts)
            
            validation_message = Message(role="user", content=[TextContent(text=str(actual_value))])
            
            for validator in validators:
                if test_results is not None:
                    test_results.total_validations += 1
                
                is_async = hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func)
                try:
                    if is_async:
                        await validator(validation_message)
                    else:
                        validator(validation_message)
                    results.append(True)
                except EvalError as e:
                    explanations.append(f"Input key '{key}': {str(e)}")
                    results.append(False)
    
    return all(results), explanations


async def eval_output(
    turn: Turn,
    agent_output_message: Message,
    test_results: EvalTestSuiteResult | None = None
) -> tuple[bool | None, list[str]]:
    """Validate the output produced by the agent during a turn.
    
    Validates top-level validators (like time, usage) and per-key validators
    (like content validation for output fields).
    
    Args:
        turn: The turn object containing output validators
        agent_output_message: The actual message output from the agent
        test_results: Optional test results object to track validation counts
        
    Returns:
        Tuple of (success, explanations) where:
        - success: True if all validators pass, False if any fail, None if no validators
        - explanations: List of error messages if validators fail
    """
    if not turn.output or not turn.output.validators:
        return None, []
    
    validators = turn.output.validators
    if not isinstance(validators, dict):
        return None, []
    
    top_level_validators = {}
    per_key_validators = {}
    
    for key, validator_list in validators.items():
        if key in ("time", "usage"):
            top_level_validators[key] = validator_list
        else:
            per_key_validators[key] = validator_list
    
    if test_results is not None:
        run_context = get_run_context()
        if run_context:
            run_context._eval_test_results = test_results
    
    explanations = []
    results = []
    
    for validator_name, validator_list in top_level_validators.items():
        if not isinstance(validator_list, list):
            validator_list = [validator_list]
        
        validator_value = None if validator_name in ("time", "usage") else agent_output_message
        for validator in validator_list:
            if test_results is not None:
                test_results.total_validations += 1
            
            is_async = hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func)
            try:
                if is_async:
                    await validator(validator_value)
                else:
                    validator(validator_value)
                results.append(True)
            except EvalError as e:
                explanations.append(f"Output {validator_name}: {str(e)}")
                results.append(False)
    
    if per_key_validators:
        content_validators = per_key_validators.get("content", [])
        if content_validators:
            if not isinstance(content_validators, list):
                content_validators = [content_validators]
            
            for validator in content_validators:
                if test_results is not None:
                    test_results.total_validations += 1
                
                validator_name = getattr(validator, "name", "unknown")
                is_async = hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func)
                try:
                    if is_async:
                        await validator(agent_output_message)
                    else:
                        validator(agent_output_message)
                    results.append(True)
                except EvalError as e:
                    explanations.append(f"Validator {validator_name}: {str(e)}")
                    results.append(False)
        
    if not results:
        return None, []
        
    return all(results), explanations


async def run_turn(
    agent: Agent,
    turn: Turn,
    test: Test,
    conversation_history: list[Message],
    test_results: EvalTestSuiteResult,
    test_file_name: str,
    test_file_dir: Path
) -> Message | None:
    """Execute a single turn in an evaluation test.
    
    Runs the agent with the turn's input, validates the response using input/output/step
    validators, and records results. Handles execution errors and validation failures.
    
    Args:
        agent: The agent to evaluate
        turn: The turn object defining input, expected output, and validators
        test: The test case containing this turn
        conversation_history: Previous messages in the conversation
        test_results: Test results object to update with validation outcomes
        test_file_name: Name of the test file (for error reporting)
        test_file_dir: Directory of the test file (for resolving file paths)
        
    Returns:
        The agent's output message, or None if execution failed
    """
    reason = []
    execution_error = None
    agent_usage = {}
    agent_output_event = None

    start_time = time.time()
    try:
        user_message = turn.input.to_message(role="user", test_file_dir=test_file_dir)
        input_dict = turn.input.model_dump(exclude={"validators"})
        input_dict.pop("prompt", None)
        
        messages = conversation_history + [user_message]
        agent_output_event = await agent(messages=messages, **input_dict).collect()
        agent_usage = agent_output_event.usage or {}
        execution_time = time.time() - start_time
        
        run_context = get_run_context()
        if run_context:
            run_context._last_usage = agent_usage
            run_context._last_execution_time = execution_time
        
        if agent_output_event.output and agent_output_event.output.content:
            agent_output = agent_output_event.output.content[0].text
        else:
            agent_output = "No output generated"
        
        if agent_output_event.error is not None:
            execution_error = agent_output_event.error
            reason.append("execution_error")
            test_results.execution_errors += 1
        
        if execution_error is None:
            run_context = get_run_context()
            if run_context and run_context._trace:
                for call_id, span in run_context._trace.items():
                    if span.path and "." in span.path:
                        path_parts = span.path.split(".")
                        if len(path_parts) >= 2 and path_parts[1] != "llm" and span.error is not None:
                            execution_error = span.error
                            reason.append("execution_error")
                            test_results.execution_errors += 1
                            break
            
    except Exception as e:
        execution_time = time.time() - start_time
        execution_error = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        reason.append("execution_error")
        test_results.execution_errors += 1
        agent_usage = {}
        agent_output = f"Execution failed: {str(e)}"
        run_context = get_run_context()
        if run_context:
            run_context._last_usage = agent_usage
            run_context._last_execution_time = execution_time

    input_passed = None
    input_explanations = []
    if turn.input and turn.input.validators and execution_error is None:
        input_passed, input_explanations = await eval_input(turn, turn.input.to_dict(), test_results)
        if input_passed is not None:
            if input_passed:
                test_results.inputs_passed += 1
            else:
                test_results.inputs_failed += 1
                reason.append("input")

    steps_passed, steps_explanations, actual_steps = None, [], []
    if turn.steps and execution_error is None:
        steps_passed, steps_explanations, actual_steps = await eval_steps(turn, test_results)
        if steps_passed:
            test_results.steps_passed += 1
        else:
            test_results.steps_failed += 1
            reason.append("steps")
        run_context = get_run_context()
        if run_context and (not hasattr(run_context, "_last_usage") or getattr(run_context, "_last_usage", None) is None):
            run_context._last_usage = agent_usage

    output_passed = None
    output_explanations = []
    if agent_output_event and agent_output_event.output:
        agent_output_message = agent_output_event.output
    else:
        agent_output_message = Message(role="assistant", content=[TextContent(text=agent_output)])
    
    if turn.output and turn.output.validators and execution_error is None:
        run_context = get_run_context()
        if run_context and (not hasattr(run_context, "_last_usage") or getattr(run_context, "_last_usage", None) is None):
            run_context._last_usage = agent_usage
        output_passed, output_explanations = await eval_output(turn, agent_output_message, test_results)
        if output_passed is not None:
            if output_passed:
                test_results.outputs_passed += 1
            else:
                test_results.outputs_failed += 1
                reason.append("output")

    if reason:
        test_results.tests_failed.append(
            EvalResult(
                test_name=test.name,
                test_path=f"{test_file_name}::{test.name}",
                input=turn.input.to_dict(),
                reason=reason,
                execution_error=execution_error,
                input_passed=input_passed,
                input_explanations=input_explanations,
                output_passed=output_passed,
                output_explanations=output_explanations,
                actual_output={
                    "text": agent_output_message.content[0].text if agent_output_message and agent_output_message.content[0].type == "text" else None,
                    "files": [c.file for c in agent_output_message.content if c.type == "file"] if agent_output_message else None
                },
                expected_output=turn.output.to_dict() if turn.output else None,
                steps_passed=steps_passed,
                steps_explanations=steps_explanations,
                actual_steps=actual_steps,
                expected_steps=turn.steps.to_dict() if turn.steps else None,
            )
        )
    
    return agent_output_message


async def eval_file(
    path: Path,
    _agent: Agent,
    test_results: EvalTestSuiteResult,
    test_name: str | None = None
) -> Any:
    """Load and run all tests from a YAML evaluation file.
    
    Parses the YAML file, validates the test suite structure, and executes each test.
    Tests without validators are treated as context-setting turns and added to
    conversation history without validation.
    
    Args:
        path: Path to the YAML evaluation file
        _agent: The agent to evaluate
        test_results: Test results object to update with outcomes
        test_name: Optional specific test name to run (if None, runs all tests)
        
    Returns:
        The updated test_results object
    """
    with open(path) as f:
        test_suite = TestSuite.model_validate(yaml.safe_load(f))
    
    test_results.total_files += 1
    test_file_dir = path.parent
    
    for test in test_suite.tests:
        if test_name is not None and test.name != test_name:
            continue
        test_results.total_tests += 1
        conversation_history = []
        for turn in test.turns:
            has_validators = (
                (turn.input and turn.input.validators) or
                (turn.output and turn.output.validators) or
                (turn.steps and turn.steps.validators)
            )
            
            if has_validators:
                await run_turn(_agent, turn, test, conversation_history, test_results, str(path.name), test_file_dir)
            else:
                conversation_history.append(turn.input.to_message(role="user", test_file_dir=test_file_dir))
                conversation_history.append(turn.output.to_message(role="assistant", test_file_dir=test_file_dir))
    
    return test_results