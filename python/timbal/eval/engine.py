import inspect
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
from .validators import semantic_output

logger = structlog.get_logger("timbal.eval.engine")



async def eval_steps(turn: Turn, agent: Agent) -> tuple[bool, list[str], list[dict]]:
    """"""
    try:
        # If turn has no steps field, no step validation needed
        if turn.steps is None:
            return True, [], []
            
        run_context = get_run_context()
        if not run_context or not run_context._trace:
            return False, ["No trace data available"], []
            
        # Extract tool calls from trace data
        actual_steps = []
        
        # First, extract from trace data (for explicit tool calls)
        for call_id, span in run_context._trace.items():
            path = span.path
            # Skip the root agent call (call_id=None or no path separators)
            if call_id is None or not path or "." not in path:
                continue
                
            path_parts = path.split(".")
            if len(path_parts) >= 2:
                # The tool name is the second part (after "agent")
                tool_name = path_parts[1]
                
                # Skip LLM calls - we only want actual tools that the LLM calls
                if tool_name == "llm":
                    continue
                
                tool_input = span.input or {}
                actual_steps.append({
                    "tool": tool_name,
                    "input": tool_input
                })
        
        # Second, extract from usage data (for tools tracked via usage, like web_search_requests)
        if hasattr(run_context, '_last_usage') and run_context._last_usage:
            for key, value in run_context._last_usage.items():
                if '_requests' in key or '_calls' in key:
                    # Extract tool name from key like "gpt-4.1-mini:web_search_requests"
                    parts = key.split(':')
                    if len(parts) == 2:
                        tool_key = parts[1]
                        if tool_key.endswith('_requests'):
                            tool_name = tool_key.replace('_requests', '')
                        elif tool_key.endswith('_calls'):
                            tool_name = tool_key.replace('_calls', '')
                        else:
                            tool_name = tool_key
                        
                        # Only add if not already in actual_steps
                        if not any(step.get("tool") == tool_name for step in actual_steps) and value > 0:
                            actual_steps.append({
                                "tool": tool_name,
                                "usage_count": value
                            })

        explanations = []
        results = []
        if turn.steps and turn.steps.validators:
            for i, validator in enumerate(turn.steps.validators):
                if hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func):
                    try:
                        await validator(actual_steps)
                        correct = True
                        logger.info("eval_steps_validator_result", validator_index=i, result="passed")
                    except EvalError as e:
                        correct = False
                        explanations.append(str(e))
                        logger.info("eval_steps_validator_result", validator_index=i, result="failed", error=str(e))
                else:
                    try:
                        validator(actual_steps)
                        correct = True
                        logger.info("eval_steps_validator_result", validator_index=i, result="passed")
                    except EvalError as e:
                        correct = False
                        explanations.append(str(e))
                        logger.info("eval_steps_validator_result", validator_index=i, result="failed", error=str(e))
                results.append(correct)
        
        final_result = all(results) if results else True
        logger.info("eval_steps_final_result", results=results, final_result=final_result, explanations=explanations)
        return final_result, explanations, actual_steps
    except Exception as e:
        # If there's an error accessing agent memory or evaluating steps
        error_msg = f"Error evaluating steps: {str(e)}"
        logger.error("eval_steps_exception", error=error_msg, exception=str(e))
        return False, [error_msg], []


async def eval_input(
    turn: Turn,
    actual_input: dict
) -> tuple[bool | None, list[str]]:
    """Evaluate the input keys of a turn using per-key validators.
    Validates keys that may be set in the input definition or by pre_hooks/agent.
    Gets actual values from run_context after pre_hook execution.
    Returns (False, explanations) if any validation fails, (True, []) if all pass.
    """
    explanations = []
    results = []
    
    if not turn.input or not turn.input.validators:
        # No validators means nothing to validate - return None equivalent (handled by caller)
        return None, []
    
    # Get per-key validators
    input_validators = turn.input.validators
    if not isinstance(input_validators, dict):
        # Not a dict means no per-key validators - return None (handled by caller)
        return None, []
    
    # Get all actual input values from run_context (after pre_hook execution)
    # This includes values set by pre_hooks
    all_actual_values = {}
    
    # First, get from turn.input (the definition)
    if turn.input:
        input_dict = turn.input.model_dump(exclude={"validators"})
        all_actual_values.update(input_dict)
    
    # Then, get from run_context (after pre_hook modifications)
    run_context = get_run_context()
    if run_context:
        agent_span = run_context.current_span()
        if agent_span and hasattr(agent_span, 'input'):
            # Get the actual input after pre_hook modifications
            context_input = agent_span.input
            if isinstance(context_input, dict):
                # Merge context input (takes precedence as it's after pre_hook)
                all_actual_values.update(context_input)
            elif context_input is not None:
                # If input is not a dict, try to extract values
                all_actual_values["_raw_input"] = context_input
    
    # Finally, merge with actual_input (from to_dict, as fallback)
    all_actual_values.update(actual_input)
    
    # Remove None values
    all_actual_values = {k: v for k, v in all_actual_values.items() if v is not None}
    
    # Validate each key that has validators
    for key, validators in input_validators.items():
        # Get the actual value for this key
        actual_value = all_actual_values.get(key)
        if actual_value is None:
            # Try to get from prompt if key is prompt
            if key == "prompt":
                actual_value = all_actual_values.get("prompt")
        
        # If still None, the key might not exist - create empty string for validation
        if actual_value is None:
            actual_value = ""
        
        # Convert prompt list to string for validation
        if key == "prompt" and isinstance(actual_value, list):
            # Join list items, converting Files to their paths
            text_parts = []
            for item in actual_value:
                if hasattr(item, 'path'):
                    text_parts.append(str(item.path))
                else:
                    text_parts.append(str(item))
            actual_value = " ".join(text_parts)
        
        # Create a message-like object for validation (validators expect Message)
        # For input validation, we validate the string value
        from ..types.content import TextContent
        validation_message = Message(role="user", content=[TextContent(text=str(actual_value))])
        
        for validator in validators:
            if hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func):
                try:
                    await validator(validation_message)
                    correct = True
                except EvalError as e:
                    correct = False
                    explanations.append(f"Input key '{key}': {str(e)}")
            else:
                try:
                    validator(validation_message)
                    correct = True
                except EvalError as e:
                    correct = False
                    explanations.append(f"Input key '{key}': {str(e)}")
            results.append(correct)
    
    return all(results), explanations


async def eval_output(
    turn: Turn,
    agent_output_message: Message
) -> tuple[bool | None, list[str]]:
    """Evaluate the output of a turn using validators."""
    explanations = []
    results = []

    if not turn.output or not turn.output.validators:
        # No validators means nothing to validate
        return None, []
    
    validators = turn.output.validators
    
    # Top-level validators - validate the entire message
    if isinstance(validators, list):
        validator_list = validators
    else:
        # Shouldn't happen with new structure, but handle it
        validator_list = []
    
    for validator in validator_list:
        if hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func):
            try:
                await validator(agent_output_message)
                correct = True
            except EvalError as e:
                correct = False
                explanations.append(str(e))
        else:
            try:
                validator(agent_output_message)
                correct = True
            except EvalError as e:
                correct = False
                explanations.append(str(e))
        results.append(correct)
        
    return all(results), explanations


def _find_matching_model(model_prefix: str, actual_usage: dict) -> tuple[str | None, str | None]:
    """
    Find matching model in actual_usage using prefix matching.
    Returns (matched_model, error_message) where error_message is set if multiple matches found.
    """
    # Extract all model names from actual_usage keys (format: model:usage_type)
    model_names = set()
    for key in actual_usage.keys():
        if ':' in key:
            model_name = key.split(':', 1)[0]
            model_names.add(model_name)
    
    # Find models that start with the prefix
    matching_models = [model for model in model_names if model.startswith(model_prefix)]
    
    if len(matching_models) == 0:
        return None, None
    elif len(matching_models) == 1:
        return matching_models[0], None
    else:
        return None, f"Multiple models match prefix '{model_prefix}': {', '.join(matching_models)}"


def eval_usage(
    turn: Turn,
    actual_usage: dict
) -> list[dict]:
    """
    For each model:key in turn.usage, compare with the corresponding value in actual_usage.
    Returns a list of dicts with comparison results.
    Flattens keys as model:usage_type (e.g., gpt-4.1:input_text_tokens)
    Supports prefix matching for model names - if model is 'gpt-4.1', it will match 'gpt-4.1-2025-04-14'.
    """
    if turn.usage is None:
        return []
    
    results = []
    # Handle both list and dict for turn.usage
    usage_items = []
    if isinstance(turn.usage, dict):
        usage_items = turn.usage.items()
    elif isinstance(turn.usage, list):
        for item in turn.usage:
            if isinstance(item, dict):
                for model, usage_types in item.items():
                    for usage_type, limits in usage_types.items():
                        usage_items.append((model, usage_type, limits))
    else:
        return []

    for model, usage_type, limits in usage_items:
        max_value = limits.get("max")
        min_value = limits.get("min")
        

        # Find the actual model name using prefix matching
        matched_model, prefix_error = _find_matching_model(model, actual_usage)
        
        if prefix_error:
            # Handle multiple matches error
            result = {
                "usage_key": f"{model}:{usage_type}",
                "max": max_value,
                "min": min_value,
                "actual": None,
                "correct": False,
                "explanation": prefix_error
            }
            results.append(result)
            continue
        
        if matched_model is None:
            # No matching model found
            usage_key = f"{model}:{usage_type}"
            actual_value = None
        else:
            # Use the matched model name
            usage_key = f"{matched_model}:{usage_type}"
            if "+" in usage_type:
                keys = usage_type.split("+")
                actual_value = sum(actual_usage.get(f"{matched_model}:{k}", 0) for k in keys)
            else:
                actual_value = actual_usage.get(f"{matched_model}:{usage_type}")
        
        # Update usage_key for display purposes to show matched model name
        if matched_model:
            display_usage_key = f"{matched_model}:{usage_type}"
        else:
            display_usage_key = f"{model}:{usage_type}"
        

        if max_value is not None and actual_value is not None and actual_value > max_value:
            correct = False
            explanation = f"Actual value {actual_value} is greater than max value {max_value} for {display_usage_key}."
        elif min_value is not None and actual_value is not None and actual_value < min_value:
            correct = False
            explanation = f"Actual value {actual_value} is less than min value {min_value} for {display_usage_key}."
        elif actual_value is not None:
            correct = True
            if min_value is not None and max_value is not None:
                explanation = f"Actual value {actual_value} is between min value {min_value} and max value {max_value} for {display_usage_key}."
            elif min_value is not None:
                explanation = f"Actual value {actual_value} is greater than or equal to min value {min_value} for {display_usage_key}."
            elif max_value is not None:
                explanation = f"Actual value {actual_value} is less than or equal to max value {max_value} for {display_usage_key}."
            else:
                explanation = f"Actual value {actual_value} for {display_usage_key} (no min/max set)."
        else:
            correct = False
            explanation = f"No actual value found for {display_usage_key}."
        result = {
            "usage_key": display_usage_key,
            "max": max_value,
            "min": min_value,
            "actual": actual_value,
            "correct": correct,
            "explanation": explanation
        }
        results.append(result)

    return results


async def run_turn(
    agent: Agent,
    turn: Turn,
    test: Test,
    conversation_history: list[Message],
    test_results: EvalTestSuiteResult,
    test_file_name: str,
    test_file_dir: Path
) -> None:
    """"""
    test_results.total_turns += 1
    user_input = turn.input
    reason = []
    execution_error = None

    # Run the agent
    try:
        # Convert Input to Message to properly handle files
        user_message = user_input.to_message(role="user", test_file_dir=test_file_dir)
        agent_output_event = await agent(messages=conversation_history, prompt=user_message).collect()
        agent_usage = agent_output_event.usage
        
        # Store usage in run_context for eval_steps to access
        run_context = get_run_context()
        run_context._last_usage = agent_usage
        if agent_output_event.output and agent_output_event.output.content and len(agent_output_event.output.content) > 0:
            agent_output = agent_output_event.output.content[0].text
        else:
            agent_output = "No output generated"
        
        # Check if there was an execution error in the agent output event
        if agent_output_event.error is not None:
            execution_error = agent_output_event.error
            reason.append("execution_error")
            test_results.execution_errors += 1
            
    except Exception as e:
        # Capture any exceptions that occur during agent execution
        execution_error = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        reason.append("execution_error")
        test_results.execution_errors += 1
        agent_usage = {}
        agent_output = f"Execution failed: {str(e)}"
        # Keep the existing run_context for error case

    # Evaluate the input keys
    input_passed = None
    input_explanations = []
    if turn.input and turn.input.validators and execution_error is None:
        actual_input_dict = user_input.to_dict()
        input_passed, input_explanations = await eval_input(turn, actual_input_dict)
        # Only count if there were actual validators to check (input_passed is not None)
        if input_passed is not None:
            if input_passed:
                test_results.inputs_passed += 1
            else:
                test_results.inputs_failed += 1
                reason.append("input")

    # Evaluate the steps
    steps_passed, steps_explanations, actual_steps = None, [], []
    if turn.steps and execution_error is None:  # Only evaluate steps if no execution error
        steps_passed, steps_explanations, actual_steps = await eval_steps(turn, agent)
        if steps_passed:
            test_results.steps_passed += 1
        else:
            test_results.steps_failed += 1
            reason.append("steps")

    output_passed = None
    output_explanations = []
    agent_output_message = Message(role="assistant", content=[TextContent(text=agent_output)])
    # Evaluate the output only if it has validators (otherwise it's a fixed record for conversation history)
    if turn.output and turn.output.validators and execution_error is None:
        output_passed, output_explanations = await eval_output(turn, agent_output_message)
        # Only count if there were actual validators to check (output_passed is not None)
        if output_passed is not None:
            if output_passed:
                test_results.outputs_passed += 1
            else:
                test_results.outputs_failed += 1
                reason.append("output")

    usage_comparisons = []
    if turn.usage:
        usage_comparisons = eval_usage(turn, agent_usage)
        if all(comparison["correct"] for comparison in usage_comparisons):
            test_results.usage_passed += 1
        else:
            test_results.usage_failed += 1
            reason.append("usage")

    if len(reason) > 0:
        test_results.tests_failed.append(
            EvalResult(
                test_name=test.name,
                test_path=f"{test_file_name}::{test.name}",
                input=user_input.to_dict(),
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
                usage_passed=all(comparison["correct"] for comparison in usage_comparisons),
                usage_explanations = [comparison["explanation"] for comparison in usage_comparisons if not comparison["correct"]],
                steps_passed=steps_passed,
                steps_explanations=steps_explanations,
                actual_steps=actual_steps,
                expected_steps=turn.steps.to_dict() if turn.steps else None,
            )
        )





async def eval_file(
    path: Path,
    _agent: Agent,
    test_results: EvalTestSuiteResult,
    test_name: str | None = None
) -> Any:
    """Parse and run all the tests in the given file. Optionally, only run a specific test by name."""
    with open(path) as f:
        test_suite = yaml.safe_load(f)

    test_suite = TestSuite.model_validate(test_suite)
    
    test_results.total_files += 1

    test_file_dir = path.parent
    
    for test in test_suite.tests:
        if test_name is not None and test.name != test_name:
            continue
        test_results.total_tests += 1
        conversation_history = []
        for turn in test.turns:
            await run_turn(_agent, turn, test, conversation_history, test_results, str(path.name), test_file_dir)
            # Add to conversation history after processing
            conversation_history.append(turn.input.to_message(role="user", test_file_dir=test_file_dir))
            # Only add expected output to conversation history if there are no validators
            if turn.output and not turn.output.validators:
                conversation_history.append(turn.output.to_message(role="assistant", test_file_dir=test_file_dir))
    
    return test_results