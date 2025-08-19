import inspect
import traceback
from pathlib import Path
from typing import Any

import structlog
import yaml

from ..core_v2 import Agent
from ..errors import EvalError
from ..state import RunContext, get_run_context, set_run_context
from ..types.chat.content import TextContent
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
            
        # For core_v2 agents, extract steps from run context tracing
        run_context = get_run_context()
        if not run_context or not run_context.tracing:
            return False, ["No tracing data available"], []
            
        # Extract tool calls from tracing data
        actual_steps = []
        for call_id, trace in run_context.tracing.items():
            path = trace.get("path", "")
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
                
                tool_input = trace.get("input", {})
                actual_steps.append({
                    "tool": tool_name,
                    "input": tool_input
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


async def eval_output(
    turn: Turn,
    agent_output_message: Message
) -> tuple[bool, list[str]]:
    """Evaluate the output of a turn. If no validators are present, use semantic_output as default."""
    explanations = []
    results = []

    # If we have output text but no validators, use semantic validation as default
    if turn.output and not turn.output.validators and turn.output.text:
        default_validator = semantic_output(turn.output.text)
        try:
            if hasattr(default_validator, "func") and inspect.iscoroutinefunction(default_validator.func):
                await default_validator(agent_output_message)
            else:
                default_validator(agent_output_message)
            results.append(True)
        except EvalError as e:
            results.append(False)
            explanations.append(str(e))
    elif turn.output and turn.output.validators:
        # Use explicit validators if provided
        for validator in turn.output.validators:
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
    test_file_name: str
) -> None:
    """"""
    test_results.total_turns += 1
    user_input = turn.input
    reason = []
    execution_error = None
    
    # Set up run context for core_v2 with conversation history
    run_context = RunContext()
    # Store conversation history in run context data for core_v2
    if conversation_history:
        run_context.data["memory"] = conversation_history
    set_run_context(run_context)

    # Run the agent
    try:
        # Convert Input to Message to properly handle files
        user_message = user_input.to_message(role="user")
        
        # Execute agent using core_v2 interface with conversation context
        if conversation_history:
            # Create system context with conversation history
            history_text = "\n".join([
                f"{msg.role}: {msg.content[0].text if msg.content else ''}" 
                for msg in conversation_history
            ])
            system_context = f"Previous conversation:\n{history_text}\n\n{agent.instructions or ''}"
            agent_output_event = await agent(prompt=user_message, system_context=system_context).collect()
        else:
            agent_output_event = await agent(prompt=user_message).collect()
        agent_usage = agent_output_event.usage
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
    # Evaluate the output
    if turn.output and execution_error is None:  # Only evaluate output if no execution error
        output_passed, output_explanations = await eval_output(turn, agent_output_message)
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
    
    for test in test_suite.tests:
        if test_name is not None and test.name != test_name:
            continue
        test_results.total_tests += 1
        conversation_history = []
        for turn in test.turns:
            await run_turn(_agent, turn, test, conversation_history, test_results, str(path.name))
            # Add to conversation history after processing
            conversation_history.append(turn.input.to_message(role="user"))
            if turn.output:
                conversation_history.append(turn.output.to_message(role="assistant"))
    
    return test_results