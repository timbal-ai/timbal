import inspect
import time
import uuid
from pathlib import Path
from typing import Any
import traceback

import structlog
import yaml

from ..core.agent import Agent
from ..errors import EvalError
from ..state.context import RunContext
from ..state.data import DataValue
from ..state.savers import InMemorySaver
from ..state.snapshot import Snapshot
from ..state import set_run_context
from ..types.chat.content import TextContent
from ..types.message import Message
from .types.result import EvalResult, EvalTestSuiteResult
from .types.test import Test
from .types.test_suite import TestSuite
from .types.turn import Turn
from .validators import semantic_output

logger = structlog.get_logger("timbal.eval.engine")



async def eval_steps(
    turn: Turn,
    agent: Agent,
    run_context: RunContext
) -> tuple[bool, list[str], list[dict]]:
    """"""
    try:
        if agent.state_saver is None:
            return False, ["No state saver available"], []
            
        last_snapshot = await agent.state_saver.get_last(path=agent.path)
        if last_snapshot is None:
            return False, ["No snapshot found"], []
            
        agent_memory = last_snapshot.data["memory"].resolve()
        actual_steps = []
        for message in agent_memory:
            if isinstance(message, dict):
                message = Message.validate(message)
            if message.content and len(message.content) > 0 and message.content[0].type == "tool_use":
                actual_steps.append(
                    {
                        "tool": message.content[0].name,
                        "input": message.content[0].input
                    }
                )

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


def eval_usage(
    turn: Turn,
    actual_usage: dict
) -> list[dict]:
    """
    For each model:key in turn.usage, compare with the corresponding value in actual_usage.
    Returns a list of dicts with comparison results.
    Flattens keys as model:usage_type (e.g., gpt-4.1:input_text_tokens)
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
        if "+" in usage_type:
            keys = usage_type.split("+")
            actual_value = sum(actual_usage.get(f"{model}:{k}", 0) for k in keys)
            usage_key = f"{model}:{usage_type}"
        else:
            actual_value = actual_usage.get(f"{model}:{usage_type}")
            usage_key = f"{model}:{usage_type}"
        if max_value is not None and actual_value is not None and actual_value > max_value:
            correct = False
            explanation = f"Actual value {actual_value} is greater than max value {max_value} for {usage_key}."
        elif min_value is not None and actual_value is not None and actual_value < min_value:
            correct = False
            explanation = f"Actual value {actual_value} is less than min value {min_value} for {usage_key}."
        elif actual_value is not None:
            correct = True
            if min_value is not None and max_value is not None:
                explanation = f"Actual value {actual_value} is between min value {min_value} and max value {max_value} for {usage_key}."
            elif min_value is not None:
                explanation = f"Actual value {actual_value} is greater than or equal to min value {min_value} for {usage_key}."
            elif max_value is not None:
                explanation = f"Actual value {actual_value} is less than or equal to max value {max_value} for {usage_key}."
            else:
                explanation = f"Actual value {actual_value} for {usage_key} (no min/max set)."
        else:
            correct = False
            explanation = f"No actual value found for {usage_key}."
        result = {
            "usage_key": usage_key,
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
    
    # Add memory to the agent
    data = {"memory": DataValue(value=conversation_history)}
    snapshot = Snapshot(
        v="0.2.0",
        id=str(uuid.uuid4()),
        path=agent.path,
        input=user_input,
        t0=int(time.time() * 1000),
        data=data,
    )
    await agent.state_saver.put(snapshot)
    run_context = RunContext(parent_id=snapshot.id)
    set_run_context(run_context)

    # Run the agent
    try:
        agent_output_event = await agent.complete(context=run_context, prompt=user_input)
        agent_usage = agent_output_event.usage
        if agent_output_event.output and agent_output_event.output.content and len(agent_output_event.output.content) > 0:
            agent_output = agent_output_event.output.content[0].text
        else:
            agent_output = "No output generated"
        run_context = RunContext(parent_id=agent_output_event.run_id)
        
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
        run_context = RunContext(parent_id=snapshot.id)

    # Evaluate the steps
    steps_passed, steps_explanations, actual_steps = None, [], []
    if turn.steps and execution_error is None:  # Only evaluate steps if no execution error
        steps_passed, steps_explanations, actual_steps = await eval_steps(turn, agent, run_context)
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
    agent: Agent,
    test_results: EvalTestSuiteResult,
    test_name: str | None = None
) -> Any:
    """Parse and run all the tests in the given file. Optionally, only run a specific test by name."""
    with open(path) as f:
        test_suite = yaml.safe_load(f)

    test_suite = TestSuite.model_validate(test_suite)
    
    for test in test_suite.tests:
        # Reset the state saver for each test
        agent.state_saver = InMemorySaver()
        if test_name is not None and test.name != test_name:
            continue
        conversation_history = []
        for i, turn in enumerate(test.turns):
            is_last_turn = (i == len(test.turns) - 1)
            if is_last_turn:
                await run_turn(agent, turn, test, conversation_history, test_results, str(path.name))
            else:
                conversation_history.append(turn.input.to_message(role="user"))
                if turn.output:
                    conversation_history.append(turn.output.to_message(role="assistant"))
    
    return test_results