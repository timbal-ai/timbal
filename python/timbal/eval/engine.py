import inspect
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
import yaml

from ..core.agent import Agent
from ..errors import EvalError
from ..state.context import RunContext
from ..state.data import DataValue
from ..state.snapshot import Snapshot
from ..types.chat.content import TextContent
from ..types.message import Message
from .types.result import EvalResult, EvalTestSuiteResult
from .types.test import Test
from .types.test_suite import TestSuite
from .types.turn import Turn

logger = structlog.get_logger("timbal.eval.engine")



async def eval_steps(
    turn: Turn,
    agent: Agent,
    run_context: RunContext
) -> None:
    """"""
    last_snapshot = agent.state_saver.get_last(path=agent.path, context=run_context)
    agent_memory = last_snapshot.data["memory"].resolve()
    actual_steps = []
    for message in agent_memory:
        if isinstance(message, dict):
            message = Message.validate(message)
        if message.content[0].type == "tool_use":
            actual_steps.append(
                {
                    "tool": message.content[0].name,
                    "input": message.content[0].input
                }
            )

    explanations = []
    results = []
    for validator in turn.steps.validators:
        if hasattr(validator, "func") and inspect.iscoroutinefunction(validator.func):
            try:
                await validator(actual_steps)
                correct = True
            except EvalError as e:
                correct = False
                explanations.append(str(e))
        else:
            try:
                validator(actual_steps)
                correct = True
            except EvalError as e:
                correct = False
                explanations.append(str(e))
        results.append(correct)
    return all(results), explanations, actual_steps


async def eval_output(
    turn: Turn,
    agent_output: str
) -> tuple[bool, list[str]]:
    """Evaluate the output of a turn. If no validators are present, use semantic_output as default."""
    explanations = []
    results = []
    agent_output_message = Message(role="assistant", content=[TextContent(text=agent_output)])

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
        
    return all(results), explanations, agent_output_message


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
                        usage_items.append((f"{model}:{usage_type}", limits))
    else:
        return []

    for usage_key, limits in usage_items:
        max_value = limits.get("max")
        min_value = limits.get("min")
        actual_value = actual_usage.get(usage_key)
        if max_value is not None and actual_value is not None and actual_value > max_value:
            correct = False
            explanation = f"Actual value {actual_value} is greater than max value {max_value} for {usage_key}."
        elif min_value is not None and actual_value is not None and actual_value < min_value:
            correct = False
            explanation = f"Actual value {actual_value} is less than min value {min_value} for {usage_key}."
        elif actual_value is not None:
            correct = True
            explanation = f"Actual value {actual_value} is between min value {min_value} and max value {max_value} for {usage_key}."
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
    agent.state_saver.put(snapshot, RunContext())
    run_context = RunContext(parent_id=snapshot.id)

    # Run the agent
    agent_output_event = await agent.complete(context=run_context, prompt=user_input)
    agent_usage = agent_output_event.usage
    agent_output = agent_output_event.output.content[0].text
    run_context = RunContext(parent_id=agent_output_event.run_id)

    # Evaluate the steps
    steps_passed, steps_explanations, actual_steps = None, [], []
    if turn.steps:
        steps_passed, steps_explanations, actual_steps = await eval_steps(turn, agent, run_context)
        if steps_passed:
            test_results.steps_passed += 1
        else:
            test_results.steps_failed += 1
            reason.append("steps")

    # Evaluate the output
    output_passed, output_explanations, agent_output_message = await eval_output(turn, agent_output)
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
                output_passed=output_passed,
                output_explanations=output_explanations,
                actual_output={
                    "text": agent_output_message.content[0].text if agent_output_message.content[0].type == "text" else None,
                    "files": [c.file for c in agent_output_message.content if c.type == "file"]
                },
                expected_output=turn.output.to_dict(),
                usage_passed=all(comparison["correct"] for comparison in usage_comparisons),
                usage_explanations= [comparison["explanation"] for comparison in usage_comparisons],
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
    test_name: str = None
) -> Any:
    """Parse and run all the tests in the given file. Optionally, only run a specific test by name."""
    with open(path) as f:
        test_suite = yaml.safe_load(f)

    test_suite = TestSuite.model_validate(test_suite)
    
    for test in test_suite.tests:
        if test_name is not None and test.name != test_name:
            continue
        conversation_history = []
        for i, turn in enumerate(test.turns):
            is_last_turn = (i == len(test.turns) - 1)
            if is_last_turn:
                await run_turn(_agent, turn, test, conversation_history, test_results, path.name)
            else:
                conversation_history.append(turn.input.to_message(role="user"))
                conversation_history.append(turn.output.to_message(role="assistant"))
    
    return test_results