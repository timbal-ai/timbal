import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any
import inspect

import structlog
import yaml
from dotenv import find_dotenv, load_dotenv

from .. import __version__
from ..logs import setup_logging
from ..state import RunContext
from ..types.models import dump
from ..server.utils import ModuleSpec, load_module

from pydantic import BaseModel, field_validator

from ..types.message import Message
from ..types.file import File

from collections.abc import Callable

from ..core.agent import Agent
from ..types.events import OutputEvent
from ..state.data import DataValue
from ..state.snapshot import Snapshot

logger = structlog.get_logger("timbal.evals")


Validator = Callable[[Message], tuple[bool, str] | bool]


def build_summary(test_results):
    summary = {
        "num_tests": 0,
        "total_turns": 0,
        "total_steps_checked": 0,
        "total_steps_failed": 0,
        "total_outputs_checked": 0,
        "total_outputs_failed": 0,
        "tests": []
    }
    tests_by_name = {}
    for result in test_results:
        test_name = result["test_name"]
        if test_name not in tests_by_name:
            tests_by_name[test_name] = []
        tests_by_name[test_name].append(result)

    summary["num_tests"] = len(tests_by_name)
    for test_name, turns in tests_by_name.items():
        test_dict = {
            "test_name": test_name,
            "failed_steps": [],
            "failed_outputs": [],
            "steps_checked": 0,
            "steps_failed": 0,
            "outputs_checked": 0,
            "outputs_failed": 0,
        }
        for i, turn in enumerate(turns):
            summary["total_turns"] += 1
            # Output check
            test_dict["outputs_checked"] += 1
            summary["total_outputs_checked"] += 1
            if not turn["output_passed"]:
                test_dict["outputs_failed"] += 1
                summary["total_outputs_failed"] += 1
                test_dict["failed_outputs"].append({
                    "turn_index": i,
                    "input": turn.get("input"),
                    "output_passed": turn.get("output_passed"),
                    "output_explanations": turn.get("output_explanations"),
                    "actual_output": turn.get("actual_output"),
                    "expected_output": turn.get("expected_output"),
                })
            # Steps check (if present)
            if "steps_passed" in turn and turn["steps_passed"] is not None:
                test_dict["steps_checked"] += 1
                summary["total_steps_checked"] += 1
                if not turn["steps_passed"]:
                    test_dict["steps_failed"] += 1
                    summary["total_steps_failed"] += 1
                    test_dict["failed_steps"].append({
                        "turn_index": i,
                        "input": turn.get("input"),
                        "steps_passed": turn.get("steps_passed"),
                        "steps_explanations": turn.get("steps_explanations"),
                        "actual_steps": turn.get("actual_steps"),
                        "expected_steps": turn.get("expected_steps"),
                    })
        summary["tests"].append(test_dict)
    return summary


def json_formatter(response: Message) -> dict:
    result_text = response.output.content[0].text.strip()
    try:
        result = json.loads(result_text)
        return {
            "correct": result.get("correct", False),
            "explanation": result.get("explanation", "No explanation provided"),
        }
    except json.JSONDecodeError:
        return {
            "correct": False,
            "explanation": f"Could not parse: {result_text}",
        }
    


agent_eval_steps = Agent(
    model="gpt-4.1-nano", 
    system_prompt="""You are a helpful assistant that evaluates if the process performed by the LLM is correct.
Compare two processes: the first is the reference (expected) process, and the second is the process to evaluate.
Determine if the second process uses appropriate tools and reasonable inputs to fulfill the user's request, even if the sequence or number of tool calls differs from the reference.
Ignore case and formatting differences in all inputs.
Consider the process correct if it achieves the intended outcome using valid tools and logic, even if the steps are not identical to the reference.
If the process is correct, return true; otherwise, return false.
Return your answer as a JSON object. (Remember: your response must be valid JSON.)
Return: {"explanation": "Brief reasoning", "correct": true/false}
""",
    after_agent_callback=json_formatter,
)

agent_eval_output = Agent(
    model="gpt-4.1-nano", 
    system_prompt="""
You are a helpful assistant that evaluates if a response is correct.
Compare two responses: the first is the reference answer, and the second is the answer to evaluate.
The second response should be considered correct if it is a helpful, relevant, and contextually appropriate reply to the user's request, and if it covers the key information or question present in the reference answer.
Do not penalize for paraphrasing, extra detail, or reasonable conversational steps if they help address the user's need.
Only mark as incorrect if the response is irrelevant, unhelpful, or fails to address the user's request.
Return your answer as a JSON object. (Remember: your response must be valid JSON.)
Return: {"explanation": "Brief reasoning", "correct": true/false}
""",
    after_agent_callback=json_formatter,
)

async def semantic_steps_validator(
    actual_steps: str, 
    expected_steps: str
) -> bool:
    """"""
    prompt = f"Steps: {actual_steps}\nExpected steps: {expected_steps}"
    step_response = await agent_eval_steps.complete(prompt)
    result = json_formatter(step_response)
    return bool(result.get("correct")), result.get("explanation"), expected_steps


async def semantic_output_validator(
    actual_output: str, 
    expected_output: str
) -> bool:
    """"""
    prompt = f"Output: {actual_output}\nExpected output: {expected_output}"
    output_response = await agent_eval_output.complete(prompt=prompt)
    result = json_formatter(output_response)
    return bool(result.get("correct")), result.get("explanation"), actual_output, expected_output


class Output(BaseModel):
    """"""
    text: str | Message | None = None
    files: list[str | File] | None = None
    validators: list[Callable] | None = None

    @field_validator("text", mode="before")
    def validate_text(cls, v):
        if v is None:
            return None
        return Message.validate(v) # TODO: It could be a list?

    @field_validator("files", mode="before")
    def validate_files(cls, v):
        if v is None:
            return None
        return [File.validate(file) for file in v]

    @field_validator("validators", mode="before")
    def validate_validators(cls, v):
        # TODO:?? Only string?
        # if isinstance(v, str):
        #     #return Message.validate(v)
        #     async def validator(output):
        #         return await semantic_output_validator(output, v)
        #     return [validator]
        # el
        if isinstance(v, dict):
            validators = []
            for validator, ref in v.items():
                if validator == "contains":
                    if not isinstance(ref, list):
                        raise ValueError("contains must be a list")
                    validators.append(lambda output: (all([x in output for x in ref]), f"Must contain: {ref}", output, ref))
                elif validator == "regex":
                    if not isinstance(ref, str): # TODO: possible to have a list of regexes?
                        raise ValueError("regex must be a string")
                    validators.append(lambda output: (bool(re.search(ref, output, re.MULTILINE)), f"Must match regex: {ref}", output, ref))
                elif validator == "exactly":
                    # TODO: specific type?
                    validators.append(lambda output: (output == ref, f"Must be exactly: {ref}", output, ref))
                elif validator == "semantic":
                    if not isinstance(ref, str):
                        raise ValueError("semantic must be a string")
                    elif isinstance(ref, list):
                        for expected_output in ref:
                            async def validator(output):
                                return await semantic_output_validator(output, expected_output)
                            validators.append(validator)
                    else:
                        async def validator(output):
                            return await semantic_output_validator(output, ref)
                        validators.append(validator)
                else:
                    logger.error("unknown_validator", validator=validator)
            return validators
        
        elif isinstance(v, Message):
            expected_text = v.content[0].text if hasattr(v, "content") and v.content else str(v)
            async def validator(output):
                return await semantic_output_validator(output, expected_text)
            return [validator]

        raise ValueError("output must be a fixed string or a dict of validators")



class Turn(BaseModel):
    """"""

    input: Message
    steps: list[Validator] | None = None
    # output: list[Validator]
    output: str | Output

    @field_validator("input", mode="before")
    def validate_input(cls, v):
        if isinstance(v, str):
            return Message.validate(v)
        elif isinstance(v, dict):
            content = []
            if "files" in v:
                if isinstance(v["files"], list):
                    for file in v["files"]:
                        content.append(File.validate(file))
                else:
                    raise ValueError("files must be a list")
            elif "text" in v:
                content.append(v["text"])
            return Message.validate(content)
        raise ValueError("input must be a str or dict")
    

    @field_validator("steps", mode="before")
    def validate_steps(cls, v):
        if isinstance(v, str):
            async def validator(steps):
                return await semantic_steps_validator(steps, v)
            return [validator]
        elif isinstance(v, dict):
            validators = []
            for validator, ref in v.items():
                # TODO: contains??????
                # - tot str
                # - posar que el contains tingui: tool, input?
                if validator == "contains":
                    if not isinstance(ref, list):
                        raise ValueError("contains must be a list")
                    validators.append(lambda steps: (all([x in steps for x in ref]), f"Must contain: {ref}", ref))
                elif validator == "semantic":
                    if not isinstance(ref, str):
                        raise ValueError("semantic must be a string")
                    elif isinstance(ref, list):
                        for expected_step in ref:
                            async def validator(steps):
                                return await semantic_steps_validator(steps, expected_step)
                            validators.append(validator)
                    else:
                        async def validator(steps):
                            return await semantic_steps_validator(steps, ref)
                        validators.append(validator)
                else: # TODO: need to add more validators?
                    logger.error("unknown_validator", validator=validator)

            return validators

        raise ValueError("steps must be a str or dict of validators")


    # @field_validator("output", mode="before")
    # def validate_output(cls, v):
    #     if isinstance(v, str):
    #         #return Message.validate(v)
    #         async def validator(output):
    #             return await semantic_output_validator(output, v)
    #         return [validator]
    #     elif isinstance(v, dict):
    #         validators = []
    #         for validator, ref in v.items():
    #             if validator == "contains":
    #                 if not isinstance(ref, list):
    #                     raise ValueError("contains must be a list")
    #                 validators.append(lambda output: all([x in output for x in ref]))
    #             elif validator == "regex":
    #                 if not isinstance(ref, str): # TODO: possible to have a list of regexes?
    #                     raise ValueError("regex must be a string")
    #                 validators.append(lambda output: bool(re.search(ref, output, re.MULTILINE)))
    #             elif validator == "exactly":
    #                 # TODO: specific type?
    #                 validators.append(lambda output: output == ref)
    #             elif validator == "semantic":
    #                 if not isinstance(ref, str):
    #                     raise ValueError("semantic must be a string")
    #                 elif isinstance(ref, list):
    #                     for expected_output in ref:
    #                         async def validator(output):
    #                             return await semantic_output_validator(output, expected_output)
    #                         validators.append(validator)
    #                 else:
    #                     async def validator(output):
    #                         return await semantic_output_validator(output, ref)
    #                     validators.append(validator)
    #             else:
    #                 logger.error("unknown_validator", validator=validator)
    #         return validators
        
    #     elif isinstance(v, Message):
    #         expected_text = v.content[0].text if hasattr(v, "content") and v.content else str(v)
    #         async def validator(output):
    #             return await semantic_output_validator(output, expected_text)
    #         return [validator]

    #     raise ValueError("output must be a fixed string or a dict of validators")


class Test(BaseModel):
    """"""

    name: str
    description: str | None = None
    turns: list[Turn]


class Config(BaseModel):
    """"""

    tests: list[Test]


def discover_files(path: Path) -> list[Path]:
    """"""
    files = []
    if path.is_dir():
        for file in path.rglob("eval*.yaml"):
            files.append(file)
    else:
        if not path.name.endswith(".yaml"):
            raise ValueError(f"Invalid evals path: {path}")
        files.append(path)

    return files


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
        if message.content[0].type == "tool_use":
            actual_steps.append(
                {
                    "tool": message.content[0].name,
                    "input": message.content[0].input
                }
            )

    explanations = []
    results = []
    expected_steps = []
    for validator in turn.steps:
        if inspect.iscoroutinefunction(validator):
            correct, explanation, expected = await validator(actual_steps)
        else:
            correct, explanation, expected = validator(actual_steps)
        results.append(correct)
        explanations.append(explanation)
        expected_steps.append(expected)
    return all(results), explanations, actual_steps, expected_steps


async def eval_output(
    turn: Turn,
    agent_output: str
) -> None:
    """"""
    explanations = []
    results = []
    actual_output = []
    expected_output = []
    for validator in turn.output.validators:
        if inspect.iscoroutinefunction(validator):
            correct, explanation, actual, expected = await validator(agent_output)
        else:
            correct, explanation = validator(agent_output)
        results.append(correct)
        explanations.append(explanation)
        actual_output.append(actual)
        expected_output.append(expected)
    return all(results), explanations, actual_output, expected_output


async def run_turn(
    agent: Agent,
    turn: Turn,
    test: Test,
    conversation_history: list[Message],
    test_results: list[dict]
) -> None:
    """"""
    user_input = turn.input
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
    agent_output = agent_output_event.output.content[0].text

    # Evaluate the steps
    steps_passed, steps_explanations = None, []
    if turn.steps:
        steps_passed, steps_explanations, actual_steps, expected_steps = await eval_steps(turn, agent, run_context)

    # Evaluate the output
    output_passed, output_explanations, actual_output, expected_output = await eval_output(turn, agent_output)

    result = {
        "test_name": test.name,
        "input": str(user_input),
        "output_passed": output_passed,
        "output_explanations": output_explanations,
        "actual_output": actual_output,
        "expected_output": expected_output,
    }
    if turn.steps:
        result["steps_passed"] = steps_passed
        result["steps_explanations"] = steps_explanations
        result["actual_steps"] = actual_steps
        result["expected_steps"] = expected_steps

    test_results.append(result)

async def eval_file(
    path: Path,
    agent: Agent
) -> Any:
    """"""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, list):
        config = {"tests": config}
    config = Config.model_validate(config)
    print(config)
    test_results = []
    for test in config.tests:
        conversation_history = []
        for i, turn in enumerate(test.turns):
            is_last_turn = (i == len(test.turns) - 1)
            # Copy for the iterations?
            # current_turn_history = conversation_history.copy()
            if is_last_turn:
                # for i in range(num_iterations):
                await run_turn(agent, turn, test, conversation_history, test_results)
            else:
                conversation_history.append(turn.input)
                conversation_history.append(turn.output)

    summary = build_summary(test_results)

    # Save to JSON
    with open("summary.json", "w") as f:
        from ..types.models import dump
        summary = dump(summary)
        json.dump(summary, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timbal evals.")
    parser.add_argument(
        "-v", 
        "--version", 
        action="store_true", 
        help="Show version and exit."
    )
    parser.add_argument(
        "--fqn",
        dest="fqn",
        type=str,
        help="Fully qualified name of the module to be run (format: path/to/file.py::object_name)",
    )
    parser.add_argument(
        "--evals",
        dest="evals",
        type=str,
        help="Path to the evals directory to be run.",
    )
    args = parser.parse_args()

    if args.version:
        print(f"timbal.evals {__version__}") # noqa: T201
        sys.exit(0)

    fqn = args.fqn
    if not fqn:
        print("No timbal app Fully Qualified Name (FQN) provided.") # noqa: T201
        sys.exit(1)

    fqn_parts = fqn.split(":")
    if len(fqn_parts) > 2:
        print("Invalid timbal app Fully Qualified Name (FQN) format. Use 'path/to/file.py:object_name' or 'path/to/file.py'") # noqa: T201
        sys.exit(1)
    elif len(fqn_parts) == 2:
        module_path, module_name = fqn_parts
        module_spec = ModuleSpec(
            path=Path(module_path).expanduser().resolve(), 
            object_name=module_name,
        )
    else:
        module_spec = ModuleSpec(
            path=Path(fqn_parts[0]).expanduser().resolve(), 
            object_name=None,
        )

    agent = load_module(module_spec)

    evals_path = args.evals
    if not evals_path:
        print("No evals path provided.") # noqa: T201
        sys.exit(1)
    evals_path = Path(evals_path).expanduser().resolve()
    if not evals_path.exists():
        print(f"Evals path {evals_path} does not exist.") # noqa: T201
        sys.exit(1)

    logger.info("loading_dotenv", path=find_dotenv())
    load_dotenv(override=True)
    setup_logging()

    files = discover_files(evals_path)
    print(files)

    for file in files:
        asyncio.run(eval_file(file, agent))
