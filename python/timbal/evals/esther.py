import asyncio
import json
import yaml
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

from timbal import Agent
from timbal.state import RunContext
from timbal.state.savers import InMemorySaver
from timbal.types.chat.content import Content
from timbal.types.message import Message
from timbal.types.field import Field
from timbal.state.snapshot import Snapshot
from timbal.state.data import DataValue

from agent import agent

EVAL_PROCESS_PROMPT = open("EVAL_PROCESS_PROMPT").read()
EVAL_RESULT_PROMPT = open("EVAL_RESULT_PROMPT").read()

@dataclass
class EvaluationAnalytics:
    total: int = 0  # Number of unique tests (scenarios)
    total_checks: int = 0  # Total number of iterations (sum of all runs)
    correct: int = 0
    failures: List[Dict] = None
    test_details: Dict[str, Dict] = None
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self):
        if self.test_details is None:
            self.test_details = defaultdict(
                lambda: {
                    "result": {"n_runs": 0, "failures": 0, "correct": 0, "failure_details": []},
                    "process": {"n_runs": 0, "failures": 0, "correct": 0, "failure_details": []},
                }
            )

    def add_failure(self, scenario: str, input_text: str, result: str, expected: Any, check_type: str, explanation: str, eval_type: str):
        failure_detail = {
            "scenario": scenario,
            "input": input_text,
            "result": result,
            "expected": expected,
            "check_type": check_type,
            "explanation": explanation,
            "type": eval_type,
        }
        self.test_details[scenario][eval_type]["failures"] += 1
        self.test_details[scenario][eval_type].setdefault("failure_details", []).append(failure_detail)

    def record_result(self, is_correct: bool, scenario: str, eval_type: str = "result"):
        if self.test_details[scenario][eval_type]["n_runs"] == 0 and eval_type == "result":
            self.total += 1
        self.test_details[scenario][eval_type]["n_runs"] += 1
        self.total_checks += 1
        if is_correct:
            self.correct += 1
            self.test_details[scenario][eval_type]["correct"] += 1

    def as_dict(self, iterations_per_test=None):
        tests_out = {}
        failed_iterations = {"result": 0, "process": 0}
        scenarios_failed_both = set()
        scenarios_failed_either = set()
        scenarios_with_result = set()
        scenarios_with_process = set()
        correct_percents = []
        failures_percents = []
        num_tests = self.total
        total_checks = self.total_checks

        for scenario_name, types_data in self.test_details.items():
            tests_out[scenario_name] = self._calculate_scenario_stats(
                types_data,
                scenarios_with_result,
                scenarios_with_process,
                scenarios_failed_both,
                scenarios_failed_either,
                correct_percents,
                failures_percents,
                failed_iterations
            )

        summary_stats = self._calculate_summary_stats(
            num_tests,
            total_checks,
            failed_iterations,
            scenarios_failed_either,
            correct_percents,
            failures_percents
        )

        return {
            "num_tests": num_tests,
            "iterations_per_test": iterations_per_test if iterations_per_test is not None else (next(iter(tests_out.values()))["result"]["iterations"] if tests_out else 0),
            "total_checks": total_checks,
            **summary_stats,
            "tests": tests_out,
        }

    def _calculate_scenario_stats(
        self,
        types_data: Dict,
        scenarios_with_result: set,
        scenarios_with_process: set,
        scenarios_failed_both: set,
        scenarios_failed_either: set,
        correct_percents: list,
        failures_percents: list,
        failed_iterations: Dict[str, int]
    ) -> Dict:
        scenario_output = {}
        failed_result_in_scenario = False
        failed_process_in_scenario = False

        if "result" in types_data:
            scenarios_with_result.add(True)
            failed_result_in_scenario = types_data["result"]["failures"] > 0
        if "process" in types_data:
            scenarios_with_process.add(True)
            failed_process_in_scenario = types_data["process"]["failures"] > 0

        if failed_result_in_scenario and failed_process_in_scenario:
            scenarios_failed_both.add(True)
            scenarios_failed_either.add(True) 
        elif failed_result_in_scenario or failed_process_in_scenario:
            scenarios_failed_either.add(True) 


        for eval_type, stats in types_data.items():
            n_runs = stats["n_runs"]
            correct = stats.get("correct", 0)
            failures = stats["failures"]
            percent_correct = (100 * correct / n_runs) if n_runs else 0
            percent_failures = (100 * failures / n_runs) if n_runs else 0

            if eval_type == "result":
                correct_percents.append(percent_correct)
                failures_percents.append(percent_failures)
            
            failed_iterations[eval_type] += failures
            scenario_output[eval_type] = {
                "iterations": n_runs,
                "correct": {
                    "count": correct,
                    "percent": percent_correct,
                    "fraction": f"{correct}/{n_runs}" if n_runs else "0/0"
                },
                "failures": {
                    "count": failures,
                    "percent": percent_failures,
                    "fraction": f"{failures}/{n_runs}" if n_runs else "0/0"
                },
                "test_failed": failures > 0,
                "failure_details": stats.get("failure_details", []),
            }
        return scenario_output

    def _calculate_summary_stats(
        self,
        num_tests: int,
        total_checks: int,
        failed_iterations: Dict[str, int],
        scenarios_failed_either: set,
        correct_percents: list,
        failures_percents: list
    ) -> Dict:
        scenarios_with_failures_overall = len(scenarios_failed_either)

        scenarios_with_failures_fraction = f"{scenarios_with_failures_overall}/{num_tests}" if num_tests else "0/0"
        scenarios_with_failures_percent = (100 * scenarios_with_failures_overall / num_tests) if num_tests else 0
        
        overall_failed_iterations = sum(failed_iterations.values())
        total_failed_iterations_fraction = f"{overall_failed_iterations}/{total_checks}" if total_checks else "0/0"
        total_failed_iterations_percent = (100 * overall_failed_iterations / total_checks) if total_checks else 0
        
        total_failed_iterations_breakdown = {}
        n_runs_per_eval_type = {"result": 0, "process": 0}

        for scenario_details in self.test_details.values():
            for eval_type in ["result", "process"]:
                if eval_type in scenario_details:
                    n_runs_per_eval_type[eval_type] += scenario_details[eval_type]["n_runs"]

        for eval_type in ["result", "process"]:
            denom = n_runs_per_eval_type[eval_type]
            total_failed_iterations_breakdown[eval_type] = {
                "count": failed_iterations[eval_type], # This is correct as failed_iterations is accumulated
                "percent": (100 * failed_iterations[eval_type] / denom) if denom else 0,
                "fraction": f"{failed_iterations[eval_type]}/{denom}" if denom else "0/0"
            }
            
        mean_correct_percent = statistics.mean(correct_percents) if correct_percents else 0
        std_correct_percent = statistics.stdev(correct_percents) if len(correct_percents) > 1 else 0
        mean_failures_percent = statistics.mean(failures_percents) if failures_percents else 0
        std_failures_percent = statistics.stdev(failures_percents) if len(failures_percents) > 1 else 0

        return {
            "total_checks_result": n_runs_per_eval_type.get("result", 0),
            "total_checks_process": n_runs_per_eval_type.get("process", 0),
            "scenarios_with_failures": {
                "count": scenarios_with_failures_overall,
                "percent": scenarios_with_failures_percent,
                "fraction": scenarios_with_failures_fraction
            },
            "total_failed_iterations": {
                "overall": {
                    "count": overall_failed_iterations,
                    "percent": total_failed_iterations_percent,
                    "fraction": total_failed_iterations_fraction
                },
                **total_failed_iterations_breakdown
            },
            "correct": {
                "mean_percent": mean_correct_percent,
                "std_percent": std_correct_percent
            },
            "failures": {
                "mean_percent": mean_failures_percent,
                "std_percent": std_failures_percent
            },
        }

eval_process = Agent(
    model="gpt-4.1-nano", 
    system_prompt=EVAL_PROCESS_PROMPT
)

eval_result = Agent(
    model="gpt-4.1-nano", 
    system_prompt=EVAL_RESULT_PROMPT
)


def check_output(agent_output: str, expected_type: str, expected_value: str) -> Optional[Tuple[bool, str]]:
    """Check if the output matches the expected value based on the check type."""
    if expected_type == "semantic":
        return None
    if isinstance(agent_output, str):
        agent_output = agent_output.strip()
    if isinstance(expected_value, str):
        expected_value = expected_value.strip()
    
    checks = {
        "exactly": lambda: (agent_output == expected_value, 
                          f"Expected exactly: '{expected_value}', got: '{agent_output}'"),
        "contains": lambda: (str(expected_value) in str(agent_output),
                           f"Expected output to contain: '{expected_value}', got: '{agent_output}'"),
        "regex": lambda: (bool(re.search(expected_value, agent_output, re.MULTILINE)),
                         f"Expected output to match regex: '{expected_value}', got: '{agent_output}'")
    }
    
    if expected_type not in checks:
        return False, f"Unknown check type: {expected_type}"
    
    try:
        return checks[expected_type]()
    except re.error as e:
        return False, f"Regex error: {e}"

async def json_format(output: str) -> dict:
    result_text = output.output.content[0].text.strip()
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
    
async def handle_check(user_input: str, output: str, check_type: str, value: Any) -> dict:
    result = check_output(output, check_type, value)
    if result is None:
        prompt = f"Question: {user_input}\nResult: {output}\nExpected result: {value}"
        output = await eval_result.complete(prompt=prompt)
        return await json_format(output)
    return {"correct": result[0], "explanation": result[1]}

async def evaluate_output(agent_output: str, expected_outputs: Any, user_input: str, 
                         scenario: Dict, stats: EvaluationAnalytics) -> bool:
    """
    Evaluate agent_output against expected_outputs (which can be dict, list, or str).
    Stops at first success, records all failures if none succeed. No inner functions.
    """
    failures = []
    found_correct = False

    if isinstance(expected_outputs, dict):
        for check_type, value in expected_outputs.items():
            values = value if isinstance(value, list) else [value]
            for idx, v in enumerate(values):
                res = await handle_check(user_input, agent_output, check_type, v)
                if res["correct"]:
                    found_correct = True
                    break
                failures.append((res, idx, v, check_type))
            if found_correct:
                break
    elif isinstance(expected_outputs, list):
        for idx, out in enumerate(expected_outputs):
            if isinstance(out, dict):
                for check_type, value in out.items():
                    res = await handle_check(user_input, agent_output, check_type, value)
                    if res["correct"]:
                        found_correct = True
                        break
                    failures.append((res, idx, value, check_type))
                if found_correct:
                    break
            else:
                res = await handle_check(user_input, agent_output, "semantic", str(out))
                if res["correct"]:
                    found_correct = True
                    break
                failures.append((res, idx, out, "semantic"))
    else:
        res = await handle_check(user_input, agent_output, "semantic", str(expected_outputs))
        if res["correct"]:
            found_correct = True
        else:
            failures.append((res, 0, expected_outputs, "semantic"))

    if found_correct:
        stats.record_result(True, scenario.get("name", ""), eval_type="result")
        return True
    for res, idx, value, check_type in failures:
        stats.record_result(False, scenario.get("name", ""), eval_type="result")
        stats.add_failure(
            scenario=scenario.get("name", ""),
            input_text=user_input,
            result=agent_output,
            expected=value,
            check_type=check_type,
            explanation=res["explanation"],
            eval_type="result"
        )
    return False


async def run_turn(turn: Dict, scenario: Dict, analytics: EvaluationAnalytics, messages: List[Message]) -> None:
    """Run and evaluate a single turn of conversation."""
    user_input = turn["input"]
    user_output = turn["output"]

    user_input = Content.model_validate(user_input)
    data = {"memory": DataValue(value=messages)}
    snapshot = Snapshot(
        v="0.2.0",
        id=str(uuid.uuid4()),
        path=agent.path,
        input=user_input,
        t0=int(time.time() * 1000),
        data=data,
    )
    
    agent.state_saver = InMemorySaver()
    agent.state_saver.put(snapshot, RunContext())
    run_context = RunContext(parent_id=snapshot.id)

    flow_output_event = await agent.complete(context=run_context, prompt=user_input)
    agent_output = flow_output_event.output.content[0].text
    run_context = RunContext(parent_id=flow_output_event.run_id)

    # Evaluate process if present
    if "steps" in turn:
        if isinstance(turn["steps"], dict) and len(turn["steps"]) == 1:
            process_type, expected_process = next(iter(turn["steps"].items()))
        else:
            process_type, expected_process = "semantic", str(turn["steps"])

        last_snapshot = agent.state_saver.get_last(path=agent.path, context=run_context)
        agent_memory = last_snapshot.data["memory"].resolve()
        tools_memory = []
        for message in agent_memory:
            if message.content[0].type == "tool_use":
                tools_memory.append(
                    {
                        "tool": message.content[0].name,
                        "input": message.content[0].input
                    }
                )
        print(tools_memory)
        
        if process_type == "semantic":
            prompt = f"Expected Process: {expected_process} \nTools Used: {tools_memory}"
            print(prompt)
            output = await eval_process.complete(prompt=prompt)
            result = await json_format(output)
            analytics.record_result(result["correct"], scenario.get("name", ""), eval_type="process")
            if not result["correct"]:
                analytics.add_failure(
                    scenario=scenario.get("name", ""),
                    input_text=user_input,
                    result=tools_memory,
                    expected=expected_process,
                    check_type=process_type,
                    explanation=result["explanation"],
                    eval_type="process"
                )
        else:
            result = check_output(tools_memory, process_type, str(expected_process))
            is_correct = result and result[0]
            explanation = result[1] if result else "Unknown check type"
            analytics.record_result(is_correct, scenario.get("name", ""), eval_type="process")
            if not is_correct:
                analytics.add_failure(
                    scenario=scenario.get("name", ""),
                    input_text=user_input,
                    result=tools_memory,
                    expected=expected_process,
                    check_type=process_type,
                    explanation=explanation,
                    eval_type="process"
                )

    # Evaluate output
    await evaluate_output(agent_output, user_output, user_input, scenario, analytics)


async def run_turn_with_semaphore(turn, scenario, analytics, messages, semaphore):
    async with semaphore:
        await run_turn(turn, scenario, analytics, messages)


async def evaluation(
    iterations: int = Field(
        default=1,
        description="Number of iterations to run for each turn"
    ),
    parallel_runs: int | None = Field(
        default=8, 
        description="Number of parallel runs to use"
    )
) -> dict:
    """Main evaluation function."""

    num_iterations = iterations.default if hasattr(iterations, "default") else iterations
    num_parallel_runs = parallel_runs.default if hasattr(parallel_runs, "default") and parallel_runs.default is not None else parallel_runs

    if num_parallel_runs is None:
        num_parallel_runs = 8
    with open("evals.yaml", "r") as evals_file:
        evals_config = yaml.safe_load(evals_file)

    analytics = EvaluationAnalytics()
    analytics.start_time = time.time()

    for scenario_config in evals_config:
        conversation_history = []
        turns_data = scenario_config.get("turns", [])

        for turn_idx, turn_data in enumerate(turns_data):
            if turn_idx == len(turns_data) - 1:
                for i in range(num_iterations):
                    current_turn_history = conversation_history.copy()
    
                    await run_turn(turn_data, scenario_config, analytics, current_turn_history)
            else:
                if "input" in turn_data:
                     conversation_history.append(Message.validate({"role": "user", "content": turn_data["input"]}))
                if "output" in turn_data:
                     conversation_history.append(Message.validate({"role": "assistant", "content": turn_data["output"]}))


    analytics.end_time = time.time()
    output_data = analytics.as_dict(iterations_per_test=num_iterations)

    with open("results.txt", "w", encoding="utf-8") as f:
        from timbal.types.models import dump
        output_data = dump(output_data)
        json.dump(output_data, f, indent=2)
    return output_data
    

if __name__ == "__main__":
    asyncio.run(evaluation())