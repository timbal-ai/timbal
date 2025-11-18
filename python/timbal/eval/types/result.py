from typing import Any

from pydantic import BaseModel


class EvalResult(BaseModel):
    test_name: str
    test_path: str
    input: str | dict
    reason: list[str] | None = None
    execution_error: dict | None = None
    input_passed: bool | None = None
    input_explanations: list[str] | None = None
    output_passed: bool | None = None
    output_explanations: list[str] | None = None
    actual_output: str | dict
    expected_output: str | dict | None = None
    steps_passed: bool | None = None
    steps_explanations: list[str] | None = None
    actual_steps: list[dict[str, Any]] | None = None
    expected_steps: str | dict | None = None
    usage_passed: bool | None = None
    usage_explanations: list[str] | None = None


class EvalTestSuiteResult(BaseModel):
    total_files: int = 0
    total_tests: int = 0
    total_turns: int = 0
    inputs_passed: int = 0
    inputs_failed: int = 0
    outputs_passed: int = 0
    outputs_failed: int = 0
    steps_passed: int = 0
    steps_failed: int = 0
    usage_passed: int = 0
    usage_failed: int = 0
    execution_errors: int = 0
    tests_failed: list[EvalResult] = []