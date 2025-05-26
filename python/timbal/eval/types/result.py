from pydantic import BaseModel
from typing import Any

class EvalResult(BaseModel):
    test_path: str
    input: str
    reason: list[str] | None = None
    output_passed: bool | None = None
    output_explanations: list[str] | None = None
    # actual_output: list[str]
    # expected_output: list[str]
    steps_passed: bool | None = None
    steps_explanations: list[str] | None = None
    # actual_steps: list[dict[str, Any]] | None = None
    # expected_steps: list[dict[str, Any]] | None = None


class EvalTestSuiteResult(BaseModel):
    total_tests: int = 0
    total_turns: int = 0
    outputs_passed: int = 0
    outputs_failed: int = 0
    steps_passed: int = 0
    steps_failed: int = 0
    usage_passed: int = 0
    usage_failed: int = 0
    tests_failed: list[EvalResult] = []