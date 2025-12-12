from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from .validators import BaseValidator, collect_validators


class Eval(BaseModel):
    """A single eval test case."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    path: Path
    name: str
    description: str | None = None
    params: dict[str, Any] = {}
    # TODO A lot of stuff
    validators: list[BaseValidator] = []

    @model_validator(mode="before")
    @classmethod
    def collect_validators(cls, data: dict[str, Any]) -> dict[str, Any]:
        root_path = data.pop("root_path")
        validators = collect_validators(data, path=root_path)
        data["validators"] = validators
        return data


class EvalError(BaseModel):
    """Error from a failed eval run."""

    type: str
    message: str
    traceback: str | None = None


class EvalResult(BaseModel):
    """Result of a single eval run."""

    eval: Eval
    passed: bool
    duration: float = 0.0
    error: EvalError | None = None
    captured_stdout: str = ""
    captured_stderr: str = ""


class EvalSummary(BaseModel):
    """Summary of all eval runs."""

    results: list[EvalResult] = []

    @computed_field
    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.results)

    @computed_field
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @computed_field
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @computed_field
    @property
    def total(self) -> int:
        return len(self.results)
