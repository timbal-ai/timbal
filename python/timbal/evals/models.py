from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SkipValidation, computed_field, model_validator

from ..core.runnable import Runnable

logger = structlog.get_logger("timbal.evals.models")


class Eval(BaseModel):
    """A single eval test case."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    path: Path
    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)

    timeout: float | None = None
    env: dict[str, str] = Field(default_factory=dict)

    runnable: Runnable

    # Parsed validators stored privately
    _validators: list = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def parse_checks(self) -> "Eval":
        if not isinstance(self.model_extra, dict):
            return self

        runnable = self.runnable
        runnable_path = runnable._path
        for k, v in self.model_extra.items():
            if k != runnable_path:
                logger.warning(f"Unexpected key '{k}' for eval {self.path}::{self.name}")
                continue
            # TODO
            print(v)

        return self


class EvalError(BaseModel):
    """Error from a failed eval run."""

    type: str
    message: str
    traceback: str | None = None


class EvalResult(BaseModel):
    """Result of a single eval run."""

    eval: SkipValidation[Eval]
    passed: bool
    duration: float = 0.0
    error: SkipValidation[EvalError | None] = None
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
