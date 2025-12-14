from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SkipValidation, computed_field, model_validator

from ..core.runnable import Runnable
from ..state.tracing.span import Span
from .validators import parse_validator

logger = structlog.get_logger("timbal.evals.models")


SPAN_PROPERTIES = frozenset(Span.model_fields.keys())
FLOW_VALIDATORS = frozenset(["seq!", "parallel!", "any!"])


class Eval(BaseModel):
    """A single eval test case."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    path: Path
    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)

    timeout: float | None = None
    env: dict[str, str] = Field(default_factory=dict)

    params: dict[str, Any] = Field(default_factory=dict)

    runnable: Runnable

    # Parsed validators stored privately
    _validators: list = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def parse_validators(self) -> "Eval":
        if not isinstance(self.model_extra, dict):
            return self

        def dfs(target: str, spec: dict[str, Any]):
            validators = []
            for k, v in spec.items():
                if k in FLOW_VALIDATORS:
                    if not isinstance(v, list):
                        logger.warning(
                            "Invalid flow validator step. Value should be a list of steps", target=target, step=v
                        )
                        continue
                    span_names = []
                    nested_validators = []
                    for step in v:
                        if isinstance(step, str):
                            span_names.append(step)
                        elif isinstance(step, dict):
                            if len(step) != 1:
                                logger.warning("Invalid flow validator step", target=target, step=step)
                                continue
                            span_name, nested_spec = next(iter(step.items()))
                            span_names.append(span_name)
                            if isinstance(nested_spec, dict):
                                nested = dfs(f"{target}.{span_name}", nested_spec)
                                nested_validators.extend(nested)
                            else:
                                logger.warning("Invalid flow validator step", target=target, step=step)
                        else:
                            logger.warning("Invalid flow validator step", target=target, step=step)
                    # Add the flow validator first, then its nested validators
                    validators.append((target, k, span_names))
                    validators.extend(nested_validators)
                elif k.endswith("!"):
                    # Try to create validator instance, fallback to tuple
                    try:
                        validator = parse_validator({"name": k, "target": target, "value": v})
                        validators.append(validator)
                    except Exception:
                        # Unknown validator, keep as tuple
                        validators.append((target, k, v))
                elif isinstance(v, dict):
                    nested = dfs(f"{target}.{k}", v)
                    validators.extend(nested)
                else:
                    logger.warning("Invalid eval property", target=target, key=k, value=v)
            return validators

        self._validators = dfs(self.runnable._path, self.model_extra)
        for validator in self._validators:
            print(validator)
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
