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

        # Track occurrence counts for each step target path
        # This allows us to distinguish validators for different occurrences of the same step
        # e.g., two "get_datetime" steps in a parallel block
        occurrence_counts: dict[str, int] = {}

        def get_next_occurrence(step_target: str) -> int:
            """Get the next occurrence index for a step target and increment the counter."""
            count = occurrence_counts.get(step_target, 0)
            occurrence_counts[step_target] = count + 1
            return count

        def dfs(target: str, spec: dict[str, Any], occurrence: int = 0):
            validators = []
            for k, v in spec.items():
                if k in FLOW_VALIDATORS:
                    if not isinstance(v, list):
                        logger.warning(
                            "Invalid flow validator step. Value should be a list of steps", target=target, step=v
                        )
                        continue
                    steps_for_flow = []
                    nested_validators = []
                    for step in v:
                        if isinstance(step, str):
                            steps_for_flow.append(step)
                            # Track occurrence even for plain step names (no validators)
                            step_target = f"{target}.{step}"
                            get_next_occurrence(step_target)
                        elif isinstance(step, dict):
                            if len(step) != 1:
                                logger.warning("Invalid flow validator step", target=target, step=step)
                                continue
                            step_key, step_spec = next(iter(step.items()))
                            if step_key in FLOW_VALIDATORS:
                                # Nested flow validator (e.g., parallel! inside seq!)
                                # Pass the entire step dict to preserve structure
                                steps_for_flow.append(step)
                                # Also extract validators from inside the nested flow
                                if isinstance(step_spec, list):
                                    for nested_step in step_spec:
                                        if isinstance(nested_step, str):
                                            # Plain step name inside nested flow
                                            nested_step_target = f"{target}.{nested_step}"
                                            get_next_occurrence(nested_step_target)
                                        elif isinstance(nested_step, dict) and len(nested_step) == 1:
                                            nested_step_name, nested_step_spec = next(iter(nested_step.items()))
                                            nested_step_target = f"{target}.{nested_step_name}"
                                            step_occurrence = get_next_occurrence(nested_step_target)
                                            if nested_step_name not in FLOW_VALIDATORS and isinstance(
                                                nested_step_spec, dict
                                            ):
                                                nested = dfs(
                                                    nested_step_target, nested_step_spec, occurrence=step_occurrence
                                                )
                                                nested_validators.extend(nested)
                            else:
                                # Span name with nested validators
                                steps_for_flow.append(step)
                                step_target = f"{target}.{step_key}"
                                step_occurrence = get_next_occurrence(step_target)
                                if isinstance(step_spec, dict):
                                    nested = dfs(step_target, step_spec, occurrence=step_occurrence)
                                    nested_validators.extend(nested)
                        else:
                            logger.warning("Invalid flow validator step", target=target, step=step)
                    # Add the flow validator first, then its nested validators
                    try:
                        flow_validator = parse_validator({"name": k, "target": target, "value": steps_for_flow})
                        validators.append(flow_validator)
                    except Exception:
                        logger.warning("Unknown flow validator", target=target, name=k)
                        continue
                    validators.extend(nested_validators)
                elif k.endswith("!"):
                    # Try to create validator instance, fallback to tuple
                    try:
                        validator = parse_validator({"name": k, "target": target, "value": v, "occurrence": occurrence})
                        validators.append(validator)
                    except Exception:
                        logger.warning("Unknown validator", target=target, name=k)
                elif isinstance(v, dict):
                    nested = dfs(f"{target}.{k}", v, occurrence=occurrence)
                    validators.extend(nested)
                else:
                    logger.warning("Invalid eval property", target=target, key=k, value=v)
            return validators

        self._validators = dfs(self.runnable._path, self.model_extra)
        return self


class EvalError(BaseModel):
    """Error from a failed eval run."""

    type: str
    message: str
    traceback: str | None = None


class ValidatorResult(BaseModel):
    """Result of a single validator execution."""

    target: str
    name: str
    value: Any = None
    passed: bool
    error: str | None = None
    traceback: str | None = None


class EvalResult(BaseModel):
    """Result of a single eval run."""

    eval: SkipValidation[Eval]
    passed: bool
    duration: float = 0.0
    error: SkipValidation[EvalError | None] = None
    validator_results: list[ValidatorResult] = Field(default_factory=list)
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
