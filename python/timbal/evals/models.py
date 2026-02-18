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

        def dfs(
            target: str,
            path_key: str,
            spec: dict[str, Any],
            step_counters: dict[str, int] | None = None,
        ):
            """
            Parse validators from a spec dict.

            Args:
                target: The current target path (e.g., "agent.get_datetime")
                path_key: The unique path key with indices (e.g., "agent.seq#0.get_datetime#0")
                spec: The spec dict containing validators and nested structures
                step_counters: Counter dict for tracking step occurrences at current level
            """
            if step_counters is None:
                step_counters = {}

            validators = []
            for k, v in spec.items():
                if k in FLOW_VALIDATORS:
                    if not isinstance(v, list):
                        logger.warning(
                            "Invalid flow validator step. Value should be a list of steps", target=target, step=v
                        )
                        continue

                    # Create flow validator with current path_key
                    steps_for_flow = []
                    nested_validators = []

                    # Track step occurrences within this flow
                    flow_step_counters: dict[str, int] = {}

                    for step in v:
                        if isinstance(step, str):
                            # Plain step name (no validators)
                            steps_for_flow.append(step)
                            # Track occurrence
                            step_count = flow_step_counters.get(step, 0)
                            flow_step_counters[step] = step_count + 1
                        elif isinstance(step, dict):
                            if len(step) != 1:
                                logger.warning("Invalid flow validator step", target=target, step=step)
                                continue
                            step_key, step_spec = next(iter(step.items()))
                            if step_key in FLOW_VALIDATORS:
                                # Nested flow validator (e.g., parallel! inside seq!)
                                steps_for_flow.append(step)
                                # Process nested flow recursively
                                if isinstance(step_spec, list):
                                    nested_flow_step_counters: dict[str, int] = {}
                                    for nested_step in step_spec:
                                        if isinstance(nested_step, str):
                                            # Plain step name inside nested flow
                                            ns_count = nested_flow_step_counters.get(nested_step, 0)
                                            nested_flow_step_counters[nested_step] = ns_count + 1
                                        elif isinstance(nested_step, dict) and len(nested_step) == 1:
                                            nested_step_name, nested_step_spec = next(iter(nested_step.items()))
                                            if nested_step_name in FLOW_VALIDATORS:
                                                # Skip nested flow validators for now
                                                continue
                                            # Get occurrence for this nested step
                                            ns_count = nested_flow_step_counters.get(nested_step_name, 0)
                                            nested_flow_step_counters[nested_step_name] = ns_count + 1
                                            if isinstance(nested_step_spec, dict):
                                                nested_target = f"{target}.{nested_step_name}"
                                                nested_path_key = f"{path_key}.{nested_step_name}#{ns_count}"
                                                nested = dfs(
                                                    nested_target,
                                                    nested_path_key,
                                                    nested_step_spec,
                                                    {},
                                                )
                                                nested_validators.extend(nested)
                            else:
                                # Span name with nested validators
                                steps_for_flow.append(step)
                                step_count = flow_step_counters.get(step_key, 0)
                                flow_step_counters[step_key] = step_count + 1
                                step_target = f"{target}.{step_key}"
                                step_path_key = f"{path_key}.{step_key}#{step_count}"
                                if isinstance(step_spec, dict):
                                    nested = dfs(step_target, step_path_key, step_spec, {})
                                    nested_validators.extend(nested)
                        else:
                            logger.warning("Invalid flow validator step", target=target, step=step)

                    # Add the flow validator
                    try:
                        flow_validator = parse_validator(
                            {
                                "name": k,
                                "target": target,
                                "value": steps_for_flow,
                                "path_key": path_key,
                            }
                        )
                        validators.append(flow_validator)
                    except Exception:
                        logger.warning("Unknown flow validator", target=target, name=k)
                        continue
                    validators.extend(nested_validators)
                elif k.endswith("!"):
                    # Value validator
                    try:
                        validator = parse_validator(
                            {
                                "name": k,
                                "target": target,
                                "value": v,
                                "path_key": path_key,
                            }
                        )
                        validators.append(validator)
                    except Exception:
                        logger.warning("Unknown validator", target=target, name=k)
                elif isinstance(v, dict):
                    # Nested property path (e.g., input.timezone)
                    nested_target = f"{target}.{k}"
                    nested_path_key = f"{path_key}.{k}"
                    nested = dfs(nested_target, nested_path_key, v, step_counters)
                    validators.extend(nested)
                else:
                    logger.warning("Invalid eval property", target=target, key=k, value=v)
            return validators

        root_path = self.runnable._path
        self._validators = dfs(root_path, root_path, self.model_extra, {})
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
    path_key: str = ""  # Unique path key with indices
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

    # ACE tracking
    ace_context: list[dict[str, Any]] = Field(default_factory=list)
    ace_active_policies: list[str] = Field(default_factory=list)
    ace_policies: list[str] = Field(default_factory=list)


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
