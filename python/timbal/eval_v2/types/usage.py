import structlog
from pydantic import BaseModel, ConfigDict, field_validator

logger = structlog.get_logger("timbal.eval.types.steps")


class Usage(BaseModel):
    """Represents the usage data for a single turn in an evaluation test.

    This model defines the expected resource or token usage for a turn, such as the number of input or output tokens
    consumed by a model or tool. It is used to specify constraints or expectations on resource usage, which can be
    programmatically checked against the agent's actual usage during evaluation.

    The `usage` attribute allows you to define one or more usage constraints (e.g., max/min tokens for a model or tool)
    that should be satisfied during the turn. These constraints are typically checked after the agent's response is
    generated, and can be used to ensure efficiency, cost control, or adherence to system limits.
    """
    model_config = ConfigDict(extra="ignore")

    usage: list[dict] = []
    """A list of usage constraint dictionaries, each specifying expected resource usage (such as token counts)
    for a model or tool. Each dictionary may define `max` and/or `min` values for a particular usage type.
    If this list is empty, no usage constraints are applied for the turn.
    """
    @field_validator("usage", mode="before")
    def validate_usage(cls, v):
        if isinstance(v, dict):
            if all(isinstance(val, dict) and ("max" in val or "min" in val) for val in v.values()):
                return v
            flat_usage = {}
            for name, value_type in v.items():
                if value_type is None:
                    continue
                if isinstance(value_type, dict):
                    for name_type, limits in value_type.items():
                        flat_usage[f"{name}:{name_type}"] = limits
            return flat_usage
        elif isinstance(v, list):
            usage = {}
            for item in v:
                if isinstance(item, dict):
                    usage.update(item)
            flat_usage = {}
            for name, value_type in usage.items():
                if value_type is None:
                    continue
                if isinstance(value_type, dict):
                    for name_type, limits in value_type.items():
                        flat_usage[f"{name}:{name_type}"] = limits
            return flat_usage
        else:
            raise ValueError("usage must be a list of dicts or a dict")