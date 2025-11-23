import structlog
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..validators import (
    Validator,
    contains_ordered_steps,
    contains_steps,
    equals_steps,
    not_contains_steps,
    semantic_steps,
)
from ..validators import (
    usage as usage_validator,
)

logger = structlog.get_logger("timbal.eval.types.steps")


def _normalize_step_items(items: list, validator_name: str) -> list[dict]:
    """Normalize step items from string or dict format to dict format with 'name' field."""
    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"name": item})
        elif isinstance(item, dict):
            if "name" not in item:
                raise ValueError(f"Invalid {validator_name} step item: {item}. Must have 'name' field.")
            normalized.append(item)
        else:
            raise ValueError(f"Invalid {validator_name} step item: {item}")
    return normalized


class Steps(BaseModel):
    """Represents the steps data for a single turn in an evaluation test.

    This model defines the expected sequence of tool-use or action steps that an agent should perform within a turn.
    It serves a dual purpose:

    1.  **Expected Steps for Validation**: When `validators` are present, this model's fields define the
        expected steps (such as tool invocations or actions) that the agent should execute. The `validators`
        are then used to programmatically check if the agent's actual steps match these expectations.

    2.  **Fixed Record/Observation**: If the `validators` list is empty, this steps instance is treated as a
        fixed record, representing a sequence of actions or steps to be added to the agent's memory or
        conversation history, influencing subsequent turns without undergoing validation itself.

    The `validators` attribute specifies how the actual steps are to be checked against the expected steps.
    """
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    validators: list[Validator] | None = None
    """A list of `Validator` instances to apply to the agent's actual steps.
    If this list is empty, the `Steps` instance is treated as a fixed record
    rather than an expected sequence of steps to be validated.
    """
    
    # Direct step specification fields (alternative to validators)
    contains: list[str] | None = None
    """List of tool names that should be present in the steps."""
    
    not_contains: list[str] | None = None  
    """List of tool names that should NOT be present in the steps."""
    
    semantic: list[str] | None = None
    """List of semantic descriptions that the steps should match."""


    @field_validator("validators", mode="before")
    def validate_validators(cls, v):
        """Converts validator specifications from a dictionary (e.g., YAML) into `Validator` instances."""
        if v is None:
            return None
            
        if isinstance(v, list) and len(v) == 0:
            return []
            
        if isinstance(v, list) and all(isinstance(item, Validator) for item in v):
            return v
            
        if not isinstance(v, dict):
            raise ValueError("validators must be a dict or list of Validator objects")
        
        validators = []
        for validator_name, validator_arg in v.items():
            if validator_name == "contains":
                validators.append(contains_steps(_normalize_step_items(validator_arg, validator_name)))
            elif validator_name == "contains_ordered":
                validators.append(contains_ordered_steps(_normalize_step_items(validator_arg, validator_name)))
            elif validator_name == "equals":
                validators.append(equals_steps(_normalize_step_items(validator_arg, validator_name)))
            elif validator_name == "usage":
                validators.append(usage_validator(validator_arg))
            elif validator_name == "semantic":
                validators.append(semantic_steps(validator_arg))
            elif validator_name == "not_contains":
                validators.append(not_contains_steps(_normalize_step_items(validator_arg, validator_name)))
            else:
                logger.warning("unknown_validator", validator=validator_name)

        return validators
    
    @model_validator(mode="after")
    def convert_direct_fields_to_validators(self):
        """Convert direct step specification fields to validators."""
        if self.validators is not None:
            return self
        
        validators = []
        
        if self.contains:
            normalized_arg = [{"name": item} for item in self.contains]
            validators.append(contains_steps(normalized_arg))
        
        if self.not_contains:
            normalized_arg = [{"name": item} for item in self.not_contains]
            validators.append(not_contains_steps(normalized_arg))
            
        if self.semantic:
            validators.append(semantic_steps(self.semantic))
        
        if validators:
            self.validators = validators
        
        return self
        
    def to_dict(self) -> dict:
        d = {}
        
        if self.contains:
            d["contains"] = self.contains
        if self.not_contains:
            d["not_contains"] = self.not_contains
        if self.semantic:
            d["semantic"] = self.semantic
            
        if not d and getattr(self, "validators", None):
            validators_dict = {}
            for v in self.validators:
                key = getattr(v, "name", str(type(v)))
                value = getattr(v, "ref", str(v))
                
                if key == "contains_steps":
                    if isinstance(value, list):
                        d["contains"] = value
                elif key == "not_contains_steps":
                    if isinstance(value, list):
                        d["not_contains"] = value
                else:
                    if key in validators_dict:
                        if isinstance(validators_dict[key], list):
                            validators_dict[key].append(value)
                        else:
                            validators_dict[key] = [validators_dict[key], value]
                    else:
                        validators_dict[key] = value
            
            if validators_dict:
                d["validators"] = validators_dict
        return d
    