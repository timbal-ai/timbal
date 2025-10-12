import structlog
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..validators import Validator, contains_steps, not_contains_steps, semantic_steps

logger = structlog.get_logger("timbal.eval.types.steps")


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
        # Handle None case
        if v is None:
            return None
            
        # Handle empty list case
        if isinstance(v, list) and len(v) == 0:
            return []
            
        # Handle case where validators are already Validator objects (direct instantiation)
        if isinstance(v, list) and all(isinstance(item, Validator) for item in v):
            return v
            
        # Handle case where validators come from dict (e.g., YAML)
        if not isinstance(v, dict):
            raise ValueError("validators must be a dict or list of Validator objects")
        
        validators = []
        for validator_name, validator_arg in v.items():
            if validator_name == "contains":
                # Handle multiple formats:
                # 1. ["tool_name"] - list of strings
                # 2. [{"name": "tool_name"}] - list of dicts with just name
                # 3. [{"name": "tool_name", "input": {...}}] - list of dicts with name and input
                normalized_arg = []
                for item in validator_arg:
                    if isinstance(item, str):
                        normalized_arg.append({"name": item})
                    elif isinstance(item, dict):
                        # Ensure the dict has a "name" field
                        if "name" not in item:
                            raise ValueError(f"Invalid contains step item: {item}. Must have 'name' field.")
                        normalized_arg.append(item)
                    else:
                        raise ValueError(f"Invalid contains step item: {item}")
                validators.append(contains_steps(normalized_arg))
            # elif validator_name == "regex":
            #     validators.append(regex(validator_arg))
            elif validator_name == "semantic":
                validators.append(semantic_steps(validator_arg))
            elif validator_name == "not_contains":
                # Handle both formats: ["tool_name"] and [{"name": "tool_name"}]
                normalized_arg = []
                for item in validator_arg:
                    if isinstance(item, str):
                        normalized_arg.append({"name": item})
                    elif isinstance(item, dict):
                        # Ensure the dict has a "name" field
                        if "name" not in item:
                            raise ValueError(f"Invalid not_contains step item: {item}. Must have 'name' field.")
                        normalized_arg.append(item)
                    else:
                        raise ValueError(f"Invalid not_contains step item: {item}")
                validators.append(not_contains_steps(normalized_arg))
            # TODO Add more validators.
            else:
                logger.warning("unknown_validator", validator=validator_name)

        return validators
    
    @model_validator(mode="after")
    def convert_direct_fields_to_validators(self):
        """Convert direct step specification fields to validators."""
        # If we already have validators from explicit validators field, don't override
        if self.validators is not None:
            return self
        
        validators = []
        
        # Convert direct field specifications to validators
        if self.contains:
            normalized_arg = [{"name": item} for item in self.contains]
            validators.append(contains_steps(normalized_arg))
        
        if self.not_contains:
            normalized_arg = [{"name": item} for item in self.not_contains]
            validators.append(not_contains_steps(normalized_arg))
            
        if self.semantic:
            validators.append(semantic_steps(self.semantic))
        
        # Only set validators if we have any, otherwise leave as None
        if validators:
            self.validators = validators
        
        return self
        
    def to_dict(self) -> dict:
        d = {}
        
        # Show direct field values first
        if self.contains:
            d["contains"] = self.contains
        if self.not_contains:
            d["not_contains"] = self.not_contains
        if self.semantic:
            d["semantic"] = self.semantic
            
        # If no direct fields, extract from validators
        if not d and getattr(self, "validators", None):
            validators_dict = {}
            for v in self.validators:
                key = getattr(v, "name", str(type(v)))
                value = getattr(v, "ref", str(v))
                
                # For contains_steps and not_contains_steps, show the expected tools with full structure
                if key == "contains_steps":
                    # Extract full structure from the validator reference
                    if isinstance(value, list):
                        # Keep the full structure including input if present
                        d["contains"] = value
                elif key == "not_contains_steps":
                    if isinstance(value, list):
                        # Keep the full structure including input if present
                        d["not_contains"] = value
                else:
                    # Group multiple validators of the same type
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
    