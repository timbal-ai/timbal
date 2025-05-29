import structlog
from pydantic import BaseModel, ConfigDict, field_validator

from ..validators import Validator, contains_steps, not_contains_steps, regex, semantic_steps

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

    validators: list[Validator] = []
    """A list of `Validator` instances to apply to the agent's actual steps.
    If this list is empty, the `Steps` instance is treated as a fixed record
    rather than an expected sequence of steps to be validated.
    """


    @field_validator("validators", mode="before")
    def validate_validators(cls, v):
        """Converts validator specifications from a dictionary (e.g., YAML) into `Validator` instances."""
        if not isinstance(v, dict):
            raise ValueError("validators must be a dict")
        
        validators = []
        for validator_name, validator_arg in v.items():
            if validator_name == "contains":
                validators.append(contains_steps(validator_arg))
            elif validator_name == "regex":
                validators.append(regex(validator_arg))
            elif validator_name == "semantic":
                validators.append(semantic_steps(validator_arg))
            elif validator_name == "not_contains":
                validators.append(not_contains_steps(validator_arg))
            # TODO Add more validators.
            else:
                logger.warning("unknown_validator", validator=validator_name)

        return validators
        
    def to_dict(self) -> dict:
        d = {}
        
        if getattr(self, "validators", None):
            validators_dict = {}
            for v in self.validators:
                key = getattr(v, "name", str(type(v)))
                value = getattr(v, "ref", str(v))
                # Group multiple validators of the same type
                if key in validators_dict:
                    if isinstance(validators_dict[key], list):
                        validators_dict[key].append(value)
                    else:
                        validators_dict[key] = [validators_dict[key], value]
                else:
                    validators_dict[key] = value
            d["validators"] = validators_dict
        return d
    