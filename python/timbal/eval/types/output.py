import structlog
from pydantic import ConfigDict, field_validator

from ..validators import Validator, contains_output, regex, semantic_output
from .input import Input

logger = structlog.get_logger("timbal.eval.types.output")


class Output(Input):
    """Represents the output data for a single turn in an evaluation test.

    This model defines the data associated with an agent's response or action
    within a turn. It serves a dual purpose:

    1.  **Expected Output for Validation**: When `validators` are present, this
        model's fields (e.g., `text`, `files` inherited from `Input`) define the
        expected response from the agent. The `validators` are then used to
        programmatically check if the agent's actual output matches these
        expectations.

    2.  **Fixed Record/Observation**: If the `validators` list is empty, this
        output is treated as a fixed record. This can be used to represent
        an observation or a pre-defined response to be added to the agent's
        memory or conversation history, influencing subsequent turns without
        undergoing validation itself.

    Inherits `text` and `files` attributes from the `Input` class, which define
    the content of the output.
    """
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    validators: list[Validator] = []
    """A list of `Validator` instances to apply to the agent's actual output.
    If this list is empty, the `Output` instance is treated as a fixed record
    rather than an expected output to be validated.
    """


    @field_validator("validators", mode="before")
    def validate_validators(cls, v):
        """Converts validator specifications from a dictionary (e.g., YAML) into `Validator` instances."""
        if not isinstance(v, dict):
            raise ValueError("validators must be a dict")
        
        validators = []
        for validator_name, validator_arg in v.items():
            if validator_name == "contains":
                validators.append(contains_output(validator_arg))
            elif validator_name == "regex":
                validators.append(regex(validator_arg))
            elif validator_name == "semantic":
                validators.append(semantic_output(validator_arg))
            # TODO Add more validators.
            else:
                logger.warning("unknown_validator", validator=validator_name)

        return validators
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        
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
    