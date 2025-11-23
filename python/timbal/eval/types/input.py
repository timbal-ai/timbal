from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...types.file import File
from ...types.message import Message
from .utils import has_file_extension, resolve_file_path, validators_to_dict
from ..validators import (
    Validator,
    contains_any_output,
    contains_output,
    equals,
    not_contains_output,
    regex,
    semantic_output,
)
from ..validators import (
    usage as usage_validator,
)


class Input(BaseModel):
    """Represents the input for a single turn in an evaluation test.

    This model defines the data provided to the agent at the start of a turn.
    The prompt is a list where elements ending with file extensions become Files,
    otherwise they are treated as text. Arbitrary keys can also be included.
    Each key can have validators to check the input value.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    prompt: str | list[str | File] | None = None
    """The prompt content as a string or list. List elements ending with file extensions become Files.
    Shortcut without key (string or list) is assigned to this field.
    Validators can be inside this key: {prompt: {value: [...], validators: {...}}} or {prompt: {validators: {...}}}
    """
    validators: dict[str, list[Any]] | None = None
    """Per-key validators. Each key in the input can have its own validators.
    Validators are extracted from within each key structure.
    """

    @model_validator(mode="before")
    @classmethod
    def from_string(cls, v: Any) -> Any:
        """Enable initializing an input directly with some value."""
        if isinstance(v, str):
            return {"prompt": [v]}
        if isinstance(v, dict):
            result = {}
            validators = {}
            top_level_validators = None
            for key, value in v.items():
                if key == "validators":
                    # Top-level validators: only "usage" is allowed for Input
                    if isinstance(value, dict):
                        has_validator_names = any(k in ("usage",) for k in value.keys())
                        
                        if has_validator_names:
                            top_level_validators = value
                        else:
                            raise ValueError("validators must be inside keys, not at top-level. Use {prompt: {value: [...], validators: {...}}} or {key: {value: ..., validators: {...}}}")
                    elif value is None:
                        top_level_validators = None
                    else:
                        raise ValueError("validators must be a dict")
                elif isinstance(value, dict) and "validators" in value:
                    # Key with inline validators: {key: {value: "...", validators: {...}}}
                    if "value" in value:
                        result[key] = value["value"]
                    else:
                        result[key] = None
                    if "validators" in value:
                        validators[key] = value["validators"]
                elif isinstance(value, dict) and any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any", "usage") for k in value.keys()):
                    # Key with inline validators directly: {key: {contains: [...]}}
                    result[key] = None
                    validators[key] = value
                else:
                    result[key] = value
            
            # Convert 'text' or 'files' to 'prompt' list for backward compatibility
            if "text" in result and "prompt" not in result:
                result["prompt"] = [result.pop("text")]
            if "files" in result:
                files = result.pop("files")
                if not isinstance(files, list):
                    files = [files]
                if "prompt" in result:
                    if isinstance(result["prompt"], list):
                        result["prompt"].extend(files)
                    else:
                        result["prompt"] = [result["prompt"]] + files
                else:
                    result["prompt"] = files
            # Also handle validators for 'text'/'files' -> 'prompt' conversion
            if validators:
                if "text" in validators and "prompt" not in validators:
                    validators["prompt"] = validators.pop("text")
                if "files" in validators and "prompt" not in validators:
                    validators["prompt"] = validators.pop("files")
            
            if "prompt" in result and isinstance(result["prompt"], str):
                result["prompt"] = [result["prompt"]]
            
            # Set validators: either top-level (usage) or per-key validators
            if top_level_validators is not None:
                result["validators"] = top_level_validators
            elif validators:
                result["validators"] = validators
            return result
        return v
    
    @field_validator("prompt", mode="before")
    def validate_prompt(cls, v):
        """Convert string to list, keep list as-is."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        return [str(v)]

    @field_validator("validators", mode="before")
    @classmethod
    def validate_validators(cls, v):
        """Converts validator specifications from a dictionary (e.g., YAML) into Validator instances.
        Supports both per-key validators (values are dicts with validator names) and top-level validators
        (keys are validator names like "time", "usage").
        """
        if v is None:
            return None
        
        if isinstance(v, dict):
            all_validators = True
            for key, value in v.items():
                if isinstance(value, list):
                    if not all(isinstance(item, Validator) for item in value):
                        all_validators = False
                        break
                elif not isinstance(value, Validator):
                    all_validators = False
                    break
            if all_validators:
                return v
        
        if not isinstance(v, dict):
            return v
        
        # Distinguish between per-key validators (values are dicts with validator names)
        # and top-level validators (keys are validator names like "usage")
        is_per_key = False
        for key, value in v.items():
            if isinstance(value, dict) and any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any", "usage") for k in value.keys()):
                is_per_key = True
                break
        
        # Process top-level validators (only "usage" for Input)
        if not is_per_key and any(k in ("usage",) for k in v.keys()):
            result = {}
            for validator_name, validator_arg in v.items():
                if validator_name == "usage":
                    result[validator_name] = [usage_validator(validator_arg)]
                else:
                    raise ValueError(f"Unknown top-level validator '{validator_name}' for Input. Only 'usage' is supported.")
            return result
        
        if not is_per_key:
            return v
        
        # Process per-key validators
        result = {}
        for key, validator_spec in v.items():
            if not isinstance(validator_spec, dict):
                raise ValueError(f"validators for key '{key}' must be a dict")
            
            validators = []
            for validator_name, validator_arg in validator_spec.items():
                if validator_name == "contains":
                    validators.append(contains_output(validator_arg))
                elif validator_name == "not_contains":
                    validators.append(not_contains_output(validator_arg))
                elif validator_name == "equals":
                    validators.append(equals(validator_arg))
                elif validator_name == "regex":
                    validators.append(regex(validator_arg))
                elif validator_name == "semantic":
                    validators.append(semantic_output(validator_arg))
                elif validator_name == "contains_any":
                    validators.append(contains_any_output(validator_arg))
                else:
                    raise ValueError(f"Validator '{validator_name}' cannot be used as a per-key validator. Use it as a top-level validator in input.validators instead.")
            
            result[key] = validators
        
        return result

    @model_validator(mode="after")
    def validate_at_least_one_field(self):
        """Ensure at least prompt is provided or there are other keys."""
        if type(self).__name__ == "Input":
            if self.prompt is None or (isinstance(self.prompt, list) and len(self.prompt) == 0):
                other_keys = {k: v for k, v in self.model_dump().items() 
                            if k not in ("prompt", "validators") and v is not None}
                if not other_keys:
                    raise ValueError("Input must have either prompt or other keys")
        return self


    def to_message(self, role: str = "user", test_file_dir: Path | None = None) -> Message:
        """Convert the Input model instance to a Message instance.
        Elements in prompt list ending with file extensions become Files, others become text.
        """
        content = []
        if self.prompt:
            prompt_list = self.prompt if isinstance(self.prompt, list) else [self.prompt]
            for item in prompt_list:
                if isinstance(item, File):
                    content.append(resolve_file_path(item, test_file_dir))
                elif isinstance(item, str):
                    if has_file_extension(item):
                        content.append(resolve_file_path(item, test_file_dir))
                    else:
                        content.append(item)
                else:
                    content.append(str(item))
        
        return Message.validate({
            "role": role, 
            "content": content,
        })
    
    def to_dict(self) -> dict:
        d = {}
        output_dict = self.model_dump(exclude={"validators"})
        for key, value in output_dict.items():
            if value is not None:
                d[key] = value
        
        if self.validators:
            for key, validators in self.validators.items():
                validators_dict = validators_to_dict(validators)
                if key in d:
                    key_value = d.pop(key)
                    d[key] = {"value": key_value, "validators": validators_dict}
                else:
                    d[key] = {"validators": validators_dict}
        return d
