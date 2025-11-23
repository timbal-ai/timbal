from pathlib import Path
from typing import Any

import structlog
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
    time as time_validator,
)
from ..validators import (
    usage as usage_validator,
)

logger = structlog.get_logger("timbal.eval.types.output")


class Output(BaseModel):
    """Represents the output data for a single turn in an evaluation test.

    This model defines the data associated with an agent's response or action
    within a turn. It serves a dual purpose:

    1.  **Expected Output for Validation**: When `validators` are present, the
        actual agent output is validated against these validators.

    2.  **Fixed Record/Observation**: If `content` is provided but no `validators`,
        this output is treated as a fixed record to be added to the agent's
        memory or conversation history, influencing subsequent turns without
        undergoing validation itself.
    
    The output can have multiple keys (like Input). The shortcut without a key
    (string or list) is assigned to `content`. Each key can have validators inside:
    - Shortcut: `"Hello"` or `["Hello"]` → assigns to `content`
    - With validators: `{content: {content: [...], validators: {...}}}`
    - Multiple keys: `{content: [...], other_key: {...}}`
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    content: list[str | File] | None = None
    """List of content items (text strings or Files) that represent the expected output.
    Used when this is a fixed record for conversation history (no validators).
    Elements ending with file extensions become Files, others are treated as text.
    Shortcut without key (string or list) is assigned to this field.
    """
    
    validators: dict[str, list[Any]] | None = None
    """Per-key validators. Each key in the output can have its own validators.
    Validators can only exist inside keys (like content or other arbitrary keys).
    Validators at top-level are not allowed - they must be associated with a key.
    """

    @model_validator(mode="before")
    @classmethod
    def from_dict_or_list(cls, v: Any) -> Any:
        """Handle different input formats for Output."""
        if v is None:
            return None
        if isinstance(v, str):
            # Single string becomes content list (shortcut without key)
            return {"content": [v]}
        if isinstance(v, list):
            # List is always content (shortcut without key, validators must be inside keys)
            return {"content": v}
        if isinstance(v, dict):
            # Handle case where output has keys with validators nested
            result = {}
            validators = {}
            for key, value in v.items():
                if key == "validators":
                    # Top-level validators: only "time" and "usage" are allowed
                    if isinstance(value, dict):
                        # Check if values are already Validator objects (from serialization)
                        all_validators = all(
                            isinstance(v, Validator) or 
                            (isinstance(v, list) and len(v) > 0 and all(isinstance(item, Validator) for item in v))
                            for v in value.values()
                        ) if value else False
                        
                        if all_validators:
                            validators.update(value)
                        else:
                            # Check if this is per-key format: {"content": {"validators": {...}}}
                            is_per_key_format = any(
                                isinstance(v, dict) and "validators" in v 
                                for v in value.values()
                            ) or any(
                                isinstance(v, dict) and any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any") for k in v.keys())
                                for v in value.values()
                            )
                            
                            if is_per_key_format:
                                # Per-key format: {validators: {content: {validators: {...}}}}
                                validators.update(value)
                            else:
                                # Check if keys are top-level validators (time, usage) only
                                has_top_level = any(k in ("time", "usage") for k in value.keys())
                                has_per_key = any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any") for k in value.keys())
                                
                                if has_per_key:
                                    raise ValueError("Validators like 'contains', 'not_contains', 'equals', 'regex', 'semantic', 'contains_any' must be inside keys (like 'content'). Only 'time' and 'usage' can be top-level validators.")
                                
                                if has_top_level:
                                    # Top-level format: {validators: {time: {...}, usage: {...}}}
                                    validators.update(value)
                                else:
                                    raise ValueError("validators must be a dict with validator names. Only 'time' and 'usage' are allowed as top-level validators.")
                    else:
                        raise ValueError("validators must be a dict")
                elif isinstance(value, dict) and "validators" in value:
                    # Key with inline validators: {key: {value: "...", validators: {...}}}
                    if "content" in value or "value" in value:
                        value_key = "content" if key == "content" else "value"
                        if value_key in value:
                            result[key] = value[value_key]
                        else:
                            result[key] = None
                    else:
                        result[key] = None
                    if "validators" in value:
                        validators[key] = value["validators"]
                elif isinstance(value, dict) and any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any", "time", "usage") for k in value.keys()):
                    # Key with inline validators directly: {key: {contains: [...]}}
                    result[key] = None
                    validators[key] = value
                else:
                    result[key] = value
            
            # Ensure content is a list if it's a string
            if "content" in result and isinstance(result["content"], str):
                result["content"] = [result["content"]]
            
            if validators:
                result["validators"] = validators
            return result
        return v

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v):
        """Convert content to list format."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            # Empty list should be None
            if len(v) == 0:
                return None
            return v
        return [str(v)]

    @field_validator("validators", mode="before")
    @classmethod
    def validate_validators(cls, v):
        """Converts validator specifications from a dictionary (e.g., YAML) into Validator instances.
        Supports both per-key validators (values are dicts with validator names) and top-level validators
        (keys are validator names like "time", "usage" only). 
        Only 'time' and 'usage' can be top-level validators. Other validators must be inside keys like 'content'.
        """
        if v is None:
            return None
        
        # If v is already a dict with Validator objects (from serialization), return as-is
        if isinstance(v, dict):
            # Check if values are already Validator objects or lists of Validators
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
        
        # Separate top-level validators (time, usage) from per-key validators (content, etc.)
        top_level_validators = {}
        per_key_validators = {}
        
        for key, value in v.items():
            if key in ("time", "usage"):
                top_level_validators[key] = value
            else:
                per_key_validators[key] = value
        
        result = {}
        
        # Process top-level validators (time, usage)
        for validator_name, validator_arg in top_level_validators.items():
            if validator_name == "time":
                result[validator_name] = [time_validator(validator_arg)]
            elif validator_name == "usage":
                result[validator_name] = [usage_validator(validator_arg)]
        
        # Process per-key validators (content, etc.)
        for key, validator_spec in per_key_validators.items():
            if not isinstance(validator_spec, dict):
                raise ValueError(f"validators for key '{key}' must be a dict")
            
            # Handle nested format: {"content": {"validators": {...}}}
            if "validators" in validator_spec:
                validator_spec = validator_spec["validators"]
            
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
                elif validator_name == "time":
                    validators.append(time_validator(validator_arg))
                elif validator_name == "usage":
                    validators.append(usage_validator(validator_arg))
                else:
                    # Unknown validator
                    logger.warning("unknown_validator", validator=validator_name)
            
            result[key] = validators
        
        return result
    

    def to_message(self, role: str = "assistant", test_file_dir: Path | None = None) -> Message:
        """Convert the Output model instance to a Message instance.
        Elements in content list ending with file extensions become Files, others become text.
        """
        content = []
        if self.content:
            for item in self.content:
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
        # Include all keys except validators
        output_dict = self.model_dump(exclude={"validators"})
        for key, value in output_dict.items():
            if value is not None:
                d[key] = value
        
        # Convert validators back to dict format inside each key
        if self.validators:
            for key, validators in self.validators.items():
                validators_dict = validators_to_dict(validators)
                if key in d:
                    key_value = d.pop(key)
                    d[key] = {}
                    if key == "content":
                        d[key]["content"] = key_value
                    else:
                        d[key]["value"] = key_value
                    d[key]["validators"] = validators_dict
                else:
                    d[key] = {"validators": validators_dict}
        return d
    