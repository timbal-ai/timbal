from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...types.file import File
from ...types.message import Message
from ..validators import Validator, contains_output, not_contains_output, equals, regex, semantic_output, contains_any_output


class Input(BaseModel):
    """Represents the input for a single turn in an evaluation test.

    This model defines the data provided to the agent at the start of a turn.
    The prompt is a list where elements ending with file extensions become Files,
    otherwise they are treated as text. Arbitrary keys can also be included.
    Each key can have validators to check the input value.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    prompt: str | list[str | File] | None = None
    """The prompt content as a string or list. List elements ending with file extensions become Files."""
    validators: dict[str, list[Any]] | None = None
    """Per-key validators. Each key in the input can have its own validators."""

    @model_validator(mode="before")
    @classmethod
    def from_string(cls, v: Any) -> Any:
        """Enable initializing an input directly with some value."""
        if isinstance(v, str):
            return {"prompt": [v]}
        if isinstance(v, dict):
            # Handle case where input has keys with validators nested
            result = {}
            validators = {}
            for key, value in v.items():
                if key == "validators":
                    validators = value
                elif isinstance(value, dict) and "validators" in value:
                    # Key has inline validators: {key: {value: "...", validators: {...}}} or {key: {validators: {...}}}
                    if "value" in value:
                        result[key] = value["value"]
                    else:
                        result[key] = None
                    if "validators" in value:
                        validators[key] = value["validators"]
                elif isinstance(value, dict) and any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any") for k in value.keys()):
                    # Key has inline validators directly: {key: {contains: [...]}}
                    result[key] = None
                    validators[key] = value
                else:
                    # Regular key-value pair
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
            # Also handle validators for 'text'/'files' -> 'prompt'
            if validators is not None:
                if "text" in validators and "prompt" not in validators:
                    validators["prompt"] = validators.pop("text")
                if "files" in validators and "prompt" not in validators:
                    if "prompt" in validators:
                        # Merge files validators into prompt validators
                        pass
                    else:
                        validators["prompt"] = validators.pop("files")
            
            # Ensure prompt is a list if it's a string
            if "prompt" in result and isinstance(result["prompt"], str):
                result["prompt"] = [result["prompt"]]
            
            if validators:
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
        """Converts validator specifications from a dictionary (e.g., YAML) into Validator instances per key.
        Only processes if it looks like per-key validators (values are dicts with validator names).
        Otherwise returns as-is for Output to handle.
        """
        if v is None:
            return None
        
        if not isinstance(v, dict):
            # Not a dict, let Output handle it or return as-is
            return v
        
        # Check if this looks like per-key validators (values are dicts with validator names)
        # or top-level validators (keys are validator names like "contains")
        is_per_key = False
        for key, value in v.items():
            # If value is a dict with validator names, it's per-key
            if isinstance(value, dict) and any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any") for k in value.keys()):
                is_per_key = True
                break
        
        # If keys are validator names (top-level), don't process here - let Output handle it
        if not is_per_key and any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any") for k in v.keys()):
            return v  # Return as-is for Output to process
        
        if not is_per_key:
            # Doesn't look like per-key validators, return as-is
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
                    raise ValueError(f"Unknown validator '{validator_name}' for key '{key}'")
            
            result[key] = validators
        
        return result

    @model_validator(mode="after")
    def validate_at_least_one_field(self):
        """Ensure at least prompt is provided or there are other keys."""
        # Only validate if this is actually an Input class, not a subclass like Output
        if type(self).__name__ == "Input":
            if self.prompt is None or (isinstance(self.prompt, list) and len(self.prompt) == 0):
                # Check if there are any other keys (excluding validators)
                other_keys = {k: v for k, v in self.model_dump().items() 
                            if k not in ("prompt", "validators") and v is not None}
                if not other_keys:
                    raise ValueError("Input must have either prompt or other keys")
        return self

    def _has_file_extension(self, s: str) -> bool:
        """Check if a string ends with a file extension."""
        if not isinstance(s, str):
            return False
        # Common file extensions
        extensions = ['.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv', '.json', 
                     '.xml', '.html', '.htm', '.png', '.jpg', '.jpeg', '.gif', '.svg',
                     '.mp3', '.mp4', '.avi', '.zip', '.tar', '.gz', '.py', '.js', '.ts',
                     '.md', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.log']
        s_lower = s.lower().strip()
        # Check if it ends with an extension (with optional query params for URLs)
        for ext in extensions:
            if s_lower.endswith(ext) or ext in s_lower.split('?')[0]:
                return True
        return False

    def to_message(self, role: str = "user", test_file_dir: Path | None = None) -> Message:
        """Convert the Input model instance to a Message instance.
        Elements in prompt list ending with file extensions become Files, others become text.
        """
        content = []
        if self.prompt:
            prompt_list = self.prompt if isinstance(self.prompt, list) else [self.prompt]
            for item in prompt_list:
                if isinstance(item, File):
                    # Already a File object
                    if test_file_dir and hasattr(item, 'path') and item.path:
                        file_path = Path(item.path)
                        if not file_path.is_absolute():
                            resolved_path = (test_file_dir / file_path).resolve()
                            item = File.validate(str(resolved_path))
                    content.append(item)
                elif isinstance(item, str):
                    # Check if it's a file path (ends with extension or is a path)
                    if self._has_file_extension(item) or '/' in item or '\\' in item:
                        # Treat as file
                        if test_file_dir:
                            file_path = Path(item)
                            if not file_path.is_absolute():
                                resolved_path = (test_file_dir / file_path).resolve()
                                file = File.validate(str(resolved_path))
                            else:
                                file = File.validate(item)
                        else:
                            file = File.validate(item)
                        content.append(file)
                    else:
                        # Treat as text
                        content.append(item)
                else:
                    # Convert to string
                    content.append(str(item))
        
        return Message.validate({
            "role": role, 
            "content": content,
        })
    
    def to_dict(self) -> dict:
        d = {}
        if self.prompt:
            d["prompt"] = self.prompt
        # Include other keys
        for key, value in self.model_dump().items():
            if key not in ("prompt", "validators") and value is not None:
                d[key] = value
        if self.validators:
            # Convert validators back to dict format
            validators_dict = {}
            for key, validators in self.validators.items():
                validators_dict[key] = {}
                for v in validators:
                    validator_name = getattr(v, "name", "unknown")
                    validator_ref = getattr(v, "ref", None)
                    if validator_name in validators_dict[key]:
                        if not isinstance(validators_dict[key][validator_name], list):
                            validators_dict[key][validator_name] = [validators_dict[key][validator_name]]
                        validators_dict[key][validator_name].append(validator_ref)
                    else:
                        validators_dict[key][validator_name] = validator_ref
            d["validators"] = validators_dict
        return d
