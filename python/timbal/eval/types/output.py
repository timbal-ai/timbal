import structlog
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...types.file import File
from ...types.message import Message
from ..validators import (
    Validator,
    contains_any_output,
    contains_output,
    equals,
    not_contains_output,
    regex,
    semantic_output,
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
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    content: list[str | File] | None = None
    """List of content items (text strings or Files) that represent the expected output.
    Used when this is a fixed record for conversation history (no validators).
    Elements ending with file extensions become Files, others are treated as text.
    """
    
    validators: dict[str, list[Any]] | list[Any] | None = None
    """Validators to apply to the agent's actual output.
    Can be:
    - Top-level validators (dict with validator names like "contains") - validates the entire message
    If this is empty and content is provided, the Output is treated as a fixed record.
    """

    @model_validator(mode="before")
    @classmethod
    def from_dict_or_list(cls, v: Any) -> Any:
        """Handle different input formats for Output."""
        if v is None:
            return None
        if isinstance(v, str):
            # Single string becomes content list
            return {"content": [v]}
        if isinstance(v, list):
            # List can be content or validators - check if it looks like validators
            if len(v) > 0 and isinstance(v[0], dict) and any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any") for k in v[0].keys()):
                # Looks like validators
                return {"validators": v}
            else:
                # Looks like content
                return {"content": v}
        if isinstance(v, dict):
            # Dict - check if it has content or validators
            result = {}
            if "content" in v:
                result["content"] = v["content"]
            if "validators" in v:
                result["validators"] = v["validators"]
            # If neither, check if keys are validator names
            if "content" not in result and "validators" not in result:
                if any(k in ("contains", "not_contains", "equals", "regex", "semantic", "contains_any") for k in v.keys()):
                    result["validators"] = v
                else:
                    # Assume it's content (list of values)
                    result["content"] = list(v.values()) if v else []
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
        
        # Top-level validators (keys are validator names like "contains", "regex", etc.)
        validators = []
        for validator_name, validator_arg in v.items():
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
            # TODO Add more validators.
            else:
                logger.warning("unknown_validator", validator=validator_name)
        
        return validators
    
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

    def to_message(self, role: str = "assistant", test_file_dir: Path | None = None) -> Message:
        """Convert the Output model instance to a Message instance.
        Elements in content list ending with file extensions become Files, others become text.
        """
        content = []
        if self.content:
            for item in self.content:
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
        if self.content:
            d["content"] = self.content
        if self.validators:
            # Convert validators back to dict format
            if isinstance(self.validators, list):
                validators_dict = {}
                for v in self.validators:
                    key = getattr(v, "name", str(type(v)))
                    value = getattr(v, "ref", str(v))
                    if key in validators_dict:
                        if isinstance(validators_dict[key], list):
                            validators_dict[key].append(value)
                        else:
                            validators_dict[key] = [validators_dict[key], value]
                    else:
                        validators_dict[key] = value
                d["validators"] = validators_dict
            else:
                # Shouldn't happen with new structure, but handle it
                d["validators"] = self.validators
        return d
    