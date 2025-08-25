from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...types.file import File
from ...types.message import Message


class Input(BaseModel):
    """Represents the input for a single turn in an evaluation test.

    This model defines the data provided to the agent at the start of a turn.
    Input may include user text, files, or both. Inputs are never validated.
    """
    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    """The main text input for the agent."""
    files: list[File | str] | None = None
    """Optional list of files to include as part of the input."""


    @model_validator(mode="before")
    @classmethod
    def from_string(cls, v: Any) -> Any:
        """Enable initializing an input directly with some value."""
        if isinstance(v, str):
            return {"text": v}
        return v
    
    @field_validator("text", mode="before")
    def validate_text(cls, v):
        """Stringify the value to enable passing this to an LLM."""
        if v is None:
            return None
        if not isinstance(v, str):
            return str(v)
        return v

    @model_validator(mode="after")
    def validate_at_least_one_field(self):
        """Ensure at least text or files is provided."""
        # Only validate if this is actually an Input class, not a subclass like Output
        if type(self).__name__ == "Input":
            if self.text is None and (self.files is None or len(self.files) == 0):
                raise ValueError("Input must have either text or files")
        return self

    @field_validator("files", mode="before")
    @classmethod
    def validate_files(cls, v):
        """Handle relative file paths by deferring validation."""
        if v is None:
            return v
        
        # Convert to list if it's a single file
        if not isinstance(v, list):
            v = [v]
        
        # For each file, if it's a string that looks like a relative path, 
        # we'll handle it specially in to_message
        processed_files = []
        for file_item in v:
            if isinstance(file_item, str) and not file_item.startswith(('http://', 'https://', 'data:', '/')):
                # This looks like a relative path, we'll handle it in to_message
                processed_files.append(file_item)
            else:
                # Let the normal File validation handle it
                processed_files.append(file_item)
        
        return processed_files


    def to_message(self, role: str = "user", test_file_dir: Path | None = None) -> Message:
        """Convert the Input model instance to a Message instance."""
        content = []
        if self.files:
            for file in self.files:
                # Handle relative path strings
                if isinstance(file, str):
                    if test_file_dir:
                        # Resolve relative path
                        file_path = Path(file)
                        if not file_path.is_absolute():
                            resolved_path = (test_file_dir / file_path).resolve()
                            file = File.validate(str(resolved_path))
                        else:
                            file = File.validate(file)
                    else:
                        # No test_file_dir provided, try to validate as-is
                        file = File.validate(file)
                # Handle File objects with relative paths
                elif test_file_dir and hasattr(file, 'path') and file.path:
                    file_path = Path(file.path)
                    if not file_path.is_absolute():
                        resolved_path = (test_file_dir / file_path).resolve()
                        # Create a new file object with the resolved path
                        file = File.validate(str(resolved_path))
                content.append(file)
        if self.text:
            content.append(self.text)
        return Message.validate({
            "role": role, 
            "content": content,
        })
    
    def to_dict(self) -> dict:
        d = {}
        if self.text:
            d["text"] = self.text
        if self.files:
            d["files"] = self.files
        return d
