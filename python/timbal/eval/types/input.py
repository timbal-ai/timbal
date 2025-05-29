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
    files: list[File] | None = None
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
        if not isinstance(v, str):
            return str(v)
        return v


    def to_message(self, role: str = "user") -> Message:
        """Convert the Input model instance to a Message instance."""
        content = []
        if self.files:
            for file in self.files:
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
