from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, field_validator


class BaseOperation(BaseModel, ABC):
    """Base class for all filesystem operation messages."""
    type: str
    path: str
    
    # This will be set by the server when it starts
    # All operations automatically inherit the same base path
    # No need to pass context through every validation call
    _base_path: ClassVar[Path | None] = None

    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate that path is within allowed directory."""
        if cls._base_path is None:
            return v
            
        try:
            full_path = cls._base_path / v
            full_path.resolve().relative_to(cls._base_path.resolve())
            return v
        except ValueError as e:
            raise ValueError("Path outside allowed directory") from e

    @abstractmethod
    async def __call__(self, base_path: Path) -> dict[str, Any]:
        """Execute the operation with the given base path."""
        pass
