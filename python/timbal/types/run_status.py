from pydantic import BaseModel
from typing import Literal


class RunStatus(BaseModel):
    code: Literal["success", "error", "cancelled", "timeout"]
    """The code associated with the run status."""
    reason: str | None = None
    """The reason for the run status."""
    message: str | None = None
    """The message associated with the run status."""
