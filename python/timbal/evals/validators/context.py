from pydantic import BaseModel, ConfigDict

from ...state.tracing.trace import Trace


class ValidationContext(BaseModel):
    """Context passed to all validators in an eval.

    Attributes:
        trace: The full execution trace being validated.
        cursor: Position in trace for sequential validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace: Trace
    cursor: int = 0
