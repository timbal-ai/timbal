from abc import ABC, abstractmethod

from ...state.tracing.trace import Trace


class BaseValidator(ABC):
    type: str
    path: str

    def __init__(self, type: str, path: str, **kwargs) -> None:  # noqa: ARG002
        self.type = type
        self.path = path

    @abstractmethod
    async def run(self, trace: Trace, **kwargs) -> bool:
        """Validate the trace. Returns True if validation passes."""
        ...
