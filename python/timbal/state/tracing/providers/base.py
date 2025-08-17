from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..data import Tracing

if TYPE_CHECKING:
    from ...context import RunContext


class TracingProvider(ABC):
    """Abstract base class for retrieving tracing data by run_id."""
    
    @classmethod
    @abstractmethod
    async def get(cls, run_id: str) -> Tracing | None:
        """Retrieve trace data for a specific run_id.
        
        Args:
            run_id: The run identifier to retrieve trace data for

        Returns:
            Tracing data or None if not found
        """
        pass

    @classmethod
    @abstractmethod
    async def put(cls, run_context: "RunContext") -> None:
        """Store trace data for a run.
        
        Args:
            run_context: The run context to store trace data for
        """
        pass
