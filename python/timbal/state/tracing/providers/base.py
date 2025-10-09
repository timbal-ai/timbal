from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..trace import Trace

if TYPE_CHECKING:
    from ...context import RunContext


class TracingProvider(ABC):
    """Abstract base class for retrieving tracing data by run_id."""
    
    @classmethod
    @abstractmethod
    async def get(cls, run_context: "RunContext") -> Trace | None:
        """Retrieve the parent run's trace data.
        
        Args:
            run_context: The run context to retrieve trace data for

        Returns:
            Trace data or None if not found
        """
        pass

    @classmethod
    @abstractmethod
    async def put(cls, run_context: "RunContext") -> None:
        """Store the trace data for a run.
        
        Args:
            run_context: The run context to store trace data for
        """
        pass
