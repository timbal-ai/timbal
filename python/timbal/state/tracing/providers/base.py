from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .. import Tracing

if TYPE_CHECKING:
    from ...context import RunContext


class TracingProvider(ABC):
    """Abstract base class for retrieving tracing data by run_id."""
    
    @classmethod
    @abstractmethod
    async def get(cls, run_context: "RunContext") -> Tracing | None:
        """Retrieve the parent run's tracing data.
        
        Args:
            run_context: The run context to retrieve tracing data for

        Returns:
            Tracing data or None if not found
        """
        pass

    @classmethod
    @abstractmethod
    async def put(cls, run_context: "RunContext") -> None:
        """Store the tracing data for a run.
        
        Args:
            run_context: The run context to store tracing data for
        """
        pass
