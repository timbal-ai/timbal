from abc import ABC, abstractmethod

from ..data import Tracing


class TracingProvider(ABC):
    """Abstract base class for retrieving tracing data by run_id."""
    
    @classmethod
    @abstractmethod
    async def get_tracing(cls, run_id: str) -> Tracing | None:
        """Retrieve trace data for a specific run_id.
        
        Args:
            run_id: The run identifier to retrieve trace data for

        Returns:
            Tracing data or None if not found
        """
        pass

    @classmethod
    @abstractmethod
    async def put_tracing(cls, run_id: str, tracing: Tracing) -> None:
        """Store trace data for a specific run_id.
        
        Args:
            run_id: The run identifier to store trace data for
            tracing: The trace data to store
        """
        pass
