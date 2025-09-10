from abc import ABC, abstractmethod
from typing import Any

from ..state.context import RunContext


class EventCollector(ABC):
    """Base abstract class for all event collectors with internal state management."""
    
    def __init__(self, run_context: RunContext, start: float):
        self._run_context = run_context
        self._start = start
    
    @classmethod
    @abstractmethod
    def can_handle(cls, event: Any) -> bool:
        """Check if this collector can handle the given event type."""
        pass
    
    @abstractmethod
    def process(self, event: Any) -> Any:
        """Process the event and update internal state.
        
        Args:
            event: The event to process
            
        Returns:
            Processed content if available for streaming, None otherwise
        """
        pass
    
    @abstractmethod
    def collect(self) -> Any:
        """Return the collected results in the format appropriate for this collector type."""
        pass
