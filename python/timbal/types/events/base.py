from typing import Any

from pydantic import BaseModel, ConfigDict

from ..models import dump
from ...state.context import RunContext


class BaseEvent(BaseModel):
    """Base class for all timbal events yielded during flow execution."""
    # Allow storing extra fields in the model and make immutable after creation.
    model_config = ConfigDict(extra="allow", frozen=True)

    type: str
    """The type of the event. This will be very useful for serializing and deserializing events."""
    run_id: str
    """The id of the run this event was emitted from."""
    path: str
    """The path of the element that yielded this event."""
    
    @classmethod
    async def build(cls, **data):
        """Factory method to create an event with pre-computed dump."""
        instance = cls(**data)
        # Compute and store the dump
        object.__setattr__(instance, '_dump', await dump(instance))
        return instance
    
    @property
    def dump(self) -> dict[str, Any]:
        """Pre-computed dump of the event."""
        if hasattr(self, '_dump'):
            return self._dump
        else:
            raise RuntimeError("Event was not created using BaseEvent.build() - use BaseEvent.build() instead of direct instantiation")
