from pydantic import BaseModel, ConfigDict


class BaseEvent(BaseModel):
    """Base class for all timbal events yielded during flow execution."""
    model_config = ConfigDict(extra="ignore", frozen=True) # Make immutable after creation

    type: str
    """The type of the event. This will be very useful for serializing and deserializing events."""
    run_id: str
    """The id of the run this event was emitted from."""
    parent_run_id: str | None = None
    """The id of the parent run (if any)."""
    path: str
    """The path of the element that yielded this event."""
    call_id: str
    """The id of the single execution in a run."""
    parent_call_id: str | None = None
    """The id of the parent call if this event comes from a nested runnable."""
    