from pydantic import BaseModel, ConfigDict


class BaseEvent(BaseModel):
    """Base class for all timbal events yielded during execution."""
    # Allow storing extra fields in the model.
    model_config = ConfigDict(extra="allow")

    type: str
    """The type of the event. This will be very useful for serializing and deserializing events."""
    run_id: str
    """The id of the run this event was emitted from."""
    path: str
    """The path of the element that yielded this event."""

    # TODO Add if we need to add status text or status audio, etc.
    # TODO Add a proper event id
