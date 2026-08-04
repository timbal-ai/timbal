from typing import Any

from ..._slots import SlotModel


class BaseEvent(SlotModel):
    """Base class for all timbal events yielded during flow execution.

    Events are plain ``__slots__`` classes (not pydantic models) because they
    are constructed on every runnable call and every streamed delta. Unknown
    keyword arguments are ignored on construction (the old pydantic config was
    ``extra="ignore"``) so wire payloads with extra keys keep deserializing.
    """

    __slots__ = ("run_id", "parent_run_id", "path", "call_id", "parent_call_id")

    type: str = ""
    """The type of the event. This will be very useful for serializing and deserializing events."""

    _FIELDS = ("type", "run_id", "parent_run_id", "path", "call_id", "parent_call_id")

    def __init__(
        self,
        *,
        run_id: str,
        path: str,
        call_id: str,
        parent_run_id: str | None = None,
        parent_call_id: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.run_id = run_id
        """The id of the run this event was emitted from."""
        self.parent_run_id = parent_run_id
        """The id of the parent run (if any)."""
        self.path = path
        """The path of the element that yielded this event."""
        self.call_id = call_id
        """The id of the single execution in a run."""
        self.parent_call_id = parent_call_id
        """The id of the parent call if this event comes from a nested runnable."""
