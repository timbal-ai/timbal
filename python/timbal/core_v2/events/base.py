from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

EventData = TypeVar("EventData", bound=BaseModel)


class Event(BaseModel, Generic[EventData]):
    """"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    # ? Add an event id
    type: str = Field(
        ..., 
        description="Type of the event.",
    )
    run_id: str = Field(
        ...,
        description="ID of the run.",
    )
    path: str = Field(
        ...,
        description="Path of the runnable component that generated the event.",
    )
    # ? The path identifies what runnable component generated the event, but... do we need another identifier to group events?
    data: EventData = Field(
        ...,
        description="Data associated with the event.",
    )
    ts: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of the event (UTC).",
    )
    # ? Some mechanism to return status feedback or something
