# ruff: noqa: F401
from typing import Annotated

from pydantic import Field

from .base import BaseEvent
from .chunk import ChunkEvent
from .delta import DeltaEvent
from .output import OutputEvent
from .start import StartEvent

# Create a discriminated union of all possible event types.
# Pydantic will use the 'type' field to determine which model to use.
Event = Annotated[
    StartEvent | OutputEvent | ChunkEvent | DeltaEvent,
    Field(discriminator="type"),
]
