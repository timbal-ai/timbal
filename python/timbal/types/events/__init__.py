# ruff: noqa: F401
from typing import Annotated

from pydantic import Field

from .approval import ApprovalEvent
from .base import BaseEvent
from .delta import DeltaEvent
from .interaction import InteractionEvent
from .output import OutputEvent
from .start import StartEvent

# Create a discriminated union of all possible event types.
# Pydantic will use the 'type' field to determine which model to use.
Event = Annotated[
    StartEvent | OutputEvent | DeltaEvent | ApprovalEvent | InteractionEvent,
    Field(discriminator="type"),
]
