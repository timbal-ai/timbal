from typing import Any

from .base import BaseEvent


class ChunkEvent(BaseEvent):
    """Event emitted when a chunk of data is received."""
    type: str = "CHUNK"

    chunk: Any
    """The chunk of output generated by the step generator."""
