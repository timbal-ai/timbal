"""ChunkEvent - Simple untyped streaming events.

.. deprecated::
    ChunkEvents will be deprecated in a future release in favor of DeltaEvents.
    DeltaEvents provide structured, typed streaming with better observability.
    Enable with TIMBAL_DELTA_EVENTS=true to migrate to the new system.
    
    See: timbal.types.events.delta for the replacement DeltaEvent system.
"""

from typing import Any, Literal

from .base import BaseEvent


class ChunkEvent(BaseEvent):
    """Event emitted when a chunk of data is received.
    
    .. deprecated::
        ChunkEvents will be deprecated in a future release. Use DeltaEvents instead
        for structured, typed streaming with semantic information about different
        content types (text, tool calls, thinking, etc.).
        
        Enable DeltaEvents by setting TIMBAL_DELTA_EVENTS=true.
        
        See :class:`~timbal.types.events.delta.DeltaEvent` for the replacement.
    
    Attributes:
        type: Event type discriminator, always "CHUNK"
        chunk: Untyped chunk of output data (can be any type)
    """
    type: Literal["CHUNK"] = "CHUNK"
    chunk: Any
