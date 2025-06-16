import asyncio
from typing import override

from ..events.base import Event
from .base import EventStore



class InMemoryEventStore(EventStore):
    """"""
    _events: list[Event] = []
    _lock: asyncio.Lock = asyncio.Lock()


    @override
    @classmethod
    async def add(cls, event: Event) -> None:
        """"""
        async with cls._lock:
            cls._events.append(event)

    
    @override
    @classmethod
    async def get(cls, type: type[Event], run_id: str, path: str) -> list[Event]:
        """"""
        async with cls._lock:
            return [
                event 
                for event in cls._events
                if event.run_id == run_id and event.path == path and isinstance(event, type)
            ]
        