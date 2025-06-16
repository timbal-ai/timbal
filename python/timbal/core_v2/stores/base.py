from abc import ABC, abstractmethod

from ..events.base import Event


class EventStore(ABC):
    """"""


    @classmethod
    @abstractmethod
    async def add(cls, event: Event) -> None:
        """"""
        pass

    
    @classmethod
    @abstractmethod
    async def get(cls, type: type[Event], run_id: str, path: str) -> list[Event]:
        """"""
        pass
        