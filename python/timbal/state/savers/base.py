from abc import ABC, abstractmethod
from typing import Any

from ..snapshot import Snapshot


class BaseSaver(ABC):
    """Base class for creating a Flow state saver.

    State savers persist Flow states within and across multiple interactions.

    Note:
        When creating a custom state saver, consider implementing async
        versions to avoid blocking the main thread.
    """
    
    @abstractmethod
    def get(self, id: str) -> Snapshot | None:
        pass


    @abstractmethod 
    def get_last(
        self, 
        n: int = 1, 
        parent_id: str | None = None, 
        group_id: str | None = None,
        flow_path: str | None = None,
    ) -> list[Snapshot]:
        pass


    @abstractmethod
    def put(self, snapshot: Snapshot) -> None:
        pass
