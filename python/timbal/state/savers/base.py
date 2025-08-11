from abc import ABC, abstractmethod

from ..snapshot import Snapshot


class BaseSaver(ABC):
    """Base class for creating a Flow state saver.

    State savers persist Flow states across multiple interactions.
    """

    @abstractmethod 
    async def get_last(self, path: str) -> Snapshot | None:
        """Retrieve the last snapshot matching the specified criteria.

        Args:
            path: Flows are nested structures. We don't ensure id uniqueness across nested subflows.
                  This path ensures uniqueness of every different BaseStep in the flow.

        Returns:
            The last snapshot matching the specified criteria.
        """
        pass

    @abstractmethod
    async def put(self, snapshot: Snapshot) -> None:
        """Store a new snapshot.

        Args:
            snapshot: The Snapshot object to store.
        """
        pass
