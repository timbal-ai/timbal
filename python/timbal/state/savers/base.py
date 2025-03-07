from abc import ABC, abstractmethod

from ..context import RunContext
from ..snapshot import Snapshot


class BaseSaver(ABC):
    """Base class for creating a Flow state saver.

    State savers persist Flow states across multiple interactions.

    Note:
        When creating a custom state saver, consider implementing async
        versions to avoid blocking the main thread.
    """


    @abstractmethod 
    def get_last(
        self, 
        path: str,
        context: RunContext,
    ) -> Snapshot | None:
        """Retrieve the last snapshot matching the specified criteria.

        Args:
            path: Flows are nested structures. We don't ensure id uniqueness across nested subflows.
                  This path ensures uniqueness of every different BaseStep in the flow.
            context: The run context. For instance, if we're looking for a specific parent id or a group id.

        Returns:
            The last snapshot matching the specified criteria.
        """
        pass


    @abstractmethod
    def put(
        self, 
        snapshot: Snapshot, 
        context: RunContext,
    ) -> None:
        """Store a new snapshot.

        Args:
            snapshot: The Snapshot object to store.
            context: The run context. Might be used for authentication to an external service.
        """
        pass
