from ..snapshot import Snapshot
from .base import BaseSaver


class InMemorySaver(BaseSaver):
    """An in-memory state saver.

    This state saver stores snapshots in memory using a python in-memory list.

    Note:
        Only use `InMemorySaver` for debugging or testing purposes.
        For production use cases, use a persistent state saver like JSONLSaver or PostgresSaver.
    """

    def __init__(self, snapshots: list[Snapshot] | None = None) -> None:
        """Initialize an InMemorySaver instance.

        Args:
            snapshots: Optional list of Snapshot objects to initialize the saver with.
                      If None, starts with an empty list.
        """
        if snapshots is not None:
            self.snapshots = snapshots
        else:
            self.snapshots: list[Snapshot] = []
    

    def get(self, id: str) -> Snapshot | None:
        """Retrieve a snapshot by its ID.

        Args:
            id: The ID of the snapshot to retrieve.

        Returns:
            The matching Snapshot object if found, None otherwise.
            If multiple snapshots exist with the same ID, returns the most recent one.
        """
        for snapshot in self.snapshots[::-1]:
            if snapshot.id == id:
                return snapshot
        return None
    

    def get_last(
        self, 
        n: int = 1,
        parent_id: str | None = None,
        group_id: str | None = None,
    ) -> list[Snapshot]:
        """Retrieve the last n snapshots matching the specified criteria.

        Args:
            n: Number of snapshots to retrieve (default: 1).
            parent_id: If provided, only returns snapshots in the ancestry chain
                      starting from this parent ID.
            group_id: If provided, only returns snapshots belonging to this group.

        Returns:
            A list of matching Snapshot objects in chronological order.
            The list may contain fewer than n items if insufficient matches are found.
        """
        last_snapshots = []
        current_parent_id = parent_id
        for snapshot in self.snapshots[::-1]:
            if len(last_snapshots) >= n:
                break

            if snapshot.group_id != group_id:
                continue

            if not current_parent_id:
                last_snapshots.append(snapshot)
            elif snapshot.id == current_parent_id:
                last_snapshots.append(snapshot)
                current_parent_id = snapshot.parent_id

        return last_snapshots[::-1]
    

    def put(self, snapshot: Snapshot) -> None:
        """Store a new snapshot.

        Args:
            snapshot: The Snapshot object to store.

        Raises:
            ValueError: If a snapshot with the same ID already exists.

        Note:
            If the snapshot has no ID, one will be automatically assigned
            based on the current number of stored snapshots.
        """
        if snapshot.id is None:
            snapshot.id = str(len(self.snapshots))
        if self.get(snapshot.id) is not None:
            raise ValueError(f"Snapshot with id {snapshot.id} already exists.")
        self.snapshots.append(snapshot)
