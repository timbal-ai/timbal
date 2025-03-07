import json
from pathlib import Path

from pydantic import TypeAdapter
from uuid_extensions import uuid7

from ...types.models import dump
from ..data import Data
from ..snapshot import Snapshot
from .base import BaseSaver


class JSONLSaver(BaseSaver):
    """A JSONL state saver.

    This state saver stores snapshots in a JSONL file by serializing every snapshot into every line.

    Note:
        Only use `JSONLSaver` for debugging or testing purposes.
        This saver was implemented to test serialization and deserialization of snapshots.
        For production use cases, use a persistent state saver like `PostgresSaver`.
    """

    def __init__(self, path: Path) -> None:
        """Initialize a JSONLSaver instance.

        Args: 
            path: Path to the JSONl file that will store the snapshots. 
        """
        self.path = path
        # Ensure the directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file if it doesn't exist
        if not self.path.exists():
            self.path.touch()


    @staticmethod 
    def _load_snapshot_from_line(line: str) -> Snapshot:
        snapshot = json.loads(line.strip())
        snapshot["data"] = {
            k: TypeAdapter(Data).validate_python(v)
            for k, v in snapshot["data"].items()
        }
        return Snapshot(**snapshot)

    
    def get(self, id: str) -> Snapshot | None:
        """Retrieve a snapshot by its ID.

        Warning:
            This method loads the entire file into memory. For production use cases with large files,
            consider implementing a streaming approach that reads the file line by line.

        Args:
            id: The ID of the snapshot to retrieve.

        Returns:
            The matching Snapshot object if found, None otherwise.
            If multiple snapshots exist with the same ID, returns the most recent one.
        """
        with open(self.path) as f:
            for line in reversed(list(f)):
                snapshot = self._load_snapshot_from_line(line)
                if snapshot.id == id:
                    return snapshot
        return None

    
    def get_last(
        self, 
        n: int = 1,
        parent_id: str | None = None,
        group_id: str | None = None,
        flow_path: str | None = None,
    ) -> list[Snapshot]:
        """Retrieve the last n snapshots matching the specified criteria.

        Warning:
            This method loads the entire file into memory. For production use cases with large files,
            consider implementing a streaming approach that reads the file line by line.

        Args:
            n: Number of snapshots to retrieve (default: 1).
            parent_id: If provided, only returns snapshots in the ancestry chain
                       starting from this parent ID.
            group_id: If provided, only returns snapshots belonging to this group.
            flow_path: If provided, only returns snapshots from this flow.

        Returns:
            A list of matching Snapshot objects in chronological order.
            The list may contain fewer than n items if insufficient matches are found.
        """
        last_snapshots = []
        current_parent_id = parent_id
        with open(self.path) as f:
            for line in reversed(list(f)):
                if len(last_snapshots) >= n:
                    break

                snapshot = self._load_snapshot_from_line(line)

                if snapshot.group_id != group_id:
                    continue

                if snapshot.flow_path != flow_path:
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
            If the snapshot has no ID, one will be automatically assigned.
        """
        if snapshot.id is None:
            snapshot.id = uuid7(as_type="str")

        if self.get(snapshot.id) is not None:
            raise ValueError(f"Snapshot with id {snapshot.id} already exists.")

        with open(self.path, "a") as f:
            snapshot_dump = dump(snapshot)
            f.write(json.dumps(snapshot_dump) + "\n")
    