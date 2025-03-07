import json
from pathlib import Path

from pydantic import TypeAdapter

from ...types.models import dump
from ..context import RunContext
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

    
    def get_last(
        self, 
        path: str,
        context: RunContext,
    ) -> Snapshot | None:
        """See base class.

        Warning:
            This method loads the entire file into memory. For production use cases with large files,
            consider implementing a streaming approach that reads the file line by line.
        """
        with open(self.path) as f:
            for line in reversed(list(f)):
                snapshot = self._load_snapshot_from_line(line)

                if snapshot.group_id != context.group_id:
                    continue

                if snapshot.path != path:
                    continue

                if context.parent_id is None:
                    return snapshot
                elif snapshot.id == context.parent_id:
                    return snapshot

        return None
    

    def put(
        self, 
        snapshot: Snapshot,
        context: RunContext, # noqa: ARG002
    ) -> None:
        """See base class."""
        # Since we're appending lines to a file and there's no intrinsic way of ensuring
        # unicity of ids, we need to check if the snapshot already exists.
        with open(self.path) as f:
            for line in reversed(list(f)):
                snapshot = self._load_snapshot_from_line(line)
                if snapshot.id == id:
                    raise ValueError(f"Snapshot with id {snapshot.id} already exists.")

        with open(self.path, "a") as f:
            snapshot_dump = dump(snapshot)
            f.write(json.dumps(snapshot_dump) + "\n")
    