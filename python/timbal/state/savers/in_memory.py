import copy

from .. import get_run_context
from ..snapshot import Snapshot
from .base import BaseSaver


class InMemorySaver(BaseSaver):
    """An in-memory state saver.

    This state saver stores snapshots in memory using a python in-memory list.

    Note:
        Only use `InMemorySaver` for debugging or testing purposes.
        For production use cases, use a persistent state saver like `PostgresSaver`.
    """

    def __init__(
        self, 
        snapshots: list[Snapshot] | None = None,
    ) -> None:
        """Initialize an InMemorySaver instance.

        Args:
            snapshots: Optional list of Snapshot objects to initialize the saver with.
                       Defaults to an empty list.
        """
        self.snapshots = snapshots if snapshots is not None else []
    

    async def get_last(self, path: str) -> Snapshot | None:
        """See base class."""
        context = get_run_context()
        if context.parent_id is None:
            return None

        for snapshot in self.snapshots[::-1]:
            if snapshot.path == path and snapshot.id == context.parent_id:
                return snapshot

        return None
    

    async def put(self, snapshot: Snapshot) -> None:
        """See base class."""
        # Since we're appending snapshots to a python list and there's no intrinsic way of ensuring
        # unicity of ids, we need to check if the snapshot already exists.
        for snapshot_i in reversed(self.snapshots):
            if snapshot_i.id == snapshot.id and snapshot_i.path == snapshot.path:
                raise ValueError(f"Snapshot with id {snapshot.id} and path {snapshot.path} already exists.")

        # TODO Fix (max recursion error / cannot pickle bufferedreader instances) with deepcopy and File objects.
        # When saving artifacts in memory, we might modify LLM memories by reference. 
        # We perform a deepcopy to avoid modifying the original snapshot data.
        self.snapshots.append(copy.deepcopy(snapshot))
