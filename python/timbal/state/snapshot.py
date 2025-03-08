from typing import Any

from pydantic import BaseModel

from .data import BaseData


class Snapshot(BaseModel):
    """Flow snapshot taken during a specific run.

    We allow for some fields to be None, so we can initialize this object without having to 
    await for the run to complete and send intermediate events with this.
    """
    
    v: str 
    """Versioning."""
    id: str
    """The snapshot's identifier (in the state saver's context).
    We allow for the id to be None, so we can defer the id assignment to the state saver later on.
    """
    parent_id: str | None = None
    """The parent snapshot's identifier (in the state saver's context).
    When null, the snapshot is a root snapshot.
    """
    path: str
    """Snapshots can be stored by any Flow instance. Users can nest as many subflows as
    they want (with the same ids). We need this to uniquely identify where this snapshot comes from.
    """
    input: Any
    """Kwargs passed to Flow.run()"""
    output: Any | None = None
    """Output of Flow._collect_outputs()"""
    t0: int 
    """Start time of the snapshot since epoch in ms."""
    t1: int | None = None
    """End time of the snapshot since epoch in ms."""
    steps: list[Any] | None = None
    """Run data of each step at the end of the run."""
    data: dict[str, BaseData] | None = None
    """Data state of the flow at the end of the run."""
