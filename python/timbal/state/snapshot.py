from typing import Any

from pydantic import BaseModel

from .data import BaseData


class Snapshot(BaseModel):
    """State snapshot taken at a specific point in time."""
    
    v: str 
    """Versioning."""
    id: str | None = None
    """The snapshot's identifier (in the state saver's context).
    We allow for the id to be None, so we can defer the id assignment to the state saver later on.
    """
    parent_id: str | None = None
    """The parent snapshot's identifier (in the state saver's context).
    When null, the snapshot is a root snapshot.
    """
    group_id: str | None = None
    """The group's identifier (in the state saver's context).
    When null, the snapshot is part of the default group (if enabled in the by the saver).
    """
    flow_path: str
    """Snapshots can be stored by any Flow instance. Users can nest as many subflows as
    they want (with the same ids). We need this to uniquely identify where this snapshot comes from.
    """
    input: Any
    """Kwargs passed to Flow.run()"""
    output: Any
    """Output of Flow._collect_outputs()"""
    t0: int 
    """Start time of the snapshot since epoch in ms."""
    t1: int
    """End time of the snapshot since epoch in ms."""
    status: str
    """Status of the run {success, failed, ...}"""
    steps: list[Any]
    """Run data of each step at the end of the run."""
    data: dict[str, BaseData]
    """Data state of the flow at the end of the run."""
