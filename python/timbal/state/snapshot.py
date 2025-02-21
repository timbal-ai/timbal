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
    When null, the snapshot is part of the default group.
    """
    ts: int 
    """The snapshot's timestamp in milliseconds since the Unix epoch."""
    data: dict[str, BaseData]
    """The snapshot's data."""
    elapsed_time: int 
    """The elapsed time in milliseconds of execution time to get to this snapshot state."""
    # TODO Add status, error, usage, etc.
