from typing import Any

from pydantic import BaseModel, ConfigDict

from .data import BaseData


class Snapshot(BaseModel):
    """Flow snapshot taken during a specific run.

    We allow for some fields to be None, so we can initialize this object without having to 
    await for the run to complete and send intermediate events with this.
    """
    model_config = ConfigDict(extra="ignore")

    v: str 
    """Versioning."""
    id: str
    """The snapshot's run identifier."""
    parent_id: str | None = None
    """The parent run's identifier. When null, the snapshot is a from a root run."""
    path: str
    """Snapshots can be stored by any Flow instance. Users can nest as many subflows as
    they want (with the same ids). We need this to uniquely identify where this snapshot comes from.
    """
    input: Any
    """Kwargs passed to Flow.run()"""
    output: Any | None = None
    """Output of Flow._collect_outputs()"""
    error: Any | None = None
    """Error raised during the execution of the flow."""
    t0: int 
    """Start time of the snapshot since epoch in ms."""
    t1: int | None = None
    """End time of the snapshot since epoch in ms."""
    steps: dict[str, Any] | None = {}
    """Run data of each step at the end of the run."""
    data: dict[str, BaseData] | None = {}
    """Data state of the flow at the end of the run."""
    usage: dict[str, int] | None = {}
    """Accumulated usage across all steps of the flow."""
