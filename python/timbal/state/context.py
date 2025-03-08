from pydantic import BaseModel, ConfigDict


class RunContext(BaseModel):
    """Context for a run.
    This is shared between all steps in a flow (including nested subflows).
    """
    # Allow for extra fields.
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    """Unique identifier for the run.
    We allow for the id to be None so the dev has more flexibility when trying things out.
    The existance of this field should be enforced when creating a Snapshot and using the state saver.
    """
    parent_id: str | None = None
    """Whether this run is a direct child of another run.
    Can be used to recursively retrieve all runs into a single list -> chat history.
    Can also be used to create a new branch from a specific run -> rewind.
    """
