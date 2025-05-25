from pydantic import BaseModel, ConfigDict

from .turn import Turn


class Test(BaseModel):
    """Represents a single evaluation test case, identified by a unique name."""
    model_config = ConfigDict(extra="ignore")

    name: str
    """The name for the test.
    This name should be unique within the list of tests.
    """
    description: str | None = None
    """Additional description for the test."""
    turns: list[Turn]
    """The list of turns to run for this test."""
