from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunContext(BaseModel):
    """Context for a run.
    This is shared between all steps in a flow (including nested subflows).
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    id: str = Field(
        ...,
        description="Unique identifier for the run.",
    )
    parent_id: str | None = Field(
        None,
        description="Whether this run is a direct child of another run.",
    )
    # TODO Add data and more stuff
    usage: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Usage data for the run. Stored as key-value pairs."
            "e.g. 'gpt-4.1:input_text_tokens': 1000"
        ),
    )


    def update_usage(self, key: str, value: int) -> None:
        """"""
        if key in self.usage:
            self.usage[key] += value
        else:
            self.usage[key] = value
