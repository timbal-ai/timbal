from typing import Any, Literal

from pydantic import BaseModel, Field

from .base import Event


class ChunkEventData(BaseModel):
    """"""
    chunk: Any = Field(
        ...,
        description="The chunk of the output.",
    )


class ChunkEvent(Event[ChunkEventData]):
    """"""
    type: Literal["chunk"] = "chunk"
