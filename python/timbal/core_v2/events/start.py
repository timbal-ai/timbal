from typing import Literal

from pydantic import BaseModel

from .base import Event


class StartEventData(BaseModel):
    """"""
    pass


class StartEvent(Event[StartEventData]):
    """"""
    type: Literal["start"] = "start"
