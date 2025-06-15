from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseCollector(ABC, BaseModel):
    """"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )


    @abstractmethod
    def handle_chunk(self, chunk: Any) -> Any | None:
        """"""
        pass

    @abstractmethod
    def collect(self) -> Any:
        """"""
        pass
