from collections import UserDict
from typing import Any

from .span import Span


class Trace(UserDict):
    
    def __init__(self, data: dict[str, Any] | list[dict[str, Any]] | None = None):
        self._root_call_id = None

        if data is None:
            super().__init__({})
        elif isinstance(data, list):
            super().__init__({})
            for record in data:
                if "call_id" not in record:
                    raise ValueError("Missing call_id in trace record")
                self.__setitem__(record["call_id"], record)
        elif isinstance(data, dict):
            super().__init__({})
            for call_id, record in data.items():
                if "call_id" not in record:
                    raise ValueError("Missing call_id in trace record")
                self.__setitem__(call_id, record)
        else:
            raise ValueError(f"Invalid trace data type: {type(data)}")
    
    def __setitem__(self, key: str, value: Span | dict[str, Any]) -> None:
        if not isinstance(key, str):
            raise ValueError(f"Invalid trace key type: {type(key)}")
        if not isinstance(value, Span | dict):
            raise ValueError(f"Invalid trace value type: {type(value)}")
        
        if isinstance(value, dict):
            if "parent_call_id" not in value:
                raise ValueError("Missing parent_call_id in trace value")
            value = Span(**value)
        
        if value.parent_call_id is None:
            if self._root_call_id is not None:
                raise ValueError("Cannot set multiple root calls in trace data")
            self._root_call_id = key

        super().__setitem__(key, value)

    def model_dump(self) -> list[dict[str, Any]]:
        """Returns a list of trace record references ready for serialization."""
        return [span.model_dump() for span in self.data.values()]

    def as_records(self) -> list[Span]:
        """Returns a list of trace record references."""
        return list(self.data.values())

    def get_path(self, path: str) -> list[Span]:
        """Returns a list of trace record references for the given path."""
        return [span for span in self.data.values() if span.path == path]

    def get_level(self, path: str) -> list[Span]:
        """Returns a list of trace record references for the given path level."""
        level = path.count(".") + 1
        return [
            span
            for span in self.data.values()
            if span.path.count(".") == level and span.path.startswith(path)
        ]
