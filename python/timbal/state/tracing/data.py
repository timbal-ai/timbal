from collections import UserDict
from typing import Any


class Tracing(UserDict):
    
    def __init__(self, data: dict[str, Any] | list[dict[str, Any]] | None = None):
        self._root_call_id = None

        if data is None:
            super().__init__({})
        elif isinstance(data, list):
            super().__init__({})
            for record in data:
                if "call_id" not in record:
                    raise ValueError("Missing call_id in tracing record")
                self.__setitem__(record["call_id"], record)
        elif isinstance(data, dict):
            super().__init__({})
            for call_id, record in data.items():
                if "call_id" not in record:
                    raise ValueError("Missing call_id in tracing record")
                self.__setitem__(call_id, record)
        else:
            raise ValueError(f"Invalid tracing data type: {type(data)}")
    
    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        if not isinstance(key, str):
            raise ValueError(f"Invalid tracing key type: {type(key)}")
        if not isinstance(value, dict):
            raise ValueError(f"Invalid tracing value type: {type(value)}")
        if not "parent_call_id" in value:
            raise ValueError("Missing parent_call_id in tracing value")
        if value["parent_call_id"] is None:
            if self._root_call_id is not None:
                raise ValueError("Cannot set multiple root calls in tracing data")
            self._root_call_id = key
        super().__setitem__(key, value)

    def as_records(self) -> list[dict[str, Any]]:
        """Returns a list of tracing record references."""
        return list(self.data.values())

    def get_path(self, path: str) -> list[dict[str, Any]]:
        """Returns a list of tracing record references for the given path."""
        return [trace for trace in self.data.values() if trace.get("path") == path]

    def get_level(self, path: str) -> list[dict[str, Any]]:
        """Returns a list of tracing record references for the given path level."""
        level = path.count(".") + 1
        return [
            trace
            for trace in self.data.values()
            if trace.get("path", "").count(".") == level and trace.get("path", "").startswith(path)
        ]
