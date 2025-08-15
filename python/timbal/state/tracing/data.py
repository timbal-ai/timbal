from collections import UserDict
from typing import Any


class Tracing(UserDict):
    """Container for execution trace data with custom serialization and display methods."""
    
    # TODO Make this method bulletproof
    def __init__(self, data: dict[str, Any] | list[dict[str, Any]] | None = None):
        """Initialize tracing container.
        
        Args:
            data: Optional initial trace data dictionary
        """
        if data is None:
            super().__init__({})
        elif isinstance(data, list):
            super().__init__({
                record["call_id"]: record
                for record in data
            })
        elif isinstance(data, dict):
            super().__init__(data)
        else:
            raise ValueError(f"Invalid tracing data type: {type(data)}")

    def to_records(self) -> list[dict[str, Any]]:
        """Convert the tracing data to a list of dictionaries, ordered by call_id (UUID v7)."""
        return [
            {"call_id": call_id, **trace}
            for call_id, trace in sorted(self.data.items(), key=lambda x: x[0] or "")
        ]

    def get_root(self) -> dict[str, Any]:
        """Get the root trace record."""
        trace_data = self.data.get(None)
        if trace_data is None:
            raise ValueError("Root trace not found")
        return {"call_id": None, **trace_data}

    def get_path(self, path: str) -> list[dict[str, Any]]:
        """Get the traces at a specific path, ordered by call_id (UUID v7)."""
        return [
            {"call_id": call_id, **trace}
            for call_id, trace in sorted(
                ((cid, t) for cid, t in self.data.items() if t.get("path") == path),
                key=lambda x: x[0] or ""
            )
        ]

    def get_level(self, path: str) -> list[dict[str, Any]]:
        """Get the traces at a specific level by path, ordered by call_id (UUID v7)."""
        level = path.count(".") + 1
        return [
            {"call_id": call_id, **trace}
            for call_id, trace in sorted(
                ((cid, t) for cid, t in self.data.items() 
                 if t.get("path", "").count(".") == level and t.get("path", "").startswith(path)),
                key=lambda x: x[0] or ""
            )
        ]
