---
title: Saver
sidebar: 'docsSidebar'
---

# State Savers

State savers in Timbal provide persistence mechanisms for storing and retrieving flow execution states. They enable memory retention across sessions and allow you to implement custom storage solutions.

## Built-in State Savers

Timbal includes some built-in state savers:

### InMemorySaver

The simplest state saver that keeps everything in memory. Useful for development and testing:

```python
from timbal import Flow
from timbal.state.savers import InMemorySaver

flow = (
   Flow()
   .add_llm(memory_id="conversation")
   .compile(state_saver=InMemorySaver())
   )
```


### JSONLSaver

Persists states to a JSONL file, providing simple file-based storage:

```python
from timbal.state.savers import JSONLSaver

flow = (
   Flow()
   .add_llm(memory_id="conversation")
   .compile(state_saver=JSONLSaver("path/to/states.jsonl"))
   )
```


## Creating Custom State Savers

You can create your own state saver by inheriting from `BaseSaver`. You just need to implement three key methods:

1. `get(id: str) -> Snapshot | None`
   - Retrieves a specific snapshot by ID
   - Returns None if not found

2. `get_last(n: int = 1, parent_id: str | None = None, group_id: str | None = None) -> list[Snapshot]`
   - Retrieves the last n snapshots matching criteria
   - Supports filtering by parent_id and group_id
   - Returns snapshots in chronological order

3. `put(snapshot: Snapshot) -> None`
   - Stores a new snapshot
   - Should assign UUID if snapshot.id is None
   - Must prevent duplicate IDs

### The Snapshot Model

The `Snapshot` class represents a point-in-time state of a flow:

```python
class Snapshot(BaseModel):
   id: str | None = None # Unique identifier
   parent_id: str | None = None # ID of parent snapshot
   group_id: str | None = None # Group identifier
   data: dict[str, Any] # State data
   metadata: dict[str, Any] # Additional metadata
```