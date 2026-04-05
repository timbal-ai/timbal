import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..trace import Trace
from .base import TracingProvider

if TYPE_CHECKING:
    from ...context import RunContext


class JsonlTracingProvider(TracingProvider):
    """JSONL file tracing provider.

    Appends one JSON record per run to a file. Each line has the form::

        {"run_id": "...", "parent_id": "...", "spans": [{...}, ...]}

    Configure via the base-class ``configured()`` method::

        provider = JsonlTracingProvider.configured(_path=Path("traces.jsonl"))
        agent = Agent(name="my_agent", model=..., tracing_provider=provider)

    Class-level attributes:

    - ``_path`` (Path | None): output file. Created on first write if absent.
    - ``_lock`` (asyncio.Lock | None): write lock. Initialised lazily per subclass.

    The file is created automatically if it does not exist.
    Multiple concurrent runs append safely via an asyncio lock.

    This is a reference implementation — a practical starting point for building
    your own provider (OTel, Langfuse, Datadog, etc.). The interface is two methods:
    ``put()`` to persist a completed run, ``get()`` to retrieve a parent run's trace
    for session chaining. Everything else is up to you.
    """

    _path: Path | None = None
    _lock: asyncio.Lock | None = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    @override
    async def get(cls, run_context: "RunContext") -> Trace | None:
        """Retrieve the parent run's trace from the JSONL file.

        Scans the file for a record whose run_id matches run_context.parent_id.
        Returns None if the file does not exist or no matching record is found.
        """
        if cls._path is None or not cls._path.exists():
            return None
        parent_id = run_context.parent_id
        if parent_id is None:
            return None
        try:
            with cls._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("run_id") == str(parent_id):
                        return Trace(record["spans"])
        except (OSError, json.JSONDecodeError):
            return None
        return None

    @classmethod
    @override
    async def _store(cls, run_context: "RunContext") -> None:
        """Append the current run's trace as a JSON line.

        Raises:
            RuntimeError: If ``_path`` has not been set via ``configured()``.
        """
        if cls._path is None:
            raise RuntimeError(
                "JsonlTracingProvider._path is not set. "
                "Use JsonlTracingProvider.configured(_path=Path(...)) to create a configured provider."
            )
        record = {
            "run_id": str(run_context.id),
            "parent_id": str(run_context.parent_id) if run_context.parent_id else None,
            "spans": run_context._trace.model_dump(),
        }
        line = json.dumps(record, default=str) + "\n"
        async with cls._get_lock():
            with cls._path.open("a", encoding="utf-8") as f:
                f.write(line)
