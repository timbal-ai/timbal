import asyncio
import fcntl
import json
import time
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

    Stores one JSON record per run in a file. Each line has the form::

        {"run_id": "...", "parent_id": "...", "spans": [{...}, ...]}

    Configure via the base-class ``configured()`` method::

        provider = JsonlTracingProvider.configured(_path=Path("traces.jsonl"))
        agent = Agent(name="my_agent", model=..., tracing_provider=provider)

    Class-level attributes:

    - ``_path`` (Path | None): output file. Created on first write if absent.
    - ``_lock`` (asyncio.Lock | None): write lock. Initialised lazily per subclass.

    The file is created automatically if it does not exist.
    Multiple concurrent runs append safely via an asyncio lock.

    .. warning::

        **Not suitable for production use.** Every ``_store()`` call reads and
        rewrites the entire file to update the record for a given run_id
        (O(n) in lines). This is fine for local development, debugging, and
        tests, but will degrade with large trace files or high-throughput
        workloads. For production, use ``PlatformTracingProvider`` or implement
        a provider backed by a proper database.

    **Why keep it at all?**

    The human-readable JSON format makes this provider uniquely valuable for
    testing serialization and deserialization correctness. Unlike
    ``InMemoryTracingProvider`` (which stores live Python objects) and
    ``SqliteTracingProvider`` (which stores opaque blobs), every round-trip
    through ``JsonlTracingProvider`` exercises the full ``model_dump()`` →
    ``json.dumps()`` → ``json.loads()`` → ``Trace(spans)`` pipeline. This
    catches bugs that would only surface in production — e.g. non-serialisable
    types in span fields, loss of precision in numeric values, or discrepancies
    between the live ``_memory_dump`` private attribute and its reconstructed
    counterpart after a reload. Use it in tests whenever you want to verify that
    spans survive a real serialization round-trip.

    This is also a reference implementation — a practical starting point for
    building your own provider (OTel, Langfuse, Datadog, etc.). The interface
    is two methods: ``put()`` to persist a completed run, ``get()`` to retrieve
    a parent run's trace for session chaining. Everything else is up to you.
    """

    _path: Path | None = None
    _lock: asyncio.Lock | None = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def _approval_claims_path(cls) -> Path:
        assert cls._path is not None
        return cls._path.with_suffix(cls._path.suffix + ".approval_claims.json")

    @classmethod
    def _approval_claims_lock_path(cls) -> Path:
        assert cls._path is not None
        return cls._path.with_suffix(cls._path.suffix + ".approval_claims.lock")

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
    async def claim_approval(cls, parent_id: str | None, approval_id: str, run_id: str) -> bool:
        """Claim ``(parent_id, approval_id)`` using a sidecar JSON file.

        JSONL itself stores one record per run, so mutating the parent record
        for a tiny approval-claim row would be awkward and race-prone. A
        sidecar file keeps the trace format unchanged while still giving local
        development and tests a durable cross-process lock via ``fcntl``.
        """
        if parent_id is None:
            return True
        if cls._path is None:
            raise RuntimeError(
                "JsonlTracingProvider._path is not set. "
                "Use JsonlTracingProvider.configured(_path=Path(...)) to create a configured provider."
            )

        def _claim() -> bool:
            claims_path = cls._approval_claims_path()
            lock_path = cls._approval_claims_lock_path()
            claims_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.parent.mkdir(parents=True, exist_ok=True)

            with lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        claims = json.loads(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else {}
                    except json.JSONDecodeError:
                        claims = {}

                    key = f"{parent_id}:{approval_id}"
                    existing = claims.get(key)
                    if existing is not None:
                        return existing.get("claimed_by_run_id") == run_id

                    claims[key] = {
                        "parent_id": parent_id,
                        "approval_id": approval_id,
                        "claimed_by_run_id": run_id,
                        "claimed_at": int(time.time() * 1000),
                    }
                    tmp_path = claims_path.with_suffix(claims_path.suffix + ".tmp")
                    tmp_path.write_text(json.dumps(claims, sort_keys=True), encoding="utf-8")
                    tmp_path.replace(claims_path)
                    return True
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        return await asyncio.to_thread(_claim)

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
        run_id = str(run_context.id)
        new_record = {
            "run_id": run_id,
            "parent_id": str(run_context.parent_id) if run_context.parent_id else None,
            "spans": run_context._trace.model_dump(),
        }
        new_line = json.dumps(new_record, default=str) + "\n"
        async with cls._get_lock():
            # Update existing record if present, otherwise append — mirrors the
            # InMemoryTracingProvider dict overwrite behaviour. Providers emit
            # intermediate snapshots on each span completion, so we keep only the
            # latest (most complete) state per run_id, preventing file bloat.
            if cls._path.exists():
                lines = cls._path.read_text(encoding="utf-8").splitlines(keepends=True)
                for i, line in enumerate(lines):
                    try:
                        if json.loads(line).get("run_id") == run_id:
                            lines[i] = new_line
                            cls._path.write_text("".join(lines), encoding="utf-8")
                            return
                    except json.JSONDecodeError:
                        continue
            with cls._path.open("a", encoding="utf-8") as f:
                f.write(new_line)
