import asyncio
import json
import sqlite3
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


class SqliteTracingProvider(TracingProvider):
    """SQLite tracing provider — local, persistent, and performant.

    Stores one row per run keyed by ``run_id``. Intermediate snapshots during
    a run are upserted in-place via ``INSERT OR REPLACE`` — O(log n) versus the
    O(n) full-file rewrite of ``JsonlTracingProvider``.

    Configure via the base-class ``configured()`` method::

        provider = SqliteTracingProvider.configured(_path=Path("traces.db"))
        agent = Agent(name="my_agent", model=..., tracing_provider=provider)

    Class-level attributes:

    - ``_path`` (Path | None): database file. Created on first write if absent.

    The database is created automatically if it does not exist. WAL journal
    mode is enabled so concurrent reads never block writes. Uses only the
    stdlib ``sqlite3`` module — no extra dependencies.

    Schema::

        runs (
            run_id    TEXT PRIMARY KEY,
            parent_id TEXT,
            spans     TEXT NOT NULL,   -- JSON array of span dicts
            stored_at INTEGER NOT NULL -- Unix ms timestamp
        )
    """

    _path: Path | None = None
    _lock: asyncio.Lock | None = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def _connect(cls) -> sqlite3.Connection:
        conn = sqlite3.connect(str(cls._path), timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @classmethod
    @override
    async def get(cls, run_context: "RunContext") -> Trace | None:
        """Retrieve the parent run's trace from the SQLite database.

        Performs an indexed primary-key lookup on ``run_id``.
        Returns None if the database does not exist or no matching record is found.
        """
        if cls._path is None or not cls._path.exists():
            return None
        parent_id = run_context.parent_id
        if parent_id is None:
            return None

        def _query() -> str | None:
            conn = cls._connect()
            try:
                row = conn.execute(
                    "SELECT spans FROM runs WHERE run_id = ?",
                    (str(parent_id),),
                ).fetchone()
            finally:
                conn.close()
            return row[0] if row else None

        try:
            spans_json = await asyncio.to_thread(_query)
        except (OSError, sqlite3.Error):
            return None

        if spans_json is None:
            return None

        try:
            return Trace(json.loads(spans_json))
        except (json.JSONDecodeError, ValueError):
            return None

    @classmethod
    @override
    async def _store(cls, run_context: "RunContext") -> None:
        """Upsert the current run's trace into the SQLite database.

        Raises:
            RuntimeError: If ``_path`` has not been set via ``configured()``.
        """
        if cls._path is None:
            raise RuntimeError(
                "SqliteTracingProvider._path is not set. "
                "Use SqliteTracingProvider.configured(_path=Path(...)) to create a configured provider."
            )

        run_id = str(run_context.id)
        parent_id = str(run_context.parent_id) if run_context.parent_id else None
        spans_json = json.dumps(run_context._trace.model_dump(), default=str)
        stored_at = int(time.time() * 1000)

        def _write() -> None:
            conn = cls._connect()
            try:
                # CREATE TABLE / INDEX are idempotent — SQLite caches the schema
                # after the first call so these are effectively free on subsequent writes.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id    TEXT    PRIMARY KEY,
                        parent_id TEXT,
                        spans     TEXT    NOT NULL,
                        stored_at INTEGER NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_parent_id ON runs (parent_id)")
                conn.execute(
                    "INSERT OR REPLACE INTO runs (run_id, parent_id, spans, stored_at) VALUES (?, ?, ?, ?)",
                    (run_id, parent_id, spans_json, stored_at),
                )
                conn.commit()
            finally:
                conn.close()

        async with cls._get_lock():
            await asyncio.to_thread(_write)
