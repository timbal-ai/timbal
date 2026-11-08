import asyncio
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..trace import Trace
from .base import TracingProvider

if TYPE_CHECKING:
    from ...context import RunContext


class _Database:
    """One connection per database file, shared by every configured provider on it.

    All access runs on worker threads and is serialized here: SQLite has one writer
    at a time anyway, and a queue in the process beats retrying ``SQLITE_BUSY``.
    """

    def __init__(self, path: Path, synchronous: str) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA synchronous={synchronous}")
        _ensure_schema(self.conn)
        # Per run: which spans are finished and committed to this file (not rewritten until the root ends),
        # and the order spans were first seen. Kept per connection: a file recreated after being deleted
        # holds none of them. Bounded: a run whose process died never ends.
        self.runs: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def alive(self) -> bool:
        return self.path.exists()

    def run_state(self, run_id: str) -> dict[str, Any]:
        state = self.runs.get(run_id)
        if state is None:
            state = self.runs[run_id] = {"stored": set(), "seq": {}}
            while len(self.runs) > _RUN_STATE_MAX:
                self.runs.popitem(last=False)
        else:
            self.runs.move_to_end(run_id)
        return state


_DATABASES: dict[tuple[str, str], _Database] = {}
_DATABASES_LOCK = threading.Lock()
_RUN_STATE_MAX = 2048


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id    TEXT    PRIMARY KEY,
            parent_id TEXT,
            spans     TEXT    NOT NULL,
            stored_at INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_parent_id ON runs (parent_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            run_id  TEXT    NOT NULL,
            call_id TEXT    NOT NULL,
            seq     INTEGER NOT NULL,
            span    TEXT    NOT NULL,
            PRIMARY KEY (run_id, call_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_claims (
            parent_id         TEXT    NOT NULL,
            approval_id       TEXT    NOT NULL,
            claimed_by_run_id TEXT    NOT NULL,
            claimed_at        INTEGER NOT NULL,
            PRIMARY KEY (parent_id, approval_id)
        )
    """)
    # Durable chat sessions (timbal.state.chat_store.SqliteChatSessionStore): one mutable row per session.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id  TEXT    PRIMARY KEY,
            agent       TEXT    NOT NULL,
            snapshot    TEXT    NOT NULL,
            version     INTEGER NOT NULL,
            lease_owner TEXT,
            lease_until REAL,
            updated_at  INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_session_inbox (
            id         TEXT    PRIMARY KEY,
            session_id TEXT    NOT NULL,
            item       TEXT    NOT NULL,
            posted_at  REAL    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS chat_session_inbox_session ON chat_session_inbox (session_id, posted_at)")
    conn.commit()


def database(path: Path | str, synchronous: str = "NORMAL") -> _Database:
    """The process's one connection to the SQLite file at ``path`` (created with the schema if absent)."""
    mode = synchronous.upper()
    if mode not in ("NORMAL", "FULL"):
        raise ValueError(f"synchronous must be 'NORMAL' or 'FULL', got {synchronous!r}")
    key = (str(Path(path).resolve()), mode)
    with _DATABASES_LOCK:
        db = _DATABASES.get(key)
        if db is None or not db.alive():  # the file was deleted under us: don't write to the old inode
            db = _DATABASES[key] = _Database(Path(path), mode)
        return db


class SqliteTracingProvider(TracingProvider):
    """SQLite tracing provider — local, persistent, and performant.

    Stores one row per span, so a save writes what changed since the last one
    instead of the whole run: the spans not yet stored finished, plus every span
    once the root has ended (a background child's late output lands then).
    One connection per file, WAL, ``synchronous=NORMAL`` by default: a commit
    survives the process crashing or being killed, not a power cut or kernel
    panic (``synchronous="FULL"`` for that, ~10x slower per save). Single
    process; several workers on one file serialize on SQLite's one writer.

    Configure via the base-class ``configured()`` method::

        provider = SqliteTracingProvider.configured(path=Path("traces.db"))
        agent = Agent(name="my_agent", model=..., tracing_provider=provider)

    Class-level attributes:

    - ``_path`` (Path | None): database file. Created on first write if absent.
    - ``synchronous`` (str): ``"NORMAL"`` (default) or ``"FULL"``.

    Uses only the stdlib ``sqlite3`` module — no extra dependencies.

    Schema::

        runs (run_id PRIMARY KEY, parent_id, spans, stored_at)
            -- spans: the whole trace as a JSON array once the run has ended
            -- ('' while it runs); earlier versions wrote it on every save
        spans (run_id, call_id, seq, span, PRIMARY KEY (run_id, call_id))
        approval_claims (parent_id, approval_id, claimed_by_run_id, claimed_at)
    """

    _path: Path | None = None
    synchronous: str = "NORMAL"

    @classmethod
    def _db(cls) -> _Database:
        if cls._path is None:
            raise RuntimeError(
                "SqliteTracingProvider._path is not set. "
                "Use SqliteTracingProvider.configured(path=Path(...)) to create a configured provider."
            )
        return database(cls._path, cls.synchronous)

    @classmethod
    @override
    async def get(cls, run_context: "RunContext") -> Trace | None:
        """Retrieve the parent run's trace (indexed lookups on ``run_id``).

        Returns None if the database does not exist or no matching record is found.
        """
        try:
            return await cls.get_strict(run_context)
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError):
            return None

    @classmethod
    @override
    async def get_strict(cls, run_context: "RunContext") -> Trace | None:
        """See base class: database and decoding errors raise."""
        if cls._path is None:
            return None
        try:
            # Not Path.exists(): it reads a failed stat (a permission error) as "no database". Checked at all
            # because connecting would create one.
            Path(cls._path).stat()
        except FileNotFoundError:
            return None
        parent_id = run_context.parent_id
        if parent_id is None:
            return None

        def _query() -> tuple[list[str], str, str | None] | None:
            db = cls._db()
            with db.lock:
                row = db.conn.execute(
                    "SELECT spans, parent_id FROM runs WHERE run_id = ?", (str(parent_id),)
                ).fetchone()
                if row is None:
                    return None
                spans = [
                    span
                    for (span,) in db.conn.execute(
                        "SELECT span FROM spans WHERE run_id = ? ORDER BY seq", (str(parent_id),)
                    )
                ]
            return spans, row[0], row[1]

        found = await asyncio.to_thread(_query)
        if found is None:
            return None
        spans, legacy, grandparent = found
        if spans:
            return Trace([json.loads(span) for span in spans], parent_id=grandparent)
        return Trace(json.loads(legacy), parent_id=grandparent) if legacy else None

    @classmethod
    @override
    async def claim_approval(cls, parent_id: str | None, approval_id: str, run_id: str) -> bool:
        """Atomically claim ``(parent_id, approval_id)``.

        SQLite enforces the single-consumer invariant with a unique primary
        key. ``INSERT OR IGNORE`` makes duplicate workers race safely: one row
        wins, every other worker observes the existing claimant and stops
        before executing the gated handler.
        """
        if parent_id is None:
            return True

        def _claim() -> bool:
            db = cls._db()
            with db.lock:
                with db.conn:  # committed, or rolled back on error: the shared connection never stays mid-transaction
                    cursor = db.conn.execute(
                        """
                        INSERT OR IGNORE INTO approval_claims
                            (parent_id, approval_id, claimed_by_run_id, claimed_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (str(parent_id), approval_id, run_id, int(time.time() * 1000)),
                    )
                if cursor.rowcount == 1:
                    return True
                row = db.conn.execute(
                    "SELECT claimed_by_run_id FROM approval_claims WHERE parent_id = ? AND approval_id = ?",
                    (str(parent_id), approval_id),
                ).fetchone()
            return bool(row and row[0] == run_id)

        return await asyncio.to_thread(_claim)

    @classmethod
    @override
    async def _store(cls, run_context: "RunContext") -> None:
        """Store the spans of the current run that changed since its last save.

        Raises:
            RuntimeError: If ``_path`` has not been set via ``configured()``.
        """
        db = cls._db()
        run_id = str(run_context.id)
        parent_id = str(run_context.parent_id) if run_context.parent_id else None
        spans = run_context._trace.as_records()
        root = next((span for span in spans if span.parent_call_id is None), None)
        ended = root is not None and root.t1 is not None

        state = db.run_state(run_id)
        order: dict[str, int] = state["seq"]
        rows = []
        finished = []
        for span in spans:
            seq = order.setdefault(span.call_id, len(order))
            if not ended and span.call_id in state["stored"]:
                continue
            rows.append((run_id, span.call_id, seq, json.dumps(span.model_dump(), default=str)))
            if span.t1 is not None:
                finished.append(span.call_id)
        stored_at = int(time.time() * 1000)
        # An ended run also gets the whole trace in runs.spans, as before: one write per run
        # keeps readers of that column working. While it runs, only its span rows are current.
        blob = "[" + ",".join(row[3] for row in rows) + "]" if ended else ""

        def _write() -> None:
            with db.lock, db.conn:  # one transaction: committed whole, or rolled back on any error
                db.conn.execute(
                    """
                    INSERT INTO runs (run_id, parent_id, spans, stored_at) VALUES (?, ?, ?, ?)
                    ON CONFLICT (run_id) DO UPDATE SET parent_id = excluded.parent_id, spans = excluded.spans,
                        stored_at = excluded.stored_at
                    """,
                    (run_id, parent_id, blob, stored_at),
                )
                db.conn.executemany(
                    "INSERT OR REPLACE INTO spans (run_id, call_id, seq, span) VALUES (?, ?, ?, ?)", rows
                )

        await asyncio.to_thread(_write)
        # Only once committed: a save that failed leaves them for the next one to write.
        if ended:
            # Later saves (background work outliving the root) rewrite everything again.
            state["stored"].clear()
        else:
            state["stored"].update(finished)
