import asyncio
import json
import os
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


class _Writes:
    """The writes of one run that are on their way to a file, in the order their snapshots were taken.

    Kept from the first of them until the last is over (committed, failed, found older than what is stored,
    or dropped before a worker took it), also once its caller has been cancelled: while one of them can still
    commit, a later save of the run has to be able to tell which of the two is newer. With none left there is
    nothing to order, and the run's next save starts another.
    """

    __slots__ = ("pending", "snapshots", "committed")

    def __init__(self) -> None:
        self.pending = 0  # not over yet
        self.snapshots = 0  # the number of the latest one taken
        self.committed = 0  # the newest one stored


class _FileOrder:
    """What orders the writes of runs to one file, shared by every connection to it (``NORMAL`` and ``FULL``
    providers on one file get a connection each).

    ``writing``: per run with a write on its way, its ``_Writes``. ``lock``: held while a snapshot is numbered,
    for a few statements; re-entrant, since a save left suspended when its loop was closed is closed by the
    garbage collector, whose cleanup asks for it on whatever thread is allocating, one that holds it included.
    ``commit``: held from the comparison with what is stored to the end of the commit, on any connection to the
    file.
    """

    __slots__ = ("commit", "lock", "writing")

    def __init__(self) -> None:
        self.writing: dict[str, _Writes] = {}
        self.lock = threading.RLock()
        self.commit = threading.Lock()


class _Database:
    """One connection per database file, shared by every configured provider on it.

    All access runs on worker threads and is serialized here: SQLite has one writer
    at a time anyway, and a queue in the process beats retrying ``SQLITE_BUSY``.
    """

    def __init__(self, path: Path, synchronous: str, order: _FileOrder | None = None) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA synchronous={synchronous}")
        _ensure_schema(self.conn)
        # Which file this is a connection to: the path can come to name another one (see alive()).
        self._file = self._file_at_path()
        # Per run: which spans are finished and committed to this file (not rewritten until the root ends),
        # and the order spans were first seen. Kept per connection: a file recreated after being deleted
        # holds none of them. Bounded: a run whose process died never ends.
        self.runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # Per run with a write on its way: what orders them (_Writes). Not in ``runs``: that cache lets go
        # of a run however many of its writes are still out, and one parked on its worker meanwhile would
        # then be compared with nothing. Bounded by the writes in flight. Shared by every connection to the
        # file (_FileOrder): two connections are two writers of the same rows. The lock is held while a
        # snapshot is numbered and read, on the event loop too: never while the connection is in use.
        self.order = _FileOrder() if order is None else order
        self.writing = self.order.writing
        self.writing_lock = self.order.lock

    def _file_at_path(self) -> tuple[int, int] | None:
        try:
            found = os.stat(self.path)
        except OSError:
            return None
        return (found.st_dev, found.st_ino)

    def alive(self) -> bool:
        """Whether the path still names the file this connection opened.

        Not just whether a file is there: one deleted and written again (by another connection, by a
        restore) is a different file, and this connection would go on writing to the one that is gone.
        """
        return self._file is not None and self._file_at_path() == self._file

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
# Per file (device and inode, which every name of the file shares: case-insensitive paths, links): what orders its
# writes, whichever connection makes them. Under _DATABASES_LOCK.
_ORDERS: dict[Any, _FileOrder] = {}
_DATABASES_LOCK = threading.Lock()
_RUN_STATE_MAX = 2048
# The parent's connections, in a forked child: kept referenced so that they are never closed there.
_INHERITED_AFTER_FORK: list[_Database] = []


def _forget_databases_after_fork() -> None:
    """In a forked child (a server that opened a file and then forked its workers): open again.

    A connection shared across a fork is two processes on one connection's file locks and WAL index, which
    SQLite does not support (lock errors, a corrupted WAL). And a lock a parent thread held at the fork,
    the registry's or a connection's, stays held in the child with nobody left to let go of it.

    The inherited connections are not closed: closing one here would let go of the file locks the parent
    holds through it. They are kept referenced instead, so that collecting them never closes them either.
    """
    global _DATABASES_LOCK
    _INHERITED_AFTER_FORK.extend(_DATABASES.values())
    _DATABASES.clear()
    _ORDERS.clear()  # its locks may be held by a parent thread
    _DATABASES_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_databases_after_fork)


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
            updated_at  INTEGER NOT NULL,
            has_work    INTEGER
        )
    """)
    # has_work: 1 when the session would run something without a client asking (SessionSnapshot.has_work),
    # 0 when not, written with every save. NULL only in a row from before the column, until it is written
    # again. Startup asks for the sessions with work: idle ones stay in the table until deleted and soon
    # outnumber them, so that question is answered from an index over the rows that are not 0.
    if "has_work" not in {column[1] for column in conn.execute("PRAGMA table_info(chat_sessions)")}:
        try:
            conn.execute("ALTER TABLE chat_sessions ADD COLUMN has_work INTEGER")
        except sqlite3.OperationalError:
            # Another process opening the file added it first.
            if "has_work" not in {column[1] for column in conn.execute("PRAGMA table_info(chat_sessions)")}:
                raise
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_sessions_work ON chat_sessions (session_id) WHERE has_work IS NOT 0"
    )
    conn.commit()


def database(path: Path | str, synchronous: str = "NORMAL") -> _Database:
    """The process's one connection to the SQLite file at ``path`` (created with the schema if absent)."""
    mode = synchronous.upper()
    if mode not in ("NORMAL", "FULL"):
        raise ValueError(f"synchronous must be 'NORMAL' or 'FULL', got {synchronous!r}")
    key = (str(Path(path).resolve()), mode)
    with _DATABASES_LOCK:
        db = _DATABASES.get(key)
    if db is not None and db.alive():  # not alive: the file was deleted under us, don't write to the old inode
        return db
    # Opened outside the registry's lock, which every database of the process goes through: a file that
    # cannot be opened now (a stalled mount) holds up its own callers, not those of every other file.
    opened = _Database(Path(path), mode)
    with _DATABASES_LOCK:
        ident = opened._file if opened._file is not None else key[0]
        order = _ORDERS.get(ident)
        if order is None:
            order = _ORDERS[ident] = _FileOrder()
        opened.order, opened.writing, opened.writing_lock = order, order.writing, order.lock
        current = _DATABASES.get(key)
        if current is not None and current is not db and current.alive():
            # Another thread opened it meanwhile: one connection per file, theirs. Only while theirs is
            # still to the file at the path: one that went stale since (the file deleted under it) is
            # replaced by this one, as a stale one always is.
            opened.conn.close()
            return current
        _DATABASES[key] = opened
        return opened


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

        # Commits are not in snapshot order: each write is on a worker thread of its own, whichever takes the
        # connection first goes first, and the write of a save whose caller was cancelled still happens. The
        # number tells an older snapshot from a newer one then. It is taken before the run is read (its spans
        # and its parent): a save numbered after this one read the run after it. The lock is held for the
        # number alone, not for the read, which other runs of the file never wait for.
        with db.writing_lock:
            writes = db.writing.get(run_id)
            if writes is None:
                writes = db.writing[run_id] = _Writes()
            writes.pending += 1
            snapshot = writes.snapshots = writes.snapshots + 1

        def over() -> None:
            """This write can no longer commit. Called with ``db.writing_lock`` held."""
            writes.pending -= 1
            if not writes.pending and db.writing.get(run_id) is writes:
                del db.writing[run_id]

        try:
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
        except BaseException:
            with db.writing_lock:
                over()  # nothing to write: this number orders nothing
            raise
        stored_at = int(time.time() * 1000)
        # An ended run also gets the whole trace in runs.spans, as before: one write per run
        # keeps readers of that column working. While it runs, only its span rows are current.
        blob = "[" + ",".join(row[3] for row in rows) + "]" if ended else ""
        began = dropped = False

        def _write() -> None:
            nonlocal began
            with db.writing_lock:
                if dropped:
                    return
                began = True
            try:
                # This connection's lock, then the file's commit lock: another connection to the file (a provider
                # with the other ``synchronous``) compares and commits the same runs. Nothing holds the commit lock
                # while it asks for a connection's, so the two cannot wait for each other.
                with db.lock, db.order.commit:
                    if writes.committed > snapshot:
                        # A newer snapshot of this run is stored. It holds everything this one does (each
                        # span as it was then or later; a span this one has and that one left out was
                        # finished and stored before), so this one would only take the run back: the root's
                        # end over the late output of background work, or a run still going over one that
                        # ended.
                        return
                    with db.conn:  # one transaction: committed whole, or rolled back on any error
                        db.conn.execute(
                            """
                            INSERT INTO runs (run_id, parent_id, spans, stored_at) VALUES (?, ?, ?, ?)
                            ON CONFLICT (run_id) DO UPDATE SET parent_id = excluded.parent_id,
                                spans = excluded.spans, stored_at = excluded.stored_at
                            """,
                            (run_id, parent_id, blob, stored_at),
                        )
                        db.conn.executemany(
                            "INSERT OR REPLACE INTO spans (run_id, call_id, seq, span) VALUES (?, ?, ?, ?)", rows
                        )
                    writes.committed = snapshot
            finally:
                with db.writing_lock:
                    over()

        try:
            await asyncio.to_thread(_write)
        except BaseException:
            # The caller was cancelled, or the write never reached a worker. One a worker has begun goes on
            # and stays counted until it is over. One it has not begun is dropped here: what orders the run's
            # writes is kept until each is over, and nothing says when a worker would get to this one.
            with db.writing_lock:
                if not began:
                    dropped = True
                    over()
            raise
        # Only once committed: a save that failed leaves them for the next one to write.
        if ended:
            # Later saves (background work outliving the root) rewrite everything again.
            state["stored"].clear()
        else:
            state["stored"].update(finished)
