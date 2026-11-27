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

    def __init__(self, path: Path | str, synchronous: str, order: _FileOrder | None = None) -> None:
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
        return _file_at(self.path)

    def alive(self) -> bool:
        """Whether the path still names the file this connection opened.

        Not just whether a file is there: one deleted and written again (by another connection, by a
        restore) is a different file, and this connection would go on writing to the one that is gone.
        """
        return self.names(self.path)

    def names(self, path: Path | str) -> bool:
        """Whether ``path`` names the file this connection opened, as :meth:`alive` asks of its own path."""
        return self._file is not None and _file_at(path) == self._file

    def run_state(self, run_id: str) -> dict[str, Any]:
        state = self.runs.get(run_id)
        if state is None:
            state = self.runs[run_id] = {"stored": set(), "seq": {}}
            while len(self.runs) > _RUN_STATE_MAX:
                self.runs.popitem(last=False)
        else:
            self.runs.move_to_end(run_id)
        return state


def _file_at(path: Path | str) -> tuple[int, int] | None:
    """The file ``path`` names now (its device and inode), or None when there is none."""
    try:
        found = os.stat(path)
    except OSError:
        return None
    return (found.st_dev, found.st_ino)


# Per spelling of a path, as given (relative, through a link, in another case, with a trailing slash): the spelling
# as a path (``str(Path(spelling))``, which drops that slash, for a stat) and Path.resolve()'s answer, which
# database() keys its connection by and opens it by. Resolving reads the file system, a call per component and a
# stat, and on Windows it opens the file (~250 us); every store and trace call asks for its database. Made once
# per spelling, and again whenever the spelling no longer names the file of the connection the remembered key has
# (the working directory changed under a relative path, a link was moved, the file was deleted): one stat per call,
# which database() made anyway to tell a file deleted and written again. A spelling a stat cannot follow (through
# a directory that is not there and back out of it, ``missing/../traces.db``, which SQLite and Path.resolve() both
# read as ``traces.db``) is resolved on every call, as before. A remembered key can outlive the path it was
# (a directory renamed with the file in it): the connection under it is still to the file the name names, which
# is all a connection needs; the path's own key then gets a second connection to that file, sharing its
# _FileOrder, as two synchronous modes or two cases of one path do. Not for chat_store._store_thread(), which
# resolves on every call: a store's calls are ordered by the path's key when the call is made, and no stat tells
# that a path came to resolve elsewhere while the file stayed the same. Plain data, so a forked child keeps it;
# thread safe as a dict.
_RESOLVED: dict[str, tuple[str, str]] = {}
_RESOLVED_MAX = 4096  # spellings; past it they are all resolved again


def _resolve(spelling: str) -> tuple[str, str]:
    target = Path(spelling)
    found = (str(target), str(target.resolve()))
    if len(_RESOLVED) >= _RESOLVED_MAX:
        _RESOLVED.clear()
    _RESOLVED[spelling] = found
    return found


def _sqlite_name(spelling: str) -> bool:
    """Whether SQLite gives ``spelling`` a meaning of its own instead of a file's: ``:memory:``, and a name that
    starts with ``file:``, which a build that reads URIs takes for one (``file:x?mode=memory``). Such a name is
    opened as given: resolved, it is an ordinary file of that name in the working directory. SQLite's own checks
    are exact and case-sensitive."""
    return spelling == ":memory:" or spelling.startswith("file:")


def _registered(spelling: str, mode: str) -> tuple[str, "_Database | None", bool]:
    """For a spelling of a path: the registry key (its resolved path), the connection registered under that key
    in ``mode`` (None when there is none), and whether that connection is to the file the spelling names now,
    which is the stat a call costs.

    A remembered key is used only while the spelling names that connection's file. Otherwise the spelling is
    resolved again, as it was on every call before, and the connection is checked against the resolved path,
    which is the path it was opened by: whatever the spelling names now is what the caller gets, and a path that
    came to name another file is never answered with the old file's connection. The registry is read without its
    lock (one dict lookup), so a thread that holds it while it opens another file holds up nothing here.
    """
    known = _RESOLVED.get(spelling)
    if known is not None:
        name, key = known
        db = _DATABASES.get((key, mode))
        if db is not None and db.names(name):
            return key, db, True
    _, key = _resolve(spelling)
    db = _DATABASES.get((key, mode))
    return key, db, db is not None and db.names(key)


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
    # The connection under the path's key, while the path names its file (a file deleted and written again is
    # another file: don't write to the old inode).
    spelling = os.fspath(path)
    resolved_path, db, names_it = _registered(spelling, mode)
    if names_it:
        return db
    key = (resolved_path, mode)
    # Opened by its resolved path, which every spelling of the path shares and a stat can follow (SQLite would
    # open ``missing/../traces.db`` as ``traces.db``, where a stat of that spelling fails): the connection, the
    # file it is to (``_file``, which finds the file's _FileOrder) and alive() go by one path. A name of SQLite's
    # own (``:memory:``, a ``file:`` URI) is opened as given, as it always was: it names no file to resolve to.
    # Outside the registry's lock, which every database of the process goes through: a file that cannot be
    # opened now (a stalled mount) holds up its own callers, not those of every other file.
    opened = _Database(spelling if _sqlite_name(spelling) else Path(resolved_path), mode)
    ident = opened._file if opened._file is not None else key[0]
    with _DATABASES_LOCK:
        order = _ORDERS.get(ident)
        if order is None:
            order = _ORDERS[ident] = _FileOrder()
    opened.order, opened.writing, opened.writing_lock = order, order.writing, order.lock
    stale = db  # what this call found under the key, not to the file at the path
    while True:
        with _DATABASES_LOCK:
            current = _DATABASES.get(key)
            if current is None or current is stale:
                _DATABASES[key] = opened
                return opened
        # Another thread opened it meanwhile: one connection per file, theirs. Only while theirs is still to
        # the file at the path: one that went stale since (the file deleted under it) is replaced by this
        # one, as a stale one always is. Checked, and this one closed, outside the registry's lock, which
        # every file's callers take: a stat or a close that is slow holds up only this file's.
        if current.names(resolved_path):
            opened.conn.close()
            return current
        stale = current


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
