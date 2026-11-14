"""Durable chat sessions: what a :class:`~timbal.core.chat_session.ChatSession` needs to come back after a restart.

The conversation itself is in the traces. A snapshot holds the session layer on top of them: the head run,
the messages and wakes no turn has taken yet, a turn that was running, whether the session is parked. One
mutable row per session, written ahead of every change (a message is durable before its receipt goes out).

A lease (owner + expiry, renewed while the session is open) keeps two processes from serving one session,
and every save is a compare-and-swap on ``version`` that also checks the lease: an owner whose lease another
process took (it expired, or renewing it failed) cannot write, so the new owner is never the one to give up.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog
from uuid_extensions import uuid7

logger = structlog.get_logger("timbal.state.chat_store")

__all__ = [  # ``PROCESS_OWNER`` is still readable (module __getattr__), for code written before process_owner()
    "process_owner",
    "ChatSessionStore",
    "InMemoryChatSessionStore",
    "SessionLeased",
    "SessionNotStored",
    "SessionSnapshot",
    "SnapshotConflict",
    "SqliteChatSessionStore",
    "storable",
]

_OWNER: tuple[int, str] | None = None


def process_owner() -> str:
    """This process, as a lease owner: ``host:pid:uuid``, unique across restarts of the same pid.

    Resolved when it is used, per pid: a server that imports this module and then forks its workers must
    not hand every child the parent's identity (each would renew, and take, the others' leases).
    """
    global _OWNER
    pid = os.getpid()
    if _OWNER is None or _OWNER[0] != pid:
        _OWNER = (pid, f"{socket.gethostname()}:{pid}:{uuid7(as_type='hex')}")
    return _OWNER[1]


def __getattr__(name: str) -> Any:
    # ``chat_store.PROCESS_OWNER`` still reads as this process's owner, at the time it is read. Importing the
    # name copies the value of that moment: after a fork it is the parent's. Call process_owner() instead.
    if name == "PROCESS_OWNER":
        return process_owner()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _holder_alive(owner: str | None) -> bool:
    """False for a lease held by a process of this host that no longer exists (``kill -9``): its lease would
    otherwise keep the session from the process restarted after it until it expires."""
    if not owner:
        return False
    host, _, rest = owner.partition(":")
    pid = rest.partition(":")[0]
    if host != socket.gethostname() or not pid.isdigit():
        return True  # another host's: only expiry tells
    return _pid_alive(int(pid))


def _pid_alive(pid: int) -> bool:
    """Whether a process of this host exists. Errs on "alive": a wrong "dead" lets two processes serve a session.

    Never ``os.kill(pid, 0)`` on Windows: there any signal but the console ones *terminates* the process.
    """
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # PermissionError: exists, another user's
    return True


def _windows_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    process_query_limited_information, still_active = 0x1000, 259
    error_invalid_parameter = 87  # what OpenProcess answers for a pid no process has
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return ctypes.get_last_error() != error_invalid_parameter  # access denied: exists
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        # An exited process whose handle someone still holds keeps its pid: its exit code tells.
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


class SessionLeased(RuntimeError):
    """Another live process holds the session."""


class SessionNotStored(RuntimeError):
    """The session's latest state could not be written (the store failed): a change made since the last write
    would not survive a restart."""


class SnapshotConflict(RuntimeError):
    """The stored snapshot is not the version this owner last wrote: another owner took the session."""


def _encode(value: Any) -> str:
    """The one JSON encoding a snapshot is written with. No ``default=`` (a value JSON cannot carry is not
    written as its ``str()``) and no NaN or Infinity (Python would write them; no other JSON reader takes them)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def storable(value: Any) -> bool:
    """Whether a snapshot can hold ``value`` as it is.

    The test a session applies to what its callers hand it (answers, turn arguments, a prompt) before it keeps
    it. It is the encoder :meth:`SessionSnapshot.to_json` writes with, so the two cannot disagree: a value this
    lets through never fails a save, and with it every save after it.
    """
    try:
        _encode(value)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


@dataclass
class SessionSnapshot:
    session_id: str
    agent: str
    """The agent's name: a session is only restored for the agent it belongs to."""
    config: dict[str, Any] = field(default_factory=dict)
    head: str | None = None
    turns: int = 0
    last_status: dict[str, Any] | None = None
    parked: bool = False
    last_prompt: Any = None
    """The dumped prompt of the last turn: a resume of a parked turn passes it again (the agent drops it)."""
    last_options: dict[str, Any] = field(default_factory=dict)
    in_flight: dict[str, Any] | None = None
    """The turn running at save time: ``run_id``, ``parent_id``, ``trigger``, ``prompt`` (dumped), ``options``,
    ``events`` (the messages it took). Cleared when it finishes; still set after a crash."""
    pending: list[dict[str, Any]] = field(default_factory=list)
    """Events no turn has taken yet (user messages, app/client events): ids and timestamps kept."""
    wakes: list[dict[str, Any]] = field(default_factory=list)
    resume: dict[str, Any] | None = None
    """Answers to a parked turn sent but not yet taken by a turn."""
    log_cursor: int = 0
    closed: bool = False
    version: int = 0

    def to_json(self) -> str:
        """Raises for a value JSON cannot carry (TypeError; ValueError for NaN and Infinity, which Python would
        write and no other JSON reader takes). Not written as its ``str()``: read back, that is a different
        value (an approval that is no longer one), and nothing would say so. The session checks what callers
        hand it before it takes it (answers, turn arguments), so a save does not get here with one; if one
        does, the save fails, is reported and retried, and what it held is never acknowledged as stored."""
        return _encode(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> SessionSnapshot:
        data = json.loads(raw)
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})

    def has_work(self) -> bool:
        """Something would run without a client asking: restore it at startup, not on first use."""
        return not self.closed and bool(self.in_flight or self.pending or self.wakes or self.resume)


class ChatSessionStore(ABC):
    """Where snapshots live. Implementations must make ``save`` and ``acquire_lease`` atomic."""

    lease_ttl_s: float = 30.0

    @abstractmethod
    async def load(self, session_id: str) -> SessionSnapshot | None: ...

    @abstractmethod
    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None, owner: str | None = None) -> int:
        """Write ``snapshot`` if the stored version is ``expect_version`` (None: the row must not exist) and, with
        ``owner``, no other owner holds the lease (expired or not: one that took it may not have written yet).
        A new row is created with ``owner``'s lease, in the same write: a session never exists unowned.
        Returns the new version; raises :class:`SnapshotConflict` otherwise."""

    @abstractmethod
    async def delete(self, session_id: str, *, expect_version: int | None = None, owner: str | None = None) -> None:
        """Remove the session. With ``expect_version`` and/or ``owner``: only while it is still at that version
        and no other owner holds its lease. Raises :class:`SnapshotConflict` otherwise (another process took
        the session and it is theirs now); a row already gone is not an error."""

    @abstractmethod
    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        """Open sessions (``with_work``: only those :meth:`SessionSnapshot.has_work` would restore eagerly)."""

    @abstractmethod
    async def acquire_lease(self, session_id: str, owner: str) -> None:
        """Take or renew the lease. Raises :class:`SessionLeased` while another owner's has not expired."""

    @abstractmethod
    async def release_lease(self, session_id: str, owner: str) -> None: ...


class InMemoryChatSessionStore(ChatSessionStore):
    """Process-local (the default): what sessions did before they were durable, through the same interface."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[SessionSnapshot, str | None, float]] = {}

    async def load(self, session_id: str) -> SessionSnapshot | None:
        row = self._rows.get(session_id)
        return SessionSnapshot.from_json(row[0].to_json()) if row is not None else None

    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None, owner: str | None = None) -> int:
        row = self._rows.get(snapshot.session_id)
        current = row[0].version if row is not None else None
        if current != expect_version:
            raise SnapshotConflict(
                f"session {snapshot.session_id}: stored version {current}, expected {expect_version}"
            )
        if owner is not None and row is not None and row[1] not in (None, owner):
            raise SnapshotConflict(f"session {snapshot.session_id}: leased by {row[1]}")
        stored = SessionSnapshot.from_json(snapshot.to_json())
        stored.version = (current or 0) + 1
        if row is not None:
            holder, until = row[1], row[2]
        else:
            holder, until = (owner, time.time() + self.lease_ttl_s) if owner is not None else (None, 0.0)
        self._rows[snapshot.session_id] = (stored, holder, until)
        return stored.version

    async def delete(self, session_id: str, *, expect_version: int | None = None, owner: str | None = None) -> None:
        row = self._rows.get(session_id)
        if row is None:
            return
        if expect_version is not None and row[0].version != expect_version:
            raise SnapshotConflict(f"session {session_id}: stored version {row[0].version}, expected {expect_version}")
        if owner is not None and row[1] not in (None, owner):
            raise SnapshotConflict(f"session {session_id}: leased by {row[1]}")
        del self._rows[session_id]

    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        return [
            sid for sid, (snap, _, _) in self._rows.items() if not snap.closed and (not with_work or snap.has_work())
        ]

    async def acquire_lease(self, session_id: str, owner: str) -> None:
        row = self._rows.get(session_id)
        if row is None:
            return
        snap, holder, until = row
        if holder not in (None, owner) and until > time.time() and _holder_alive(holder):
            raise SessionLeased(f"session {session_id} is held by {holder}")
        self._rows[session_id] = (snap, owner, time.time() + self.lease_ttl_s)

    async def release_lease(self, session_id: str, owner: str) -> None:
        row = self._rows.get(session_id)
        if row is not None and row[1] == owner:
            self._rows[session_id] = (row[0], None, 0.0)


class SqliteChatSessionStore(ChatSessionStore):
    """Snapshots in the ``chat_sessions`` table of a SQLite file, next to
    :class:`~timbal.state.tracing.providers.SqliteTracingProvider`'s runs (same connection when it is the
    same file)."""

    def __init__(self, path: Path | str, *, synchronous: str = "NORMAL") -> None:
        self.path = Path(path)
        self.synchronous = synchronous

    def _db(self):
        from .tracing.providers.sqlite import database

        return database(self.path, self.synchronous)

    async def _run(self, fn):
        def locked():
            db = self._db()
            with db.lock:
                return fn(db.conn)

        return await asyncio.to_thread(locked)

    async def load(self, session_id: str) -> SessionSnapshot | None:
        def query(conn):
            return conn.execute(
                "SELECT snapshot, version FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

        row = await self._run(query)
        if row is None:
            return None
        snapshot = SessionSnapshot.from_json(row[0])
        snapshot.version = row[1]
        return snapshot

    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None, owner: str | None = None) -> int:
        new_version = (expect_version or 0) + 1
        body = SessionSnapshot.from_json(snapshot.to_json())
        body.version = new_version
        raw, now = body.to_json(), int(time.time() * 1000)
        lease_until = time.time() + self.lease_ttl_s if owner is not None else None

        def write(conn):
            if expect_version is None:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO chat_sessions "
                    "(session_id, agent, snapshot, version, updated_at, lease_owner, lease_until) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (snapshot.session_id, snapshot.agent, raw, new_version, now, owner, lease_until),
                )
            elif owner is None:
                cursor = conn.execute(
                    "UPDATE chat_sessions SET snapshot = ?, version = ?, updated_at = ? "
                    "WHERE session_id = ? AND version = ?",
                    (raw, new_version, now, snapshot.session_id, expect_version),
                )
            else:
                cursor = conn.execute(
                    "UPDATE chat_sessions SET snapshot = ?, version = ?, updated_at = ? "
                    "WHERE session_id = ? AND version = ? AND (lease_owner IS NULL OR lease_owner = ?)",
                    (raw, new_version, now, snapshot.session_id, expect_version, owner),
                )
            conn.commit()
            return cursor.rowcount

        if await self._run(write) != 1:
            raise SnapshotConflict(
                f"session {snapshot.session_id}: not at version {expect_version}"
                + (f", or leased by another owner than {owner}" if owner is not None else "")
            )
        return new_version

    async def delete(self, session_id: str, *, expect_version: int | None = None, owner: str | None = None) -> None:
        clauses, params = ["session_id = ?"], [session_id]
        if expect_version is not None:
            clauses.append("version = ?")
            params.append(expect_version)
        if owner is not None:
            clauses.append("(lease_owner IS NULL OR lease_owner = ?)")
            params.append(owner)

        def write(conn):
            # One statement: the check and the delete cannot be split by another process's write.
            cursor = conn.execute(f"DELETE FROM chat_sessions WHERE {' AND '.join(clauses)}", params)
            gone = cursor.rowcount == 1 or (
                conn.execute("SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone() is None
            )
            conn.commit()
            return gone

        if not await self._run(write):
            raise SnapshotConflict(
                f"session {session_id}: not at version {expect_version}, or leased by another owner than {owner}"
            )

    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        def query(conn):
            return conn.execute("SELECT session_id, snapshot FROM chat_sessions").fetchall()

        ids = []
        for session_id, raw in await self._run(query):
            try:
                snapshot = SessionSnapshot.from_json(raw)
            except Exception as e:
                # One row that cannot be read (cut short, written by something else) must not hide every
                # session next to it, at startup and on each later look.
                logger.error("chat_session_snapshot_unreadable", session_id=session_id, error=str(e))
                continue
            if not snapshot.closed and (not with_work or snapshot.has_work()):
                ids.append(session_id)
        return ids

    async def acquire_lease(self, session_id: str, owner: str) -> None:
        now, until = time.time(), time.time() + self.lease_ttl_s

        def write(conn):
            cursor = conn.execute(
                "UPDATE chat_sessions SET lease_owner = ?, lease_until = ? "
                "WHERE session_id = ? AND (lease_owner IS NULL OR lease_owner = ? OR lease_until < ?)",
                (owner, until, session_id, owner, now),
            )
            if cursor.rowcount == 1:
                conn.commit()
                return None
            held = conn.execute("SELECT lease_owner FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
            if held is not None and not _holder_alive(held[0]):
                # Held by a dead process of this host: take it, unless someone else took it meanwhile.
                cursor = conn.execute(
                    "UPDATE chat_sessions SET lease_owner = ?, lease_until = ? WHERE session_id = ? AND lease_owner = ?",
                    (owner, until, session_id, held[0]),
                )
                if cursor.rowcount == 1:
                    conn.commit()
                    return None
            conn.commit()
            return held

        held = await self._run(write)
        if held is not None:
            raise SessionLeased(f"session {session_id} is held by {held[0]}")

    async def release_lease(self, session_id: str, owner: str) -> None:
        def write(conn):
            conn.execute(
                "UPDATE chat_sessions SET lease_owner = NULL, lease_until = NULL "
                "WHERE session_id = ? AND lease_owner = ?",
                (session_id, owner),
            )
            conn.commit()

        await self._run(write)
