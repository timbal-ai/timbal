"""Durable chat sessions: what a :class:`~timbal.core.chat_session.ChatSession` needs to come back after a restart.

The conversation itself is in the traces. A snapshot holds the session layer on top of them: the head run,
the messages and wakes no turn has taken yet, a turn that was running, whether the session is parked. One
mutable row per session, written ahead of every change (a message is durable before its receipt goes out).

A lease (owner + expiry, renewed while the session is open) keeps two processes from serving one session,
and every save is a compare-and-swap on ``version``: an owner that lost its lease cannot overwrite the new one.
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

from uuid_extensions import uuid7

__all__ = [
    "PROCESS_OWNER",
    "ChatSessionStore",
    "InMemoryChatSessionStore",
    "SessionLeased",
    "SessionSnapshot",
    "SnapshotConflict",
    "SqliteChatSessionStore",
]

PROCESS_OWNER = f"{socket.gethostname()}:{os.getpid()}:{uuid7(as_type='hex')}"
"""This process, as a lease owner: unique across restarts of the same pid."""


def _holder_alive(owner: str | None) -> bool:
    """False for a lease held by a process of this host that no longer exists (``kill -9``): its lease would
    otherwise keep the session from the process restarted after it until it expires."""
    if not owner:
        return False
    host, _, rest = owner.partition(":")
    pid = rest.partition(":")[0]
    if host != socket.gethostname() or not pid.isdigit():
        return True  # another host's: only expiry tells
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, another user's
    return True


class SessionLeased(RuntimeError):
    """Another live process holds the session."""


class SnapshotConflict(RuntimeError):
    """The stored snapshot is not the version this owner last wrote: another owner took the session."""


@dataclass
class SessionSnapshot:
    session_id: str
    agent: str
    """The agent's name: a session is only restored for the agent it belongs to."""
    config: dict[str, Any] = field(default_factory=dict)
    head: str | None = None
    root_run_id: str | None = None
    """The thread's root run, once known (:attr:`ChatSession.root_run_id`); absent from older snapshots."""
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
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"), default=str)

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
    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None) -> int:
        """Write ``snapshot`` if the stored version is ``expect_version`` (None: the row must not exist).
        Returns the new version; raises :class:`SnapshotConflict` otherwise."""

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...

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

    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None) -> int:
        row = self._rows.get(snapshot.session_id)
        current = row[0].version if row is not None else None
        if current != expect_version:
            raise SnapshotConflict(
                f"session {snapshot.session_id}: stored version {current}, expected {expect_version}"
            )
        stored = SessionSnapshot.from_json(snapshot.to_json())
        stored.version = (current or 0) + 1
        owner, until = (row[1], row[2]) if row is not None else (None, 0.0)
        self._rows[snapshot.session_id] = (stored, owner, until)
        return stored.version

    async def delete(self, session_id: str) -> None:
        self._rows.pop(session_id, None)

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

    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None) -> int:
        new_version = (expect_version or 0) + 1
        body = SessionSnapshot.from_json(snapshot.to_json())
        body.version = new_version
        raw, now = body.to_json(), int(time.time() * 1000)

        def write(conn):
            if expect_version is None:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO chat_sessions (session_id, agent, snapshot, version, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (snapshot.session_id, snapshot.agent, raw, new_version, now),
                )
            else:
                cursor = conn.execute(
                    "UPDATE chat_sessions SET snapshot = ?, version = ?, updated_at = ? "
                    "WHERE session_id = ? AND version = ?",
                    (raw, new_version, now, snapshot.session_id, expect_version),
                )
            conn.commit()
            return cursor.rowcount

        if await self._run(write) != 1:
            raise SnapshotConflict(f"session {snapshot.session_id}: not at version {expect_version}")
        return new_version

    async def delete(self, session_id: str) -> None:
        def write(conn):
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            conn.commit()

        await self._run(write)

    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        def query(conn):
            return conn.execute("SELECT session_id, snapshot FROM chat_sessions").fetchall()

        ids = []
        for session_id, raw in await self._run(query):
            snapshot = SessionSnapshot.from_json(raw)
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
